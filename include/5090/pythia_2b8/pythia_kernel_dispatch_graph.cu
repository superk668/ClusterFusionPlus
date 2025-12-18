/**
 * CUDA Graph-compatible dispatch for Pythia decoder layer.
 * 
 * Key differences from regular dispatch:
 * 1. TensorMaps are created with max_seq_len (not current seq_len)
 * 2. All buffers are pre-allocated
 * 3. No CPU-GPU synchronization within decode step
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cudaTypedefs.h>
#include <torch/extension.h>
#include <unordered_map>
#include "config.h"

// External declaration of kernel defined in pythia_kernel_dispatch.cu
extern __global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) PythiaDecoderLayerKernel(
    half* output,
    half* k_output,
    half* v_output,
    half* input,
    half* ln_weight,
    half* ln_bias,
    half* qkv_bias,
    half* o_bias,
    float* cos,
    float* sin,
    half* k_cache,
    half* v_cache,
    half* post_ln_weight,
    half* post_ln_bias,
    half* mlp_up_weight,
    half* mlp_up_bias,
    half* mlp_down_weight,
    half* mlp_down_bias,
    half* mlp_intermediate,
    half* post_ln_buffer,
    const __grid_constant__ CUtensorMap tensor_map,
    const __grid_constant__ CUtensorMap tensor_map_k_cache,
    const __grid_constant__ CUtensorMap tensor_map_v_cache,
    const __grid_constant__ CUtensorMap tensor_map_weight_o,
    const __grid_constant__ CUtensorMap tensor_map_mlp_up,
    const __grid_constant__ CUtensorMap tensor_map_mlp_down,
    const uint32_t SEQ_LEN,
    const uint32_t KV_DIM_PER_BLOCK
);

// Static context for CUDA Graph execution
struct PythiaGraphContext {
    // Tensor maps (created once with max_seq_len)
    CUtensorMap tensor_map_weight;
    CUtensorMap tensor_map_k_cache;
    CUtensorMap tensor_map_v_cache;
    CUtensorMap tensor_map_weight_o;
    CUtensorMap tensor_map_mlp_up;
    CUtensorMap tensor_map_mlp_down;
    
    // Static buffers
    torch::Tensor output;
    torch::Tensor k_out;
    torch::Tensor v_out;
    torch::Tensor mlp_intermediate;
    torch::Tensor post_ln_buffer;
    
    uint32_t max_shmem_size;
    bool initialized = false;
};

// Thread-local context (one per layer typically)
static std::unordered_map<int64_t, PythiaGraphContext> g_contexts;

/**
 * Create static context for CUDA Graph execution.
 * Call once per layer before graph capture.
 */
void pythia_2b8_create_graph_context_sm120(
    int64_t context_id,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_down_weight,
    int64_t max_seq_len
) {
    auto& ctx = g_contexts[context_id];
    auto device = k_cache.device();
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(device);
    
    // Allocate static buffers
    ctx.output = torch::zeros({1, HIDDEN_DIM}, options);
    ctx.k_out = torch::zeros({1, HEAD_NUM, HEAD_DIM}, options);
    ctx.v_out = torch::zeros({1, HEAD_NUM, HEAD_DIM}, options);
    ctx.mlp_intermediate = torch::zeros({FFN_DIM}, options);
    ctx.post_ln_buffer = torch::zeros({HIDDEN_DIM}, options);
    
    // Get raw pointers
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());
    half* mlp_up_weight_ptr = reinterpret_cast<half*>(mlp_up_weight.data_ptr<at::Half>());
    half* mlp_down_weight_ptr = reinterpret_cast<half*>(mlp_down_weight.data_ptr<at::Half>());
    
    // Create tensor maps with max_seq_len (allows reuse for any seq_len <= max)
    constexpr uint32_t rank = 2;
    
    // QKV weight tensor map
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HEAD_NUM * HEAD_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride[rank] = {1, 1};
    
    cuTensorMapEncodeTiled(
        &ctx.tensor_map_weight,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank, weight_qkv_ptr, size, stride, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // K cache tensor map - use max_seq_len
    uint64_t size_k_cache[rank] = {HIDDEN_DIM, (uint64_t)max_seq_len};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};

    cuTensorMapEncodeTiled(
        &ctx.tensor_map_k_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank, k_cache_ptr, size_k_cache, stride_k_cache, box_size_k_cache, elem_stride_k_cache,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // V cache tensor map
    cuTensorMapEncodeTiled(
        &ctx.tensor_map_v_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank, v_cache_ptr, size_k_cache, stride_k_cache, box_size_k_cache, elem_stride_k_cache,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // Output projection weight tensor map
    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_o[rank] = {1, 1};
    
    cuTensorMapEncodeTiled(
        &ctx.tensor_map_weight_o,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank, weight_o_ptr, size_weight_o, stride_weight_o, box_size_weight_o, elem_stride_weight_o,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // MLP up weight tensor map
    uint64_t size_mlp_up[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_mlp_up[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_mlp_up[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_up[rank] = {1, 1};
    
    cuTensorMapEncodeTiled(
        &ctx.tensor_map_mlp_up,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank, mlp_up_weight_ptr, size_mlp_up, stride_mlp_up, box_size_mlp_up, elem_stride_mlp_up,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // MLP down weight tensor map
    uint64_t size_mlp_down[rank] = {FFN_DIM, HIDDEN_DIM};
    uint64_t stride_mlp_down[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_mlp_down[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_down[rank] = {1, 1};
    
    cuTensorMapEncodeTiled(
        &ctx.tensor_map_mlp_down,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank, mlp_down_weight_ptr, size_mlp_down, stride_mlp_down, box_size_mlp_down, elem_stride_mlp_down,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // Set kernel attributes
    cudaFuncSetAttribute(PythiaDecoderLayerKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    ctx.max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(PythiaDecoderLayerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ctx.max_shmem_size);
    
    ctx.initialized = true;
}

/**
 * Execute one decode step using pre-created context.
 * This function can be captured in a CUDA Graph.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_graph_decode_step_sm120(
    int64_t context_id,
    torch::Tensor input,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor qkv_bias,
    torch::Tensor o_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_bias,
    torch::Tensor mlp_down_bias,
    int64_t current_seq_len
) {
    auto& ctx = g_contexts[context_id];
    
    if (!ctx.initialized) {
        throw std::runtime_error("Context not initialized. Call pythia_create_graph_context first.");
    }
    
    // CRITICAL: Zero output buffer before kernel (kernel uses atomicAdd)
    ctx.output.zero_();
    
    // Get pointers
    half* o_ptr = reinterpret_cast<half*>(ctx.output.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(ctx.k_out.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(ctx.v_out.data_ptr<at::Half>());
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* layernorm_weight_ptr = reinterpret_cast<half*>(layernorm_weight.data_ptr<at::Half>());
    half* layernorm_bias_ptr = reinterpret_cast<half*>(layernorm_bias.data_ptr<at::Half>());
    half* qkv_bias_ptr = reinterpret_cast<half*>(qkv_bias.data_ptr<at::Half>());
    half* o_bias_ptr = reinterpret_cast<half*>(o_bias.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());
    half* post_ln_weight_ptr = reinterpret_cast<half*>(post_ln_weight.data_ptr<at::Half>());
    half* post_ln_bias_ptr = reinterpret_cast<half*>(post_ln_bias.data_ptr<at::Half>());
    half* mlp_up_bias_ptr = reinterpret_cast<half*>(mlp_up_bias.data_ptr<at::Half>());
    half* mlp_down_bias_ptr = reinterpret_cast<half*>(mlp_down_bias.data_ptr<at::Half>());
    half* mlp_up_weight_ptr = nullptr;  // Already in TensorMap
    half* mlp_down_weight_ptr = nullptr;  // Already in TensorMap
    half* mlp_intermediate_ptr = reinterpret_cast<half*>(ctx.mlp_intermediate.data_ptr<at::Half>());
    half* post_ln_buffer_ptr = reinterpret_cast<half*>(ctx.post_ln_buffer.data_ptr<at::Half>());

    const uint32_t seq_len = static_cast<uint32_t>(current_seq_len);
    const uint32_t KV_DIM_PER_BLOCK = ((seq_len + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    void* kernel_args[] = {
        &o_ptr, &k_ptr, &v_ptr, &input_ptr,
        &layernorm_weight_ptr, &layernorm_bias_ptr, &qkv_bias_ptr, &o_bias_ptr,
        &cos_ptr, &sin_ptr, &k_cache_ptr, &v_cache_ptr,
        &post_ln_weight_ptr, &post_ln_bias_ptr,
        &mlp_up_weight_ptr, &mlp_up_bias_ptr, &mlp_down_weight_ptr, &mlp_down_bias_ptr,
        &mlp_intermediate_ptr, &post_ln_buffer_ptr,
        (void*)&ctx.tensor_map_weight, (void*)&ctx.tensor_map_k_cache, 
        (void*)&ctx.tensor_map_v_cache, (void*)&ctx.tensor_map_weight_o,
        (void*)&ctx.tensor_map_mlp_up, (void*)&ctx.tensor_map_mlp_down,
        (void*)&seq_len, (void*)&KV_DIM_PER_BLOCK
    };
    
    cudaLaunchCooperativeKernel(
        (void*)PythiaDecoderLayerKernel,
        grid, block,
        kernel_args,
        ctx.max_shmem_size
    );
    
    return std::make_tuple(ctx.output, ctx.k_out, ctx.v_out);
}

/**
 * Destroy context and free resources.
 */
void pythia_2b8_destroy_graph_context_sm120(int64_t context_id) {
    g_contexts.erase(context_id);
}
