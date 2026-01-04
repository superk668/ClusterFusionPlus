/**
 * ClusterFusion Pythia-2.8B Split Kernel Dispatch
 * 
 * This file dispatches the split version of the decoder layer kernel:
 *   Kernel 1: PythiaAttentionMlpUpKernel (LayerNorm → QKV → RoPE → Attention → Output Proj → Post-LN → MLP Up → GELU)
 *   Kernel 2: PythiaMlpDownKernel (MLP Down → Cluster Reduce → Final Residual)
 * 
 * Key difference from fused kernel:
 *   - No cooperative launch needed (no grid.sync())
 *   - Uses cudaLaunchKernelEx for cluster launch
 *   - Kernel launch boundary provides synchronization
 */

#include "kernel_attention.cuh"
#include "kernel_mlp.cuh"
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_decoder_layer_split_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor bias_qkv,
    torch::Tensor weight_o,
    torch::Tensor bias_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_up_bias,
    torch::Tensor mlp_down_weight,
    torch::Tensor mlp_down_bias,
    int64_t current_seq_len
) 
{
    // Set kernel attributes for both kernels
    cudaFuncSetAttribute(PythiaAttentionMlpUpKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(PythiaMlpDownKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(PythiaAttentionMlpUpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    cudaFuncSetAttribute(PythiaMlpDownKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({1, HIDDEN_DIM}, 0, options);
    torch::Tensor k = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    torch::Tensor v = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    
    // Intermediate buffers
    torch::Tensor mlp_intermediate = torch::empty({FFN_DIM}, options);
    torch::Tensor post_ln_buffer = torch::empty({HIDDEN_DIM}, options);
    torch::Tensor attn_output = torch::zeros({1, HIDDEN_DIM}, options);  // For attention output (atomicAdd target)
    
    // Get pointers
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());
    half* attn_output_ptr = reinterpret_cast<half*>(attn_output.data_ptr<at::Half>());

    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* bias_qkv_ptr = reinterpret_cast<half*>(bias_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* bias_o_ptr = reinterpret_cast<half*>(bias_o.data_ptr<at::Half>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());
    half* layernorm_weight_ptr = reinterpret_cast<half*>(layernorm_weight.data_ptr<at::Half>());
    half* layernorm_bias_ptr = reinterpret_cast<half*>(layernorm_bias.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());
    
    half* post_ln_weight_ptr = reinterpret_cast<half*>(post_ln_weight.data_ptr<at::Half>());
    half* post_ln_bias_ptr = reinterpret_cast<half*>(post_ln_bias.data_ptr<at::Half>());
    half* mlp_up_weight_ptr = reinterpret_cast<half*>(mlp_up_weight.data_ptr<at::Half>());
    half* mlp_up_bias_ptr = reinterpret_cast<half*>(mlp_up_bias.data_ptr<at::Half>());
    half* mlp_down_weight_ptr = reinterpret_cast<half*>(mlp_down_weight.data_ptr<at::Half>());
    half* mlp_down_bias_ptr = reinterpret_cast<half*>(mlp_down_bias.data_ptr<at::Half>());
    half* mlp_intermediate_ptr = reinterpret_cast<half*>(mlp_intermediate.data_ptr<at::Half>());
    half* post_ln_buffer_ptr = reinterpret_cast<half*>(post_ln_buffer.data_ptr<at::Half>());

    const uint32_t SEQ_LEN = static_cast<uint32_t>(current_seq_len);
    const uint32_t seq_len = SEQ_LEN;
    const uint32_t KV_DIM_PER_BLOCK = ((seq_len + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);
    
    // Create TensorMaps
    constexpr uint32_t rank = 2;
    
    // QKV weight tensor map
    CUtensorMap tensor_map_weight{};
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HEAD_NUM * HEAD_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_weight, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, weight_qkv_ptr, 
        size, stride, box_size, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // K cache tensor map
    CUtensorMap tensor_map_k_cache{};
    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_k_cache, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, k_cache_ptr,
        size_k_cache, stride_k_cache, box_size_k_cache, elem_stride_k_cache, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // V cache tensor map
    CUtensorMap tensor_map_v_cache{};
    uint64_t size_v_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_v_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_v_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_v_cache[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_v_cache, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, v_cache_ptr,
        size_v_cache, stride_v_cache, box_size_v_cache, elem_stride_v_cache, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Output projection weight tensor map
    CUtensorMap tensor_map_weight_o{};
    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    uint32_t elem_stride_weight_o[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_weight_o, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, weight_o_ptr,
        size_weight_o, stride_weight_o, box_size_weight_o, elem_stride_weight_o, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // MLP up weight tensor map
    CUtensorMap tensor_map_mlp_up{};
    uint64_t size_mlp_up[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_mlp_up[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_mlp_up[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_up[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_mlp_up, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, mlp_up_weight_ptr,
        size_mlp_up, stride_mlp_up, box_size_mlp_up, elem_stride_mlp_up, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // MLP down weight tensor map
    CUtensorMap tensor_map_mlp_down{};
    uint64_t size_mlp_down[rank] = {FFN_DIM, HIDDEN_DIM};
    uint64_t stride_mlp_down[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_mlp_down[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_down[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_mlp_down, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, mlp_down_weight_ptr,
        size_mlp_down, stride_mlp_down, box_size_mlp_down, elem_stride_mlp_down, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    
    // ==================== Kernel 1: Attention + MLP Up ====================
    // Uses cudaLaunchKernelEx with cluster attribute (NOT cooperative launch)
    cudaLaunchConfig_t config1 = {};
    config1.gridDim = grid;
    config1.blockDim = block;
    config1.dynamicSmemBytes = max_shmem_size;
    
    cudaLaunchAttribute attrs1[1];
    attrs1[0].id = cudaLaunchAttributeClusterDimension;
    attrs1[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs1[0].val.clusterDim.y = 1;
    attrs1[0].val.clusterDim.z = 1;
    config1.attrs = attrs1;
    config1.numAttrs = 1;
    
    void* kernel1_args[] = {
        &attn_output_ptr,       // attention output
        &k_ptr, &v_ptr,
        &mlp_intermediate_ptr,  // MLP up output
        &input_ptr,
        &layernorm_weight_ptr, &layernorm_bias_ptr,
        &bias_qkv_ptr, &bias_o_ptr,
        &cos_ptr, &sin_ptr,
        &k_cache_ptr, &v_cache_ptr,
        &post_ln_weight_ptr, &post_ln_bias_ptr,
        &mlp_up_bias_ptr, &post_ln_buffer_ptr,
        (void*)&tensor_map_weight, (void*)&tensor_map_k_cache,
        (void*)&tensor_map_v_cache, (void*)&tensor_map_weight_o,
        (void*)&tensor_map_mlp_up,
        (void*)&seq_len, (void*)&KV_DIM_PER_BLOCK
    };
    
    cudaLaunchKernelExC(&config1, (void*)PythiaAttentionMlpUpKernel, kernel1_args);
    
    // ==================== Kernel 2: MLP Down ====================
    // Kernel launch provides implicit synchronization (replaces grid.sync())
    cudaLaunchConfig_t config2 = {};
    config2.gridDim = grid;
    config2.blockDim = block;
    config2.dynamicSmemBytes = max_shmem_size;
    
    cudaLaunchAttribute attrs2[1];
    attrs2[0].id = cudaLaunchAttributeClusterDimension;
    attrs2[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs2[0].val.clusterDim.y = 1;
    attrs2[0].val.clusterDim.z = 1;
    config2.attrs = attrs2;
    config2.numAttrs = 1;
    
    void* kernel2_args[] = {
        &o_ptr,                 // final output
        &input_ptr,             // original input for residual
        &attn_output_ptr,       // attention output for residual
        &mlp_intermediate_ptr,  // MLP up output
        &mlp_down_bias_ptr,
        (void*)&tensor_map_mlp_down
    };
    
    cudaLaunchKernelExC(&config2, (void*)PythiaMlpDownKernel, kernel2_args);
    
    cudaDeviceSynchronize();
    return std::make_tuple(o, k, v);
}

// ==================== Attention-Only Kernel ====================
// Returns: (attn_output, mlp_intermediate, k_new, v_new)
// - attn_output: [1, HIDDEN_DIM] - attention output (before adding to residual)
// - mlp_intermediate: [FFN_DIM] - GELU(MLP_up(post_ln(x)))
// - k_new, v_new: [1, NUM_HEADS, HEAD_DIM] - new K/V for cache update
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_attention_only_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor bias_qkv,
    torch::Tensor weight_o,
    torch::Tensor bias_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_up_bias,
    int64_t current_seq_len
) 
{
    cudaFuncSetAttribute(PythiaAttentionMlpUpKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(PythiaAttentionMlpUpKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor k = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    torch::Tensor v = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    torch::Tensor mlp_intermediate = torch::empty({FFN_DIM}, options);
    torch::Tensor post_ln_buffer = torch::empty({HIDDEN_DIM}, options);
    torch::Tensor attn_output = torch::zeros({1, HIDDEN_DIM}, options);

    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());
    half* attn_output_ptr = reinterpret_cast<half*>(attn_output.data_ptr<at::Half>());
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* weight_qkv_ptr = reinterpret_cast<half*>(weight_qkv.data_ptr<at::Half>());
    half* bias_qkv_ptr = reinterpret_cast<half*>(bias_qkv.data_ptr<at::Half>());
    half* weight_o_ptr = reinterpret_cast<half*>(weight_o.data_ptr<at::Half>());
    half* bias_o_ptr = reinterpret_cast<half*>(bias_o.data_ptr<at::Half>());
    half* k_cache_ptr = reinterpret_cast<half*>(k_cache.data_ptr<at::Half>());
    half* v_cache_ptr = reinterpret_cast<half*>(v_cache.data_ptr<at::Half>());
    half* layernorm_weight_ptr = reinterpret_cast<half*>(layernorm_weight.data_ptr<at::Half>());
    half* layernorm_bias_ptr = reinterpret_cast<half*>(layernorm_bias.data_ptr<at::Half>());
    float* cos_ptr = reinterpret_cast<float*>(cos.data_ptr<float>());
    float* sin_ptr = reinterpret_cast<float*>(sin.data_ptr<float>());
    half* post_ln_weight_ptr = reinterpret_cast<half*>(post_ln_weight.data_ptr<at::Half>());
    half* post_ln_bias_ptr = reinterpret_cast<half*>(post_ln_bias.data_ptr<at::Half>());
    half* mlp_up_weight_ptr = reinterpret_cast<half*>(mlp_up_weight.data_ptr<at::Half>());
    half* mlp_up_bias_ptr = reinterpret_cast<half*>(mlp_up_bias.data_ptr<at::Half>());
    half* mlp_intermediate_ptr = reinterpret_cast<half*>(mlp_intermediate.data_ptr<at::Half>());
    half* post_ln_buffer_ptr = reinterpret_cast<half*>(post_ln_buffer.data_ptr<at::Half>());

    const uint32_t SEQ_LEN = static_cast<uint32_t>(current_seq_len);
    const uint32_t seq_len = SEQ_LEN;
    const uint32_t KV_DIM_PER_BLOCK = ((seq_len + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);

    // Create TensorMaps
    constexpr uint32_t rank = 2;
    
    CUtensorMap tensor_map_weight{};
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HEAD_NUM * HEAD_DIM};
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_weight, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, weight_qkv_ptr, 
        size, stride, box_size, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUtensorMap tensor_map_k_cache{};
    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    cuTensorMapEncodeTiled(&tensor_map_k_cache, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, k_cache_ptr,
        size_k_cache, stride_k_cache, box_size_k_cache, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUtensorMap tensor_map_v_cache{};
    cuTensorMapEncodeTiled(&tensor_map_v_cache, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, v_cache_ptr,
        size_k_cache, stride_k_cache, box_size_k_cache, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUtensorMap tensor_map_weight_o{};
    uint64_t size_weight_o[rank] = {HIDDEN_DIM, HIDDEN_DIM};
    uint64_t stride_weight_o[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_weight_o[rank] = {HEAD_DIM, TMA_LOAD_ONCE};
    cuTensorMapEncodeTiled(&tensor_map_weight_o, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, weight_o_ptr,
        size_weight_o, stride_weight_o, box_size_weight_o, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUtensorMap tensor_map_mlp_up{};
    uint64_t size_mlp_up[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_mlp_up[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_mlp_up[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    cuTensorMapEncodeTiled(&tensor_map_mlp_up, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, mlp_up_weight_ptr,
        size_mlp_up, stride_mlp_up, box_size_mlp_up, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = max_shmem_size;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    void* kernel_args[] = {
        &attn_output_ptr, &k_ptr, &v_ptr,
        &mlp_intermediate_ptr, &input_ptr,
        &layernorm_weight_ptr, &layernorm_bias_ptr,
        &bias_qkv_ptr, &bias_o_ptr,
        &cos_ptr, &sin_ptr,
        &k_cache_ptr, &v_cache_ptr,
        &post_ln_weight_ptr, &post_ln_bias_ptr,
        &mlp_up_bias_ptr, &post_ln_buffer_ptr,
        (void*)&tensor_map_weight, (void*)&tensor_map_k_cache,
        (void*)&tensor_map_v_cache, (void*)&tensor_map_weight_o,
        (void*)&tensor_map_mlp_up,
        (void*)&seq_len, (void*)&KV_DIM_PER_BLOCK
    };
    
    cudaLaunchKernelExC(&config, (void*)PythiaAttentionMlpUpKernel, kernel_args);
    cudaDeviceSynchronize();
    
    return std::make_tuple(attn_output, mlp_intermediate, k, v);
}

// ==================== MLP-Down-Only Kernel ====================
// Takes mlp_intermediate from attention_only, computes MLP down + residual
// Returns: output [1, HIDDEN_DIM]
torch::Tensor pythia_2b8_mlp_only_sm120(
    torch::Tensor input,           // [1, HIDDEN_DIM] - original input for residual
    torch::Tensor attn_output,     // [1, HIDDEN_DIM] - attention output for residual
    torch::Tensor mlp_intermediate, // [FFN_DIM] - GELU(MLP_up(post_ln(x)))
    torch::Tensor mlp_down_weight,
    torch::Tensor mlp_down_bias
) 
{
    cudaFuncSetAttribute(PythiaMlpDownKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(PythiaMlpDownKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor output = torch::full({1, HIDDEN_DIM}, 0, options);

    half* output_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());
    half* input_ptr = reinterpret_cast<half*>(input.data_ptr<at::Half>());
    half* attn_output_ptr = reinterpret_cast<half*>(attn_output.data_ptr<at::Half>());
    half* mlp_intermediate_ptr = reinterpret_cast<half*>(mlp_intermediate.data_ptr<at::Half>());
    half* mlp_down_weight_ptr = reinterpret_cast<half*>(mlp_down_weight.data_ptr<at::Half>());
    half* mlp_down_bias_ptr = reinterpret_cast<half*>(mlp_down_bias.data_ptr<at::Half>());

    // Create TensorMap for MLP down weight
    constexpr uint32_t rank = 2;
    CUtensorMap tensor_map_mlp_down{};
    uint64_t size_mlp_down[rank] = {FFN_DIM, HIDDEN_DIM};
    uint64_t stride_mlp_down[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_mlp_down[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride[rank] = {1, 1};
    cuTensorMapEncodeTiled(&tensor_map_mlp_down, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, mlp_down_weight_ptr,
        size_mlp_down, stride_mlp_down, box_size_mlp_down, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 grid(HEAD_NUM * CLUSTER_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = max_shmem_size;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_SIZE;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    void* kernel_args[] = {
        &output_ptr, &input_ptr, &attn_output_ptr,
        &mlp_intermediate_ptr, &mlp_down_bias_ptr,
        (void*)&tensor_map_mlp_down
    };
    
    cudaLaunchKernelExC(&config, (void*)PythiaMlpDownKernel, kernel_args);
    cudaDeviceSynchronize();
    
    return output;
}
