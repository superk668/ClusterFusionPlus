#include "kernel.cuh"
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_6b9_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,      // [3 * num_heads * head_dim, hidden_dim] = [12288, 4096]
    torch::Tensor bias_qkv,        // [3 * num_heads * head_dim] = [12288]
    torch::Tensor weight_o,        // [hidden_dim, hidden_dim] = [4096, 4096]
    torch::Tensor bias_o,          // [hidden_dim]
    torch::Tensor k_cache,         // Full cache buffer [max_seq_len, hidden_dim]
    torch::Tensor v_cache,         // Full cache buffer [max_seq_len, hidden_dim]
    torch::Tensor layernorm_weight,  // [hidden_dim] = [4096]
    torch::Tensor layernorm_bias,    // [hidden_dim] = [4096]
    torch::Tensor cos,
    torch::Tensor sin,
    // MLP weights
    torch::Tensor post_ln_weight,    // [hidden_dim]
    torch::Tensor post_ln_bias,      // [hidden_dim]
    torch::Tensor mlp_up_weight,     // [ffn_dim, hidden_dim] = [16384, 4096]
    torch::Tensor mlp_up_bias,       // [ffn_dim]
    torch::Tensor mlp_down_weight,   // [hidden_dim, ffn_dim] = [4096, 16384]
    torch::Tensor mlp_down_bias,     // [hidden_dim]
    int64_t current_seq_len          // Current sequence length (before appending new token)
) 
{
    cudaFuncSetAttribute(Pythia6b9DecoderLayerKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    // Shared memory layout: weight(2*TMA*MAX) + local_qkv(3*HEAD) + input_shmem(DIM_PER_BLOCK) + reduction
    uint32_t max_shmem_size = 128 * sizeof(char) + (2 * TMA_LOAD_ONCE * MAX_SMEM_DIM + DIM_PER_BLOCK + 3 * HEAD_DIM) * sizeof(half) + DIM_BLOCK_REDUCE * sizeof(float);
    cudaFuncSetAttribute(Pythia6b9DecoderLayerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shmem_size);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, 0);
    torch::Tensor o = torch::full({1, HIDDEN_DIM}, 0, options);
    torch::Tensor k = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    torch::Tensor v = torch::full({1, HEAD_NUM, HEAD_DIM}, 0, options);
    
    // Intermediate buffer for MLP up projection output
    torch::Tensor mlp_intermediate = torch::empty({FFN_DIM}, options);
    // Buffer for post-attention LayerNorm output (input to MLP)
    torch::Tensor post_ln_buffer = torch::empty({HIDDEN_DIM}, options);
    
    half* o_ptr = reinterpret_cast<half*>(o.data_ptr<at::Half>());
    half* k_ptr = reinterpret_cast<half*>(k.data_ptr<at::Half>());
    half* v_ptr = reinterpret_cast<half*>(v.data_ptr<at::Half>());

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
    
    // MLP pointers
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
    const uint32_t max_cache_size = static_cast<uint32_t>(k_cache.size(0));
    const uint32_t KV_DIM_PER_BLOCK = ((seq_len + CLUSTER_SIZE - 1) / CLUSTER_SIZE + (TMA_LOAD_ONCE_ATTN - 1)) & ~(TMA_LOAD_ONCE_ATTN - 1);
    
    CUtensorMap tensor_map_weight{};
    CUtensorMap tensor_map_k_cache{};
    CUtensorMap tensor_map_v_cache{};
    CUtensorMap tensor_map_weight_o{};
    
    // QKV weight tensor map
    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {HIDDEN_DIM, 3 * HEAD_NUM * HEAD_DIM};  // {4096, 12288}
    uint64_t stride[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size[rank] = {TMA_LOAD_ONCE, HEAD_DIM};          // {64, 128}
    uint32_t elem_stride[rank] = {1, 1};
    
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_weight,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_qkv_ptr,                 
        size,                       
        stride,                     
        box_size,                   
        elem_stride,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // K cache tensor map
    uint64_t size_k_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_k_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_k_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_k_cache[rank] = {1, 1};

    CUresult res_k_cache = cuTensorMapEncodeTiled(
        &tensor_map_k_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        k_cache_ptr,                
        size_k_cache,                      
        stride_k_cache,                     
        box_size_k_cache,                   
        elem_stride_k_cache,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // V cache tensor map
    uint64_t size_v_cache[rank] = {HIDDEN_DIM, SEQ_LEN};
    uint64_t stride_v_cache[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_v_cache[rank] = {HEAD_DIM, TMA_LOAD_ONCE / 2};
    uint32_t elem_stride_v_cache[rank] = {1, 1};

    CUresult res_v_cache = cuTensorMapEncodeTiled(
        &tensor_map_v_cache,               
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                      
        v_cache_ptr,                
        size_v_cache,                      
        stride_v_cache,                     
        box_size_v_cache,                   
        elem_stride_v_cache,                
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
    
    CUresult res_weight_o = cuTensorMapEncodeTiled(
        &tensor_map_weight_o,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        weight_o_ptr,                 
        size_weight_o,                       
        stride_weight_o,                     
        box_size_weight_o,                   
        elem_stride_weight_o,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // MLP up weight tensor map
    CUtensorMap tensor_map_mlp_up{};
    uint64_t size_mlp_up[rank] = {HIDDEN_DIM, FFN_DIM};
    uint64_t stride_mlp_up[rank - 1] = {HIDDEN_DIM * sizeof(half)};
    uint32_t box_size_mlp_up[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_up[rank] = {1, 1};
    
    CUresult res_mlp_up = cuTensorMapEncodeTiled(
        &tensor_map_mlp_up,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        mlp_up_weight_ptr,                 
        size_mlp_up,                       
        stride_mlp_up,                     
        box_size_mlp_up,                   
        elem_stride_mlp_up,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // MLP down weight tensor map
    CUtensorMap tensor_map_mlp_down{};
    uint64_t size_mlp_down[rank] = {FFN_DIM, HIDDEN_DIM};
    uint64_t stride_mlp_down[rank - 1] = {FFN_DIM * sizeof(half)};
    uint32_t box_size_mlp_down[rank] = {TMA_LOAD_ONCE, HEAD_DIM};
    uint32_t elem_stride_mlp_down[rank] = {1, 1};
    
    CUresult res_mlp_down = cuTensorMapEncodeTiled(
        &tensor_map_mlp_down,                
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        rank,                       
        mlp_down_weight_ptr,                 
        size_mlp_down,                       
        stride_mlp_down,                     
        box_size_mlp_down,                   
        elem_stride_mlp_down,                
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    dim3 grid(HEAD_NUM * CLUSTER_SIZE); 
    dim3 block(BLOCK_SIZE);

    cudaDeviceSynchronize();
    
    // Use cooperative launch for grid.sync() support
    void* kernel_args[] = {
        &o_ptr, &k_ptr, &v_ptr, &input_ptr,
        &layernorm_weight_ptr, &layernorm_bias_ptr, &bias_qkv_ptr, &bias_o_ptr,
        &cos_ptr, &sin_ptr, &k_cache_ptr, &v_cache_ptr,
        &post_ln_weight_ptr, &post_ln_bias_ptr,
        &mlp_up_weight_ptr, &mlp_up_bias_ptr, &mlp_down_weight_ptr, &mlp_down_bias_ptr,
        &mlp_intermediate_ptr, &post_ln_buffer_ptr,
        (void*)&tensor_map_weight, (void*)&tensor_map_k_cache, 
        (void*)&tensor_map_v_cache, (void*)&tensor_map_weight_o,
        (void*)&tensor_map_mlp_up, (void*)&tensor_map_mlp_down,
        (void*)&seq_len, (void*)&KV_DIM_PER_BLOCK
    };
    
    cudaLaunchCooperativeKernel(
        (void*)Pythia6b9DecoderLayerKernel,
        grid, block,
        kernel_args,
        max_shmem_size
    );
    
    cudaDeviceSynchronize();
    return std::make_tuple(o, k, v);
}



