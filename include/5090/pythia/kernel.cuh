#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h> 
#include "../../dsm.cuh"
#include "config.h"
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// Pythia uses Neox-style RoPE with rotary_pct=0.25
#define NEOX_STYLE_ROPE

__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) PythiaDecoderLayerKernel(
    half* output,           // [1, hidden_dim]
    half* k_output,         // [1, num_heads, head_dim]
    half* v_output,         // [1, num_heads, head_dim]
    half* input,            // [1, hidden_dim]
    half* ln_weight,        // [hidden_dim] - LayerNorm weight
    half* ln_bias,          // [hidden_dim] - LayerNorm bias
    half* qkv_bias,         // [3 * num_heads * head_dim] - QKV projection bias
    half* o_bias,           // [hidden_dim] - Output projection bias
    float* cos,             // [head_dim]
    float* sin,             // [head_dim]
    half* k_cache,
    half* v_cache,
    const __grid_constant__ CUtensorMap tensor_map,
    const __grid_constant__ CUtensorMap tensor_map_k_cache,
    const __grid_constant__ CUtensorMap tensor_map_v_cache,
    const __grid_constant__ CUtensorMap tensor_map_weight_o,
    const uint32_t SEQ_LEN,
    const uint32_t KV_DIM_PER_BLOCK
) {
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank();
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id = tid % WARP_SIZE; 
    const uint32_t warp_id = tid / WARP_SIZE;
    const uint32_t tile_row = tid / NUM_THREAD_PER_ROW_2;
    const uint32_t tile_col = tid % NUM_THREAD_PER_ROW_2;

    // Init shared memory
    extern __shared__ uint8_t shmem_base[];
    half* weight = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_qkv = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;
    half* input_shmem = local_qkv + 3 * HEAD_DIM;
    float* reduction = reinterpret_cast<float*>(input_shmem + DIM_PER_BLOCK);

    __shared__ float cluster_local_sum, cluster_local_max, cluster_local_mean;

    // Init registers
    float local_sum = 0.0f, local_mean = 0.0f, eps = 1e-5f, var_rcp = 0.0f, tmp = 0.0f;
    float local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0f;
    float softmax_scale = __frsqrt_rn(HEAD_DIM) * 1.44269504088896340736f;
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    float q_rope, q_rope_1, k_rope, k_rope_1, cos_reg, sin_reg;
    uint32_t size;
    uint32_t src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];
    
    // Init barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[4];
    barrier::arrival_token token[4];
    __shared__ uint64_t barrier_mem;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_mem));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[2], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[3], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    block.sync();

    // Precompute indices
    uint input_idx = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    
    // For interleaved QKV: head_id's Q starts at row head_id * 3 * HEAD_DIM
    uint head_qkv_offset = head_id * QKV_HEAD_STRIDE;  // head_id * 240

    // ==================== LayerNorm ====================
    // LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    // Step 1: Compute mean
    local_sum = 0.0f;
    for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        for (int di = 0; di < 8; di++)
            local_sum += __half2float(reg_input[di]);
    }
    // Warp reduction for sum
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0) {
        reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < NUM_WARPS) 
        local_sum = reduction[tid];
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;
    cluster.sync();
    
    // Cluster reduction for mean
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    local_mean = cluster_local_sum / HIDDEN_DIM;
    if (tid == 0)
        cluster_local_mean = local_mean;
    cluster.sync();
    local_mean = cluster_local_mean;

    // Step 2: Compute variance
    local_sum = 0.0f;
    for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        for (int di = 0; di < 8; di++) {
            float val = __half2float(reg_input[di]) - local_mean;
            local_sum += val * val;
    }
    }
    // Warp reduction for variance
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    }
    if (lane_id == 0) {
        reduction[warp_id] = local_sum;
    }
    block.sync(); 
    if (tid < NUM_WARPS) 
        local_sum = reduction[tid];
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    } 
    if (tid == 0)
        cluster_local_sum = local_sum;
    cluster.sync();
    
    // Cluster reduction for variance
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    var_rcp = __frsqrt_rn(cluster_local_sum / HIDDEN_DIM + eps);

    // Step 3: Normalize and apply weight/bias
    for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) { 
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&ln_weight[cluster_block_st_id + d]);
        half __align__(16) reg_bias[8];
        *(uint4*)(&reg_bias[0]) = *(uint4*)(&ln_bias[cluster_block_st_id + d]);
        for (int i = 0; i < 8; i++) {
            float normalized = (__half2float(reg_input[i]) - local_mean) * var_rcp;
            reg_input[i] = __float2half(normalized * __half2float(reg_weight[i]) + __half2float(reg_bias[i]));
        }
        *(uint4*)(&input_shmem[d]) = *(uint4*)(&reg_input[0]);
    }
    block.sync();
    
    // ==================== QKV Projection with Interleaved Layout ====================
    // Interleaved: head i's Q at rows [i*240, i*240+80), K at [i*240+80, i*240+160), V at [i*240+160, i*240+240)
    // TMA coords: (in_dim_offset, out_dim_offset)
    
    // Compute Q for this head
    // NOTE: ALL threads must participate in barrier operations to avoid deadlock
    tmp = 0.0f;
    
    // Preload first block - all threads participate in barrier
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_block_st_id, head_qkv_offset, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_block_st_id + id * TMA_LOAD_ONCE, head_qkv_offset, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        // Only threads with valid weight_idx compute
        if (weight_idx < HEAD_DIM) {
            for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    if (weight_idx < HEAD_DIM) {
        for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
    if (lane_id % NUM_THREAD_PER_ROW == 0 && weight_idx < HEAD_DIM) {
        local_qkv[weight_idx] = __float2half(tmp);
    }
    block.sync();

    // Compute K for this head
    tmp = 0.0f;
    
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_block_st_id, head_qkv_offset + HEAD_DIM, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_block_st_id + id * TMA_LOAD_ONCE, head_qkv_offset + HEAD_DIM, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (weight_idx < HEAD_DIM) {
            for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    if (weight_idx < HEAD_DIM) {
        for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
    if (lane_id % NUM_THREAD_PER_ROW == 0 && weight_idx < HEAD_DIM) {
        local_qkv[HEAD_DIM + weight_idx] = __float2half(tmp);
    }
    block.sync();

    // Compute V for this head
    tmp = 0.0f;
    
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map, cluster_block_st_id, head_qkv_offset + 2 * HEAD_DIM, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map, cluster_block_st_id + id * TMA_LOAD_ONCE, head_qkv_offset + 2 * HEAD_DIM, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (weight_idx < HEAD_DIM) {
            for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) { 
                *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + (id - 1) * TMA_LOAD_ONCE + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    if (weight_idx < HEAD_DIM) {
        for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) { 
            *(uint4*)(&reg_input[0]) = *(uint4*)(&input_shmem[input_idx + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
    if (lane_id % NUM_THREAD_PER_ROW == 0 && weight_idx < HEAD_DIM) {
        local_qkv[2 * HEAD_DIM + weight_idx] = __float2half(tmp);
    }
    block.sync();

    // ==================== Cluster Reduce for QKV ====================
    size = (HEAD_DIM * 3) * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, local_qkv, weight);
    cluster.sync();

    // ==================== Add QKV Bias ====================
    // bias_qkv layout: interleaved [Q0_bias, K0_bias, V0_bias, Q1_bias, ...]
    if (tid < HEAD_DIM) {
        // Q bias
        float q_val = __half2float(local_qkv[tid]) + __half2float(qkv_bias[head_qkv_offset + tid]);
        local_qkv[tid] = __float2half(q_val);
        // K bias
        float k_val = __half2float(local_qkv[HEAD_DIM + tid]) + __half2float(qkv_bias[head_qkv_offset + HEAD_DIM + tid]);
        local_qkv[HEAD_DIM + tid] = __float2half(k_val);
        // V bias
        float v_val = __half2float(local_qkv[2 * HEAD_DIM + tid]) + __half2float(qkv_bias[head_qkv_offset + 2 * HEAD_DIM + tid]);
        local_qkv[2 * HEAD_DIM + tid] = __float2half(v_val);
    }
    block.sync();
    
    // ==================== RoPE (Neox-style, rotary_pct=0.25) ====================
    // Only first ROTARY_DIM (20) dimensions get RoPE
    if (tid < ROTARY_DIM) {
        q_rope = __half2float(local_qkv[tid]);
        k_rope = __half2float(local_qkv[HEAD_DIM + tid]);
        cos_reg = cos[tid];
        sin_reg = sin[tid];
        
        // Neox-style: rotate half
        if (tid < ROTARY_DIM / 2) {
            q_rope_1 = __half2float(local_qkv[ROTARY_DIM / 2 + tid]);
            k_rope_1 = __half2float(local_qkv[HEAD_DIM + ROTARY_DIM / 2 + tid]);
        } else {
            q_rope_1 = __half2float(local_qkv[tid - ROTARY_DIM / 2]);
            k_rope_1 = __half2float(local_qkv[HEAD_DIM + tid - ROTARY_DIM / 2]);
        }
    }
    block.sync();
    
    if (tid < ROTARY_DIM) {
        if (tid < ROTARY_DIM / 2) {
            local_qkv[tid] = __float2half(q_rope * cos_reg - q_rope_1 * sin_reg);
            local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg - k_rope_1 * sin_reg);
        } else {
            local_qkv[tid] = __float2half(q_rope * cos_reg + q_rope_1 * sin_reg);
            local_qkv[HEAD_DIM + tid] = __float2half(k_rope * cos_reg + k_rope_1 * sin_reg);
        }
    }
    // Dimensions from ROTARY_DIM to HEAD_DIM remain unchanged
    block.sync();

    // ==================== Write K, V back to cache ====================
    // Write new K, V directly to cache at position SEQ_LEN (append to cache)
    // k_cache layout: [max_seq_len, HIDDEN_DIM] = [max_seq_len, num_heads * head_dim]
    // For head_id, write to offset: SEQ_LEN * HIDDEN_DIM + head_id * HEAD_DIM
    cluster.sync();
    if (cluster_block_id == 0 && tid < HEAD_DIM) {
        uint32_t cache_offset = SEQ_LEN * HIDDEN_DIM + head_id * HEAD_DIM + tid;
        k_cache[cache_offset] = local_qkv[HEAD_DIM + tid];
        v_cache[cache_offset] = local_qkv[2 * HEAD_DIM + tid];
        
        // Also write to k_output, v_output for debugging/verification (optional)
        k_output[head_id * HEAD_DIM + tid] = local_qkv[HEAD_DIM + tid];
        v_output[head_id * HEAD_DIM + tid] = local_qkv[2 * HEAD_DIM + tid];
    }
    cluster.sync();

    // ==================== Flash Decoding ====================
    local_sum = 0.0f;
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    // Bounds check for HEAD_DIM=80: only load if input_idx_2 < HEAD_DIM
    if (input_idx_2 < HEAD_DIM) {
    *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
    } else {
        // Zero out reg_input for threads beyond HEAD_DIM
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            reg_input[i] = __float2half(0.0f);
        }
    }
    block.sync();

    // Preload kv_cache
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_k_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar[2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
        token[2] = bar[2].arrive();
    }

    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_k_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        pre_max = local_max;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            // TMA auto-fills OOB positions with 0, so no bounds check needed for loading
            if (input_idx_2 < HEAD_DIM) {
                *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            } else {
                for (int i = 0; i < NUM_PER_THREAD; i++) reg_weight[i] = __float2half(0.0f);
            }
            qk[j] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
            #pragma unroll
            for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
                qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
            }
            qk[j] = qk[j] * softmax_scale;
            local_max = max(local_max, qk[j]);
        }
        scale = ptx_exp2(pre_max - local_max);
        local_sum *= scale;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            if ((KV_DIM_PER_BLOCK * cluster_block_id + (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
                qk[j] = ptx_exp2(qk[j] - local_max);
                local_sum += qk[j];
            }
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            reg_reduce[j] = reg_reduce[j] * scale;
        }
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[2 + id % 2]);
            token[2 + id % 2] = cuda::device::barrier_arrive_tx(bar[2 + id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[2 + id % 2] = bar[2 + id % 2].arrive();
        }
        bar[2 + (id - 1) % 2].wait(std::move(token[2 + (id - 1) % 2]));
        for (int j = 0; j < DEC_TILE; j++) {
            if (input_idx_2 < HEAD_DIM) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
            } else {
                for (int i = 0; i < NUM_PER_THREAD; i++) reg_weight[i] = __float2half(0.0f);
            }
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
            }
        }
    }
    if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
        bar[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2].wait(std::move(token[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2]));
    } else {
        bar[0].wait(std::move(token[0]));
    }
    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        // TMA auto-fills OOB positions with 0, so no bounds check needed for loading
        if (input_idx_2 < HEAD_DIM) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        } else {
            for (int i = 0; i < NUM_PER_THREAD; i++) reg_weight[i] = __float2half(0.0f);
        }
        qk[j] = 0.0f;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            qk[j] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[j] += __shfl_xor_sync(0xffffffff, qk[j], mask);
        }
        qk[j] = qk[j] * softmax_scale;
        local_max = max(local_max, qk[j]);
    }
    scale = ptx_exp2(pre_max - local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        if ((KV_DIM_PER_BLOCK * cluster_block_id + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j) < SEQ_LEN) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        }
    }
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
        bar[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2 + 2].wait(std::move(token[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2 + 2]));
    } else {
        bar[2].wait(std::move(token[2]));
    }
    for (int j = 0; j < DEC_TILE; j++) {
        if (input_idx_2 < HEAD_DIM) {
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[((KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2) * TMA_LOAD_ONCE_NUM + TMA_LOAD_ONCE_NUM_ATTN + (weight_idx_2 + j) * HEAD_DIM + input_idx_2]);
        } else {
            for (int i = 0; i < NUM_PER_THREAD; i++) reg_weight[i] = __float2half(0.0f);
        }
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = reg_reduce[d] + qk[j] * __half2float(reg_weight[d]);
        }
    }
    block.sync();

    // Process KV of current token
    if (cluster_block_id == 0 && warp_id == 0) {
        if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
            pre_max = local_max;
            if (input_idx_2 < HEAD_DIM) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[HEAD_DIM + input_idx_2]); 
            } else {
                for (int i = 0; i < NUM_PER_THREAD; i++) reg_weight[i] = __float2half(0.0f);
            }
            qk[0] = 0.0f;
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                qk[0] += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_2 >> 1); mask > 0; mask >>= 1) {
            qk[0] += __shfl_xor_sync(0xffffffff, qk[0], mask);
        }
        if (lane_id / NUM_THREAD_PER_ROW_2 == 1) {
            qk[0] = qk[0] * softmax_scale;
            local_max = max(local_max, qk[0]); 
            scale = ptx_exp2(pre_max - local_max);
            local_sum *= scale;
            qk[0] = ptx_exp2(qk[0] - local_max);
            local_sum += qk[0];
            #pragma unroll
            for (int j = 0; j < NUM_PER_THREAD; j++) {
                reg_reduce[j] = reg_reduce[j] * scale;
            }
            if (input_idx_2 < HEAD_DIM) {
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_2]);
            } else {
                for (int i = 0; i < NUM_PER_THREAD; i++) reg_weight[i] = __float2half(0.0f);
            }
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                reg_reduce[d] = reg_reduce[d] + qk[0] * __half2float(reg_weight[d]);
            }
        }
    }
    block.sync();

    // Bounds check for writing to shared memory
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        if (tile_col * NUM_PER_THREAD + i < HEAD_DIM) {
        weight[tile_row * HEAD_DIM + tile_col * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
        }
    }
    if (lane_id % NUM_THREAD_PER_ROW_2 == 0) {
        reduction[tile_row * 2] = local_max;
        reduction[tile_row * 2 + 1] = local_sum;
    }
    block.sync();
    for(int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0, local_max = 0.0;
    #pragma unroll
    for(int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        // Bounds check for reading from shared memory
        if (tile_col * NUM_PER_THREAD < HEAD_DIM) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        } else {
            for (int i = 0; i < NUM_PER_THREAD; i++) reg_input[i] = __float2half(0.0f);
        }
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = ptx_exp2(m - local_max);
        s *= scale;
        local_sum = local_sum * ptx_exp2(pre_max - local_max) + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = reg_reduce[d] * ptx_exp2(pre_max - local_max) + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    pre_max = local_max;
    if(tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    // ClusterReduce: local_max
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_max = cluster_local_max;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_max, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            *dst_shmem = fmaxf(*dst_shmem, local_max);
        }
        cluster.sync();
    }
    scale = ptx_exp2(pre_max - cluster_local_max);
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if(tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();
    // ClusterReduce: local_sum
    for (int i = 1; i < cluster.num_blocks() - 1; i++) {
        if (tid == 0) {
            local_sum = cluster_local_sum;
            int dst_cta = (cluster_block_id + i) % cluster.num_blocks();
            dst_shmem = cluster.map_shared_rank(&cluster_local_sum, dst_cta);  
        }
        cluster.sync();
        if (tid == 0) {
            atomicAdd(dst_shmem, local_sum);
        }
        cluster.sync();
    }
    // Add epsilon to prevent division by zero when all exp values underflow
    float safe_sum = cluster_local_sum + 1e-10f;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(safe_sum);
    }
    // Only threads that cover valid HEAD_DIM indices write output
    if(tid < NUM_THREAD_PER_ROW_2 && tid * NUM_PER_THREAD < HEAD_DIM) {
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            if (tid * NUM_PER_THREAD + i < HEAD_DIM) {
            local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
            }
        }
    }
    block.sync();

    // ClusterReduce attention output
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[2 * HEAD_DIM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, HEAD_DIM, cluster_block_id,  
        src_addr, dst_addr, bar_ptr, 
        neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);
    cluster.sync();

    // ==================== Output Projection ====================
    // TMA coords: (in_dim_offset, out_dim_offset)
    // weight_o shape: [out_dim=hidden, in_dim=hidden]
    // We load from in_dim = head_id*HEAD_DIM, out_dim = cluster_block_st_id
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_weight_o, head_id * HEAD_DIM, cluster_block_st_id, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    for (int id = 1; id < DIM_PER_BLOCK / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_weight_o, head_id * HEAD_DIM, cluster_block_st_id + id * TMA_LOAD_ONCE, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        tmp = 0.0;
        for (int j = 0; j < HEAD_DIM; j += NUM_PER_ROW_3) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
            *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[(id - 1) % 2 * TMA_LOAD_ONCE_NUM + weight_idx_3 * HEAD_DIM + input_idx_3 + j]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
            }
        }
        #pragma unroll
        for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
            tmp += __shfl_down_sync(0xffffffff, tmp, mask);
        }
        if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
            atomicAdd(&output[cluster_block_st_id + weight_idx_3 + (id - 1) * TMA_LOAD_ONCE], __float2half(tmp));
        }
        block.sync();
    }
    bar[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(DIM_PER_BLOCK / TMA_LOAD_ONCE - 1) % 2]));
    tmp = 0.0;
    for (int j = 0; j < HEAD_DIM; j += NUM_PER_ROW_3) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[2 * HEAD_DIM + input_idx_3 + j]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&weight[TMA_LOAD_ONCE_NUM + weight_idx_3 * HEAD_DIM + input_idx_3 + j]);
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            tmp += __half2float(reg_input[d]) * __half2float(reg_weight[d]);
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW_3 >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW_3 == 0) {
        atomicAdd(&output[cluster_block_st_id + weight_idx_3 + ((DIM_PER_BLOCK / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE], __float2half(tmp));
    }

    // ==================== Add Output Bias ====================
    // Use atomicAdd to avoid race conditions
    // Each block adds bias for its portion of the output
    for (int d = tid; d < DIM_PER_BLOCK; d += BLOCK_SIZE) {
        int out_idx = cluster_block_st_id + d;
        // Only the first cluster (head 0) adds the bias to avoid adding it 32 times
        if (head_id == 0) {
            atomicAdd(&output[out_idx], o_bias[out_idx]);
        }
        
    }
}
