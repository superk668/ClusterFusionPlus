/**
 * ClusterFusion Pythia-2.8B Decoder Layer Kernel
 * 
 * Fuses LayerNorm + QKV Projection + RoPE + Flash Decoding + Output Projection
 *       + Post-LN + MLP (Up + GELU + Down) + Residual Connection
 * 
 * Architecture: 32 clusters (one per head), 4 blocks per cluster, 128 threads per block
 * Optimizations:
 *   - Single-pass LayerNorm using Var(x) = E[x²] - E[x]²
 *   - Tree reduction for cluster-level reductions
 *   - TMA for efficient weight loading with double buffering
 *   - Cluster-level distributed shared memory for QKV/attention reductions
 */

#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h> 
#include "../../dsm.cuh"
#include "config.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

// Fast exp2 approximation using PTX
__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Fast tanh using PTX exp2: tanh(x) = (e^2x - 1) / (e^2x + 1)
// -10.0 <= x <= 10.0 to avoid overflow
__forceinline__ __device__ float ptx_tanh(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    constexpr float LOG2_E = 1.44269504088896340736f;
    float exp_2x = ptx_exp2(2.0f * x * LOG2_E);
    return (exp_2x - 1.0f) * __frcp_rn(exp_2x + 1.0f);
}

// Fast GELU using PTX: GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__forceinline__ __device__ float ptx_gelu(float x) {
    constexpr float GELU_CONST = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float GELU_COEF = 0.044715f;
    float x3 = x * x * x;
    float inner = GELU_CONST * (x + GELU_COEF * x3);
    return x * 0.5f * (1.0f + ptx_tanh(inner));
}

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) PythiaDecoderLayerKernel(
    // Output tensors
    half* output,           // [1, HIDDEN_DIM] - final output
    half* k_output,         // [1, NUM_HEADS, HEAD_DIM] - new K for cache
    half* v_output,         // [1, NUM_HEADS, HEAD_DIM] - new V for cache
    // Input tensor
    half* input,            // [1, HIDDEN_DIM]
    // LayerNorm weights
    half* ln_weight,        // [HIDDEN_DIM]
    half* ln_bias,          // [HIDDEN_DIM]
    // Attention weights
    half* qkv_bias,         // [3 * NUM_HEADS * HEAD_DIM] - interleaved QKV bias
    half* o_bias,           // [HIDDEN_DIM]
    // RoPE embeddings
    float* cos,             // [HEAD_DIM] - padded, only first ROTARY_DIM used
    float* sin,             // [HEAD_DIM]
    // KV cache
    half* k_cache,          // [max_seq_len, HIDDEN_DIM]
    half* v_cache,          // [max_seq_len, HIDDEN_DIM]
    // MLP weights
    half* post_ln_weight,   // [HIDDEN_DIM]
    half* post_ln_bias,     // [HIDDEN_DIM]
    half* mlp_up_weight,    // [FFN_DIM, HIDDEN_DIM]
    half* mlp_up_bias,      // [FFN_DIM]
    half* mlp_down_weight,  // [HIDDEN_DIM, FFN_DIM]
    half* mlp_down_bias,    // [HIDDEN_DIM]
    // Intermediate buffers
    half* mlp_intermediate, // [FFN_DIM] - MLP up projection output
    half* post_ln_buffer,   // [HIDDEN_DIM] - post-attention LN output
    // TMA descriptors
    const __grid_constant__ CUtensorMap tensor_map,
    const __grid_constant__ CUtensorMap tensor_map_k_cache,
    const __grid_constant__ CUtensorMap tensor_map_v_cache,
    const __grid_constant__ CUtensorMap tensor_map_weight_o,
    const __grid_constant__ CUtensorMap tensor_map_mlp_up,
    const __grid_constant__ CUtensorMap tensor_map_mlp_down,
    // Runtime parameters
    const uint32_t SEQ_LEN,
    const uint32_t KV_DIM_PER_BLOCK
) {
    // ==================== Thread/Block Indexing ====================
    cg::grid_group grid             = cg::this_grid();
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = grid.cluster_rank();      // 0..31 (one cluster per head)
    const uint32_t cluster_block_id = cluster.block_rank();     // 0..3  (blocks within cluster)
    const uint32_t tid              = block.thread_rank();      // 0..127
    const uint32_t lane_id          = tid % WARP_SIZE;
    const uint32_t warp_id          = tid / WARP_SIZE;
    const uint32_t tile_row         = tid / NUM_THREAD_PER_ROW_2;
    const uint32_t tile_col         = tid % NUM_THREAD_PER_ROW_2;

    // ==================== Shared Memory Layout ====================
    extern __shared__ uint8_t shmem_base[];
    half* weight      = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_qkv   = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;
    half* input_shmem = local_qkv + 3 * HEAD_DIM;
    float* reduction  = reinterpret_cast<float*>(input_shmem + DIM_PER_BLOCK);

    __shared__ float cluster_local_sum, cluster_local_sum_sq, cluster_local_max;

    // ==================== Register Allocation ====================
    float local_sum = 0.0f, local_mean = 0.0f, var_rcp = 0.0f, tmp = 0.0f;
    float local_max = -CUDART_INF_F, pre_max = -CUDART_INF_F, scale = 0.0f;
    const float eps = 1e-5f;
    const float softmax_scale = 0.16129820913429784f;  // 1/sqrt(80) * log2(e)
    
    half __align__(16) reg_input[NUM_PER_THREAD], reg_weight[NUM_PER_THREAD];
    float reg_reduce[NUM_PER_THREAD];
    float* dst_shmem;
    float q_rope, q_rope_1, k_rope, k_rope_1, cos_reg, sin_reg;
    uint32_t size, src_addr, dst_addr, neighbor_dst_bar = 0;
    float __align__(16) qk[DEC_TILE];

    // ==================== Barrier Initialization ====================
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

    // ==================== Precompute Thread Indices ====================
    uint input_idx   = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx  = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;
    uint input_idx_2 = (lane_id % NUM_THREAD_PER_ROW_2) * NUM_PER_THREAD;
    uint weight_idx_2 = warp_id * NUM_ROW_PER_WARP_2 + (lane_id / NUM_THREAD_PER_ROW_2) * DEC_TILE;
    uint input_idx_3 = (lane_id % NUM_THREAD_PER_ROW_3) * NUM_PER_THREAD;
    uint weight_idx_3 = warp_id * NUM_ROW_PER_WARP_3 + lane_id / NUM_THREAD_PER_ROW_3;
    uint cluster_block_st_id = cluster_block_id * DIM_PER_BLOCK;
    uint head_qkv_offset = head_id * QKV_HEAD_STRIDE;

    // ==================== LayerNorm (Single-Pass Optimization) ====================
    // Compute sum and sum_sq in one pass using Var(x) = E[x²] - E[x]²
    local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        #pragma unroll
        for (int di = 0; di < 8; di++) {
            float val = __half2float(reg_input[di]);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Block-level reduction
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, mask);
    }
    if (lane_id == 0) {
        reduction[warp_id * 2] = local_sum;
        reduction[warp_id * 2 + 1] = local_sum_sq;
    }
    block.sync();
    if (tid < NUM_WARPS) {
        local_sum = reduction[tid * 2];
        local_sum_sq = reduction[tid * 2 + 1];
    }
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, mask);
    }
    if (tid == 0) {
        cluster_local_sum = local_sum;
        cluster_local_sum_sq = local_sum_sq;
    }
    cluster.sync();

    // Cluster-level tree reduction (2 steps for CLUSTER_SIZE=4)
    if (tid == 0) {
        uint32_t partner = cluster_block_id ^ 2;
        float* partner_sum = cluster.map_shared_rank(&cluster_local_sum, partner);
        float* partner_sum_sq = cluster.map_shared_rank(&cluster_local_sum_sq, partner);
        cluster_local_sum += *partner_sum;
        cluster_local_sum_sq += *partner_sum_sq;
    }
    cluster.sync();
    if (tid == 0) {
        uint32_t partner = cluster_block_id ^ 1;
        float* partner_sum = cluster.map_shared_rank(&cluster_local_sum, partner);
        float* partner_sum_sq = cluster.map_shared_rank(&cluster_local_sum_sq, partner);
        cluster_local_sum += *partner_sum;
        cluster_local_sum_sq += *partner_sum_sq;
    }
    cluster.sync();

    // Compute statistics
    local_mean = cluster_local_sum / HIDDEN_DIM;
    float variance = cluster_local_sum_sq / HIDDEN_DIM - local_mean * local_mean;
    var_rcp = __frsqrt_rn(variance + eps);

    // Apply normalization with weight and bias
    for (int d = tid * 8; d < DIM_PER_BLOCK; d += BLOCK_SIZE * 8) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&input[cluster_block_st_id + d]);
        *(uint4*)(&reg_weight[0]) = *(uint4*)(&ln_weight[cluster_block_st_id + d]);
        half __align__(16) reg_bias[8];
        *(uint4*)(&reg_bias[0]) = *(uint4*)(&ln_bias[cluster_block_st_id + d]);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float normalized = (__half2float(reg_input[i]) - local_mean) * var_rcp;
            reg_input[i] = __float2half(normalized * __half2float(reg_weight[i]) + __half2float(reg_bias[i]));
        }
        *(uint4*)(&input_shmem[d]) = *(uint4*)(&reg_input[0]);
    }
    block.sync();

    // ==================== QKV Projection (Interleaved Layout) ====================
    // Weight layout: [Q0,K0,V0,Q1,K1,V1,...] where each has HEAD_DIM=80 dims
    
    // Compute Q
    tmp = 0.0f;
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

    // Compute K
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

    // Compute V
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

    // Cluster reduce QKV partial results
    size = (HEAD_DIM * 3) * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,
        src_addr, dst_addr, bar_ptr,
        neighbor_dst_bar, local_qkv, weight);
    cluster.sync();

    // ==================== Add QKV Bias ====================
    if (tid < HEAD_DIM) {
        float q_val = __half2float(local_qkv[tid]) + __half2float(qkv_bias[head_qkv_offset + tid]);
        local_qkv[tid] = __float2half(q_val);
        float k_val = __half2float(local_qkv[HEAD_DIM + tid]) + __half2float(qkv_bias[head_qkv_offset + HEAD_DIM + tid]);
        local_qkv[HEAD_DIM + tid] = __float2half(k_val);
        float v_val = __half2float(local_qkv[2 * HEAD_DIM + tid]) + __half2float(qkv_bias[head_qkv_offset + 2 * HEAD_DIM + tid]);
        local_qkv[2 * HEAD_DIM + tid] = __float2half(v_val);
    }
    block.sync();

    // ==================== RoPE (Neox-style, rotary_pct=0.25) ====================
    // Only first ROTARY_DIM=20 dimensions get rotary embedding
    if (tid < ROTARY_DIM) {
        q_rope = __half2float(local_qkv[tid]);
        k_rope = __half2float(local_qkv[HEAD_DIM + tid]);
        cos_reg = cos[tid];
        sin_reg = sin[tid];
        
        // Neox-style rotate_half: swap and negate first half
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
    block.sync();

    // ==================== Write K,V to Cache ====================
    cluster.sync();
    if (cluster_block_id == 0 && tid < HEAD_DIM) {
        uint32_t cache_offset = SEQ_LEN * HIDDEN_DIM + head_id * HEAD_DIM + tid;
        k_cache[cache_offset] = local_qkv[HEAD_DIM + tid];
        v_cache[cache_offset] = local_qkv[2 * HEAD_DIM + tid];
        k_output[head_id * HEAD_DIM + tid] = local_qkv[HEAD_DIM + tid];
        v_output[head_id * HEAD_DIM + tid] = local_qkv[2 * HEAD_DIM + tid];
    }
    cluster.sync();

    // ==================== Flash Decoding ====================
    // Online softmax attention with TMA-accelerated KV cache loading
    local_sum = 0.0f;
    for (int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;

    // Load Q into registers (with bounds check for HEAD_DIM=80)
    if (input_idx_2 < HEAD_DIM) {
        *(uint4*)(&reg_input[0]) = *(uint4*)(&local_qkv[input_idx_2]);
    } else {
        for (int i = 0; i < NUM_PER_THREAD; i++)
            reg_input[i] = __float2half(0.0f);
    }
    block.sync();

    // Preload first KV cache tile
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_k_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[TMA_LOAD_ONCE_NUM_ATTN], &tensor_map_v_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK, bar[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar[2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
    } else {
        token[0] = bar[0].arrive();
        token[2] = bar[2].arrive();
    }

    // Process KV cache tiles with online softmax
    for (int id = 1; id < KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_k_cache, head_id * HEAD_DIM, cluster_block_id * KV_DIM_PER_BLOCK + id * TMA_LOAD_ONCE_ATTN, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE_ATTN);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        pre_max = local_max;

        // Compute Q·K^T for this tile
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
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
            int seq_idx = KV_DIM_PER_BLOCK * cluster_block_id + (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
            if (seq_idx < SEQ_LEN) {
                local_max = max(local_max, qk[j]);
            }
        }

        // Online softmax rescaling
        scale = (local_max > -CUDART_INF_F) ? ptx_exp2(pre_max - local_max) : 0.0f;
        local_sum *= scale;
        #pragma unroll
        for (int j = 0; j < DEC_TILE; j++) {
            int seq_idx = KV_DIM_PER_BLOCK * cluster_block_id + (id - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
            if (seq_idx < SEQ_LEN) {
                qk[j] = ptx_exp2(qk[j] - local_max);
                local_sum += qk[j];
            } else {
                qk[j] = 0.0f;
            }
        }
        #pragma unroll
        for (int j = 0; j < NUM_PER_THREAD; j++) {
            reg_reduce[j] = reg_reduce[j] * scale;
        }

        // Load V and accumulate attention output
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

    // Process last KV cache tile
    if (KV_DIM_PER_BLOCK > TMA_LOAD_ONCE_ATTN) {
        bar[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2].wait(std::move(token[(KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) % 2]));
    } else {
        bar[0].wait(std::move(token[0]));
    }
    pre_max = local_max;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
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
        int seq_idx = KV_DIM_PER_BLOCK * cluster_block_id + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
        if (seq_idx < SEQ_LEN) {
            local_max = max(local_max, qk[j]);
        }
    }
    scale = (local_max > -CUDART_INF_F) ? ptx_exp2(pre_max - local_max) : 0.0f;
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < DEC_TILE; j++) {
        int seq_idx = KV_DIM_PER_BLOCK * cluster_block_id + (KV_DIM_PER_BLOCK / TMA_LOAD_ONCE_ATTN - 1) * TMA_LOAD_ONCE_ATTN + weight_idx_2 + j;
        if (seq_idx < SEQ_LEN) {
            qk[j] = ptx_exp2(qk[j] - local_max);
            local_sum += qk[j];
        } else {
            qk[j] = 0.0f;
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

    // Process current token's KV (in local_qkv, not yet in cache)
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
            scale = (pre_max > -CUDART_INF_F) ? ptx_exp2(pre_max - local_max) : 0.0f;
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

    // Store partial attention output to shared memory
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

    // Block-level reduction of attention outputs
    for (int i = 0; i < NUM_PER_THREAD; i++)
        reg_reduce[i] = 0.0f;
    local_sum = 0.0f;
    local_max = -CUDART_INF_F;
    #pragma unroll
    for (int j = 0; j < DIM_BLOCK_REDUCE / 2; j++) {
        if (tile_col * NUM_PER_THREAD < HEAD_DIM) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&weight[j * HEAD_DIM + tile_col * NUM_PER_THREAD]);
        } else {
            for (int i = 0; i < NUM_PER_THREAD; i++) reg_input[i] = __float2half(0.0f);
        }
        float m = reduction[j * 2], s = reduction[j * 2 + 1];
        pre_max = local_max;
        local_max = max(m, local_max);
        scale = (m > -CUDART_INF_F && local_max > -CUDART_INF_F) ? ptx_exp2(m - local_max) : 0.0f;
        s *= scale;
        float rescale = (pre_max > -CUDART_INF_F && local_max > -CUDART_INF_F) ? ptx_exp2(pre_max - local_max) : 0.0f;
        local_sum = local_sum * rescale + s;
        #pragma unroll
        for (int d = 0; d < NUM_PER_THREAD; d++) {
            reg_reduce[d] = reg_reduce[d] * rescale + __half2float(reg_input[d]) * scale;
        }
    }
    block.sync();

    // Cluster-level reduction for attention (tree pattern)
    pre_max = local_max;
    if (tid == 0) {
        cluster_local_max = local_max;
    }
    cluster.sync();
    
    // Tree reduction for max
    if (tid == 0) {
        uint32_t partner = cluster_block_id ^ 2;
        float* partner_max = cluster.map_shared_rank(&cluster_local_max, partner);
        cluster_local_max = fmaxf(cluster_local_max, *partner_max);
    }
    cluster.sync();
    if (tid == 0) {
        uint32_t partner = cluster_block_id ^ 1;
        float* partner_max = cluster.map_shared_rank(&cluster_local_max, partner);
        cluster_local_max = fmaxf(cluster_local_max, *partner_max);
    }
    cluster.sync();

    scale = (pre_max > -CUDART_INF_F && cluster_local_max > -CUDART_INF_F) ? ptx_exp2(pre_max - cluster_local_max) : 0.0f;
    local_sum *= scale;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * scale;
    }
    if (tid == 0) {
        cluster_local_sum = local_sum;
    }
    cluster.sync();

    // Tree reduction for sum
    if (tid == 0) {
        uint32_t partner = cluster_block_id ^ 2;
        float* partner_sum = cluster.map_shared_rank(&cluster_local_sum, partner);
        cluster_local_sum += *partner_sum;
    }
    cluster.sync();
    if (tid == 0) {
        uint32_t partner = cluster_block_id ^ 1;
        float* partner_sum = cluster.map_shared_rank(&cluster_local_sum, partner);
        cluster_local_sum += *partner_sum;
    }
    cluster.sync();

    // Normalize attention output
    float safe_sum = cluster_local_sum + 1e-10f;
    #pragma unroll
    for (int j = 0; j < NUM_PER_THREAD; j++) {
        reg_reduce[j] = reg_reduce[j] * __frcp_rn(safe_sum);
    }
    if (tid < NUM_THREAD_PER_ROW_2 && tid * NUM_PER_THREAD < HEAD_DIM) {
        #pragma unroll
        for (int i = 0; i < NUM_PER_THREAD; i++) {
            if (tid * NUM_PER_THREAD + i < HEAD_DIM) {
                local_qkv[2 * HEAD_DIM + tid * NUM_PER_THREAD + i] = __float2half(reg_reduce[i]);
            }
        }
    }
    block.sync();

    // Cluster reduce attention output
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&local_qkv[2 * HEAD_DIM]));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::ATTN>(
        size, tid, HEAD_DIM, cluster_block_id,
        src_addr, dst_addr, bar_ptr,
        neighbor_dst_bar, &local_qkv[2 * HEAD_DIM], weight);
    cluster.sync();

    // ==================== Output Projection ====================
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

    // Add output bias (only head 0 to avoid 32x addition)
    for (int d = tid; d < DIM_PER_BLOCK; d += BLOCK_SIZE) {
        int out_idx = cluster_block_st_id + d;
        if (head_id == 0) {
            atomicAdd(&output[out_idx], o_bias[out_idx]);
        }
    }

    // ==================== Post-Attention LayerNorm ====================
    // Reuse mean/var from input LayerNorm (Pythia parallel residual)
    cluster.sync();
    for (int d = tid; d < DIM_PER_BLOCK; d += BLOCK_SIZE) {
        int idx = cluster_block_st_id + d;
        float normalized = (__half2float(input[idx]) - local_mean) * var_rcp;
        post_ln_buffer[idx] = __float2half(normalized * __half2float(post_ln_weight[idx]) + __half2float(post_ln_bias[idx]));
    }
    block.sync();

    // ==================== MLP Up Projection + GELU ====================
    uint32_t global_block_id = head_id * CLUSTER_SIZE + cluster_block_id;
    uint32_t ffn_block_offset = global_block_id * HEAD_DIM;

    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_mlp_up, 0, ffn_block_offset, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    tmp = 0.0f;
    for (int id = 1; id < HIDDEN_DIM / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_mlp_up, id * TMA_LOAD_ONCE, ffn_block_offset, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (weight_idx < HEAD_DIM) {
            #pragma unroll 8
            for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
                *(uint4*)(&reg_input[0]) = *(uint4*)(&post_ln_buffer[(id - 1) * TMA_LOAD_ONCE + input_idx + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(reg_input[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
    }
    bar[(HIDDEN_DIM / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(HIDDEN_DIM / TMA_LOAD_ONCE - 1) % 2]));
    if (weight_idx < HEAD_DIM) {
        #pragma unroll 8
        for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
            *(uint4*)(&reg_input[0]) = *(uint4*)(&post_ln_buffer[((HIDDEN_DIM / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + input_idx + i]);
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

    // Apply GELU using fast PTX approximation
    if (lane_id % NUM_THREAD_PER_ROW == 0 && weight_idx < HEAD_DIM) {
        float val = tmp + __half2float(mlp_up_bias[ffn_block_offset + weight_idx]);
        mlp_intermediate[ffn_block_offset + weight_idx] = __float2half(ptx_gelu(val));
    }

    // ==================== MLP Down Projection ====================
    grid.sync();  // Wait for all blocks to finish MLP up

    uint32_t down_out_offset = head_id * HEAD_DIM;
    uint32_t ffn_input_offset = cluster_block_id * (FFN_DIM / CLUSTER_SIZE);
    uint32_t ffn_input_per_block = FFN_DIM / CLUSTER_SIZE;

    tmp = 0.0f;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_mlp_down, ffn_input_offset, down_out_offset, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    half __align__(16) mlp_in[NUM_PER_THREAD];
    for (int id = 1; id < ffn_input_per_block / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_mlp_down, ffn_input_offset + id * TMA_LOAD_ONCE, down_out_offset, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (weight_idx < HEAD_DIM) {
            #pragma unroll 8
            for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
                *(uint4*)(&mlp_in[0]) = *(uint4*)(&mlp_intermediate[ffn_input_offset + (id - 1) * TMA_LOAD_ONCE + input_idx + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(mlp_in[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
    }
    bar[(ffn_input_per_block / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(ffn_input_per_block / TMA_LOAD_ONCE - 1) % 2]));
    if (weight_idx < HEAD_DIM) {
        #pragma unroll 8
        for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
            *(uint4*)(&mlp_in[0]) = *(uint4*)(&mlp_intermediate[ffn_input_offset + ((ffn_input_per_block / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + input_idx + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(mlp_in[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
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

    // Cluster reduce MLP down partial results
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,
        src_addr, dst_addr, bar_ptr,
        neighbor_dst_bar, local_qkv, weight);
    cluster.sync();

    // ==================== Final Residual Connection ====================
    // Pythia parallel residual: output = input + attn_output + mlp_output
    if (cluster_block_id == 0) {
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            int out_idx = down_out_offset + d;
            float mlp_val = __half2float(local_qkv[d]) + __half2float(mlp_down_bias[out_idx]);
            float input_val = __half2float(input[out_idx]);
            float attn_val = __half2float(output[out_idx]);
            output[out_idx] = __float2half(input_val + attn_val + mlp_val);
        }
    }
}
