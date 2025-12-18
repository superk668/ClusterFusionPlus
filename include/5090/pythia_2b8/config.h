// Pythia-2.8b model configuration
// QKV Layout: INTERLEAVED per head [Q0, K0, V0, Q1, K1, V1, ...]
// Normalization: LayerNorm (with bias), NOT RMSNorm
// QKV projection: has bias

#define HEAD_DIM 80     // Pythia-2.8b: 2560 / 32 = 80
#define HEAD_NUM 32     // Pythia-2.8b: 32 attention heads
#define FFN_DIM 10240   // Pythia-2.8b: intermediate_size
#define HIDDEN_DIM 2560 // Pythia-2.8b: hidden_size

// Pythia-specific: only 25% of head_dim uses RoPE (Neox-style)
#define ROTARY_DIM 20   // HEAD_DIM * 0.25 = 80 * 0.25 = 20

// Interleaved QKV layout: each head has Q,K,V contiguous
// Head i: Q at [i*3*HEAD_DIM, i*3*HEAD_DIM+HEAD_DIM)
//         K at [i*3*HEAD_DIM+HEAD_DIM, i*3*HEAD_DIM+2*HEAD_DIM)
//         V at [i*3*HEAD_DIM+2*HEAD_DIM, i*3*HEAD_DIM+3*HEAD_DIM)
#define QKV_HEAD_STRIDE (3 * HEAD_DIM)  // 240

#define NUM_WARPS 4
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)  // 128 threads per block
#define CLUSTER_SIZE 4
#define NUM_PER_THREAD 8

// QKV computation: use configuration matching head_dim=80
// Each warp handles 20 output dimensions (HEAD_DIM / NUM_WARPS = 80/4 = 20)
// 4 threads collaborate on each output (WARP_SIZE / 20 ≈ 1.6, use 1 for simplicity)
#define NUM_ROW_PER_WARP 20  // HEAD_DIM / NUM_WARPS
#define NUM_THREAD_PER_ROW 1 // Each output computed by 1 thread

// NUM_PER_ROW: Elements processed per row iteration
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW)  // 8 * 1 = 8

#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)  // 640
#define FFN_DIM_PER_CLUSTER (FFN_DIM / HEAD_NUM)   // 320

// Shared memory sizing: TMA loads TMA_LOAD_ONCE x HEAD_DIM tiles
// MAX_SMEM_DIM should be HEAD_DIM for weight buffer
#define MAX_SMEM_DIM HEAD_DIM                      // 80 (optimized from 320)

// TMA configuration for QKV weights
// Load TMA_LOAD_ONCE rows of input dimension, HEAD_DIM columns of output dimension
#define TMA_LOAD_ONCE 64
#define TMA_LOAD_ONCE_MAX 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * HEAD_DIM)  // 64 * 80 = 5120
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))

// Attention TMA configuration
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 2)  // 32
#define TMA_LOAD_ONCE_NUM_ATTN ((TMA_LOAD_ONCE * HEAD_DIM) / 2)  // 2560
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))

// Decoding phase computation constants
// For HEAD_DIM=80, we need 10 threads per row (10 * 8 = 80)
// But shfl reduction needs power-of-2 mask. We use 16 threads per row,
// with the extra 6 threads contributing 0 to the reduction.
#define NUM_THREAD_PER_ROW_2 16  // Power of 2, covers 128 elements but we only use 80
#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE_ATTN / NUM_WARPS)  // 32 / 4 = 8
#define DIM_BLOCK_REDUCE ((2 * BLOCK_SIZE + NUM_THREAD_PER_ROW_2 - 1) / NUM_THREAD_PER_ROW_2)  // ceil(256/16) = 16
#define DEC_TILE ((NUM_ROW_PER_WARP_2 * NUM_THREAD_PER_ROW_2 + WARP_SIZE - 1) / WARP_SIZE)  // ceil(8*16/32) = 4

// Output projection constants
#define NUM_ROW_PER_WARP_3 (TMA_LOAD_ONCE / NUM_WARPS)  // 64 / 4 = 16
#define NUM_THREAD_PER_ROW_3 (WARP_SIZE / NUM_ROW_PER_WARP_3)  // 32 / 16 = 2
#define NUM_PER_ROW_3 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_3)  // 8 * 2 = 16
