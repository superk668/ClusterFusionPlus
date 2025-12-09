
#define HEAD_DIM 80     // Pythia-2.8b: 2560 / 32 = 80
#define HEAD_NUM 32     // Pythia-2.8b: 32 attention heads
#define FFN_DIM 10240   // Pythia-2.8b: intermediate_size
#define HIDDEN_DIM 2560 // Pythia-2.8b: hidden_size
// #define SEQ_LEN 2048  // max_position_embeddings

// Pythia-specific: only 25% of head_dim uses RoPE
#define ROTARY_DIM 20   // HEAD_DIM * 0.25 = 80 * 0.25 = 20

#define NUM_WARPS 4
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)  // 128 threads per block
#define CLUSTER_SIZE 4 // 2 4
#define NUM_PER_THREAD 8

// Attention computation constants - adjusted for HEAD_DIM=80
// For HEAD_DIM=80, NUM_THREAD_PER_ROW=1 causes warp reduction to fail
// 
// New strategy for correctness (sacrificing some performance):
// Use 4 threads per output dimension, allowing proper warp reduction
// NUM_ROW_PER_WARP = WARP_SIZE / NUM_THREAD_PER_ROW = 32 / 4 = 8
// This means each warp computes 8 output dimensions
// For HEAD_DIM=80, we need 80/8 = 10 warps, but we only have 4
// Solution: Each warp does multiple passes, or threads compute multiple outputs
//
// Better approach: NUM_THREAD_PER_ROW = 4, NUM_ROW_PER_WARP = 8
// Each warp handles 8 output dimensions with 4 threads per dimension
// 4 warps * 8 rows = 32 outputs per pass
// Need 80/32 = 3 passes (last pass handles 80-64=16 outputs)
// NUM_PER_ROW = 4 * 8 = 32, which gives proper TMA loop coverage
#define NUM_THREAD_PER_ROW 4  // Use 4 threads per output dimension
#define NUM_ROW_PER_WARP (WARP_SIZE / NUM_THREAD_PER_ROW)  // 32/4 = 8

// NUM_PER_ROW: Elements processed per row (each thread processes NUM_PER_THREAD elements)
#define NUM_PER_ROW (NUM_PER_THREAD * NUM_THREAD_PER_ROW)  // 8 * 4 = 32

#define DIM_PER_BLOCK (HIDDEN_DIM / CLUSTER_SIZE)  // 640
// #define KV_DIM_PER_BLOCK (SEQ_LEN / CLUSTER_SIZE) 
#define FFN_DIM_PER_CLUSTER (FFN_DIM / HEAD_NUM)  // 320
#define MAX_SMEM_DIM FFN_DIM_PER_CLUSTER  // 320

// TMA (Tensor Memory Accelerator) load configuration
#define TMA_LOAD_ONCE 64 // 8 16 32 64 128 256
#define TMA_LOAD_ONCE_MAX 256
#define TMA_LOAD_ONCE_NUM (TMA_LOAD_ONCE * HEAD_DIM)  // 64 * 80 = 5120
#define TMA_LOAD_ONCE_SIZE (TMA_LOAD_ONCE_NUM * sizeof(half))
#define TMA_LOAD_ONCE_ATTN (TMA_LOAD_ONCE / 2)  // 32
#define TMA_LOAD_ONCE_NUM_ATTN ((TMA_LOAD_ONCE * HEAD_DIM) / 2)  // 2560
#define TMA_LOAD_ONCE_SIZE_ATTN (TMA_LOAD_ONCE_NUM_ATTN * sizeof(half))
#define TMA_LOAD_ONCE_NUM_FFN (TMA_LOAD_ONCE * TMA_LOAD_ONCE_MAX)
#define TMA_LOAD_ONCE_NUM_FFN_TOTAL (TMA_LOAD_ONCE * FFN_DIM_PER_CLUSTER)  // 64 * 320
#define TMA_LOAD_ONCE_SIZE_FFN (TMA_LOAD_ONCE_NUM_FFN_TOTAL * sizeof(half))

// Decoding phase computation constants - adjusted for HEAD_DIM=80
// NUM_THREAD_PER_ROW_2: For decoding, threads per row
#define NUM_THREAD_PER_ROW_2 (HEAD_DIM / NUM_PER_THREAD)  // 80 / 8 = 10

// NUM_ROW_PER_WARP_2: Rows per warp in decoding
#define NUM_ROW_PER_WARP_2 (TMA_LOAD_ONCE_ATTN / NUM_WARPS)  // 32 / 4 = 8

// DIM_BLOCK_REDUCE: Dimension for block-level reduction
// Must be >= NUM_WARPS for proper reduction, rounded up
// Original formula may produce odd numbers causing sync issues
#define DIM_BLOCK_REDUCE ((2 * BLOCK_SIZE + NUM_THREAD_PER_ROW_2 - 1) / NUM_THREAD_PER_ROW_2)  // ceil(2*128/10) = 26

// DEC_TILE: Decoding tile size
// This needs careful adjustment for HEAD_DIM=80
// Using floor division: 8 / 3 = 2, but need to ensure it's at least 1
#define DEC_TILE ((NUM_ROW_PER_WARP_2 * NUM_THREAD_PER_ROW_2 + WARP_SIZE - 1) / WARP_SIZE)  // ceil(8*10/32) = 3

// Additional computation constants
#define NUM_ROW_PER_WARP_3 (TMA_LOAD_ONCE / NUM_WARPS)  // 64 / 4 = 16
#define NUM_THREAD_PER_ROW_3 (WARP_SIZE / NUM_ROW_PER_WARP_3)  // 32 / 16 = 2
#define NUM_PER_ROW_3 (NUM_PER_THREAD * NUM_THREAD_PER_ROW_3)  // 8 * 2 = 16
