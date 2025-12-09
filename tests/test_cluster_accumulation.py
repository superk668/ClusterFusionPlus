import torch
import torch.nn as nn

def test_cluster_accumulation():
    """Verify that cluster blocks need to accumulate results."""
    
    # Model configuration
    hidden_size = 2560
    num_heads = 32
    head_dim = 80
    cluster_size = 4
    block_size = hidden_size // cluster_size  # 640
    
    print("=== Testing Cluster Accumulation ===")
    print(f"Hidden size: {hidden_size}")
    print(f"Cluster size: {cluster_size}")
    print(f"Block size: {block_size}")
    print()
    
    # Generate test data
    torch.manual_seed(42)
    input_hidden = torch.randn(1, hidden_size, dtype=torch.float16, device='cuda')
    weight_qkv = torch.randn(3 * num_heads * head_dim, hidden_size, dtype=torch.float16, device='cuda')
    layernorm_weight = torch.randn(hidden_size, dtype=torch.float16, device='cuda')
    
    # Normalize input (RMSNorm)
    eps = 1e-5
    variance = input_hidden.pow(2).mean(-1, keepdim=True)
    hidden_normed = input_hidden * torch.rsqrt(variance + eps) * layernorm_weight
    
    # Reference: Full computation
    qkv_ref = torch.matmul(hidden_normed, weight_qkv.t())  # (1, 7680)
    q_ref = qkv_ref[:, :hidden_size].reshape(1, num_heads, head_dim)  # (1, 32, 80)
    
    print("Reference Q[0, 1, :10] (head 1, first 10 dims):")
    print(q_ref[0, 1, :10])
    print()
    
    # Simulate cluster block computation
    print("=== Simulating Cluster Block Computation ===")
    
    head_id = 1
    head_start = head_id * head_dim  # 80
    head_end = (head_id + 1) * head_dim  # 160
    
    q_accumulated = torch.zeros(head_dim, dtype=torch.float32, device='cuda')
    
    for block_id in range(cluster_size):
        block_start = block_id * block_size
        block_end = (block_id + 1) * block_size
        
        # Extract input for this block
        block_input = hidden_normed[0, block_start:block_end]  # (640,)
        
        # Extract weight for Q projection of this head and this block
        weight_q_head_block = weight_qkv[head_start:head_end, block_start:block_end]  # (80, 640)
        
        # Compute partial result
        q_partial = torch.matmul(block_input, weight_q_head_block.t())  # (80,)
        
        print(f"Block {block_id} partial Q[:10]: {q_partial[:10]}")
        
        # Accumulate
        q_accumulated += q_partial.float()
    
    print()
    print("Accumulated Q[:10] (float32):")
    print(q_accumulated[:10])
    print()
    print("Reference Q[:10] (float16):")
    print(q_ref[0, 1, :10])
    print()
    print("Accumulated Q[:10] (converted to float16):")
    print(q_accumulated.half()[:10])
    print()
    print("Difference (max abs error):")
    print((q_accumulated.half()[:10] - q_ref[0, 1, :10]).abs().max())
    print()
    
    # Check full vector
    full_diff = (q_accumulated.half() - q_ref[0, 1, :]).abs().max()
    print(f"Full vector max abs error: {full_diff}")
    print()
    
    if full_diff < 1.0:
        print("✓ Cluster accumulation working correctly!")
    else:
        print("✗ Cluster accumulation has issues")
    
    # Now check what kernel debug output shows
    print()
    print("=== Understanding Kernel Debug Output ===")
    print("From test_result.txt, we see:")
    print("  Before RoPE Q values change order after 3rd call")
    print("  This suggests cluster blocks are NOT accumulating correctly")
    print()
    print("Expected behavior:")
    print("  1. Each cluster block computes partial Q/K/V")
    print("  2. Blocks should write to SAME output location and accumulate")
    print("  3. But current implementation might overwrite instead of accumulate")
    print()
    print("Check in kernel.cuh:")
    print("  local_qkv[warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW] = __float2half(tmp);")
    print("  This WRITES to local shared memory - no accumulation!")
    print()
    print("Problem: Each cluster block writes its partial result to LOCAL shared memory,")
    print("but there's no cross-cluster accumulation happening!")

if __name__ == "__main__":
    test_cluster_accumulation()
