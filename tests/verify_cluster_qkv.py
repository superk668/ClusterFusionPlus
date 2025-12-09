import torch
import numpy as np

# Test cluster accumulation manually
torch.manual_seed(42)

hidden_size = 2560
num_heads = 32
head_dim = 80
cluster_size = 4
block_size = hidden_size // cluster_size  # 640

# Generate test data
input_hidden = torch.randn(1, hidden_size, dtype=torch.float16, device='cuda')
weight_qkv = torch.randn(3 * num_heads * head_dim, hidden_size, dtype=torch.float16, device='cuda')
layernorm_weight = torch.randn(hidden_size, dtype=torch.float16, device='cuda')

# Normalize
eps = 1e-5
variance = input_hidden.pow(2).mean(-1, keepdim=True)
hidden_normed = input_hidden * torch.rsqrt(variance + eps) * layernorm_weight

print("Normalized input (first 80 dims):")
print(hidden_normed[0, :80])
print()

# Full reference QKV
qkv_full = torch.matmul(hidden_normed, weight_qkv.t())  # (1, 7680)
q_full = qkv_full[:, :hidden_size].reshape(1, num_heads, head_dim)  # (1, 32, 80)

print("Reference Q[head 1] (full computation):")
print(q_full[0, 1, :])
print()

# Now simulate what kernel does: each cluster block computes partial result
print("=== Simulating Kernel Cluster Blocks ===")
head_id = 1
head_start = head_id * head_dim  # 80
head_end = (head_id + 1) * head_dim  # 160

q_accumulated = torch.zeros(head_dim, dtype=torch.float32, device='cuda')

for block_id in range(cluster_size):
    block_start = block_id * block_size
    block_end = (block_id + 1) * block_size
    
    # Extract input for this block
    block_input = hidden_normed[0, block_start:block_end]  # (640,)
    
    # Extract weight: weight_qkv is (7680, 2560)
    # For Q of head 1: rows 80:160, columns for this block
    weight_q_head_block = weight_qkv[head_start:head_end, block_start:block_end]  # (80, 640)
    
    # Compute partial result: Q_partial = input @ W^T
    q_partial = torch.matmul(block_input, weight_q_head_block.t())  # (80,)
    
    print(f"Block {block_id}:")
    print(f"  Input range: [{block_start}:{block_end}]")
    print(f"  Input[0:5]: {block_input[:5]}")
    print(f"  Weight[0:5, 0:5]:")
    print(weight_q_head_block[:5, :5])
    print(f"  Partial Q[0:5]: {q_partial[:5]}")
    print()
    
    q_accumulated += q_partial.float()

print("Accumulated Q (after summing all blocks):")
print(q_accumulated[:10])
print()
print("Reference Q:")
print(q_full[0, 1, :10])
print()
print("Difference:")
print((q_accumulated[:10].half() - q_full[0, 1, :10]).abs())
print()
print("Max difference:", (q_accumulated.half() - q_full[0, 1, :]).abs().max().item())
