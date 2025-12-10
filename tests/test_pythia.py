import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import clusterfusion

# Pythia-2.8b model parameters
hidden_size = 2560
num_heads = 32
seqlen = 2047  # Start with small seqlen for debugging
head_dim = hidden_size // num_heads  # 80
ffn_dim = 10240
rotary_dim = head_dim // 4  # 20 (rotary_pct = 0.25)

torch.manual_seed(42)

# Enable Debug print
debug = 0
print_head = 1
if debug:
    test_run = 1
else:
    test_run = 10000

def initialize_rope_embeddings(rotary_dim):
    """
    Initialize RoPE embeddings for only rotary_dim dimensions.
    Kernel expects HEAD_DIM size, so we pad with identity (cos=1, sin=0).
    """
    angles = (torch.rand((1, rotary_dim), dtype=torch.float32) * (2 * torch.pi)).to(0)
    h_cos = torch.cos(angles)
    h_sin = torch.sin(angles)
    
    # Pad to HEAD_DIM size
    padding_size = head_dim - rotary_dim
    cos_padding = torch.ones((1, padding_size), dtype=torch.float32).to(0)
    sin_padding = torch.zeros((1, padding_size), dtype=torch.float32).to(0)
    
    h_cos = torch.cat([h_cos, cos_padding], dim=-1)
    h_sin = torch.cat([h_sin, sin_padding], dim=-1)
    
    return h_cos, h_sin

def apply_neox_style_rotary_pos_emb_partial(q, k, cos, sin, rotary_dim):
    """
    Apply Neox-style RoPE only to the first rotary_dim dimensions.
    For Pythia, rotary_pct=0.25, so only first 20 dims use RoPE.
    """
    cos = cos[:, :rotary_dim].unsqueeze(1)
    sin = sin[:, :rotary_dim].unsqueeze(1)
    
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    
    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
    
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def pythia_decode_reference(hidden, ln_weight, ln_bias, eps, kv_cache, 
                            weight_qkv, bias_qkv, weight_o, 
                            head_dim, cos, sin, rotary_dim):
    """
    Pythia decoding reference implementation.
    - Uses LayerNorm (with bias), not RMSNorm
    - QKV layout is INTERLEAVED: [Q0, K0, V0, Q1, K1, V1, ...]
    - QKV projection has bias
    """
    if debug:
        print("----------------------------- python begin -----------------------------")

    # LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, keepdim=True, unbiased=False)
    hidden_normed = (hidden - mean) / torch.sqrt(var + eps) * ln_weight + ln_bias
    
    if debug: 
        print("normed ref (first 80):", hidden_normed[..., 0:80])

    # QKV projection: output = input @ weight.T + bias
    # weight_qkv shape: [7680, 2560], bias_qkv shape: [7680]
    qkv = torch.matmul(hidden_normed, weight_qkv.t()) + bias_qkv  # [1, 7680]
    
    # Interleaved layout: reshape to [num_heads, 3, head_dim]
    # qkv[0:80] = Q head 0, qkv[80:160] = K head 0, qkv[160:240] = V head 0
    # qkv[240:320] = Q head 1, ...
    qkv = qkv.view(num_heads, 3, head_dim)  # [32, 3, 80]
    q = qkv[:, 0, :].unsqueeze(0)   # [1, 32, 80]
    k_new = qkv[:, 1, :].unsqueeze(0)  # [1, 32, 80]
    v_new = qkv[:, 2, :].unsqueeze(0)  # [1, 32, 80]

    if debug: 
        print("before RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 72: 80]}")
        print(f"k_new, head_id = {print_head}: first 8, last 8")
        print(f"{k_new[0, print_head, 0: 8]}")
        print(f"{k_new[0, print_head, 72: 80]}")

    # Apply RoPE only to first rotary_dim dimensions (Neox-style)
    q, k_new = apply_neox_style_rotary_pos_emb_partial(q, k_new, cos, sin, rotary_dim)

    if debug: 
        print("after RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 72: 80]}")

    q = q.reshape(num_heads, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0)  # [seqlen+1, num_heads, head_dim]
    v = torch.cat((kv_cache[1], v_new), dim=0)
    
    # Attention (use PyTorch native since FlashInfer doesn't support head_dim=80)
    q = q.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
    k = k.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seqlen+1, head_dim]
    v = v.transpose(0, 1).unsqueeze(0)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    o = torch.matmul(attn_weights, v)  # [1, num_heads, 1, head_dim]
    o = o.squeeze(0).squeeze(1)  # [num_heads, head_dim]
    
    if debug:
        print("attn output O")
        print(f"o, head_id = {print_head}:")
        print(f"{o[print_head, 0: 80]}")
    
    # Output projection
    o = torch.matmul(o.view(1, num_heads * head_dim), weight_o.t())
    
    if debug:
        print("final output o")
        print(o[0, 0:8])
        print(o[0, 2552:2560])
        print("-----------------------------  python end  -----------------------------")
    
    return o.detach()

def generate_random_weights(shape):
    """Generate random weights scaled appropriately"""
    return (torch.randn(shape) * 0.1).to(0).half()

def test_pythia_decode_correctness():
    """Test Pythia decoder layer correctness against reference implementation"""
    print(f"Testing Pythia-2.8b decoder layer")
    print(f"hidden_size: {hidden_size}, num_heads: {num_heads}, head_dim: {head_dim}")
    print(f"seqlen: {seqlen}, rotary_dim: {rotary_dim}")
    
    # Generate random weights and inputs
    input_tensor = generate_random_weights((1, hidden_size))
    
    # QKV weight: [7680, 2560] - interleaved layout
    weight_qkv = generate_random_weights((3 * num_heads * head_dim, hidden_size))
    bias_qkv = generate_random_weights((3 * num_heads * head_dim,))
    
    # Output projection weight and bias: [2560, 2560] and [2560]
    weight_o = generate_random_weights((hidden_size, hidden_size))
    # To match reference implementation (no bias), set bias_o to zeros
    bias_o = torch.zeros((hidden_size,), device=weight_o.device, dtype=weight_o.dtype)
    
    # LayerNorm weight and bias
    ln_weight = generate_random_weights((1, hidden_size))
    ln_bias = generate_random_weights((1, hidden_size))

    # KV cache: [2, seqlen, num_heads * head_dim]
    kv_cache_full = generate_random_weights((2, seqlen, num_heads * head_dim))

    # RoPE embeddings
    cos, sin = initialize_rope_embeddings(rotary_dim)
    
    # Our ClusterFusion kernel
    print("\n=== Running ClusterFusion Pythia Kernel ===")
    o_ours = []
    for i in range(test_run):
        input_clone = input_tensor.clone()
        output, k, v = clusterfusion.pythia_decoder_layer(
            input_clone,          
            weight_qkv,                          
            bias_qkv,             # QKV bias
            weight_o,
            bias_o,
            kv_cache_full[0],
            kv_cache_full[1],           
            ln_weight,
            ln_bias,              # LayerNorm bias
            cos,                   
            sin,
            seqlen                # Current sequence length
        )
        o_ours.append(output)
        if i == 0:
            print(f"First run output shape: {output.shape}")
            print(f"First run k shape: {k.shape}, v shape: {v.shape}")

    # Reference implementation
    print("\n=== Running Reference Implementation ===")
    eps = 1e-5
    ln_weight_flat = ln_weight.reshape((hidden_size,))
    ln_bias_flat = ln_bias.reshape((hidden_size,))

    # KV cache for reference: [2, seqlen, num_heads, head_dim]
    kv_cache_k = kv_cache_full[0].view(seqlen, num_heads, head_dim)
    kv_cache_v = kv_cache_full[1].view(seqlen, num_heads, head_dim)
    kv_cache_ref = torch.stack([kv_cache_k, kv_cache_v], dim=0)  # [2, seqlen, num_heads, head_dim]
    
    input_ref = input_tensor.clone()
    o_ref = pythia_decode_reference(
        input_ref, ln_weight_flat, ln_bias_flat, eps,
        kv_cache_ref, weight_qkv, bias_qkv, weight_o,
        head_dim, cos, sin, rotary_dim
    )
    
    print(f"\nReference output shape: {o_ref.shape}")
    print(f"Reference output abs mean: {o_ref.abs().mean().item()}")
    
    # Compare outputs
    print("\n=== Comparison ===")
    print("Ours[..., 0:80]:", o_ours[0][..., 0:80])
    print("Ref[..., 0:80]:", o_ref[..., 0:80])
    
    max_error_list = []
    mae_list = []
    mse_list = []
    
    for i in range(test_run):
        diff = (o_ours[i] - o_ref).abs()
        mae = diff.mean()
        mae_list.append(mae)
        mse = (diff ** 2).mean()
        mse_list.append(mse)
        max_error = diff.max()
        max_error_list.append(max_error)

    print(f"\n=== Error Statistics over {test_run} runs ===")
    print(f"Max MSE: {max(mse_list).item():.6f}")
    print(f"Min MSE: {min(mse_list).item():.6f}")
    print(f"Max MAE: {max(mae_list).item():.6f}")
    print(f"Min MAE: {min(mae_list).item():.6f}")
    print(f"Max absolute error: {max(max_error_list).item():.6f}")
    print(f"Min absolute error: {min(max_error_list).item():.6f}")
    
    avg_mae = sum(mae_list).item() / len(mae_list)
    print(f"\nAverage MAE: {avg_mae:.6f}")
    if avg_mae < 0.01:
        print("✓ TEST PASSED: Average error is acceptable")
    else:
        print("✗ TEST FAILED: Average error is too large")

if __name__ == "__main__":
    test_pythia_decode_correctness()
