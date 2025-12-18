"""
Test Pythia-6.9B decoder layer kernel correctness.

Pythia-6.9B vs LLaMA comparison:
- Both: HEAD_DIM=128, HEAD_NUM=32, HIDDEN_DIM=4096
- Pythia-6.9B: FFN_DIM=16384, LLaMA: FFN_DIM=12288
- Pythia-6.9B: LayerNorm with bias, LLaMA: RMSNorm (no bias)
- Pythia-6.9B: Partial RoPE (ROTARY_DIM=32), LLaMA: Full RoPE (HEAD_DIM=128)
- Pythia-6.9B: QKV interleaved layout with bias, LLaMA: no bias

Pythia-6.9B vs Pythia-2.8B comparison:
- 6.9B: HEAD_DIM=128, 2.8B: HEAD_DIM=80
- 6.9B: ROTARY_DIM=32, 2.8B: ROTARY_DIM=20
- 6.9B: FFN_DIM=16384, 2.8B: FFN_DIM=10240
- Both: LayerNorm with bias, interleaved QKV, parallel residual
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
import clusterfusion

# Pythia-6.9B model parameters
hidden_size = 4096
num_heads = 32
seqlen = 129  # Start with small seqlen for debugging
head_dim = hidden_size // num_heads  # 128
ffn_dim = 16384
rotary_dim = head_dim // 4  # 32 (rotary_pct = 0.25)

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
    Kernel expects rotary_dim size with float32.
    """
    angles = (torch.rand((1, rotary_dim), dtype=torch.float32) * (2 * torch.pi)).to(0)
    h_cos = torch.cos(angles)
    h_sin = torch.sin(angles)
    return h_cos.squeeze(0), h_sin.squeeze(0)  # [rotary_dim]


def apply_neox_style_rotary_pos_emb_partial(q, k, cos, sin, rotary_dim):
    """
    Apply Neox-style RoPE only to the first rotary_dim dimensions.
    For Pythia, rotary_pct=0.25, so only first 32 dims use RoPE.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, rotary_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
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


def gelu_approx(x):
    """GELU approximation used in kernel (matches PTX implementation)"""
    GELU_CONST = 0.7978845608028654  # sqrt(2/pi)
    GELU_COEF = 0.044715
    return x * 0.5 * (1.0 + torch.tanh(GELU_CONST * (x + GELU_COEF * x**3)))


def pythia_6b9_decode_reference(hidden, ln_weight, ln_bias, post_ln_weight, post_ln_bias, eps, kv_cache, 
                                 weight_qkv, bias_qkv, weight_o, bias_o,
                                 mlp_up_weight, mlp_up_bias, mlp_down_weight, mlp_down_bias,
                                 head_dim, cos, sin, rotary_dim):
    """
    Pythia-6.9B decoding reference implementation with parallel residual.
    Similar to LLaMA but with:
    - LayerNorm (with bias), not RMSNorm
    - Partial RoPE (only first rotary_dim=32 dimensions)
    - QKV interleaved layout with bias
    - Parallel residual: output = input + attn(ln(input)) + mlp(post_ln(input))
    """
    if debug:
        print("----------------------------- python begin -----------------------------")

    # LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, keepdim=True, unbiased=False)
    hidden_normed = (hidden - mean) / torch.sqrt(var + eps) * ln_weight + ln_bias
    
    # Post-attention LayerNorm for MLP (same input, different weights)
    post_ln_normed = (hidden - mean) / torch.sqrt(var + eps) * post_ln_weight + post_ln_bias
    
    if debug: 
        print("normed ref (first 128):", hidden_normed[..., 0:128])

    # QKV projection: output = input @ weight.T + bias
    # weight_qkv shape: [12288, 4096], bias_qkv shape: [12288]
    qkv = torch.matmul(hidden_normed, weight_qkv.t()) + bias_qkv  # [1, 12288]
    
    # Interleaved layout: reshape to [num_heads, 3, head_dim]
    # For head i: Q at [i*3*128 : i*3*128+128], K at [i*3*128+128 : i*3*128+256], V at [i*3*128+256 : i*3*128+384]
    qkv = qkv.view(num_heads, 3, head_dim)  # [32, 3, 128]
    q = qkv[:, 0, :].unsqueeze(0)   # [1, 32, 128]
    k_new = qkv[:, 1, :].unsqueeze(0)  # [1, 32, 128]
    v_new = qkv[:, 2, :].unsqueeze(0)  # [1, 32, 128]

    if debug: 
        print("before RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 120: 128]}")
        print(f"k_new, head_id = {print_head}: first 8, last 8")
        print(f"{k_new[0, print_head, 0: 8]}")
        print(f"{k_new[0, print_head, 120: 128]}")

    # Apply RoPE only to first rotary_dim dimensions (Neox-style)
    q, k_new = apply_neox_style_rotary_pos_emb_partial(q, k_new, cos, sin, rotary_dim)

    if debug: 
        print("after RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 120: 128]}")

    q = q.reshape(num_heads, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0)  # [seqlen+1, num_heads, head_dim]
    v = torch.cat((kv_cache[1], v_new), dim=0)
    
    # Attention using PyTorch native
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
        print(f"{o[print_head, 0: 128]}")
    
    # Attention output projection
    attn_out = torch.matmul(o.view(1, num_heads * head_dim), weight_o.t()) + bias_o
    
    # MLP: up -> gelu -> down
    mlp_up = torch.matmul(post_ln_normed, mlp_up_weight.t()) + mlp_up_bias  # [1, 16384]
    mlp_gelu = gelu_approx(mlp_up.float()).half()
    mlp_out = torch.matmul(mlp_gelu, mlp_down_weight.t()) + mlp_down_bias  # [1, 4096]
    
    # Parallel residual: output = input + attn_out + mlp_out
    final_output = hidden + attn_out + mlp_out
    
    if debug:
        print("final output o")
        print(final_output[0, 0:8])
        print(final_output[0, 4088:4096])
        print("-----------------------------  python end  -----------------------------")
    
    return final_output.detach()


def generate_random_weights(shape):
    """Generate random weights scaled appropriately"""
    return (torch.randn(shape) * 0.1).to(0).half()


def test_pythia_6b9_decode_correctness():
    """Test Pythia-6.9B decoder layer correctness against reference implementation"""
    print(f"Testing Pythia-6.9B decoder layer with MLP")
    print(f"hidden_size: {hidden_size}, num_heads: {num_heads}, head_dim: {head_dim}")
    print(f"seqlen: {seqlen}, rotary_dim: {rotary_dim}, ffn_dim: {ffn_dim}")
    print(f"Comparison with LLaMA: same HEAD_DIM=128, but with LayerNorm + bias + partial RoPE")
    
    # Generate random weights and inputs
    input_tensor = generate_random_weights((1, hidden_size))
    
    # QKV weight: [12288, 4096] - interleaved layout
    weight_qkv = generate_random_weights((3 * num_heads * head_dim, hidden_size))
    bias_qkv = generate_random_weights((3 * num_heads * head_dim,))
    
    # Output projection weight and bias: [4096, 4096] and [4096]
    weight_o = generate_random_weights((hidden_size, hidden_size))
    bias_o = generate_random_weights((hidden_size,))
    
    # LayerNorm weight and bias
    ln_weight = generate_random_weights((1, hidden_size))
    ln_bias = generate_random_weights((1, hidden_size))
    
    # Post-attention LayerNorm weight and bias (Pythia parallel residual)
    post_ln_weight = generate_random_weights((1, hidden_size))
    post_ln_bias = generate_random_weights((1, hidden_size))
    
    # MLP weights
    mlp_up_weight = generate_random_weights((ffn_dim, hidden_size))
    mlp_up_bias = generate_random_weights((ffn_dim,))
    mlp_down_weight = generate_random_weights((hidden_size, ffn_dim))
    mlp_down_bias = generate_random_weights((hidden_size,))

    # KV cache: [2, seqlen, num_heads * head_dim] = [2, seqlen, 4096]
    kv_cache_full = generate_random_weights((2, seqlen, num_heads * head_dim))

    # RoPE embeddings (kernel expects float32, shape [rotary_dim])
    cos, sin = initialize_rope_embeddings(rotary_dim)
    
    # Our ClusterFusion kernel
    print("\n=== Running ClusterFusion Pythia-6.9B Kernel ===")
    o_ours = []
    for i in range(test_run):
        input_clone = input_tensor.clone()
        output, k, v = clusterfusion.pythia_6b9_decoder_layer(
            input_clone,          
            weight_qkv,                          
            bias_qkv,             # QKV bias
            weight_o,
            bias_o,
            kv_cache_full[0],
            kv_cache_full[1],           
            ln_weight.squeeze(0),  # [hidden_size]
            ln_bias.squeeze(0),    # [hidden_size]
            cos,                   # [rotary_dim] float32
            sin,
            # MLP weights
            post_ln_weight.squeeze(0),
            post_ln_bias.squeeze(0),
            mlp_up_weight,
            mlp_up_bias,
            mlp_down_weight,
            mlp_down_bias,
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
    post_ln_weight_flat = post_ln_weight.reshape((hidden_size,))
    post_ln_bias_flat = post_ln_bias.reshape((hidden_size,))

    # KV cache for reference: [2, seqlen, num_heads, head_dim]
    kv_cache_k = kv_cache_full[0].view(seqlen, num_heads, head_dim)
    kv_cache_v = kv_cache_full[1].view(seqlen, num_heads, head_dim)
    kv_cache_ref = torch.stack([kv_cache_k, kv_cache_v], dim=0)  # [2, seqlen, num_heads, head_dim]
    
    input_ref = input_tensor.clone()
    o_ref = pythia_6b9_decode_reference(
        input_ref, ln_weight_flat, ln_bias_flat, 
        post_ln_weight_flat, post_ln_bias_flat, eps,
        kv_cache_ref, weight_qkv, bias_qkv, weight_o, bias_o,
        mlp_up_weight, mlp_up_bias, mlp_down_weight, mlp_down_bias,
        head_dim, cos, sin, rotary_dim
    )
    
    print(f"\nReference output shape: {o_ref.shape}")
    print(f"Reference output abs mean: {o_ref.abs().mean().item()}")
    
    # Compare outputs
    print("\n=== Comparison ===")
    print("Ours[..., 0:128]:", o_ours[0][..., 0:128])
    print("Ref[..., 0:128]:", o_ref[..., 0:128])
    
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
    test_pythia_6b9_decode_correctness()
