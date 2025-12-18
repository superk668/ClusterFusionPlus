"""
Correct PyTorch implementation matching HuggingFace Pythia-6.9B exactly
Based on actual HuggingFace source code analysis
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
import time

MODEL_NAME = "EleutherAI/pythia-6.9b"

# Pythia-6.9B configuration
NUM_HEADS = 32
HEAD_DIM = 128    # 4096 / 32 = 128
HIDDEN_SIZE = 4096
ROTARY_DIM = 32   # HEAD_DIM * 0.25 = 32

def pythia_layer_forward(
    hidden_states,  # [1, 1, hidden_size]
    k_cache, v_cache,  # [1, num_heads, seq_len, head_dim]
    layer,
    cos, sin,  # [1, 1, rotary_dim]
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
):
    """
    PyTorch implementation matching HuggingFace Pythia layer exactly
    """
    hidden_size = hidden_states.shape[-1]
    residual = hidden_states  # [1, 1, hidden_size]
    
    # ========== Attention Branch ==========
    # 1. Input LayerNorm
    attn_input = F.layer_norm(
        hidden_states, (hidden_size,), 
        layer.input_layernorm.weight, 
        layer.input_layernorm.bias, 
        eps=1e-5
    )
    
    # 2. QKV projection
    # HuggingFace GPTNeoX layout: [batch, seq, num_heads, 3*head_dim] -> transpose -> chunk
    qkv = F.linear(attn_input, layer.attention.query_key_value.weight, layer.attention.query_key_value.bias)
    qkv = qkv.view(1, 1, num_heads, 3 * head_dim)  # [1, 1, 32, 384]
    qkv = qkv.transpose(1, 2)  # [1, 32, 1, 384]
    q, k_new, v_new = qkv.chunk(3, dim=-1)  # Each [1, 32, 1, 128]
    
    # 3. Apply RoPE using HuggingFace's function
    q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
    
    # 4. Concatenate with KV cache
    k = torch.cat([k_cache, k_new], dim=2)  # [1, 32, seq_len+1, 128]
    v = torch.cat([v_cache, v_new], dim=2)
    
    # 5. Attention
    scale = head_dim ** -0.5
    attn_scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [1, 32, 1, seq_len+1]
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v.float()).half()  # [1, 32, 1, 128]
    
    # 6. Output projection
    attn_output = attn_output.transpose(1, 2)  # [1, 1, 32, 128]
    attn_output = attn_output.reshape(1, 1, num_heads * head_dim)  # [1, 1, hidden_size]
    attn_output = F.linear(attn_output, layer.attention.dense.weight, layer.attention.dense.bias)
    
    # ========== MLP Branch (parallel with attention!) ==========
    # LayerNorm on ORIGINAL input (not attention output!)
    mlp_input = F.layer_norm(
        residual, (hidden_size,), 
        layer.post_attention_layernorm.weight, 
        layer.post_attention_layernorm.bias, 
        eps=1e-5
    )
    
    # MLP
    mlp_hidden = F.linear(mlp_input, layer.mlp.dense_h_to_4h.weight, layer.mlp.dense_h_to_4h.bias)
    mlp_hidden = F.gelu(mlp_hidden)
    mlp_output = F.linear(mlp_hidden, layer.mlp.dense_4h_to_h.weight, layer.mlp.dense_4h_to_h.bias)
    
    # ========== Parallel Residual ==========
    # Pythia: x = x + attn(ln1(x)) + mlp(ln2(x))
    output = residual + attn_output + mlp_output
    
    return output, k_new, v_new

def generate_pytorch(model, tokenizer, prompt, num_new_tokens=20):
    """Generate using pure PyTorch implementation"""
    print(f"\n{'='*80}")
    print(f"PyTorch Implementation (Pythia-6.9B)")
    print(f"Prompt: '{prompt}', Generating {num_new_tokens} tokens")
    print(f"{'='*80}\n")
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    
    # Prefill with HuggingFace
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = [next_token.item()]
    print(f"First token: {next_token.item()} ('{tokenizer.decode([next_token.item()])}')")
    
    num_layers = len(model.gpt_neox.layers)
    
    # Keep KV cache in HuggingFace format: [batch, heads, seq, head_dim]
    kv_caches = list(past_key_values)  # Convert tuple to list
    
    # Decoding
    for step in range(num_new_tokens - 1):
        current_position = prompt_length + step
        
        # Embedding
        hidden_states = model.gpt_neox.embed_in(next_token).half()  # [1, 1, 4096]
        
        # Get position embeddings from model's rotary_emb
        position_ids = torch.tensor([[current_position]], device=device)
        cos, sin = model.gpt_neox.rotary_emb(hidden_states, position_ids)
        
        # Through all layers
        new_kv_caches = []
        for layer_idx in range(num_layers):
            layer = model.gpt_neox.layers[layer_idx]
            k_cache = kv_caches[layer_idx][0]  # [1, 32, seq_len, 128]
            v_cache = kv_caches[layer_idx][1]
            
            # Debug: track norms
            if step == 0 and layer_idx < 3:
                print(f"  PyTorch Layer {layer_idx} input norm: {hidden_states.float().norm().item():.2f}")
            
            hidden_states, new_k, new_v = pythia_layer_forward(
                hidden_states, k_cache, v_cache,
                layer, cos, sin,
                NUM_HEADS, HEAD_DIM
            )
            
            # Update cache - concatenate already done in layer, just save new KV
            new_k_cache = torch.cat([k_cache, new_k], dim=2)
            new_v_cache = torch.cat([v_cache, new_v], dim=2)
            new_kv_caches.append((new_k_cache, new_v_cache))
        
        kv_caches = new_kv_caches
        
        # Final LayerNorm
        hidden_states = F.layer_norm(
            hidden_states, (HIDDEN_SIZE,),
            model.gpt_neox.final_layer_norm.weight,
            model.gpt_neox.final_layer_norm.bias,
            eps=1e-5
        )
        
        # Logits
        logits = model.embed_out(hidden_states.squeeze(1))
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: token={next_token.item()}")
    
    full_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    
    print(f"\n{'='*80}")
    print(f"Generated text:\n{generated_text}")
    print(f"{'='*80}\n")
    
    return generated_text, generated_ids

if __name__ == "__main__":
    print("Loading Pythia-6.9B model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    prompt = "The meaning of life is"
    num_tokens = 20  # Reduce for faster testing
    
    # PyTorch implementation
    start = time.time()
    text_pytorch, ids_pytorch = generate_pytorch(model, tokenizer, prompt, num_tokens)
    time_pytorch = time.time() - start
    
    # HuggingFace reference
    print(f"{'='*80}")
    print(f"HuggingFace Reference")
    print(f"{'='*80}\n")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        start = time.time()
        output_ids_hf = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False, use_cache=True)
        time_hf = time.time() - start
    
    text_hf = tokenizer.decode(output_ids_hf[0], skip_special_tokens=True)
    ids_hf = output_ids_hf[0].tolist()
    
    print(f"Generated text:\n{text_hf}")
    
    # Compare
    print(f"\n{'='*80}")
    print(f"Comparison")
    print(f"{'='*80}")
    print(f"PyTorch time: {time_pytorch:.3f}s")
    print(f"HuggingFace time: {time_hf:.3f}s")
    print(f"\nText match: {text_pytorch == text_hf}")
    
    ids_pytorch_full = input_ids[0].tolist() + ids_pytorch
    print(f"Token IDs match: {ids_pytorch_full == ids_hf}")
    
    if ids_pytorch_full != ids_hf:
        print(f"\nPyTorch IDs: {ids_pytorch_full}")
        print(f"HuggingFace IDs: {ids_hf}")
        
        # Find first mismatch
        for i, (p, h) in enumerate(zip(ids_pytorch_full, ids_hf)):
            if p != h:
                print(f"\nFirst mismatch at position {i}:")
                print(f"  PyTorch: {p} ('{tokenizer.decode([p])}')")
                print(f"  HuggingFace: {h} ('{tokenizer.decode([h])}')")
                break
