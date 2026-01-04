"""
Benchmark ClusterFusion full decoder (attention + MLP) vs HuggingFace across
different numbers of generated tokens. Timing excludes one-time setup
(weight extraction, cache allocation, prefill). Only decode time is compared.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-2.8b"
TOKEN_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]
PROMPT = "The meaning of life is"

# Check if torch.compile is available
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile')


def compute_rope_embeddings(position, rotary_dim, head_dim, base=10000, device="cuda:0"):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    position_tensor = torch.tensor([position], dtype=torch.float32, device=device)
    freqs = torch.outer(position_tensor, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    # pad to HEAD_DIM
    padding_size = head_dim - rotary_dim
    cos = torch.cat([cos, torch.ones((1, padding_size), device=device)], dim=-1)
    sin = torch.cat([sin, torch.zeros((1, padding_size), device=device)], dim=-1)
    return cos, sin


def precompute_rope_embeddings(max_position, rotary_dim, head_dim, base=10000, device="cuda:0"):
    """Precompute all RoPE embeddings up to max_position."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    positions = torch.arange(0, max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    # pad to HEAD_DIM
    padding_size = head_dim - rotary_dim
    cos = torch.cat([cos, torch.ones((max_position, padding_size), device=device)], dim=-1)
    sin = torch.cat([sin, torch.zeros((max_position, padding_size), device=device)], dim=-1)
    return cos, sin  # [max_position, head_dim]


def prepare_setup(model, tokenizer, prompt, num_new_tokens):
    """Prefill + weight extraction + cache allocation. Returns setup_time and state."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        first_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    num_layers = len(model.gpt_neox.layers)
    hidden_size = 2560
    max_seq_len = prompt_length + num_new_tokens

    all_weights = []
    kv_caches = []
    for layer_idx in range(num_layers):
        layer = model.gpt_neox.layers[layer_idx]
        weights = {
            "ln_weight": layer.input_layernorm.weight.data.unsqueeze(0).half(),
            "ln_bias": layer.input_layernorm.bias.data.unsqueeze(0).half(),
            "qkv_weight": layer.attention.query_key_value.weight.data.half(),
            "qkv_bias": layer.attention.query_key_value.bias.data.half(),
            "o_weight": layer.attention.dense.weight.data.half(),
            "o_bias": layer.attention.dense.bias.data.half(),
            "post_ln_weight": layer.post_attention_layernorm.weight.data.unsqueeze(0).half(),
            "post_ln_bias": layer.post_attention_layernorm.bias.data.unsqueeze(0).half(),
            "mlp_up_weight": layer.mlp.dense_h_to_4h.weight.data.half(),
            "mlp_up_bias": layer.mlp.dense_h_to_4h.bias.data.half(),
            "mlp_down_weight": layer.mlp.dense_4h_to_h.weight.data.half(),
            "mlp_down_bias": layer.mlp.dense_4h_to_h.bias.data.half(),
        }
        all_weights.append(weights)

        k = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1).contiguous()
        v = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1).contiguous()
        k = k.reshape(k.shape[0], -1)
        v = v.reshape(v.shape[0], -1)

        k_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
        v_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
        k_cache_full[: k.shape[0]] = k
        v_cache_full[: v.shape[0]] = v
        kv_caches.append((k_cache_full, v_cache_full, k.shape[0]))

    torch.cuda.synchronize()
    setup_time = time.time() - start
    return {
        "input_ids": input_ids,
        "prompt_length": prompt_length,
        "first_token": first_token,
        "all_weights": all_weights,
        "kv_caches": kv_caches,
        "setup_time": setup_time,
    }


def decode_clusterfusion(model, prompt, num_new_tokens, state, use_cuda_graph=False):
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)
    num_heads = 32
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20

    next_token = state["first_token"]
    generated_ids = [next_token.item()]
    input_ids = state["input_ids"]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [
        (k.clone(), v.clone(), cur_len)
        for (k, v, cur_len) in state["kv_caches"]
    ]

    # Precompute all RoPE embeddings
    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope_embeddings(max_position, rotary_dim, head_dim, device=device)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            # Use precomputed RoPE
            cos = all_cos[current_position:current_position+1]
            sin = all_sin[current_position:current_position+1]

            for layer_idx in range(num_layers):
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]

                hidden_states, _, _ = clusterfusion.pythia_2b8_decoder_layer(
                    hidden_states,
                    weights["qkv_weight"],
                    weights["qkv_bias"],
                    weights["o_weight"],
                    weights["o_bias"],
                    k_cache_full,
                    v_cache_full,
                    weights["ln_weight"],
                    weights["ln_bias"],
                    cos,
                    sin,
                    weights["post_ln_weight"],
                    weights["post_ln_bias"],
                    weights["mlp_up_weight"],
                    weights["mlp_up_bias"],
                    weights["mlp_down_weight"],
                    weights["mlp_down_bias"],
                    current_len,
                )
                kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)

            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                (hidden_size,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, generated_ids


def decode_clusterfusion_graph_context(model, prompt, num_new_tokens, state):
    """
    Decode using pre-created graph contexts (TensorMaps created once).
    This reduces CPU overhead by reusing TensorMaps across steps.
    """
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20

    next_token = state["first_token"]
    generated_ids = [next_token.item()]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [
        (k.clone(), v.clone(), cur_len)
        for (k, v, cur_len) in state["kv_caches"]
    ]

    # Precompute all RoPE embeddings
    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope_embeddings(max_position, rotary_dim, head_dim, device=device)

    # Create graph contexts for all layers (one-time TensorMap creation)
    max_seq_len = prompt_length + num_new_tokens
    for layer_idx in range(num_layers):
        weights = all_weights[layer_idx]
        k_cache_full, v_cache_full, _ = kv_caches[layer_idx]
        clusterfusion.pythia_2b8_create_graph_context(
            layer_idx,  # context_id
            k_cache_full,
            v_cache_full,
            weights["qkv_weight"],
            weights["o_weight"],
            weights["mlp_up_weight"],
            weights["mlp_down_weight"],
            max_seq_len,
        )

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            cos = all_cos[current_position:current_position+1]
            sin = all_sin[current_position:current_position+1]

            for layer_idx in range(num_layers):
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]

                output, _, _ = clusterfusion.pythia_2b8_graph_decode_step(
                    layer_idx,  # context_id
                    hidden_states,
                    weights["ln_weight"],
                    weights["ln_bias"],
                    weights["qkv_bias"],
                    weights["o_bias"],
                    cos,
                    sin,
                    k_cache_full,
                    v_cache_full,
                    weights["post_ln_weight"],
                    weights["post_ln_bias"],
                    weights["mlp_up_bias"],
                    weights["mlp_down_bias"],
                    current_len,
                )
                # Clone because output is a static buffer that will be overwritten
                hidden_states = output.clone()
                kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)

            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                (hidden_size,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start

    # Clean up contexts
    for layer_idx in range(num_layers):
        clusterfusion.pythia_2b8_destroy_graph_context(layer_idx)

    return decode_time, generated_ids


def decode_clusterfusion_split(model, prompt, num_new_tokens, state):
    """
    Decode using split kernels (Attention+MLPUp and MLPDown as separate kernels).
    No cooperative launch needed, uses regular kernel launches.
    """
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20

    next_token = state["first_token"]
    generated_ids = [next_token.item()]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [
        (k.clone(), v.clone(), cur_len)
        for (k, v, cur_len) in state["kv_caches"]
    ]

    # Precompute all RoPE embeddings
    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope_embeddings(max_position, rotary_dim, head_dim, device=device)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            cos = all_cos[current_position:current_position+1]
            sin = all_sin[current_position:current_position+1]

            for layer_idx in range(num_layers):
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]

                hidden_states, _, _ = clusterfusion.pythia_2b8_decoder_layer_split(
                    hidden_states,
                    weights["qkv_weight"],
                    weights["qkv_bias"],
                    weights["o_weight"],
                    weights["o_bias"],
                    k_cache_full,
                    v_cache_full,
                    weights["ln_weight"],
                    weights["ln_bias"],
                    cos,
                    sin,
                    weights["post_ln_weight"],
                    weights["post_ln_bias"],
                    weights["mlp_up_weight"],
                    weights["mlp_up_bias"],
                    weights["mlp_down_weight"],
                    weights["mlp_down_bias"],
                    current_len,
                )
                kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)

            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                (hidden_size,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, generated_ids


def decode_hf(model, input_ids, num_new_tokens):
    """
    HuggingFace decode-only timing.
    First do prefill OUTSIDE timing, then measure only decode phase.
    """
    # Prefill outside timing
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    
    generated_ids = input_ids[0].tolist() + [next_token.item()]
    
    # Decode-only timing
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_new_tokens - 1):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())
    
    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, generated_ids


def decode_cuda_attn_pytorch_mlp(model, prompt, num_new_tokens, state):
    """
    Hybrid: CUDA Attention+MLPUp kernel + PyTorch MLP Down.
    This measures the actual performance of using only the attention kernel.
    """
    import torch.nn.functional as F
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20

    next_token = state["first_token"]
    generated_ids = [next_token.item()]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [
        (k.clone(), v.clone(), cur_len)
        for (k, v, cur_len) in state["kv_caches"]
    ]

    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope_embeddings(max_position, rotary_dim, head_dim, device=device)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            cos = all_cos[current_position:current_position+1]
            sin = all_sin[current_position:current_position+1]

            for layer_idx in range(num_layers):
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]
                input_for_residual = hidden_states.clone()

                # CUDA kernel: Attention + MLP Up
                attn_output, mlp_intermediate, _, _ = clusterfusion.pythia_2b8_attention_only(
                    hidden_states,
                    weights["qkv_weight"],
                    weights["qkv_bias"],
                    weights["o_weight"],
                    weights["o_bias"],
                    k_cache_full,
                    v_cache_full,
                    weights["ln_weight"],
                    weights["ln_bias"],
                    cos,
                    sin,
                    weights["post_ln_weight"],
                    weights["post_ln_bias"],
                    weights["mlp_up_weight"],
                    weights["mlp_up_bias"],
                    current_len,
                )
                
                # PyTorch: MLP Down + Residual
                mlp_down = F.linear(mlp_intermediate, weights["mlp_down_weight"], weights["mlp_down_bias"])
                hidden_states = input_for_residual + attn_output + mlp_down
                
                kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)

            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                (hidden_size,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, generated_ids


def decode_pytorch_attn_cuda_mlp(model, prompt, num_new_tokens, state):
    """
    Hybrid: PyTorch Attention+MLPUp + CUDA MLP Down kernel.
    This measures the actual performance of using only the MLP kernel.
    """
    import torch.nn.functional as F
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)
    num_heads = 32
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20

    next_token = state["first_token"]
    generated_ids = [next_token.item()]
    prompt_length = state["prompt_length"]
    all_weights = state["all_weights"]
    kv_caches = [
        (k.clone(), v.clone(), cur_len)
        for (k, v, cur_len) in state["kv_caches"]
    ]

    max_position = prompt_length + num_new_tokens
    all_cos, all_sin = precompute_rope_embeddings(max_position, rotary_dim, head_dim, device=device)

    def apply_rotary_emb(x, cos, sin, rotary_dim):
        """Apply rotary embeddings to first rotary_dim dimensions."""
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        cos_t = cos[:, :rotary_dim].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, rotary_dim]
        sin_t = sin[:, :rotary_dim].unsqueeze(0).unsqueeze(0)
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        cos_t = cos_t[..., ::2]
        sin_t = sin_t[..., ::2]
        x_rot_out = torch.cat([
            x1 * cos_t - x2 * sin_t,
            x1 * sin_t + x2 * cos_t
        ], dim=-1)
        return torch.cat([x_rot_out, x_pass], dim=-1)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            cos = all_cos[current_position:current_position+1]
            sin = all_sin[current_position:current_position+1]

            for layer_idx in range(num_layers):
                layer = model.gpt_neox.layers[layer_idx]
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]
                input_for_residual = hidden_states.clone()

                # PyTorch: LayerNorm
                ln_out = F.layer_norm(
                    hidden_states.float(),
                    (hidden_size,),
                    weights["ln_weight"].squeeze(0).float(),
                    weights["ln_bias"].squeeze(0).float(),
                    eps=1e-5,
                ).half()

                # PyTorch: QKV projection
                qkv = F.linear(ln_out, weights["qkv_weight"], weights["qkv_bias"])
                qkv = qkv.view(1, 1, num_heads, 3, head_dim)
                q = qkv[:, :, :, 0, :]
                k = qkv[:, :, :, 1, :]
                v = qkv[:, :, :, 2, :]

                # PyTorch: RoPE
                q = apply_rotary_emb(q, cos, sin, rotary_dim)
                k = apply_rotary_emb(k, cos, sin, rotary_dim)

                # Update KV cache
                k_flat = k.reshape(1, -1)
                v_flat = v.reshape(1, -1)
                k_cache_full[current_len] = k_flat
                v_cache_full[current_len] = v_flat

                # PyTorch: Attention
                k_cache = k_cache_full[:current_len+1].view(current_len+1, num_heads, head_dim)
                v_cache = v_cache_full[:current_len+1].view(current_len+1, num_heads, head_dim)
                q = q.squeeze(1)  # [1, num_heads, head_dim]
                scores = torch.einsum('bhd,shd->bhs', q.float(), k_cache.float()) / (head_dim ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                attn_out = torch.einsum('bhs,shd->bhd', attn_weights, v_cache.float()).half()

                # PyTorch: Output projection
                attn_out_flat = attn_out.view(1, -1)
                attn_output = F.linear(attn_out_flat, weights["o_weight"], weights["o_bias"])

                # PyTorch: Post-LayerNorm
                post_ln = F.layer_norm(
                    ln_out.float(),  # Pythia uses parallel residual, so post_ln input is ln_out
                    (hidden_size,),
                    weights["post_ln_weight"].squeeze(0).float(),
                    weights["post_ln_bias"].squeeze(0).float(),
                    eps=1e-5,
                ).half()

                # PyTorch: MLP Up + GELU
                mlp_up = F.linear(post_ln, weights["mlp_up_weight"], weights["mlp_up_bias"])
                mlp_intermediate = F.gelu(mlp_up, approximate='tanh')

                # CUDA kernel: MLP Down + Residual
                hidden_states = clusterfusion.pythia_2b8_mlp_only(
                    input_for_residual,
                    attn_output,
                    mlp_intermediate.squeeze(0),  # [FFN_DIM]
                    weights["mlp_down_weight"],
                    weights["mlp_down_bias"],
                )

                kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)

            hidden_states = torch.nn.functional.layer_norm(
                hidden_states,
                (hidden_size,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start
    return decode_time, generated_ids


def main():
    print("=" * 80)
    print("Pythia-2.8B Benchmark: ClusterFusion vs HuggingFace")
    print("=" * 80)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Model: {MODEL_NAME}")
    print(f"Params: hidden=2560, heads=32, head_dim=80, layers=32")

    # Warmup
    print("\nWarming up...")
    warmup_state = prepare_setup(model, tokenizer, PROMPT, 8)
    decode_clusterfusion(model, PROMPT, 8, warmup_state)
    warmup_state = prepare_setup(model, tokenizer, PROMPT, 8)
    decode_clusterfusion_graph_context(model, PROMPT, 8, warmup_state)
    warmup_state = prepare_setup(model, tokenizer, PROMPT, 8)
    decode_clusterfusion_split(model, PROMPT, 8, warmup_state)
    warmup_state = prepare_setup(model, tokenizer, PROMPT, 8)
    decode_cuda_attn_pytorch_mlp(model, PROMPT, 8, warmup_state)
    warmup_state = prepare_setup(model, tokenizer, PROMPT, 8)
    decode_pytorch_attn_cuda_mlp(model, PROMPT, 8, warmup_state)
    decode_hf(model, warmup_state["input_ids"], 8)
    torch.cuda.synchronize()

    results = []
    for num_tokens in TOKEN_COUNTS:
        state = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cf_time, ids_kernel = decode_clusterfusion(model, PROMPT, num_tokens, state)
        
        # Re-prepare state for graph context test (since KV caches are modified)
        state2 = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cf_graph_time, ids_graph = decode_clusterfusion_graph_context(model, PROMPT, num_tokens, state2)
        
        # Re-prepare state for split kernel test
        state3 = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cf_split_time, ids_split = decode_clusterfusion_split(model, PROMPT, num_tokens, state3)
        
        # Hybrid test: CUDA Attention + PyTorch MLP
        state4 = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cuda_attn_time, ids_cuda_attn = decode_cuda_attn_pytorch_mlp(model, PROMPT, num_tokens, state4)
        
        # Hybrid test: PyTorch Attention + CUDA MLP
        state5 = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cuda_mlp_time, ids_cuda_mlp = decode_pytorch_attn_cuda_mlp(model, PROMPT, num_tokens, state5)
        
        hf_time, ids_hf = decode_hf(model, state["input_ids"], num_tokens)

        results.append(
            {
                "tokens": num_tokens,
                "cf_decode_s": cf_time,
                "cf_graph_s": cf_graph_time,
                "cf_split_s": cf_split_time,
                "cuda_attn_s": cuda_attn_time,
                "cuda_mlp_s": cuda_mlp_time,
                "hf_decode_s": hf_time,
                "speedup_cf": hf_time / cf_time if cf_time > 0 else float("inf"),
                "speedup_graph": hf_time / cf_graph_time if cf_graph_time > 0 else float("inf"),
                "speedup_split": hf_time / cf_split_time if cf_split_time > 0 else float("inf"),
                "speedup_cuda_attn": hf_time / cuda_attn_time if cuda_attn_time > 0 else float("inf"),
                "speedup_cuda_mlp": hf_time / cuda_mlp_time if cuda_mlp_time > 0 else float("inf"),
                "match": ids_hf == (state["input_ids"][0].tolist() + ids_kernel),
                "match_graph": ids_hf == (state["input_ids"][0].tolist() + ids_graph),
                "match_split": ids_hf == (state["input_ids"][0].tolist() + ids_split),
            }
        )

    print("\n" + "=" * 110)
    print("Results (decode time only, excluding prefill/setup)")
    print("=" * 110)
    header = f"{'tokens':>8} | {'CF(s)':>8} | {'Graph(s)':>8} | {'Split(s)':>8} | {'HF(s)':>8} | {'CF↑':>6} | {'Graph↑':>6} | {'Split↑':>6} | match"
    print(header)
    print("-" * 110)
    for r in results:
        match_str = "✅" if r['match'] and r['match_graph'] and r['match_split'] else "⚠️"
        print(
            f"{r['tokens']:8d} | {r['cf_decode_s']:8.3f} | {r['cf_graph_s']:8.3f} | {r['cf_split_s']:8.3f} | {r['hf_decode_s']:8.3f} | {r['speedup_cf']:5.2f}x | {r['speedup_graph']:5.2f}x | {r['speedup_split']:5.2f}x | {match_str}"
        )
    
    # Summary
    print("\n" + "=" * 110)
    print("Summary")
    print("=" * 110)
    avg_cf = sum(r['speedup_cf'] for r in results) / len(results)
    avg_graph = sum(r['speedup_graph'] for r in results) / len(results)
    avg_split = sum(r['speedup_split'] for r in results) / len(results)
    print(f"Average Fused speedup: {avg_cf:.2f}x")
    print(f"Average Graph speedup: {avg_graph:.2f}x")
    print(f"Average Split speedup: {avg_split:.2f}x")
    print(f"Max Graph speedup:     {max(r['speedup_graph'] for r in results):.2f}x")
    print(f"Max Split speedup:     {max(r['speedup_split'] for r in results):.2f}x")
    
    # Analysis
    print("\n" + "=" * 110)
    print("Analysis: Fused vs Split")
    print("=" * 110)
    print(f"Graph vs Fused improvement: +{(avg_graph/avg_cf - 1)*100:.1f}%")
    print(f"Split vs Fused overhead:    {(avg_split/avg_cf - 1)*100:+.1f}%")
    
    # Actual Hybrid Measurements
    print("\n" + "=" * 110)
    print("Hybrid Configuration Results (Actual Measurements)")
    print("=" * 110)
    print("CUDA Attn+Up + PT Down: Uses CUDA kernel for Attention+MLPUp, PyTorch for MLPDown+Residual")
    print("PT Attn+Up + CUDA Down: Uses PyTorch for Attention+MLPUp, CUDA kernel for MLPDown+Residual")
    print()
    print(f"{'Tokens':>8} | {'CUDA Attn':>12} | {'CUDA MLP':>12} | {'Attn↑':>8} | {'MLP↑':>8}")
    print(f"{'':>8} | {'+PT Down(s)':>12} | {'+PT Attn(s)':>12} | {'':>8} | {'':>8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['tokens']:>8} | {r['cuda_attn_s']:>12.3f} | {r['cuda_mlp_s']:>12.3f} | "
              f"{r['speedup_cuda_attn']:>7.2f}x | {r['speedup_cuda_mlp']:>7.2f}x")
    
    # Summary of hybrid speedups
    print("-" * 70)
    avg_cuda_attn = sum(r['speedup_cuda_attn'] for r in results) / len(results)
    avg_cuda_mlp = sum(r['speedup_cuda_mlp'] for r in results) / len(results)
    print(f"\nAverage CUDA Attn+Up only: {avg_cuda_attn:.2f}x (主要加速来源)")
    print(f"Average CUDA MLPDown only: {avg_cuda_mlp:.2f}x")
    
    if avg_cf > 1:
        attn_contribution = (avg_cuda_attn - 1) / (avg_cf - 1) * 100
        mlp_contribution = (avg_cuda_mlp - 1) / (avg_cf - 1) * 100 if avg_cuda_mlp > 1 else 0
        print(f"\n结论: Attention+MLPUp 贡献了 {attn_contribution:.0f}% 的加速")
        print(f"      MLPDown 贡献了 {mlp_contribution:.0f}% 的加速")


if __name__ == "__main__":
    main()


