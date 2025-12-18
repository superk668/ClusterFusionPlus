"""
Benchmark ClusterFusion Pythia-6.9B decoder vs HuggingFace.
Timing excludes one-time setup (weight extraction, cache allocation, prefill).
Only decode time is compared.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-6.9b"
TOKEN_COUNTS = [16, 32, 64, 128, 256, 512, 1024, 2048]
PROMPT = "The meaning of life is"

# Pythia-6.9B parameters
HIDDEN_SIZE = 4096
NUM_HEADS = 32
HEAD_DIM = 128
ROTARY_DIM = 32
FFN_DIM = 16384


def precompute_rope_embeddings(max_position, rotary_dim, device="cuda:0"):
    """Precompute all RoPE embeddings up to max_position."""
    base = 10000
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(0, max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin  # [max_position, rotary_dim]


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
    max_seq_len = prompt_length + num_new_tokens

    all_weights = []
    kv_caches = []
    for layer_idx in range(num_layers):
        layer = model.gpt_neox.layers[layer_idx]
        weights = {
            "ln_weight": layer.input_layernorm.weight.data.half(),
            "ln_bias": layer.input_layernorm.bias.data.half(),
            "qkv_weight": layer.attention.query_key_value.weight.data.half(),
            "qkv_bias": layer.attention.query_key_value.bias.data.half(),
            "o_weight": layer.attention.dense.weight.data.half(),
            "o_bias": layer.attention.dense.bias.data.half(),
            "post_ln_weight": layer.post_attention_layernorm.weight.data.half(),
            "post_ln_bias": layer.post_attention_layernorm.bias.data.half(),
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

        k_cache_full = torch.zeros((max_seq_len, HIDDEN_SIZE), dtype=torch.float16, device=device)
        v_cache_full = torch.zeros((max_seq_len, HIDDEN_SIZE), dtype=torch.float16, device=device)
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


def decode_clusterfusion(model, prompt, num_new_tokens, state, use_graph=False):
    device = next(model.parameters()).device
    num_layers = len(model.gpt_neox.layers)

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
    all_cos, all_sin = precompute_rope_embeddings(max_position, ROTARY_DIM, device=device)

    # Create graph contexts if using graph mode
    if use_graph:
        for layer_idx in range(num_layers):
            weights = all_weights[layer_idx]
            k_cache_full, v_cache_full, _ = kv_caches[layer_idx]
            clusterfusion.pythia_6b9_create_graph_context(
                layer_idx,
                k_cache_full,
                v_cache_full,
                weights["qkv_weight"],
                weights["o_weight"],
                weights["mlp_up_weight"],
                weights["mlp_down_weight"],
                max_position,
            )

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for step in range(num_new_tokens - 1):
            current_position = prompt_length + step
            hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
            # Use precomputed RoPE
            cos = all_cos[current_position:current_position+1].squeeze(0)
            sin = all_sin[current_position:current_position+1].squeeze(0)

            for layer_idx in range(num_layers):
                weights = all_weights[layer_idx]
                k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]

                if use_graph:
                    hidden_states, _, _ = clusterfusion.pythia_6b9_graph_decode_step(
                        layer_idx,
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
                else:
                    hidden_states, _, _ = clusterfusion.pythia_6b9_decoder_layer(
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
                (HIDDEN_SIZE,),
                model.gpt_neox.final_layer_norm.weight.data,
                model.gpt_neox.final_layer_norm.bias.data,
                eps=1e-5,
            )
            logits = model.embed_out(hidden_states)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids.append(next_token.item())

    torch.cuda.synchronize()
    decode_time = time.time() - start
    
    # Cleanup graph contexts
    if use_graph:
        for layer_idx in range(num_layers):
            clusterfusion.pythia_6b9_destroy_graph_context(layer_idx)
    
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


def main():
    print("=" * 60)
    print("Pythia-6.9B Benchmark: ClusterFusion vs HuggingFace")
    print("=" * 60)
    
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Model: {MODEL_NAME}")
    print(f"Params: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}, head_dim={HEAD_DIM}, layers={len(model.gpt_neox.layers)}")

    # Warmup
    print("\nWarming up...")
    warmup_state = prepare_setup(model, tokenizer, PROMPT, 8)
    decode_clusterfusion(model, PROMPT, 8, warmup_state, use_graph=False)
    decode_clusterfusion(model, PROMPT, 8, warmup_state, use_graph=True)
    decode_hf(model, warmup_state["input_ids"], 8)
    torch.cuda.synchronize()

    results = []
    for num_tokens in TOKEN_COUNTS:
        print(f"\nTesting {num_tokens} tokens...")
        
        state = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cf_time, ids_kernel = decode_clusterfusion(model, PROMPT, num_tokens, state, use_graph=False)
        
        state_graph = prepare_setup(model, tokenizer, PROMPT, num_tokens)
        cf_graph_time, ids_graph = decode_clusterfusion(model, PROMPT, num_tokens, state_graph, use_graph=True)
        
        hf_time, ids_hf = decode_hf(model, state["input_ids"], num_tokens)
        
        # Check if generated tokens match
        expected_ids = state["input_ids"][0].tolist() + ids_kernel
        match = ids_hf == expected_ids
        
        expected_ids_graph = state_graph["input_ids"][0].tolist() + ids_graph
        match_graph = ids_hf == expected_ids_graph
        
        # Count matching tokens
        min_len = min(len(ids_hf), len(expected_ids))
        matching = sum(1 for i in range(min_len) if ids_hf[i] == expected_ids[i])
        
        min_len_graph = min(len(ids_hf), len(expected_ids_graph))
        matching_graph = sum(1 for i in range(min_len_graph) if ids_hf[i] == expected_ids_graph[i])

        results.append({
            "tokens": num_tokens,
            "cf_decode_s": cf_time,
            "cf_graph_s": cf_graph_time,
            "hf_decode_s": hf_time,
            "speedup": hf_time / cf_time if cf_time > 0 else float("inf"),
            "speedup_graph": hf_time / cf_graph_time if cf_graph_time > 0 else float("inf"),
            "match": match,
            "match_graph": match_graph,
            "matching_tokens": matching,
            "matching_graph": matching_graph,
            "total_tokens": len(ids_hf),
        })

    print("\n" + "=" * 90)
    print("Results (decode time only, excluding prefill/setup)")
    print("=" * 90)
    header = f"{'tokens':>8} | {'CF(s)':>8} | {'Graph(s)':>8} | {'HF(s)':>8} | {'CF↑':>6} | {'Graph↑':>6} | {'match'}"
    print(header)
    print("-" * 90)
    for r in results:
        detail = f"CF:{r['matching_tokens']}/{r['total_tokens']}, Graph:{r['matching_graph']}/{r['total_tokens']}"
        print(
            f"{r['tokens']:8d} | {r['cf_decode_s']:8.3f} | {r['cf_graph_s']:8.3f} | {r['hf_decode_s']:8.3f} | {r['speedup']:6.2f}x | {r['speedup_graph']:6.2f}x | {detail}"
        )

    # Show sample generation
    print("\n" + "=" * 90)
    print("Sample generation (first prompt)")
    print("=" * 90)
    state = prepare_setup(model, tokenizer, PROMPT, 20)
    _, ids_kernel = decode_clusterfusion(model, PROMPT, 20, state, use_graph=False)
    
    state_graph = prepare_setup(model, tokenizer, PROMPT, 20)
    _, ids_graph = decode_clusterfusion(model, PROMPT, 20, state_graph, use_graph=True)
    
    _, ids_hf = decode_hf(model, state["input_ids"], 20)
    
    kernel_text = tokenizer.decode(state["input_ids"][0].tolist() + ids_kernel)
    graph_text = tokenizer.decode(state_graph["input_ids"][0].tolist() + ids_graph)
    hf_text = tokenizer.decode(ids_hf)
    
    print(f"\n[HuggingFace]:")
    print(f"  {hf_text}")
    print(f"\n[ClusterFusion]:")
    print(f"  {kernel_text}")
    print(f"\n[ClusterFusion+Graph]:")
    print(f"  {graph_text}")
    print(f"\n[Match CF]: {hf_text == kernel_text}")
    print(f"[Match Graph]: {hf_text == graph_text}")


if __name__ == "__main__":
    main()

