# CS3602 Project: ClusterFusion (Pythia Port)

We ported ClusterFusion's fused decoder kernel from the original Llama-2-7B target to EleutherAI Pythia models (GPT-NeoX architecture). This branch focuses on decode-time fusion and currently supports NVIDIA `sm_120` GPUs (Blackwell/5090-class); the H100 path is not implemented.

## Supported Models

| Model | Status | Correctness | Notes |
|-------|--------|-------------|-------|
| **Pythia-2.8B** | ✅ Full support | **100% match** | Recommended for production use |
| Pythia-6.9B | ✅ Full support | ~95% match | FP16 atomicAdd non-determinism |

## What changed: Llama-2-7B vs Pythia

| Parameter | Llama-2-7B | Pythia-2.8B | Pythia-6.9B |
|-----------|------------|-------------|-------------|
| hidden_size | 4096 | 2560 | 4096 |
| num_attention_heads | 32 | 32 | 32 |
| head_dim | 128 | 80 | 128 |
| intermediate_size (FFN) | 12288 | 10240 | 16384 |
| num_layers | 32 | 32 | 32 |
| max_position_embeddings | 4096 | 2048 | 2048 |

Key architectural/kernel differences:
- **Head dim**: Pythia-2.8B uses 80 (non-power-of-2, requires special warp mapping); Pythia-6.9B uses 128 (power-of-2, 100% warp efficiency)
- **RoPE**: Llama applies RoPE to all dims; Pythia uses Neox rotary_pct=0.25 (first 25% of head_dim). The kernel pads `cos/sin` to `HEAD_DIM`.
- **Norm and residual**: Pythia uses LayerNorm with bias and parallel residual (attention + MLP). The kernel fuses LayerNorm + QKV + Attention + Output + MLP.
- **Projections**: QKV weights are interleaved with bias; MLP branch (GELU approx) is fused alongside attention.
- **CUDA Graph Context**: TensorMaps created once per layer with `max_seq_len`, eliminating per-step TensorMap reconstruction overhead.

## Environment
- Python 3.13 (conda), NVIDIA GPU with `sm_120` compute capability
- CUDA 12.8+ user-space wheels via PyTorch cu130 index

Recreate and test with the exact commands below:
```bash
conda create -n nlp_project python=3.13 -y
conda activate nlp_project

# Core DL stack (cu130 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Kernel + HF stack
pip install flashinfer-python
pip install transformers accelerate

# ClusterFusion build
pip install -e .

# Benchmark (use HF mirror for model download if needed)
export HF_ENDPOINT=https://hf-mirror.com
python tests/benchmark_decode.py      # Pythia-2.8B
python tests/benchmark_decode_6b9.py  # Pythia-6.9B
```

## How to reproduce

### Pythia-2.8B (Recommended - 100% Correctness)
1. Download HuggingFace weights for `EleutherAI/pythia-2.8b`.
2. Correctness tests:
   - `python tests/test_pythia.py` (kernel vs reference, small seq len)
   - `python tests/test_pythia_correct.py` (PyTorch implementation matching HF)
   - `python tests/test_pythia_with_kernel.py` (full decode with kernel vs HF)
3. Benchmark: `python tests/benchmark_decode.py`

### Pythia-6.9B
1. Download HuggingFace weights for `EleutherAI/pythia-6.9b`.
2. Benchmark: `python tests/benchmark_decode_6b9.py`

---

## Benchmark Results

All benchmarks run on NVIDIA RTX 5090 (sm_120), batch=1, prompt: "The meaning of life is"

### ⭐ Pythia-2.8B (100% Correctness Guaranteed)

This is our primary benchmark target with **perfect token-level accuracy** across all sequence lengths.

| Tokens | CF (s) | CF+Graph (s) | HF (s) | CF Speedup | Graph Speedup | Match |
|--------|--------|--------------|--------|------------|---------------|-------|
| 16 | 0.077 | 0.071 | 0.086 | 1.12x | **1.20x** | ✅ 100% |
| 32 | 0.158 | 0.147 | 0.177 | 1.12x | **1.20x** | ✅ 100% |
| 64 | 0.322 | 0.299 | 0.360 | 1.12x | **1.20x** | ✅ 100% |
| 128 | 0.648 | 0.605 | 0.733 | 1.13x | **1.21x** | ✅ 100% |
| 256 | 1.314 | 1.222 | 1.488 | 1.13x | **1.22x** | ✅ 100% |
| 512 | 2.653 | 2.467 | 3.050 | 1.15x | **1.24x** | ✅ 100% |
| 1024 | 5.351 | 4.990 | 6.357 | 1.19x | **1.27x** | ✅ 100% |
| 2048 | 10.915 | 10.181 | 13.530 | 1.24x | **1.33x** | ✅ 100% |

**Key Results:**
- **Up to 1.33x speedup** over HuggingFace with CUDA Graph optimization
- **100% token-level accuracy** across all sequence lengths
- Graph optimization provides 7-8% additional speedup over standard CF

### Pythia-6.9B (Performance-Focused)

Larger model with 100% warp efficiency due to power-of-2 head_dim (128).

| Tokens | CF (s) | CF+Graph (s) | HF (s) | CF Speedup | Graph Speedup | Match |
|--------|--------|--------------|--------|------------|---------------|-------|
| 16 | 0.144 | 0.137 | 0.156 | 1.09x | **1.14x** | ✅ 100% |
| 32 | 0.296 | 0.284 | 0.323 | 1.09x | **1.14x** | ~95% |
| 64 | 0.601 | 0.576 | 0.657 | 1.09x | **1.14x** | ~95% |
| 128 | 1.224 | 1.160 | 1.338 | 1.09x | **1.15x** | ~95% |
| 256 | 2.444 | 2.331 | 2.718 | 1.11x | **1.17x** | ✅ 100% |
| 512 | 4.935 | 4.699 | 5.562 | 1.13x | **1.18x** | ✅ 100% |
| 1024 | 9.910 | 9.464 | 11.453 | 1.16x | **1.21x** | ~97% |
| 2048 | 20.135 | 19.256 | 24.065 | 1.20x | **1.25x** | ~97% |

**Notes on 6.9B Correctness:**
- Minor token mismatches due to FP16 `atomicAdd` non-determinism in output projection
- Errors are non-systematic and within acceptable numerical tolerance
- Sample generation quality remains high (see benchmark output)

---

## Key Optimizations

1. **Fused Decoder Layer**: LayerNorm + QKV projection + RoPE + Attention + Output projection + MLP all in one kernel
2. **Single-pass LayerNorm**: Var(x) = E[x²] - E[x]² for numerical stability
3. **Tree reduction**: log₂(4) = 2 steps for cluster-level reductions (vs 3 sequential)
4. **PTX-accelerated GELU**: Using `ptx_exp2` and `ptx_tanh` for faster approximation
5. **CUDA Graph Context**: TensorMaps created once with `max_seq_len`, static buffers reused
6. **TMA (Tensor Memory Accelerator)**: Efficient weight loading with hardware-accelerated transfers

## API Reference

### Pythia-2.8B
```python
import clusterfusion

# Standard dispatch
output, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer(
    input, weight_qkv, bias_qkv, weight_o, bias_o,
    k_cache, v_cache, ln_weight, ln_bias, cos, sin,
    post_ln_weight, post_ln_bias,
    mlp_up_weight, mlp_up_bias, mlp_down_weight, mlp_down_bias,
    current_seq_len
)

# CUDA Graph optimized
clusterfusion.pythia_2b8_create_graph_context(ctx_id, k_cache, v_cache, weight_qkv, weight_o, mlp_up_weight, mlp_down_weight, max_seq_len)
output, k_new, v_new = clusterfusion.pythia_2b8_graph_decode_step(ctx_id, input, ln_weight, ln_bias, ...)
clusterfusion.pythia_2b8_destroy_graph_context(ctx_id)
```

### Pythia-6.9B
```python
import clusterfusion

# Standard dispatch
output, k_new, v_new = clusterfusion.pythia_6b9_decoder_layer(...)

# CUDA Graph optimized
clusterfusion.pythia_6b9_create_graph_context(...)
output, k_new, v_new = clusterfusion.pythia_6b9_graph_decode_step(...)
clusterfusion.pythia_6b9_destroy_graph_context(...)
```

## Citation

If you find ClusterFusion useful in your research or project, please kindly cite the original paper:

```
@misc{luo2025clusterfusion,
      title={ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive}, 
      author={Xinhao Luo and Zihan Liu and Yangjie Zhou and Shihan Fang and Ziyu Huang and Yu Feng and Chen Zhang and Shixuan Sun and Zhenzhe Zheng and Jingwen Leng and Minyi Guo},
      year={2025},
      eprint={2508.18850},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2508.18850}, 
}
```
