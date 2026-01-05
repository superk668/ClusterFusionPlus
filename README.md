# ClusterFusion++: Expanding Cluster-Level Fusion to Full Transformer-Block Decoding

Prior work expands fusion scope by leveraging thread-block clusters and on-chip inter-block collectives to fuse attention-side operators (QKV projection, attention, and output projection). In this project, we develop **ClusterFusion++**, a **CUDA-level** extension that expands fusion further to the **full Transformer decoder block** for GPT-NeoX/Pythia models: LayerNorm $\rightarrow$ QKV $\rightarrow$ RoPE $\rightarrow$ decode attention $\rightarrow$ output projection $\rightarrow$ Post-LN $\rightarrow$ MLP $\rightarrow$ residual. We additionally engineer a CUDA-Graph-compatible execution mode with persistent Tensor Memory Accelerator (TMA) descriptors to reduce per-step overhead. 

On an NVIDIA RTX 5090-class GPU, ClusterFusion++ improves throughput by 1.34 $\times$ for Pythia-2.8B and achieves similar gains for Pythia-6.9B, while maintaining high output fidelity (near-token-identical generation, with minor non-determinism from FP16 atomics).

## Environment
- Python 3.13 (conda), NVIDIA GPU with `sm_120` compute capability, 5090 suggested for best performance
- CUDA 12.8+ user-space wheels via PyTorch cu130 index

Install from source:
```bash
conda create -n ClusterFusion python=3.13 -y
conda activate ClusterFusion

git clone https://github.com/superk668/ClusterFusionPlus.git
cd ClusterFusionPlus

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install flashinfer-python
pip install transformers accelerate datasets

pip install -e .
```

## Evaluation

```bash
# Benchmark tests
python tests/benchmark_full.py       # Complete benchmark suite (TTFT, TPOT, PPL)
python tests/benchmark_decode.py     # Pythia-2.8B decode benchmark
python tests/benchmark_decode_6b9.py # Pythia-6.9B decode benchmark

# Quality verification
python tests/verify_lossless.py      # Verify correctness and characterize non-determinism
python tests/evaluate_decode_quality.py  # Decode quality metrics (Token Match, Top-K, MAE)

# Unit tests
python tests/test_pythia.py          # Pythia-2.8B kernel correctness
python tests/test_pythia_6b9.py      # Pythia-6.9B kernel correctness
python tests/test_llama.py           # LLaMA kernel correctness
```

---

## Benchmark Results

![TPOT Benchmark](assets/tpot.png)

We use time per output token (TPOT) as the metric for end-to-end evaluation. 
For TPOT, ClusterFusion++ achieves 1.21 $\times$, 1.25 $\times$, 1.26 $\times$, 1.30 $\times$, 1.34 $\times$ speedup on different sequence length over the baseline for Pythia-2.8B, and 1.19 $\times$, 1.24 $\times$, 1.26 $\times$, 1.29 $\times$, 1.34 $\times$ speedup for Pythia-6.9B.
Refer to our paper for more details.


## Key Optimizations

1. **Fused Decoder Layer**: All operations in one kernel (LayerNorm → QKV → RoPE → Attention → Output → MLP → Residual)
2. **Single-pass LayerNorm**: Var(x) = E[x²] - E[x]² for numerical stability
3. **Tree Reduction**: log₂(4) = 2 steps for cluster-level reductions
4. **PTX-accelerated GELU**: Using `ptx_exp2` and `ptx_tanh`
5. **TMA (Tensor Memory Accelerator)**: Hardware-accelerated weight loading
6. **CUDA Graph Context**: TensorMaps created once, static buffers reused (7-8% additional speedup)

---

## API Reference

### Pythia-2.8B
```python
import clusterfusion

# Fused kernel (cooperative launch with grid.sync)
output, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer(
    input, weight_qkv, bias_qkv, weight_o, bias_o,
    k_cache, v_cache, ln_weight, ln_bias, cos, sin,
    post_ln_weight, post_ln_bias,
    mlp_up_weight, mlp_up_bias, mlp_down_weight, mlp_down_bias,
    current_seq_len
)

# Split kernel (two regular launches, no grid.sync needed)
output, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer_split(...)

# CUDA Graph optimized (pre-created TensorMaps)
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

