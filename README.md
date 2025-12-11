# CS3602 Project: ClusterFusion (Pythia Port)

We ported ClusterFusion’s fused decoder kernel from the original Llama-2-7B target to EleutherAI Pythia-2.8B (GPT-NeoX). This branch focuses on decode-time fusion and currently supports NVIDIA `sm_120` GPUs (Blackwell/5090-class); the H100 path is not implemented.

## What changed: Llama-2-7B vs Pythia-2.8B

| Parameter | Llama-2-7B | Pythia-2.8B |
|-----------|------------|-------------|
| hidden_size | 4096 | 2560 |
| num_attention_heads | 32 | 32 |
| head_dim | 128 | 80 |
| intermediate_size (FFN) | 12288 | 10240 |
| num_layers | 32 | 32 |
| max_position_embeddings | 4096 | 2048 |

Key architectural/kernel differences:
- Head dim 128 → 80: warp mapping updated (4 warps, 20 rows/warp) and alignment for `uint4` vectorized loads.
- RoPE: Llama applies RoPE to all 128 dims; Pythia uses Neox rotary_pct=0.25 (first 20 dims). The kernel pads `cos/sin` to `HEAD_DIM`.
- Norm and residual: Pythia uses LayerNorm with bias and parallel residual (attention + MLP). The kernel keeps RMSNorm-style math for simplicity while leaving the final LayerNorm in PyTorch.
- Projections: QKV weights are interleaved with bias; MLP branch (GELU approx) is fused alongside attention.
- Scope: sglang batching removed; only `sm_120` kernels are wired up.

## Environment
- Python 3.13 (conda), NVIDIA GPU with `sm_120` compute capability
- CUDA 13.1 user-space wheels via PyTorch cu130 index

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
python tests/benchmark_decode.py
```

## How to reproduce
1. Download HuggingFace weights for `EleutherAI/pythia-2.8b`.
2. Recreate the above environment.
3. Correctness:
   - `python tests/test_pythia.py` (kernel vs reference, small seq len).
   - `python tests/test_pythia_correct.py` (PyTorch implementation matching HF).
   - `python tests/test_pythia_with_kernel.py` (full decode with kernel vs HF + speed).
4. Benchmark decode-only:
   - `python tests/benchmark_decode.py` compares ClusterFusion vs HuggingFace across token counts (prefill/setup excluded).

## Benchmark results (sm_120, batch=1, prompt: "The meaning of life is")
Command used (with mirror):  
`conda activate nlp_project && export HF_ENDPOINT=https://hf-mirror.com && python tests/benchmark_decode.py`

Decode-only timings (setup excluded):
```
tokens | CF_fused(s) | HF(s) | speedup_cf | match | setup(s)
   16  |      0.126  | 0.089 |     0.70   | True  | 0.214
   32  |      0.167  | 0.177 |     1.06   | True  | 0.007
   64  |      0.342  | 0.359 |     1.05   | True  | 0.007
  128  |      0.684  | 0.733 |     1.07   | True  | 0.007
  256  |      1.387  | 1.487 |     1.07   | True  | 0.007
  512  |      2.803  | 3.055 |     1.09   | True  | 0.007
 1024  |      5.638  | 6.398 |     1.13   | True  | 0.008
 2048  |     11.592  |13.568 |     1.17   | True  | 0.010
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
