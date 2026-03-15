# Hierarchical Routed Sinkformer (HRS)

**Geometry-Shaped Representations for Compute-Adaptive Language Modeling**

HRS is a transformer architecture organized around a core principle: *computation should be proportional to relevance*. Instead of applying global attention uniformly, HRS routes tokens through a hierarchy of compute tiers based on learned relevance scores.

## Architecture

**Core:**
- **Dual-head backbone** — generative (CE) + locality (InfoNCE) heads create representational tension that prevents embedding collapse
- **Learned router** — per-token soft routing via Nash equilibrium between competing objectives (not optimal transport)
- **Tiered compute** — convolution (local), experts (specialized), attention (global), sink (interference reduction)
- **Phased training** — differential learning rates sequence component activation across 4-5 phases

**Extensions:**
- **Temporal Routing Cache (TRC)** — causal moving average smooths routing across adjacent tokens
- **Engrams** — compressed thread memory with bounded growth (the decisive component)

## Results

Eight-configuration ablation on WikiText-103 (50K steps, RTX 5070 Ti):

| # | Configuration | Val PPL | Best PPL | vs Baseline |
|---|---------------|--------:|---------:|------------:|
| 1 | dense_baseline | 23.83 | — | — |
| 2 | + dual-head | 24.08 | — | +1.0% |
| 3 | + router + tiers | 31.63 | — | +32.7% |
| 4 | + sink channel | 31.90 | — | +33.9% |
| 5 | + phased training | 37.84 | — | +58.8% |
| 6 | + engrams | 12.04 | 10.64 | -49.5% |
| 7 | + engram refinement | 10.51 | **9.19** | **-55.9%** |
| 8 | + TRC (window=8) | 10.05 | **9.43** | **-57.8%** |

**Key findings:**
- Routing alone increases perplexity (the "router tax") — routing is infrastructure for compression, not an independent win
- Engrams are the decisive component: adding them flips a 59% regression into a 50-61% improvement
- Best perplexity (9.19) occurs at end of Phase 4; Phase 5 consistently regresses
- Dual-objective routing is a Nash equilibrium, not optimal transport (Sinkhorn fails theoretically, not just practically)

## MPAR Cross-Prompt Retrieval Experiment

**Mean-Pooled Activation Retrieval** — tests whether mid-layer hidden states are abstract enough to bridge storage and retrieval prompts that differ in surface form.

**Setup:** Extract hidden states from a target layer, mean pool across token positions to produce an MPAR vector. Store 50 fact-bearing prompts, then retrieve using 50 semantically matched but lexically different query prompts. Measure top-k accuracy and separation ratio (correct distance / nearest incorrect distance — below 1.0 means the right answer is genuinely closer).

| Experiment | Model | Best Layer | Top-1 | Top-3 | Top-5 | Sep Ratio |
|------------|-------|-----------|------:|------:|------:|----------:|
| v1 prompts | GPT-2 small | 3 | 16% | 20% | 26% | 3.28 |
| v1 prompts | Mistral 7B (4-bit) | 32 | 36% | 54% | 68% | 1.15 |
| **v2 prompts** | **Mistral 7B (4-bit)** | **24** | **94%** | **98%** | **98%** | **0.69** |

**Key findings:**
- Prompt design dominates model scale — v2 prompts (richer context, uniform length, shared domain vocabulary between storage/retrieval) jump from 36% to 94% top-1 on the same model
- Three-quarter depth (layer 24 of 32) produces the most abstract representations, not the final layer
- Separation ratios cross below 1.0 at all layers with v2 prompts, confirming correct matches are genuinely closer than incorrect ones
- Remaining 3 failures are semantically confusable pairs (rent/lease for same apartment, glucose/cholesterol blood tests, flight code/project code)

```bash
# GPT-2 baseline
python3 mpar_experiment.py --device cuda

# Mistral 7B with v1 prompts
python3 mpar_experiment_7b.py

# Mistral 7B with v2 prompts (best results)
python3 mpar_experiment_7b_v2.py
```

## Expert Isomorphism Experiment

Tests whether the 262K single-neuron experts in a PEER network converge to a shared transformation. Each expert computes `sigma(u_i^T x) * v_i` — the hypothesis is that u vectors (input/activation weights) cluster tightly while v vectors (output weights) diverge.

**Result: Hypothesis falsified.** Both u and v weight matrices are maximally dispersed (effective rank ~64/64, pairwise cosine similarity ~0.001). If anything, u has *more* variance (59%) than v (41%). The experts are genuinely distinct.

However, a shared-trunk architecture (one u for all experts, distinct v per expert) achieves 38.2% parameter reduction and actually *improves* val perplexity (191 vs 204) because collapsing u acts as regularization against overfitting on WikiText-2.

| Metric | Baseline | Shared Trunk |
|--------|----------|-------------|
| Parameters | 175.6M | 108.5M |
| Val PPL | 203.8 | 191.2 |
| Test PPL | 239.9 | 226.6 |
| Router Entropy | 0.104 | 0.098 |

```bash
# Full 5-phase experiment (train baseline, SVD analysis, shared trunk, router analysis, compression)
python3 expert_isomorphism.py --baseline-steps 15000 --finetune-steps 10000
```

## Files

| File | Description |
|------|-------------|
| `model.py` | HRS transformer model (backbone, tier integration, engram injection) |
| `router.py` | Learned token router with TRC, balance/entropy/FLOPs losses |
| `tiers.py` | Tiered compute operators (conv, expert MoE, attention, sink) |
| `engram.py` | Engram encoder and cross-attention injector |
| `losses.py` | Locality (InfoNCE) and engram reconstruction losses |
| `config.py` | All configuration dataclasses and ablation presets |
| `train.py` | Training loop with phased protocol and differential LRs |
| `data.py` | WikiText-103 data loading |
| `metrics.py` | Effective rank, routing entropy, tier distribution tracking |
| `mpar_experiment.py` | MPAR retrieval with GPT-2 + shared utility functions |
| `mpar_experiment_7b.py` | MPAR retrieval with Mistral 7B (4-bit quantization) |
| `mpar_experiment_7b_v2.py` | MPAR retrieval with v2 enriched prompts |
| `mpar_prompts_v2.py` | 50 redesigned prompt pairs with richer context |
| `expert_isomorphism.py` | 5-phase PEER expert isomorphism experiment |
| `paper.md` | Full paper with theoretical framework and experimental results |

## Usage

```bash
# Single ablation config
python3 train.py --ablation dense_baseline
python3 train.py --ablation full_hrs

# With TRC enabled
python3 train.py --ablation full_hrs --trc-window 8 --run-name full_hrs_trc

# Full ablation sweep
./run_v4.sh   # configs 1-6
./run_v5.sh   # configs 7-8
```

## Requirements

- PyTorch (with CUDA)
- Hugging Face `datasets` and `transformers` (for WikiText-103 and GPT-2 tokenizer)
- ~16GB VRAM for default config (batch_size=24, seq_len=512, d_model=512)

## Paper

See [paper.md](paper.md) for the full theoretical framework, training protocol, and detailed experimental analysis.

## Author

Michael Bee ([@mbonsign](https://medium.com/@mbonsign))
