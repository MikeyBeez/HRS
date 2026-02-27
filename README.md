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
