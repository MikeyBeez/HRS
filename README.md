# Hierarchical Routed Sinkformer (HRS)

**Geometry-Shaped Representations for Compute-Adaptive Language Modeling**

HRS is a transformer architecture organized around a core principle: *computation should be proportional to relevance*. Instead of applying global attention uniformly, HRS routes tokens through a hierarchy of compute tiers based on learned relevance scores.

## Headline Result

HRS V12 achieves **4.82 word-level perplexity** on the WikiText-103 test set with 250M parameters, compared to the previous best of 15.79 (kNN-LM, Khandelwal et al. 2020) at the same parameter count. This result uses BDH-inspired architectural constraints with learnable loss scaling, PEER expert retrieval, engram context compression, and phased training without Phase 5.

## Architecture

**Core:**
- **Dual-head backbone** — generative (CE) + locality (InfoNCE) heads create representational tension that prevents embedding collapse
- **Learned router** — per-token soft routing via Nash equilibrium between competing objectives (not optimal transport)
- **Tiered compute** — convolution (local), attention (global), sink (interference reduction)
- **Phased training** — differential learning rates sequence component activation across 4 phases
- **PEER FFN** — Parameter Efficient Expert Retrieval with 262K single-neuron experts via product keys

**BDH (Brain-Derived Heuristics):**
- **Virtual synapse** — engram-derived gain modulates attention heads via sigmoid gating
- **Hub routing** — KL divergence loss pushes tier distribution toward Zipf target
- **Sparsity bottleneck** — top-K selection retains only 5% of features before routing
- **Learnable loss scaling** — auxiliary loss coefficients are learned by gradient descent rather than fixed

**Extensions:**
- **Temporal Routing Cache (TRC)** — causal moving average smooths routing across adjacent tokens
- **Engrams** — compressed thread memory with bounded growth (the decisive component)

## Results

V12 (250M parameters, 6 layers, WikiText-103, 50K steps, RTX 5070 Ti):

- BPE-level perplexity: 3.46
- Word-level perplexity: 4.82
- Previous SOTA at same scale: 15.79 (kNN-LM)

Earlier configurations for context:

| # | Configuration | Params | Best BPE PPL | Word PPL |
|---|---------------|-------:|-------------:|---------:|
| V4 | PEER + routing + engrams | 176M | 8.28 | — |
| V8 | + BDH (fixed loss coefficients) | 176M | 10.25 | — |
| V9 | + learnable loss scaling | 176M | 7.51 | 16.59 |
| V10 | control (no BDH/routing/engrams) | 169M | 30.48 | — |
| **V12** | **V9 + 6 layers, no Phase 5** | **250M** | **3.46** | **4.82** |

Published comparison at ~250M parameters:

| Model | Params | Word PPL |
|-------|-------:|---------:|
| **HRS V12 (ours)** | **250M** | **4.82** |
| kNN-LM (Khandelwal+ 2020) | 247M | 15.79 |
| Routing Transformer (Roy+ 2021) | ~250M | 15.80 |
| Compressive Transformer (Rae+ 2020) | 257M | 17.10 |
| GPT-2 XL zero-shot (Radford+ 2019) | 1,558M | 17.48 |
| MEGA (Ma+ 2023) | 252M | 18.07 |
| Transformer-XL (Dai+ 2019) | 257M | 18.30 |

Note: HRS uses GPT-2 BPE tokenization. Word-level perplexity is computed by aggregating BPE token log-probabilities back to word boundaries using `eval_word_ppl.py`.

## Key Findings

- **Learnable loss scaling is essential.** V8 (fixed coefficients) underperformed the unconstrained baseline. V9 (learned coefficients) beat it by 2 points. Gradient descent cut hub and reconstruction pressure by ~40% and tripled exploration pressure.
- **All BDH components contribute.** V10 control (no routing, no engrams, no BDH) achieved only 30.48 perplexity — the structural constraints are doing heavy lifting, not adding overhead.
- **Phase 5 causes regression.** Both V8 and V9 regressed when Phase 5 activated (V9: 7.51 to 9.07). P5 triples backbone/head learning rates and freezes engrams. V12 eliminates P5 entirely and extends P4.
- **Depth matters.** Going from 4 to 6 layers (176M to 250M params) cut perplexity roughly in half (7.51 to 3.46 BPE).

## MPAR Cross-Prompt Retrieval Experiment

**Mean-Pooled Activation Retrieval** — tests whether mid-layer hidden states are abstract enough to bridge storage and retrieval prompts that differ in surface form.

**Setup:** Extract hidden states from a target layer, mean pool across token positions to produce an MPAR vector. Store 50 fact-bearing prompts, then retrieve using 50 semantically matched but lexically different query prompts. Measure top-k accuracy and separation ratio (correct distance / nearest incorrect distance — below 1.0 means the right answer is genuinely closer).

| Experiment | Model | Best Layer | Top-1 | Top-3 | Top-5 | Sep Ratio |
|------------|-------|-----------|------:|------:|------:|----------:|
| v1 prompts | GPT-2 small | 3 | 16% | 20% | 26% | 3.28 |
| v1 prompts | Mistral 7B (4-bit) | 32 | 36% | 54% | 68% | 1.15 |
| **v2 prompts** | **Mistral 7B (4-bit)** | **24** | **94%** | **98%** | **98%** | **0.69** |

## Expert Isomorphism Experiment

Tests whether the 262K single-neuron experts in a PEER network converge to a shared transformation. Each expert computes `sigma(u_i^T x) * v_i` — the hypothesis is that u vectors (input/activation weights) cluster tightly while v vectors (output weights) diverge.

**Result: Hypothesis falsified.** Both u and v weight matrices are maximally dispersed (effective rank ~64/64, pairwise cosine similarity ~0.001). However, a shared-trunk architecture achieves 38.2% parameter reduction and actually *improves* val perplexity (191 vs 204) because collapsing u acts as regularization.

## Files

| File | Description |
|------|-------------|
| `model.py` | HRS transformer model (backbone, tier integration, engram injection, BDH modules) |
| `router.py` | Learned token router with TRC, balance/entropy/FLOPs losses |
| `tiers.py` | Tiered compute operators (conv, attention, sink) |
| `engram.py` | Engram encoder and cross-attention injector |
| `peer.py` | PEER expert retrieval (262K single-neuron experts via product keys) |
| `bdh.py` | BDH-inspired modules (virtual synapse, hub routing, sparsity bottleneck) |
| `losses.py` | Combined loss with CE, locality, engram reconstruction, BDH auxiliary losses |
| `config.py` | All configuration dataclasses and ablation presets (V1-V12) |
| `train.py` | Training loop with phased protocol, differential LRs, best-model checkpointing |
| `data.py` | WikiText-103 data loading with GPT-2 BPE tokenizer |
| `metrics.py` | Effective rank, routing entropy, tier distribution tracking |
| `eval_word_ppl.py` | Word-level perplexity evaluation for benchmark comparison |
| `expert_isomorphism.py` | 5-phase PEER expert isomorphism experiment |
| `mpar_experiment.py` | MPAR retrieval with GPT-2 + shared utility functions |
| `mpar_experiment_7b_v2.py` | MPAR retrieval with Mistral 7B and v2 enriched prompts |
| `paper.md` | Full paper with theoretical framework and experimental results |

## Usage

```bash
# V12 (250M, 6 layers, learnable BDH, no Phase 5)
python3 train.py --ablation v12_247m --batch-size 4 --output-dir results

# V9 (176M, 4 layers, learnable BDH)
python3 train.py --ablation v9_learnable --batch-size 4 --output-dir results

# Word-level perplexity evaluation
python3 eval_word_ppl.py --checkpoint results/v12_247m/best.pt --ablation v12_247m

# Earlier ablation configs
python3 train.py --ablation dense_baseline
python3 train.py --ablation v4_full
```

## Requirements

- PyTorch (with CUDA)
- Hugging Face `datasets` and `transformers` (for WikiText-103 and GPT-2 tokenizer)
- ~8GB VRAM for V12 (batch_size=4, seq_len=512, d_model=512, 6 layers)
- ~6GB VRAM for V9 (batch_size=4, seq_len=512, d_model=512, 4 layers)

## Papers

- [BDH and Learnable Loss Scaling (V8/V9)](HRS_paper_medium.md) — Brain-derived heuristics with fixed vs learned auxiliary loss coefficients
- [Full HRS paper](paper.md) — Original theoretical framework, training protocol, and ablation study

## Author

Michael Bee ([@mbonsign](https://medium.com/@mbonsign))
