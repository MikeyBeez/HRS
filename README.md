# Hierarchical Routed Sinkformer (HRS)

**Geometry-Shaped Representations for Compute-Adaptive Language Modeling**

HRS is a transformer architecture organized around a core principle: *computation should be proportional to relevance*. Instead of applying global attention uniformly, HRS routes tokens through a hierarchy of compute tiers based on learned relevance scores.

## Headline Result

HRS V16 achieves **1.71 BPE perplexity** and a **MAUVE score of 0.905** on WikiText-103 with 510M parameters (PEER + engram, no routing), trained in ~12 hours on a single RTX 5070 Ti.

Key finding: the engram functions as *training scaffolding* — it shapes better representations during training but should be disabled at inference, where it adds noise. With engrams off, generation quality is prompt-length-invariant (MAUVE 0.905-0.906). See the [V16 article](article_peer_engram.md) for full analysis.

**Important caveat:** This is BPE (subword) perplexity, not word-level perplexity. Published WikiText-103 benchmarks (kNN-LM at 15.79, Transformer-XL at 18.3) use word-level tokenization with adaptive softmax. BPE models have a systematic advantage when converting to word-level perplexity because they predict multiple easier subword tokens per word. A direct comparison requires retraining with word-level tokenization. See the [V12 writeup](v12_article.txt) for full discussion.

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

V12 (250M parameters, 6 layers, WikiText-103, 100K steps, RTX 5070 Ti):

- BPE perplexity: 3.32 (primary metric)
- Word-level perplexity via BPE aggregation: 10.43 (not directly comparable to word-level models)

Earlier configurations for context:

| # | Configuration | Params | Best BPE PPL | MAUVE | Notes |
|---|---------------|-------:|-------------:|:-----:|-------|
| V4 | PEER + routing + engrams | 176M | 8.28 | — | unconstrained baseline |
| V8 | + BDH (fixed loss coefficients) | 176M | 10.25 | — | constraints hurt with wrong weights |
| V9 | + learnable loss scaling | 176M | 7.51 | — | constraints help with right weights |
| V10 | control (no BDH/routing/engrams) | 169M | 30.48 | — | proves components are necessary |
| V12 | V9 + 6 layers, no Phase 5 | 250M | 3.32 | — | extended to 100K steps |
| V13 | V12 + 5% sparsity | 250M | 4.53 | — | reduced sparsity, generation still poor |
| V14 | V13 + 2-tier routing (attn+sink) | 250M | 4.92 | — | removed conv tier, 48% sink |
| V15 | vanilla + routing + engrams, no PEER | 164M | 1.79 | — | good PPL, generation still poor |
| **V16** | **PEER + engram, no routing/BDH** | **510M** | **1.71** | **0.905** | **engram as training scaffolding** |

## Key Findings

- **Learnable loss scaling is essential.** V8 (fixed coefficients) underperformed the unconstrained baseline. V9 (learned coefficients) beat it by 2 points. Gradient descent cut hub and reconstruction pressure by ~40% and tripled exploration pressure.
- **All BDH components contribute.** V10 control (no routing, no engrams, no BDH) achieved only 30.48 perplexity.
- **Phase 5 causes regression.** Both V8 and V9 regressed when Phase 5 activated (V9: 7.51 to 9.07). P5 triples backbone/head learning rates and freezes engrams. V12 eliminates P5 entirely.
- **Depth matters.** Going from 4 to 6 layers (176M to 250M params) cut perplexity roughly in half.
- **Multi-path routing hurts generation.** V13-V14 showed that routing tokens through conv/attn/sink tiers fragments the representation space, causing incoherent autoregressive generation despite good teacher-forced PPL.
- **Engram is training scaffolding, not a runtime component.** V16 MAUVE benchmark shows the model generates better text (0.905) with engrams disabled at inference than enabled (0.806-0.888). The engram shapes better representations during training but adds noise at inference.
- **Engram dropout is essential.** Without dropout, models fail catastrophically on short prompts (< 128 tokens) where no engrams are produced. 10% engram dropout during training enables prompt-length-invariant generation.

## Running the Experiments

### Requirements

- Python 3.10+
- PyTorch (with CUDA)
- Hugging Face `datasets` and `transformers`
- ~8GB VRAM for V12 (batch_size=4, seq_len=512, 6 layers)
- ~6GB VRAM for V9 and earlier (batch_size=4, seq_len=512, 4 layers)

```bash
pip install torch datasets transformers
```

### V16 (510M, PEER + engram, best result)

Training (~12 hours):
```bash
python3 train.py --ablation v16_peer_engram --batch-size 4 --output-dir results
```

Generation quality check:
```bash
python3 generate_sample.py v16_peer_engram
```

MAUVE benchmark:
```bash
pip install mauve-text
python3 benchmark_mauve.py v16_peer_engram
```

### V12 (250M, 6 layers, previous best)

Initial training (50K steps, ~9.5 hours):
```bash
python3 train.py --ablation v12_247m --batch-size 4 --output-dir results
```

Extended training to 100K steps (~10 more hours, resume from checkpoint):
```bash
python3 train.py --ablation v12_247m --batch-size 4 --output-dir results \
    --max-steps 100000 --lr 6e-5 --resume results/v12_247m/checkpoint_50000.pt
```

### V9 (176M, 4 layers, learnable BDH)

```bash
python3 train.py --ablation v9_learnable --batch-size 4 --output-dir results
```

### V8 (176M, 4 layers, fixed BDH coefficients)

```bash
python3 train.py --ablation v8_bdh --batch-size 4 --output-dir results
```

### V4 baseline (176M, no BDH constraints)

```bash
python3 train.py --ablation v4_full --output-dir results
```

### V10 control (no routing, no engrams, no BDH)

```bash
python3 train.py --ablation v10_control --batch-size 4 --output-dir results
```

### Dense baseline (no routing, no PEER, no engrams)

```bash
python3 train.py --ablation dense_baseline --output-dir results
```

### Evaluation

BPE and word-level perplexity with sanity checks:
```bash
python3 eval_word_ppl_v2.py --checkpoint results/v12_247m/best.pt \
    --ablation v12_247m --no-overlap --sanity-check
```

### Other experiments

```bash
# MPAR cross-prompt retrieval (Mistral 7B)
python3 mpar_experiment_7b_v2.py

# Expert isomorphism experiment
python3 expert_isomorphism.py --baseline-steps 15000 --finetune-steps 10000
```

## Files

| File | Description |
|------|-------------|
| `model.py` | HRS transformer (backbone, tier integration, engram injection, BDH modules) |
| `router.py` | Learned token router with TRC, balance/entropy/FLOPs losses |
| `tiers.py` | Tiered compute operators (conv, attention, sink) |
| `engram.py` | Engram encoder and cross-attention injector |
| `peer.py` | PEER expert retrieval (262K single-neuron experts via product keys) |
| `bdh.py` | Virtual synapse, hub routing loss, sparsity bottleneck |
| `losses.py` | Combined loss with CE, locality, engram reconstruction, BDH auxiliary losses |
| `config.py` | All configuration dataclasses and ablation presets (V1-V16) |
| `train.py` | Training loop with phased protocol, differential LRs, best-model checkpointing |
| `data.py` | WikiText-103 data loading with GPT-2 BPE tokenizer |
| `metrics.py` | Effective rank, routing entropy, tier distribution tracking |
| `generate_sample.py` | Generation quality checker with WikiText context seeding |
| `benchmark_mauve.py` | MAUVE benchmark for evaluating generation quality |
| `eval_word_ppl_v2.py` | BPE and word-level perplexity evaluation with sanity checks |
| `expert_isomorphism.py` | PEER expert isomorphism experiment |
| `mpar_experiment_7b_v2.py` | MPAR retrieval with Mistral 7B and v2 enriched prompts |
| `v12_article.txt` | Full V12 writeup for publication |

## Papers

- [V16 PEER + Engram Results](article_peer_engram.md) — 1.71 BPE perplexity, MAUVE 0.905, engram-as-scaffolding finding
- [V12 Results and Analysis](v12_article.txt) — 3.32 BPE perplexity, Phase 5 diagnosis, tokenization discussion
- [BDH and Learnable Loss Scaling (V8/V9)](HRS_paper_medium.md) — Brain-derived heuristics with fixed vs learned coefficients
- [Full HRS paper](paper.md) — Original theoretical framework, training protocol, and ablation study

## Author

Michael Bee ([@mbonsign](https://medium.com/@mbonsign))
