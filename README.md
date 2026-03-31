# Hierarchical Routed Sinkformer (HRS)

**Geometry-Shaped Representations for Compute-Adaptive Language Modeling**

HRS is a transformer architecture organized around a core principle: *computation should be proportional to relevance*. Instead of applying global attention uniformly, HRS routes tokens through a hierarchy of compute tiers based on learned relevance scores.

## Headline Result

HRS V18 achieves **MAUVE 0.950** on WikiText-103 with 510M parameters (PEER + cross-attention engram + entropy-gated retrieval), trained in ~11.4 hours on a single RTX 5070 Ti.

V18 fixes the causal attention leakage bug from V16 by isolating the engram via cross-attention instead of sequence prepending. An entropy-gated retrieval system stores engrams for surprising text and retrieves them when the model is confused, pushing MAUVE past V17's 0.943 baseline. In needle-in-a-haystack evaluation, the retrieval system finds the correct document among 20 distractors with 100% accuracy.

See the [V18 article](article_v18_crossattn.md) for full analysis.

**Previous headline:** V16 achieved **1.71 BPE perplexity** and MAUVE 0.905 with engrams disabled. See the [V16 article](article_peer_engram.md).

**Important caveat:** Perplexity is BPE (subword), not word-level. Published WikiText-103 benchmarks use word-level tokenization. BPE models have a systematic advantage when converting to word-level perplexity. See the [V12 writeup](v12_article.txt) for discussion.

## Architecture

**Core:**
- **Dual-head backbone** — generative (CE) + locality (InfoNCE) heads create representational tension that prevents embedding collapse
- **Learned router** — per-token soft routing via Nash equilibrium between competing objectives (not optimal transport)
- **Tiered compute** — convolution (local), attention (global), sink (interference reduction)
- **Phased training** — differential learning rates sequence component activation across 4 phases
- **PEER FFN** — Parameter Efficient Expert Retrieval with 262K single-neuron experts via product keys

**V18 Cross-Attention Engram:**
- **Cross-attention injection** — engram enters via dedicated cross-attention blocks at alternating layers, structurally isolated from the causal self-attention path
- **Learned gates** — sigmoid-gated output (settled at 0.27–0.33) lets the model control engram influence per-layer
- **Categorization head** — topic classification objective (50 categories, α=0.1) gives the engram a discriminative training signal
- **EMA buffer** — corpus-level engram updated every 100 steps via exponential moving average

**V18-EGR (Entropy-Gated Retrieval):**
- **Entropy as write trigger** — high-entropy segments (>4.0 bits) stored as engram vectors
- **Entropy as read trigger** — rolling entropy spikes during generation trigger nearest-neighbor retrieval
- **Cross-attention injection** — retrieved engrams temporarily replace the trained buffer
- **100% needle-in-a-haystack retrieval** — correct document found at mean rank 1.2 among 20 distractors

**BDH (Brain-Derived Heuristics, V8–V14):**
- **Virtual synapse** — engram-derived gain modulates attention heads via sigmoid gating
- **Hub routing** — KL divergence loss pushes tier distribution toward Zipf target
- **Sparsity bottleneck** — top-K selection retains only 5% of features before routing
- **Learnable loss scaling** — auxiliary loss coefficients are learned by gradient descent rather than fixed

## Results

| # | Configuration | Params | Best BPE PPL | MAUVE | Notes |
|---|---------------|-------:|-------------:|:-----:|-------|
| **V18+EGR** | **PEER + cross-attn engram + retrieval** | **512M** | **23.3** | **0.950** | **entropy-gated retrieval, project best MAUVE** |
| V18 | PEER + cross-attn engram + categorization | 512M | 23.3 | 0.915–0.941 | fixes V16 leakage bug |
| V17 | PEER only, no engram (baseline) | 499M | 21.4 | 0.933–0.943 | clean ablation baseline |
| V16 | PEER + prepend engram | 510M | 1.71 | 0.806–0.906 | engram as training scaffolding |
| V12 | V9 + 6 layers, no Phase 5 | 250M | 3.32 | — | extended to 100K steps |
| V9 | + learnable loss scaling | 176M | 7.51 | — | constraints help with right weights |
| V4 | PEER + routing + engrams | 176M | 8.28 | — | unconstrained baseline |
| V10 | control (no BDH/routing/engrams) | 169M | 30.48 | — | proves components are necessary |

## Key Findings

- **Cross-attention fixes the engram leakage bug.** V16 prepended engram tokens to the sequence, causing causal attention leakage (MAUVE dropped 0.10). V18's cross-attention isolation eliminates this — MAUVE changes by only 0.003 with engrams active.
- **Entropy-gated retrieval improves generation quality.** Storing engrams for high-entropy text and retrieving them during entropy spikes pushes MAUVE from 0.913 to 0.950 at 500-token prompts — above V17's 0.943 baseline.
- **Retrieval works, grounded generation doesn't (yet).** NIAH test: 100% retrieval accuracy at rank 1.2. But the model can't translate retrieved context into factual generation — it needs retrieval-augmented training.
- **Entropy is a universal control signal.** Shannon entropy serves three roles: data curation (filter noise), memory write (store surprises), and memory read (retrieve when confused).
- **Learned gates find balanced operating points.** V18's cross-attention gates converged to 0.27–0.33, neither fully open nor closed. The model chose to use the engram as a gentle topic prior.
- **PEER is the real story.** V17 (PEER only, no engram) achieves MAUVE 0.933–0.943 — competitive with models several times its size. PEER's sparse routing makes full attention affordable.
- **Consumer hardware is sufficient.** All experiments ran on a single RTX 5070 Ti (~$600). Training takes 11–12 hours. VRAM usage peaks at 12.5 GB.

## Running the Experiments

### Requirements

- Python 3.10+
- PyTorch (with CUDA)
- Hugging Face `datasets` and `transformers`
- scikit-learn (for V18 category clustering)
- mauve-text (for MAUVE benchmarks)
- ~12.5GB VRAM for V18 (batch_size=4, seq_len=512, 6 layers, d=1024)

```bash
pip install torch datasets transformers scikit-learn mauve-text
```

### V18 (512M, PEER + cross-attention engram, current best)

Training (~11.4 hours):
```bash
python train.py --ablation v18_cross_attn --output-dir results
```

MAUVE benchmark:
```bash
python benchmark_mauve_v18.py
```

### V18-EGR (Entropy-Gated Retrieval)

Populate the engram store from WikiText-103 validation set (~2 minutes):
```bash
python populate_store.py --threshold 4.0 --output engram_store_data
```

MAUVE benchmark with retrieval:
```bash
python benchmark_mauve_egr.py --store engram_store_data --threshold 4.0
```

Needle-in-a-haystack test (~15 minutes):
```bash
python niah_egr.py --n-distractors 20 --threshold 4.0
```

Retrieval evaluation:
```bash
python evaluate_retrieval.py --store engram_store_data --threshold 4.0
```

### V16 (510M, PEER + prepend engram)

```bash
python train.py --ablation v16_peer_engram --output-dir results
python benchmark_mauve.py v16_peer_engram
```

### V17 (499M, PEER only baseline)

```bash
python train.py --ablation v17_peer_only --output-dir results
```

### V12 (250M, 6 layers)

```bash
python train.py --ablation v12_247m --batch-size 4 --output-dir results
```

### Evaluation

BPE and word-level perplexity:
```bash
python eval_word_ppl_v2.py --checkpoint results/v12_247m/best.pt \
    --ablation v12_247m --no-overlap --sanity-check
```

## Files

| File | Description |
|------|-------------|
| `model.py` | HRS transformer (backbone, tiers, cross-attention engram, categorization head) |
| `router.py` | Learned token router with TRC, balance/entropy/FLOPs losses |
| `tiers.py` | Tiered compute operators (conv, attention, sink) |
| `engram.py` | Engram encoder, injectors, cross-attention block, categorization head |
| `peer.py` | PEER expert retrieval (262K single-neuron experts via product keys) |
| `bdh.py` | Virtual synapse, hub routing loss, sparsity bottleneck |
| `losses.py` | Combined loss with CE, locality, reconstruction, categorization |
| `config.py` | All configuration dataclasses and ablation presets (V1–V18) |
| `train.py` | Training loop with phased protocol, differential LRs, engram buffer updates |
| `data.py` | WikiText-103 loading with GPT-2 BPE tokenizer and category labels |
| `metrics.py` | Effective rank, routing entropy, tier distribution tracking |
| `engram_store.py` | Engram vector store with cosine similarity retrieval |
| `entropy_monitor.py` | Rolling entropy computation and threshold monitoring |
| `retrieval_engine.py` | Entropy-gated engram retrieval engine for V18 inference |
| `populate_store.py` | Pre-populate engram store from WikiText-103 |
| `evaluate_retrieval.py` | Retrieval system evaluation (perplexity, trigger stats) |
| `niah_egr.py` | Needle-in-a-haystack test for entropy-gated retrieval |
| `benchmark_mauve.py` | MAUVE benchmark (V16-style) |
| `benchmark_mauve_v18.py` | MAUVE benchmark for V18 cross-attention engram |
| `benchmark_mauve_egr.py` | MAUVE benchmark for V18 + entropy-gated retrieval |
| `generate_sample.py` | Generation quality checker with WikiText context seeding |
| `eval_word_ppl_v2.py` | BPE and word-level perplexity evaluation |

## Papers

- [V18 Cross-Attention Engram + Entropy-Gated Retrieval](article_v18_crossattn.md) — MAUVE 0.950, cross-attention fix, NIAH results
- [V16 PEER + Engram Results](article_peer_engram.md) — 1.71 BPE perplexity, MAUVE 0.905, engram-as-scaffolding finding
- [V12 Results and Analysis](v12_article.txt) — 3.32 BPE perplexity, Phase 5 diagnosis, tokenization discussion
- [BDH and Learnable Loss Scaling (V8/V9)](HRS_paper_medium.md) — Brain-derived heuristics with fixed vs learned coefficients
- [Full HRS paper](paper.md) — Original theoretical framework, training protocol, and ablation study

## Author

Michael Bee ([@mbonsign](https://medium.com/@mbonsign))
