# Hierarchical Routed Sinkformer (HRS)

**Geometry-Shaped Representations for Compute-Adaptive Language Modeling**

HRS is a transformer architecture organized around a core principle: *computation should be proportional to relevance*. Instead of applying global attention uniformly, HRS routes tokens through a hierarchy of compute tiers based on learned relevance scores.

## Headline Results

**V18 Cross-Attention Engram.** Fixes the causal attention leakage bug from V16 by isolating the engram via cross-attention. MAUVE 0.915–0.941 with engram active (vs V16's 0.806 failure mode). See the [V18 article](article_v18_crossattn.md).

**Topic-Routed Context Assembly.** Uses the model's own hidden-state representations to organize context by topic instead of recency. Engram cosine similarity achieves 97.3% topic accuracy at article scale (512 tokens) but only 71.5% at sentence scale — a signal-to-noise scaling law. MAUVE improves to 0.962, but a deeper evaluation suite reveals this is distributional contamination, not genuine quality improvement: held-out perplexity worsens (38.1 vs 35.8) and an LLM judge (Llama 3.1) prefers baseline 28-22. See the [context curation paper](paper_context_curation.md).

**Exponential Kernel Attention.** Replacing dot-product attention with an exponential kernel (negative squared Euclidean distance) on Tiny Shakespeare: +3.4% topic separation at play-level (256 chars), tied at line-level. The kernel shapes representations differently where signal is adequate, but can't rescue the short-text noise floor. Exponential kernel also achieves slightly lower best val loss (1.613 vs 1.630).

**Key methodological finding:** MAUVE alone is insufficient for evaluating context engineering systems. A distributional metric can be inflated by distributional contamination — injecting reference-distribution text into the context window. Conditional metrics (held-out perplexity, LLM-as-judge) are necessary complements.

**Previous headline:** V16 achieved **1.71 BPE perplexity** and MAUVE 0.905 with engrams disabled. See the [V16 article](article_peer_engram.md).

**Important caveat:** Perplexity is BPE (subword), not word-level. Published WikiText-103 benchmarks use word-level tokenization. See the [V12 writeup](v12_article.txt) for discussion.

## Architecture

**Core:**
- **Dual-head backbone** — generative (CE) + locality (InfoNCE) heads
- **PEER FFN** — Parameter Efficient Expert Retrieval with 262K single-neuron experts via product keys
- **Phased training** — differential learning rates across 4 phases

**V18 Cross-Attention Engram:**
- **Cross-attention injection** — engram enters via dedicated cross-attention blocks at alternating layers, structurally isolated from the causal self-attention path
- **Learned gates** — sigmoid-gated output (settled at 0.27–0.33) lets the model control engram influence per-layer
- **Categorization head** — topic classification objective gives the engram a discriminative training signal
- **EMA buffer** — corpus-level engram updated every 100 steps via exponential moving average

**V18-EGR (Entropy-Gated Retrieval):**
- **Entropy as write/read trigger** — high-entropy text stored, retrieved when generation entropy spikes
- **100% needle-in-a-haystack retrieval** at mean rank 1.2 among 20 distractors

**Topic-Routed Context Assembly:**
- **Online engram clustering** — prompts clustered by topic in real time, no predefined taxonomy
- **Evolving centroids** — cluster identity drifts as conversation develops
- **Auto-merge** — clusters that converge are automatically combined
- **User-toggleable topics** — named clusters users can enable/disable
- **Configurable active slots** — 2 for simple conversations, 6-7 for multidisciplinary work

## Results

| # | Configuration | Params | Best BPE PPL | MAUVE | Notes |
|---|---------------|-------:|-------------:|:-----:|-------|
| V18+Topic | PEER + cross-attn + topic routing | 512M | 38.1* | 0.962 | *MAUVE inflated by distributional contamination |
| V18+EGR | PEER + cross-attn + entropy retrieval | 512M | 23.3 | 0.950 | Entropy-gated retrieval |
| V18 | PEER + cross-attn engram + categorization | 512M | 23.3 | 0.915–0.941 | Fixes V16 leakage bug |
| V17 | PEER only, no engram (baseline) | 499M | 21.4 | 0.933–0.943 | Clean ablation baseline |
| V16 | PEER + prepend engram | 510M | 1.71 | 0.806–0.906 | Engram as training scaffolding |
| V12 | V9 + 6 layers, no Phase 5 | 250M | 3.32 | — | Extended to 100K steps |

*Topic routing MAUVE of 0.962 is misleading — held-out perplexity worsens and LLM judge prefers baseline 28-22.

## Key Findings

- **Cross-attention fixes the engram leakage bug.** V16 prepend caused MAUVE to drop 0.10. V18 cross-attention changes MAUVE by only 0.003.
- **Engram similarity achieves 97.3% topic accuracy at 512 tokens** — linearly separable, no complex infrastructure needed. At sentence scale: 71.5%.
- **A learned classifier cannot beat a fixed cosine threshold** — confirming the failure is representational (signal-to-noise), not algorithmic.
- **MAUVE can be inflated by distributional contamination.** Topic routing improved MAUVE from 0.919 to 0.962 while worsening perplexity from 35.8 to 38.1. Four independent metrics (perplexity, semantic similarity, LLM judge, routing correlation) agree the "improvement" is illusory.
- **Engrams encode semantics, not vocabulary.** 100% adversarial routing accuracy — metaphorical cross-domain prompts cluster with their literal counterparts.
- **Exponential kernel attention improves topic separation by 3.4%** at play-level on Shakespeare but is tied at line-level. The kernel shapes representations where signal is adequate.
- **The categorization head fails at 3.3% accuracy** despite training loss of 1.2 — sequence-level label noise prevents generalizable classification.
- **Consumer hardware is sufficient.** All experiments on a single RTX 5070 Ti (~$600). Training takes 11-12 hours. VRAM peaks at 12.5 GB.

## Running the Experiments

### Requirements

```bash
pip install torch datasets transformers scikit-learn mauve-text
```

### V18 (PEER + cross-attention engram)

```bash
python train.py --ablation v18_cross_attn --output-dir results   # ~11.4 hours
python benchmark_mauve_v18.py                                     # ~3 hours
```

### Entropy-Gated Retrieval

```bash
python populate_store.py --threshold 4.0 --output engram_store_data   # ~2 min
python benchmark_mauve_egr.py --store engram_store_data               # ~4 hours
python niah_egr.py --n-distractors 20                                 # ~15 min
```

### Topic-Routed Context Assembly

```bash
python benchmark_mauve_topic.py --threshold 0.5                    # ~2 hours
python eval_topic_routing.py --threshold 0.5                       # ~30 min
python eval_accuracy.py --ollama-host 192.168.12.125 --ollama-model llama3.1  # ~20 min
python benchmark_topic_routing.py --threshold 0.4                  # ~1 min
```

### Topic Classifier Training

```bash
python train_topic_classifier.py --n-articles 2000                 # article-scale pairs
python train_topic_classifier_short.py --n-articles 3000           # sentence-scale pairs
```

### Exponential Kernel Attention

```bash
python exp_kernel_attention.py --n-steps 10000                     # ~20 min
```

### Previous Versions

```bash
python train.py --ablation v16_peer_engram --output-dir results    # V16
python train.py --ablation v17_peer_only --output-dir results      # V17
python train.py --ablation v12_247m --output-dir results           # V12
```

## Files

### Model & Training
| File | Description |
|------|-------------|
| `model.py` | HRS transformer (backbone, tiers, cross-attention engram, categorization head) |
| `peer.py` | PEER expert retrieval (262K single-neuron experts via product keys) |
| `engram.py` | Engram encoder, injectors, cross-attention block, categorization head |
| `config.py` | All configuration dataclasses and ablation presets (V1–V18) |
| `train.py` | Training loop with phased protocol, differential LRs, engram buffer updates |
| `data.py` | WikiText-103 loading with GPT-2 BPE tokenizer and category labels |
| `losses.py` | Combined loss with CE, locality, reconstruction, categorization |
| `router.py` | Learned token router with TRC, balance/entropy/FLOPs losses |
| `tiers.py` | Tiered compute operators (conv, attention, sink) |
| `bdh.py` | Virtual synapse, hub routing loss, sparsity bottleneck |
| `metrics.py` | Effective rank, routing entropy, tier distribution tracking |

### Entropy-Gated Retrieval
| File | Description |
|------|-------------|
| `engram_store.py` | Engram vector store with cosine similarity retrieval |
| `entropy_monitor.py` | Rolling entropy computation and threshold monitoring |
| `retrieval_engine.py` | Entropy-gated engram retrieval engine for V18 inference |
| `populate_store.py` | Pre-populate engram store from WikiText-103 |
| `evaluate_retrieval.py` | Retrieval system evaluation (perplexity, trigger stats) |
| `niah_egr.py` | Needle-in-a-haystack test for entropy-gated retrieval |

### Topic-Routed Context
| File | Description |
|------|-------------|
| `topic_context.py` | TopicContextManager: online clustering, active buffer, user toggles, auto-merge |
| `train_topic_classifier.py` | Article-length pair analysis and classifier training |
| `train_topic_classifier_short.py` | Short-prompt pair analysis and classifier training |
| `benchmark_topic_routing.py` | Seven stress tests (drift, overlap, adversarial, fork) |
| `eval_topic_routing.py` | Four-metric evaluation suite (perplexity, coherence, repetition, routing correlation) |
| `eval_accuracy.py` | Accuracy evaluation with LLM-as-judge (ollama) support |
| `eval_categorization.py` | Categorization head evaluation (negative result) |
| `benchmark_mauve_topic.py` | MAUVE benchmark for topic-routed context |

### Benchmarks & Generation
| File | Description |
|------|-------------|
| `benchmark_mauve.py` | MAUVE benchmark (V16-style) |
| `benchmark_mauve_v18.py` | MAUVE benchmark for V18 cross-attention engram |
| `benchmark_mauve_egr.py` | MAUVE benchmark for V18 + entropy-gated retrieval |
| `exp_kernel_attention.py` | Exponential kernel vs dot product attention experiment |
| `generate_sample.py` | Generation quality checker with WikiText context seeding |
| `eval_word_ppl_v2.py` | BPE and word-level perplexity evaluation |

## Papers

- [Topic-Routed Context Assembly](paper_context_curation.md) — "Your Transformer Already Knows What It's Talking About." Engram-based topic routing, seven stress tests, distributional contamination finding, LLM-as-judge evaluation
- [V18 Cross-Attention Engram + EGR](article_v18_crossattn.md) — Cross-attention fix, entropy-gated retrieval, NIAH evaluation
- [V16 PEER + Engram](article_peer_engram.md) — 1.71 BPE perplexity, MAUVE 0.905, engram-as-scaffolding finding
- [V12 Results](v12_article.txt) — 3.32 BPE perplexity, Phase 5 diagnosis, tokenization discussion
- [BDH and Learnable Loss Scaling](HRS_paper_medium.md) — Brain-derived heuristics with fixed vs learned coefficients
- [Full HRS paper](paper.md) — Original theoretical framework, training protocol, and ablation study

## Author

Michael Bee ([@mbonsign](https://medium.com/@mbonsign))
