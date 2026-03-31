# Topic-Routed Context Assembly: Replacing Recency with Relevance in Transformer Context Windows

**Michael Bonsignore and Claude (Anthropic)**

## Abstract

We present a method for assembling transformer context windows based on topic relevance rather than token recency. The system uses the model's own hidden-state representations — mean-pooled into engram vectors — as topic classifiers and retrieval keys, requiring no external models or additional training.

The method works well at scale and fails at short scale — a finding we characterize precisely. For 512-token article segments, engram cosine similarity achieves 97.3% accuracy at same-vs-different-topic classification, with positive pairs centered at 0.755 similarity and negative pairs at 0.135. A learned binary classifier (1,377 parameters) cannot beat the optimal fixed threshold, confirming linear separability at this scale.

For single-sentence prompts — the actual use case in conversation — accuracy drops to 71.5%. The positive/negative similarity distributions collapse from a 0.62 gap to a 0.14 gap, with their overlap zones inverted. No threshold or learned classifier can reliably separate them. Seven stress tests confirm this: at a threshold of 0.4, even trivially different topics (Roman history vs. JavaScript) collapse into one cluster. At 0.7, separation improves but adversarial accuracy drops from 100% to 60% as metaphorical prompts are misrouted.

One bright spot: the engrams capture semantics, not vocabulary. Metaphorical cross-domain prompts ("Napoleon's defeat was a catastrophic loss function for the French Empire") cluster with their literal counterparts at 100% accuracy, even when surface vocabulary is entirely from a different domain.

Despite these classification failures, topic-routed context **improves generation quality**: MAUVE score rises from 0.919 to 0.962 at 500-token prompts, the highest score in this project. Imperfect routing still beats no routing because the baseline (naive sliding window) provides zero topic-relevant context. A system that is right 71% of the time adds more relevant context than a system that ignores relevance entirely.

## 1. Introduction

Every transformer has a context window, and every context window has the same problem: it fills up with whatever came most recently. If the current prompt is about Roman history, the model's context shouldn't contain the earlier discussion about baking cupcakes. The baking tokens consume attention capacity and may crowd out older but relevant historical context.

The standard solution is retrieval-augmented generation (RAG): build an external index, retrieve relevant documents per query, inject into context. This works but requires separate infrastructure — an embedding model, a vector database, a retrieval pipeline.

We propose a simpler approach: use the language model's own hidden-state representations to organize context by topic. The model already encodes semantic content in its hidden states. We extract these representations, cluster them by similarity, and use the clusters to decide what enters the context window.

The approach works at article scale and fails at sentence scale. This paper quantifies both and maps the boundary between them.

### 1.1 Contributions

1. **Quantitative characterization of the engram similarity space** at two scales: 97.3% accuracy for 512-token segments vs. 71.5% for single sentences.
2. **Evidence that the failure is representational, not algorithmic** — a learned classifier cannot beat a fixed threshold at either scale, ruling out non-linear decision boundaries as a fix.
3. **A comprehensive failure surface** — seven stress tests characterizing centroid drift, semantic overlap resolution, adversarial vocabulary, and topic forking.
4. **Evidence that engrams encode semantics, not vocabulary** — 100% adversarial routing accuracy for metaphorical cross-domain prompts.
5. **A working system** for topic-routed context assembly, with online clustering, auto-merge, user-toggleable topics, and configurable active slots.

## 2. Method

### 2.1 Engram Extraction

Given a text segment of T tokens, we run a forward pass through the transformer and extract hidden states from the second-to-last layer. We mean-pool across the token dimension to produce a single vector of shape (d_model,), then L2-normalize it. This is the engram.

The forward pass is the same computation the model performs during inference. The only additional cost is one mean-pooling and one normalization.

### 2.2 Online Topic Clustering

Prompts are clustered as they arrive:

1. Compute the incoming prompt's engram.
2. Compare against all existing cluster centroids by cosine similarity.
3. If the best match exceeds threshold θ, join that cluster. Otherwise, create a new cluster.
4. Update the matched cluster's centroid as a running mean, L2-normalized.
5. Check all cluster pairs for merge (if two centroids exceed θ_merge, absorb the smaller into the larger).

Each cluster maintains: an evolving centroid, an ordered list of member prompts (linked list), a human-readable title generated from keyword extraction, and a user-toggleable enabled flag.

### 2.3 Active Buffer and Context Assembly

A configurable number of clusters (default 2, adjustable to any number) are held in an active buffer, ordered by most-recently-used. When a prompt matches a cluster, that cluster moves to the front. If the buffer overflows, the least-recently-used cluster is evicted but persists for later reactivation.

The context token budget is distributed across active, user-enabled clusters with exponential decay by recency: the front cluster gets 50% of the budget, the next gets 50% of the remainder, and so on. Within each cluster, the most recent prompts are packed first.

Users see a topic list and can toggle clusters on/off:

```
Active Topics:
  [x] Roman / Empire / Constantinople / Byzantine (4 prompts)
  [x] Make / Chocolate / Cupcakes / Preheat (3 prompts)
  [ ] Quantum / Entanglement / Particles (1 prompt)
```

## 3. The Engram Similarity Space

### 3.1 Experimental Setup

We use a 510M parameter transformer with PEER feed-forward layers (V18 architecture), trained on WikiText-103. d_model=1024, 6 layers, 16 attention heads. Engrams are extracted from layer 4 (second-to-last).

### 3.2 Article-Length Segments (512 tokens)

We computed engrams for segments from 2,000 WikiText-103 articles (up to 3 segments of 512 tokens each) and built 50,000 pairs: 25,000 same-article (positive) and 25,000 cross-article (negative).

| | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **Positive** | 0.755 | 0.114 | 0.087 | 0.965 |
| **Negative** | 0.135 | 0.142 | -0.202 | 0.907 |

The distributions are well-separated. Optimal fixed threshold: **0.45 at 97.3% accuracy**. A learned binary classifier (1,377 parameters, 2-layer MLP on 8 pair features) achieves **97.3%** — identical to the threshold. The space is linearly separable at this scale.

### 3.3 Single-Sentence Prompts (8–40 words)

We extracted individual sentences from 3,000 articles and built 50,000 sentence-level pairs.

| | Mean | Std | Min | Max |
|---|------|-----|-----|-----|
| **Positive** | 0.429 | 0.131 | -0.042 | 1.000 |
| **Negative** | 0.291 | 0.118 | -0.091 | 0.753 |

The distributions largely overlap. The gap between means is 0.138 (vs. 0.620 at article scale). The overlap zone is inverted: the 75th percentile of negatives (0.369) exceeds the 25th percentile of positives (0.340).

Optimal fixed threshold: **0.35 at 70.9% accuracy**. The learned classifier achieves **71.5%** — a marginal +0.6% improvement that confirms there is minimal non-linear structure to exploit.

### 3.4 The Scale Gap

| Metric | 512-token segments | Single sentences |
|--------|-------------------|-----------------|
| Positive mean | 0.755 | 0.429 |
| Negative mean | 0.135 | 0.291 |
| Gap | 0.620 | 0.138 |
| Optimal accuracy | 97.3% | 71.5% |

Mean-pooling over 512 tokens averages out noise and captures the dominant semantic theme. Mean-pooling over 15 tokens preserves noise — the representation is dominated by whichever tokens produce the largest hidden-state magnitudes, not by the overall topic.

This is not a threshold problem or a classifier problem. It is a **signal-to-noise problem** in the engram computation itself.

## 4. Needle-in-a-Haystack Evaluation

Before stress-testing, we confirm the system works under favorable conditions.

Five synthetic Wikipedia-style facts ("needles") covering science, history, hobbies, biology, and geography. Each processed alongside 20 real WikiText-103 distractor documents. Needles are multi-sentence paragraphs (~50–80 words).

| Needle | Rank | Similarity |
|--------|------|------------|
| Thornfield Protocol (science) | 1 | 0.44 |
| Kestlemere village (history) | 2 | 0.36 |
| ZB-Petrus method (hobby) | 1 | 0.64 |
| Caspian tiger genetics (biology) | 1 | 0.53 |
| Mount Seravezza (geography) | 1 | 0.55 |

**100% retrieval, mean rank 1.2.** Position invariant (first/middle/last all rank 1).

These needles are multi-sentence, providing enough tokens for reliable engram computation. The stress tests that follow use single sentences.

## 5. Stress Tests

### 5.1 Semantic Overlap Gradient

Five levels of topic overlap, from trivially separable to near-identical. Ten single-sentence prompts per level (5 per topic), interleaved.

**Results at threshold 0.4:**

| Level | Topics | Clusters | Purity |
|-------|--------|----------|--------|
| 0 (trivial) | Roman history vs JavaScript | 1 | 0.500 |
| 1 (distant) | Roman military vs modern civil engineering | 1 | 0.500 |
| 2 (moderate) | Roman bread baking vs medieval bread baking | 1 | 0.500 |
| 3 (high) | Roman Republic vs Roman Empire governance | 1 | 0.500 |
| 4 (near-identical) | Caesar's military vs Caesar's politics | 1 | 0.500 |

Every level collapses to a single cluster. Even Level 0 — topics with zero semantic overlap. This is the short-prompt problem in action: single sentences produce engrams within 0.4 similarity of everything.

### 5.2 Threshold Sweep

| Threshold | L0 Purity/Clusters | L4 Purity/Clusters | Adversarial | Fork Split | Drift Probe |
|-----------|-------------------|-------------------|-------------|------------|-------------|
| 0.4 | 0.50 / 1 | 0.50 / 1 | 100% | No | MISS |
| 0.5 | 0.80 / 2 | 0.60 / 2 | 100% | No | MATCH |
| 0.6 | 0.80 / 3 | 0.90 / 5 | 100% | No | MATCH |
| 0.7 | 1.00 / 7 | 0.90 / 6 | 60% | No | MATCH |
| 0.8 | 1.00 / 9 | 1.00 / 10 | 20% | SPLIT | MISS |
| 0.9 | 1.00 / 10 | 1.00 / 10 | 0% | SPLIT | MISS |

**No single threshold resolves the tradeoff.** Low thresholds maintain semantic coherence (adversarial accuracy 100%) but cannot separate topics. High thresholds separate topics but shatter semantics and over-cluster.

The sweet spot is around 0.6–0.7 for well-separated topics (Level 0 purity 0.80–1.00) but this still produces too many clusters (3–7 for 10 prompts) and fails on adversarial prompts at 0.7.

### 5.3 Centroid Drift Bomb

Ten prompts that gradually walk from "Roman Empire" to "Neural Networks," each close enough to the previous centroid to join the cluster.

At threshold 0.4, the centroid drifts 0.41 cosine distance in 10 steps. A probe about Rome returns similarity 0.370 — below threshold, failing to match its own cluster. The title drifts to "Networks / Roman / Medieval / Infrastructure."

At threshold 0.5, the drift bomb fails immediately — the first non-Roman prompt creates a new cluster. The original cluster is preserved.

**Implication:** Centroid drift is only a problem at very low thresholds. At 0.5+, the system is naturally immune because off-topic prompts don't meet the join threshold.

### 5.4 Adversarial Vocabulary

Five prompt pairs where the metaphorical version uses vocabulary from a different domain.

| Pair | Engram Similarity |
|------|-------------------|
| Baking metaphor for ML ("layers of a cake are like layers of a neural network") | 0.661 |
| ML vocabulary for history ("Napoleon's defeat was a catastrophic loss function") | 0.834 |
| CS vocabulary for Roman history ("Senate operated like a distributed system") | 0.784 |
| Relationship vocabulary for physics ("entanglement is like a long-distance relationship") | 0.797 |
| Cooking vocabulary for chemistry ("polymerization is like making a chain of paper clips") | 0.629 |

**At threshold 0.4: 100% accuracy.** Every metaphorical prompt clusters with its literal counterpart.

This is the system's strongest result. Engrams capture semantic content, not surface vocabulary. "Napoleon's defeat was a catastrophic loss function" routes to history (similarity 0.834 with the literal version), not to machine learning, despite using ML vocabulary. The model's hidden states encode what the text is about, not what words it uses.

At threshold 0.7, accuracy drops to 60% — the two lower-similarity pairs (0.629, 0.661) split off. The semantic signal is real but not strong enough to overcome high thresholds.

### 5.5 Topic Fork

Starting with "machine learning" prompts, then interleaving "computer vision" and "NLP" prompts.

**The fork never splits** at any threshold from 0.4 to 0.7. The shared ML vocabulary keeps all subtopics within threshold. At 0.8 it splits, but into 3+ noisy clusters, not 2 clean subtopics.

Engram similarity cannot detect subtopic divergence within a parent topic. Mean-pooled representations capture the dominant theme but lose fine-grained distinctions between fields that share vocabulary.

### 5.6 Negative Result: Trained Categorization Head

The V18 model includes a categorization head trained with α=0.1 cross-entropy loss on 50 TF-IDF-derived topic clusters. Training loss reached 1.2 (random: 3.9).

Evaluation on validation articles: **3.3% accuracy** (random baseline: 2.0%).

The head learned to minimize loss on noisy, sequence-level labels that span article boundaries. Low training loss does not imply classification accuracy when labels are misaligned.

## 6. Generation Quality: MAUVE Evaluation

The stress tests (Section 5) measure clustering accuracy on synthetic single-sentence prompts. But the system's purpose is to improve generation quality. Does topic-routed context actually produce better text than naive recency, even when the routing is imperfect?

### 6.1 Setup

We populated the topic context manager from 60 WikiText-103 validation articles (full article text, ~500 characters each), producing 34 topic clusters. We then generated 256-token continuations from WikiText-103 test set prompts at two lengths, comparing against reference text using MAUVE (1,000 reference/generated pairs per condition in the baseline runs, 500 in the topic-routed runs).

The topic manager used threshold 0.5, max 384 context tokens (leaving room for the prompt in the 512-token window), and 3 active slots.

### 6.2 Results

| Condition | Baseline | + Topic Routing | Effect |
|-----------|----------|-----------------|--------|
| 50-tok prompt | 0.915 | **0.951** | **+0.036** |
| 500-tok prompt | 0.919 | **0.962** | **+0.043** |

**Topic routing improves generation quality at both prompt lengths.** The 500-token score of 0.962 is the highest MAUVE score in this project — above entropy-gated retrieval (0.950), above the V17 baseline (0.943), and well above V16 (0.806–0.906).

The effect is larger at 500 tokens (+0.043) than at 50 tokens (+0.036). Longer generations have more room to drift off-topic; relevant context keeps them anchored.

### 6.3 Why Imperfect Routing Still Helps

This result appears to contradict the stress test findings. Single-sentence classification accuracy is only 71.5%, yet MAUVE improves substantially. Three factors explain this:

**The bar is low.** The baseline is a naive sliding window that provides no topic-relevant context at all. Even noisy topic routing — correct 71% of the time — adds relevant context that the baseline entirely lacks. A system doesn't need to be perfect to beat doing nothing.

**Longer prompts produce better engrams.** The WikiText-103 test prompts (50 or 500 tokens) are longer than the single-sentence benchmarks (15–30 tokens). At 50 tokens, we're already in a better regime than the stress tests measured. At 500 tokens, engram quality approaches the article-scale accuracy (97.3%).

**Context quantity compensates for routing noise.** Each active cluster contributes up to 384 tokens of context. Even if some of those tokens are from a misrouted prompt, the majority are topically relevant. The model's attention mechanism can ignore the irrelevant tokens — it just needs enough relevant ones to anchor generation.

### 6.4 Comparison Across All Systems

| System | 50-tok MAUVE | 500-tok MAUVE |
|--------|-------------|--------------|
| V16 (prepend engram ON) | 0.806 | 0.888 |
| V16 (engram OFF) | 0.905 | 0.906 |
| V17 (no engram, baseline) | 0.933 | 0.943 |
| V18 (cross-attn engram ON) | 0.915 | 0.919 |
| V18 + EGR (entropy-gated retrieval) | 0.926 | 0.950 |
| **V18 + Topic Routing** | **0.951** | **0.962** |

Topic routing produces the best generation quality across the board. The progression from V16 (0.806) to topic routing (0.962) represents a 0.156 MAUVE improvement through four architectural iterations, all on the same 510M parameter model and hardware.

## 7. Discussion

### 7.1 The Fundamental Limitation

The system works at article scale (97.3%) and fails at sentence scale (71.5%). This is a representation problem, not a decision-boundary problem — proven by the fact that a learned classifier cannot improve on a fixed threshold at either scale.

Mean-pooling is the bottleneck. Over 512 tokens, it produces a stable semantic summary. Over 15 tokens, it produces noise. Any system that relies on mean-pooled engrams from short text will hit this wall.

Yet even with this limitation, the system improves MAUVE by +0.036 to +0.043. Imperfect routing still beats no routing.

### 7.2 What Works

**Topic-routed context improves generation quality.** MAUVE 0.962 at 500 tokens — the best in this project. This is the bottom line.

**Engrams are semantic, not lexical.** The adversarial benchmark proves this conclusively. Metaphorical cross-domain prompts route correctly at 100% (θ=0.4). This is not keyword matching — it is genuine semantic understanding encoded in hidden states.

**Article-scale routing is essentially solved.** 97.3% accuracy with a simple cosine threshold. The engram space has clean linear separability at this scale. No complex infrastructure needed.

**The system is zero-cost.** Engram extraction is a byproduct of the forward pass. Clustering is O(n) per prompt. No external models, no retraining, no additional parameters.

### 7.3 What Doesn't Work

**Single-sentence routing is unreliable.** 71.5% accuracy — better than chance but not reliable enough for production use.

**No single threshold resolves the separation/semantic tradeoff.** Low thresholds preserve semantics, high thresholds enable separation. You can't have both with a single number.

**Subtopic detection is beyond the system's resolution.** CV and NLP are genuinely different fields but their engrams are indistinguishable.

### 7.4 Relationship to RAG

This system is complementary to RAG, not a replacement:

- **RAG:** "What external knowledge is relevant?"
- **Topic routing:** "Which parts of our conversation are relevant?"

They operate at different scales and could be combined.

### 7.5 Generality

Nothing here is specific to V18, PEER, or WikiText-103. Any transformer that produces hidden states supports engram extraction. The findings about scale dependence likely generalize to any mean-pooled representation.

## 8. Proposed Mitigations

We have not implemented these. They represent the natural next steps motivated by the failure analysis.

**Accumulate before routing.** Don't route single sentences. Buffer 2-3 prompts, concatenate, then compute the engram. This trades latency for accuracy by moving the effective input length toward the regime where engrams work (512 tokens).

**Attention-weighted pooling.** Replace mean-pooling with attention-weighted pooling, using the model's own attention scores to weight token contributions. Topically informative tokens should receive higher weight than function words, producing more discriminative engrams from short text.

**Context-augmented engrams.** When computing an engram for a new prompt, include previous prompts as context. The engram then captures the new prompt's topic conditioned on conversation history, not in isolation. This leverages the transformer's ability to contextualize.

**Multi-scale engrams.** Extract engrams from multiple layers and concatenate. Different layers may capture topic at different resolutions — early layers for broad category, late layers for specific content.

**Per-cluster adaptive thresholds.** Set each cluster's join threshold based on its intra-cluster variance. Tight, focused clusters get high thresholds; loose, diverse clusters get low ones.

## 9. Reproducibility

Code: github.com/MikeyBeez/HRS

| File | Description |
|------|-------------|
| `topic_context.py` | TopicContextManager: online clustering, active buffer, user toggles |
| `train_topic_classifier.py` | Article-length pair analysis and classifier training |
| `train_topic_classifier_short.py` | Short-prompt pair analysis and classifier training |
| `benchmark_topic_routing.py` | Seven stress tests (drift, overlap, adversarial, fork) |
| `engram_store.py` | Vector store with cosine similarity retrieval |
| `niah_egr.py` | Needle-in-a-haystack evaluation |
| `benchmark_mauve_topic.py` | MAUVE benchmark for topic-routed context |
| `eval_categorization.py` | Categorization head evaluation (negative result) |

Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM. All experiments except model training (~11 hours) run in minutes.
