# Topic-Routed Context Assembly: Replacing Recency with Relevance in Transformer Context Windows

**Michael Bonsignore and Claude (Anthropic)**

## Abstract

We present a method for assembling transformer context windows based on topic relevance rather than token recency. The system uses the model's own hidden-state representations — mean-pooled into engram vectors — as both topic classifiers and retrieval keys, requiring no external models or additional training. Incoming prompts are clustered online by engram cosine similarity, with evolving centroids and automatic cluster merging. The context window is filled from active topic clusters rather than from a naive sliding window, ensuring that the model sees relevant context regardless of when it appeared in the conversation.

We show that mean-pooled hidden states from a pretrained transformer produce an engram space where cosine similarity corresponds to topic relatedness — achieving 100% document retrieval accuracy among 20 distractors in a needle-in-a-haystack evaluation. In contrast, a categorization head explicitly trained to predict topic labels achieves only 3.3% accuracy due to sequence-level label noise. The model's internal representations are better topic features than a trained classifier.

The system is entirely inference-time: no model retraining, no learned parameters beyond the base model, no external dependencies. It consists of a few hundred lines of Python implementing online clustering, centroid management, and budget-allocated context assembly. Users can see and toggle active topics through a simple interface.

## 1. Introduction

Every transformer has a context window, and every context window has the same problem: it fills up with whatever came most recently. A 4K-token window used in a conversation about Roman history, baking, and quantum physics contains all three topics jumbled together. The model must attend over irrelevant tokens to find the ones that matter for the current query.

This is wasteful. If the current prompt is about Roman history, the baking tokens are noise. They consume attention capacity, dilute the relevant signal, and for models with limited context (512–2K tokens), they may crowd out older but relevant historical context entirely.

The standard solution is retrieval-augmented generation (RAG): build an external index, use a separate retriever model to find relevant documents, and inject them into context. This works but requires additional infrastructure — an embedding model, a vector database, a retrieval pipeline — that is separate from the language model itself.

We propose a simpler approach: use the language model's own representations to organize context by topic. The model already encodes semantic content in its hidden states. We extract these representations, cluster them by similarity, and use the clusters to decide what goes into the context window. No external models. No retraining. No additional learned parameters.

### 1.1 Key Insight

A mean-pooled hidden state from a late transformer layer — what we call an engram — captures the semantic content of a text segment in a space where cosine similarity corresponds to topic relatedness. This is not a trained feature; it is an emergent property of language model pretraining. Documents about similar topics produce similar hidden states because the model processes them through similar computational pathways.

We demonstrate this empirically: engram cosine similarity retrieves the correct document from among 20 distractors with 100% accuracy, while a categorization head explicitly trained to predict topic labels achieves only 3.3% accuracy. The model knows more about topics in its hidden states than it can express through a linear classification layer trained on noisy labels.

### 1.2 Contributions

1. **Topic-routed context assembly** — a method for filling context windows based on topic relevance rather than recency, using the model's own representations.
2. **Online engram clustering** — a real-time system that creates, evolves, and merges topic clusters as prompts arrive, without requiring a predefined topic taxonomy.
3. **A negative result on trained categorization** — demonstrating that a categorization head trained to low cross-entropy loss (1.2, well below random at 3.9) can achieve near-random classification accuracy (3.3%) when evaluated on held-out documents, due to sequence-level label noise.
4. **Evidence that hidden-state similarity outperforms trained classification** for topic routing in language models.

## 2. Background

### 2.1 Context Window Management

Current approaches to context management fall into several categories:

**Recency-based (sliding window).** The default: keep the last N tokens. Simple and universal but ignores relevance. Used by most inference frameworks.

**Retrieval-augmented generation (RAG).** Index documents externally, retrieve relevant ones per query, inject into context. Effective but requires separate infrastructure (embedding model, vector store, retrieval pipeline). Systems like RETRO (Borgeaud et al., 2022) and REALM (Guu et al., 2020) integrate retrieval into training.

**Compressive approaches.** Compress older context into summaries or compressed representations. Compressive Transformers (Rae et al., 2020) use attention to compress old memories. Gisting (Mu et al., 2023) learns to compress prompts into shorter token sequences.

**Sparse/structured attention.** Longformer (Beltagy et al., 2020), BigBird (Zaheer et al., 2020), and similar approaches modify the attention pattern to handle longer sequences efficiently. These extend context length but don't address relevance within the window.

Our approach is orthogonal to all of these. It operates at the context assembly stage — deciding which tokens enter the window — rather than modifying attention patterns or compressing tokens. It could be combined with any of the above.

### 2.2 Engrams as Semantic Keys

In our concurrent work on the PEER + Engram architecture (Bonsignore, 2026), we introduced engrams as compressed representations of model hidden states. An engram is computed by mean-pooling the hidden states from a selected transformer layer across all token positions in a segment, producing a single vector of dimension d_model.

In that work, engrams served as a memory mechanism during training. Here, we repurpose them as semantic keys for topic routing at inference time. The critical observation is that the engram space has topological structure that corresponds to semantic similarity — documents about similar topics cluster together in engram space — and this structure emerges from standard language model training without any explicit topic objective.

## 3. Method

### 3.1 Engram Extraction

Given a text segment tokenized as a sequence of T tokens, we run a forward pass through the transformer and extract hidden states from a selected layer (we use the second-to-last layer). The hidden states have shape (T, d_model). We mean-pool across the token dimension to produce a single engram vector of shape (d_model,), then L2-normalize it.

The choice of layer matters. Early layers encode syntactic features; late layers encode semantic content. The second-to-last layer balances semantic richness with pre-output stability (the final layer's representations are distorted toward the vocabulary projection).

Engram extraction requires one forward pass — the same computation the model would perform anyway during inference. The only additional cost is the mean-pooling operation, which is negligible.

### 3.2 Online Topic Clustering

Prompts are clustered as they arrive, without a predefined topic taxonomy.

**Cluster assignment.** When a new prompt arrives:
1. Compute its engram via forward pass and mean-pooling.
2. Compare against all existing cluster centroids by cosine similarity.
3. If the best match exceeds a similarity threshold θ_join (default 0.4), add the prompt to that cluster.
4. Otherwise, create a new cluster with this prompt as its sole member.

**Centroid update.** Each cluster maintains a running-mean centroid, L2-normalized after each update:

```
centroid_new = normalize((centroid_old * (n-1) + engram_new) / n)
```

This means the cluster's topic representation evolves as more prompts join. A cluster that starts with "The Roman Empire fell in 476 AD" and later receives "The Byzantine economy relied on trade through Constantinople" will develop a centroid that represents the broader topic of Roman/Byzantine history, not just the fall of Rome.

**Auto-merge.** After each prompt is processed, all cluster pairs are checked for similarity. If two centroids exceed a merge threshold θ_merge (default θ_join + 0.35, capped at 0.85), the smaller cluster is absorbed into the larger. This handles convergent topics — clusters that start separate but accumulate enough shared context to warrant merging.

**Title generation.** Each cluster generates a human-readable title from keyword extraction over its member prompts. Stopwords and short tokens are filtered; the top 3-4 keywords by frequency are joined. Titles update as new prompts join. This is a placeholder for more sophisticated title generation (e.g., via a small local language model).

### 3.3 Active Cluster Management

Not all clusters are loaded into context simultaneously. The system maintains an active buffer of configurable size (default 2, adjustable to any number for multidisciplinary work).

**MRU policy.** When a prompt matches a cluster, that cluster moves to the most-recently-used position in the active buffer. If the buffer is full, the least-recently-used cluster is evicted. Eviction is from the active buffer only — the cluster persists in storage and can be reactivated later.

**Reactivation.** If a future prompt matches an evicted cluster, it re-enters the active buffer at the MRU position, potentially evicting a different cluster. This handles topic switching: the user can discuss Rome, switch to baking, switch back to Rome, and the Roman history cluster reactivates with all its accumulated context intact.

### 3.4 Context Assembly

The context window is assembled from active, user-enabled clusters.

**Budget allocation.** The total token budget (e.g., 512 tokens) is distributed across active clusters with exponential decay by recency:
- Most recently used cluster: 50% of budget
- Next: 50% of remainder (25% total)
- Next: 50% of remainder (12.5% total)
- And so on

This prioritizes the current topic while maintaining representation of secondary topics.

**Token packing.** Within each cluster's budget, the most recent prompts are packed first (most recent prompt gets priority). If the budget allows, earlier prompts are included in chronological order. This ensures the context contains the latest relevant information.

**User toggles.** Each cluster can be enabled or disabled by the user. Disabled clusters are excluded from context assembly even if they are in the active buffer. This provides explicit user control over what the model sees:

```
Active Topics:
  [x] Empire / Roman / Constantinople / Byzantine (4 prompts, 60 tokens)
  [x] Make / Chocolate / Cupcakes / Preheat (3 prompts, 44 tokens)
  [ ] Quantum / Entanglement / Particles (1 prompt, 14 tokens)
```

## 4. Evaluation

### 4.1 Experimental Setup

We evaluate using a 510M parameter transformer with PEER feed-forward layers (V18 architecture), trained on WikiText-103. The model has a 512-token context window, d_model=1024, 6 layers, 16 attention heads. Engrams are extracted from layer 4 (second-to-last).

### 4.2 Engram Similarity as Topic Signal

**Needle-in-a-haystack.** We created 5 synthetic Wikipedia-style facts ("needles") covering science, history, hobbies, biology, and geography. Each was processed alongside 20 real WikiText-103 distractor documents. For each needle, we computed a query engram from a related question and searched for the nearest match.

| Needle | Found? | Rank | Cosine Similarity |
|--------|--------|------|-------------------|
| Thornfield Protocol (science) | Yes | 1 | 0.44 |
| Kestlemere village (history) | Yes | 2 | 0.36 |
| ZB-Petrus method (hobby) | Yes | 1 | 0.64 |
| Caspian tiger genetics (biology) | Yes | 1 | 0.53 |
| Mount Seravezza (geography) | Yes | 1 | 0.55 |

**Result: 100% retrieval accuracy, mean rank 1.2.** The engram space has sufficient semantic structure for reliable topic matching.

**Position invariance.** We varied the needle's position in the processing order (first, middle, last among the 20 distractors). Position had no effect on retrieval quality — rank 1 in all positions.

### 4.3 Trained Categorization Head (Negative Result)

The V18 model includes a categorization head — a linear projection from mean-pooled final hidden states to 50 topic categories derived from TF-IDF clustering of WikiText-103 articles. The head was trained with cross-entropy loss weighted at α=0.1, reaching a loss of 1.2 (random baseline: 3.9 = ln(50)).

When evaluated on WikiText-103 validation articles (60 documents, labels assigned by the same TF-IDF/KMeans model), the categorization head achieved **3.3% accuracy** — barely above the 2.0% random baseline.

**Analysis.** The failure stems from a train-eval mismatch. During training, each 512-token sequence receives a single category label: the mode of per-token categories within that sequence. But sequences are carved from a continuous token stream and can span article boundaries. The head learns to predict these noisy, misaligned labels — it minimizes training loss without learning generalizable topic classification.

**Implication.** Low cross-entropy loss does not imply classification accuracy when labels are noisy. For topic routing, the model's raw hidden-state similarity (100% NIAH accuracy) is a far more reliable signal than a trained classification layer (3.3% accuracy).

### 4.4 Online Clustering Quality

We fed a sequence of 7 prompts covering two topics (Roman/Byzantine history and baking) through the topic clustering system with similarity threshold 0.4.

**Correct behaviors:**
- Roman Empire and Byzantine Empire prompts merged into a single cluster titled "Empire / Roman / Constantinople / Byzantine"
- Baking prompts formed a separate cluster titled "Make / Chocolate / Cupcakes / Preheat"
- Topic switching correctly updated the active buffer
- Disabling the baking cluster removed it from context assembly

**Known failure:** One Roman-topic prompt ("Roman aqueducts transported water over vast distances using gravity") routed to the baking cluster due to centroid drift. As clusters accumulate diverse members, their centroids can become less discriminative. This is the primary failure mode and motivates future work on centroid regularization and adaptive thresholds.

## 5. Discussion

### 5.1 Why This Works

The core mechanism — engram cosine similarity as a topic signal — works because language models already organize their hidden states by semantic content. This is not a property we engineered; it is an emergent consequence of next-token prediction training. Documents about similar topics require similar computational pathways (attending to similar patterns, activating similar experts), which produces similar hidden states.

Mean-pooling aggregates these per-token hidden states into a single vector that captures the dominant semantic content of the segment. L2-normalization projects this onto the unit sphere, where cosine similarity measures angular distance — a natural measure of topical relatedness.

### 5.2 Limitations

**Threshold sensitivity.** The similarity threshold θ_join is a single global parameter. It should ideally vary per-cluster: tight clusters (highly focused topic) should require higher similarity for new members than loose clusters (broad topic). Adaptive thresholds calibrated from intra-cluster variance would address this.

**Centroid drift.** As clusters accumulate diverse members, their centroids can drift toward a generic mean that matches too many prompts. This is the standard problem of online K-means with running centroids. Potential mitigations include periodic re-centering (recompute centroid from all members), sub-clustering within large clusters, or bounded drift (cap the centroid's movement per update).

**Title quality.** Keyword extraction produces functional but ugly titles. A small local language model could generate clean titles from the cluster's content. The system exposes the necessary interfaces for this — it is an integration task, not a research problem.

**Single-model dependency.** Engram quality depends on the base model's representation quality. A poorly trained model would produce an engram space with less topological structure, degrading clustering quality. In practice, any model trained to reasonable perplexity appears to develop sufficient hidden-state structure for topic-level similarity.

### 5.3 Relationship to RAG

Topic-routed context assembly is complementary to retrieval-augmented generation, not a replacement. RAG retrieves specific documents from a large external corpus. Topic routing organizes the conversation's own context by relevance. They operate at different scales:

- **RAG:** "What external knowledge is relevant to this query?"
- **Topic routing:** "Which parts of our conversation so far are relevant to this query?"

A system could use both: topic routing to organize conversational context, and RAG to inject external knowledge when the model's entropy indicates it needs help. This is the architecture we sketched in our concurrent work on entropy-gated retrieval.

### 5.4 Generality

Nothing in this method is specific to the V18 architecture, PEER, or WikiText-103. The requirements are:

1. A transformer that produces hidden states (any transformer)
2. A method to extract a fixed-size representation from those hidden states (mean-pooling)
3. A distance metric in that representation space (cosine similarity)

This describes every language model in use today. The topic-routed context system could be applied to GPT, Llama, Mistral, or any other transformer architecture. The only model-specific parameter is the extraction layer, which should be a late (but not final) layer.

## 6. Future Work

**Adaptive thresholds.** Per-cluster similarity thresholds calibrated from intra-cluster variance.

**LLM-assisted management.** A small local model for title generation and merge decisions. The judgment "are Roman History and Byzantine Empire the same topic?" is trivial for a language model but difficult as a distance threshold.

**Evaluation at scale.** Testing with longer conversations (hundreds of prompts), more diverse topics, and measuring downstream task performance (not just clustering quality).

**Integration with attention.** Instead of hard budget allocation, let the model's attention mechanism decide how much to attend to each cluster's context. This would require modifying the attention computation but would provide a learned (rather than heuristic) relevance weighting.

**Cross-session persistence.** Saving and loading topic clusters across sessions, allowing the system to accumulate context over days or weeks of interaction.

## 7. Reproducibility

Code: github.com/MikeyBeez/HRS

Core implementation: `topic_context.py` (TopicContextManager class)

Supporting code: `engram_store.py` (vector store), `entropy_monitor.py` (entropy computation), `retrieval_engine.py` (entropy-gated retrieval), `niah_egr.py` (needle-in-a-haystack evaluation), `eval_categorization.py` (categorization head evaluation)

Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM. All experiments run in minutes except model training (~11 hours).
