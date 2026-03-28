# Plan: Entropy-Based SlimPajama Curation + Topic Labeling

## Goal

Clean SlimPajama (627B tokens) using Llama 7B as a quality scorer AND topic labeler in a single pass. The hypothesis: examples that produce high Shannon entropy under Llama 7B are noisy, confusing, or poorly written. These should be rewritten by Llama or removed. Simultaneously, Llama generates a topic label for each document, which becomes the training signal for a topic classification head in the PEER 2B model. The result is a higher-quality, topic-annotated training set.

## Why entropy?

Shannon entropy of the model's output distribution measures uncertainty. When Llama 7B reads a training example and is highly uncertain about what comes next at many positions, that signals:
- Poorly structured text (incoherent, garbled, boilerplate)
- Noisy formatting (HTML artifacts, encoding errors, tables rendered as text)
- Domain mismatch (highly specialized jargon that doesn't generalize)
- Contradictory or confusing content

Low-entropy examples are text that a strong model finds predictable — well-structured, clear, coherent. These are the examples we want to train on.

## Pipeline

### Phase 1: Score

1. Download SlimPajama in chunks (it's sharded, ~627B tokens total)
2. Load Llama 7B (or Llama 2 7B) in 4-bit quantization (~4GB VRAM) on the 5070 Ti
3. For each text example:
   - Tokenize and run through Llama 7B
   - Compute mean Shannon entropy across all positions (bits, base 2)
   - Record: (example_id, mean_entropy, max_entropy, source_domain, text_length)
4. Store scores in a lightweight database or parquet file
5. Estimate: ~1000 tokens/sec scoring throughput at batch=1 with 4-bit Llama → ~175M tokens/hour → 627B tokens would take ~3,600 hours at full scale

**Scaling strategy:** Don't score all 627B tokens. Score a representative sample:
- 1B tokens (~6 hours) gives a good entropy distribution per source domain
- Use the distribution to set thresholds, then apply thresholds to filter without scoring everything
- Or: score the first N tokens of each example (e.g., 512 tokens) as a proxy for the whole document

### Phase 1b: Topic labeling (same Llama pass)

During the same forward pass that computes entropy, generate a topic label for each document:

1. After scoring entropy, prompt Llama 7B with:
   ```
   What is the primary topic of this text? Respond with a single short phrase.

   Text: [first 512 tokens]

   Topic:
   ```
2. Record the raw topic string alongside the entropy score
3. This adds ~20-30 tokens of generation per document to the scoring pass — minimal overhead

### Phase 2: Analyze

1. Plot entropy distributions by source domain (CommonCrawl, Wikipedia, GitHub, Books, ArXiv, StackExchange)
2. Identify the entropy threshold that separates clean from noisy text
3. Manually inspect examples at different entropy levels to validate the threshold
4. Expected: Wikipedia and Books will have low entropy, CommonCrawl will have a long high-entropy tail

### Phase 2b: Build topic taxonomy

1. Embed all raw topic labels using a sentence embedding model (e.g., all-MiniLM-L6-v2, runs on CPU)
2. Cluster embeddings with k-means into ~500-1000 canonical topics
3. Map each document to its canonical topic ID
4. Manually inspect cluster centers to verify they make sense (e.g., "naval engineering", "molecular biology", "JavaScript frameworks")
5. Save: (document_id, entropy, raw_topic, canonical_topic_id)

### Phase 3: Clean

For examples above the entropy threshold, two strategies:

**Strategy A: Remove.** Simply drop high-entropy examples. Fast, simple, may lose some genuinely difficult but valuable text (e.g., technical papers that are hard to predict but worth learning from).

**Strategy B: Rewrite.** Feed high-entropy examples to Llama 7B (or a larger model if available) with a prompt like "Rewrite the following text to be clear and well-structured while preserving the factual content." This preserves the information while cleaning up the presentation. Slower and more expensive but retains coverage.

**Recommended: Start with Strategy A** (removal) for the first training run. Test Strategy B on a small subset to see if it's worth the cost.

### Phase 4: Assemble

1. Combine cleaned/filtered examples into a new dataset
2. Each example carries its canonical_topic_id as metadata
3. Maintain the source domain balance from SlimPajama (or intentionally rebalance — e.g., upweight Wikipedia and StackExchange, downweight CommonCrawl)
4. Target: 100B tokens for the first PEER 2B training run
5. Save as tokenized tensors (GPT-2 BPE) with topic IDs for the classification head

## Resource estimates

| Phase | Time | VRAM | Disk |
|-------|------|------|------|
| Download SlimPajama (full) | ~12 hours | — | ~800GB |
| Download 100B token subset | ~2 hours | — | ~130GB |
| Score 1B tokens with Llama 7B 4-bit | ~6 hours | ~5GB | ~1GB scores |
| Score 100B tokens (first 512 tok each) | ~50 hours | ~5GB | ~50GB scores |
| Rewrite (Strategy B, if used) | Days | ~5GB | ~same as input |
| Final tokenized dataset (100B tokens) | — | — | ~200GB |

All fits on the 5070 Ti (16GB) and the 7.3TB data drive (6.8TB free).

## Dependencies

```bash
pip install bitsandbytes accelerate  # for 4-bit Llama
# Llama 7B weights from HuggingFace (requires access approval)
# Alternative: use Mistral 7B (no approval needed) or Llama 2 7B
```

## Open questions

1. What entropy threshold to use? Need Phase 2 analysis first.
2. Should we use Llama 7B or Mistral 7B as the scorer? Mistral is newer and doesn't require Meta approval.
3. For rewriting (Strategy B), is 7B sufficient quality or do we need a larger model?
4. Should we score with our own V17 PEER model instead of Llama? It's smaller but might correlate well enough.
5. Domain balance: should we keep SlimPajama's original mix or rebalance toward higher-quality sources?

## After curation: PEER 2B Architecture

### Model config

- **d_model:** 2048
- **d_ff:** 8192 (for dimension compatibility, though PEER replaces the MLP)
- **n_heads:** 16 (with GQA: 16 query heads, 4 KV heads — cuts KV cache 4x)
- **n_layers:** 12-16 (TBD based on VRAM fit)
- **Context:** 2048 tokens minimum, 4096 if it fits
- **Normalization:** RMSNorm (replacing LayerNorm)
- **Positional encoding:** RoPE (already have, extend for longer context)
- **PEER:** 1024^2 = 1M experts per layer, 8 heads x 16 top-k = 128 active/token
- **KV cache:** yes (already implemented)
- **Target total params:** ~2B

### Two heads

1. **Main head (CE):** standard next-token prediction, cross-entropy loss
2. **Topic head:** small MLP classifier (d_model -> 256 -> n_topics), predicts canonical topic ID from segment representations
   - Operates on segment-level: split each 2048-token sequence into 4 segments of 512, each gets a topic prediction
   - Cross-entropy loss against LLM-generated topic labels
   - Loss weight: ~0.1 relative to main CE
   - **Dropped at inference** — the topic-aware representations stay in the weights
   - **Purpose:** teaches the model to internally represent "what kind of text am I processing," enabling future context curation (keep/discard/retrieve decisions)

### What's NOT included (dead ends from V8-V16)

- No engram (training crutch — human eval showed V17 without it is better)
- No multi-path routing (conv/attn/sink tiers fragmented representations)
- No sparsity bottleneck (redundant with PEER's natural 99.95% sparsity)
- No BDH (virtual synapse, hub routing, learnable loss scaling — all added complexity without generation benefit)

### Training plan

- Train on curated, topic-labeled SlimPajama (~100B tokens)
- Phased LR schedule (proven in V16/V17)
- Effective batch size 32+ (grad accumulation)
- Compare against a 2B dense baseline on the same data
- If hypothesis holds: 2B PEER competes with 7-8B dense models
- Inference target: real-time generation on RTX 5070 Ti
