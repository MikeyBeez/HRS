# Plan: Entropy-Based SlimPajama Curation

## Goal

Clean SlimPajama (627B tokens) using Llama 7B as a quality scorer. The hypothesis: examples that produce high Shannon entropy under Llama 7B are noisy, confusing, or poorly written. These should be rewritten by Llama or removed. The result is a higher-quality training set for scaling PEER to 2B parameters.

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

### Phase 2: Analyze

1. Plot entropy distributions by source domain (CommonCrawl, Wikipedia, GitHub, Books, ArXiv, StackExchange)
2. Identify the entropy threshold that separates clean from noisy text
3. Manually inspect examples at different entropy levels to validate the threshold
4. Expected: Wikipedia and Books will have low entropy, CommonCrawl will have a long high-entropy tail

### Phase 3: Clean

For examples above the entropy threshold, two strategies:

**Strategy A: Remove.** Simply drop high-entropy examples. Fast, simple, may lose some genuinely difficult but valuable text (e.g., technical papers that are hard to predict but worth learning from).

**Strategy B: Rewrite.** Feed high-entropy examples to Llama 7B (or a larger model if available) with a prompt like "Rewrite the following text to be clear and well-structured while preserving the factual content." This preserves the information while cleaning up the presentation. Slower and more expensive but retains coverage.

**Recommended: Start with Strategy A** (removal) for the first training run. Test Strategy B on a small subset to see if it's worth the cost.

### Phase 4: Assemble

1. Combine cleaned/filtered examples into a new dataset
2. Maintain the source domain balance from SlimPajama (or intentionally rebalance — e.g., upweight Wikipedia and StackExchange, downweight CommonCrawl)
3. Target: 100B tokens for the first PEER 2B training run
4. Save as tokenized tensors (GPT-2 BPE) for direct use by train.py

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

## After curation: Scale PEER to 2B

Once we have a clean 100B token dataset:
- Scale PEER to 2B total params (larger expert tables, possibly deeper)
- Train on the curated dataset
- Compare against a 2B dense baseline on the same data
- If hypothesis holds: 2B PEER competes with 7-8B dense models
