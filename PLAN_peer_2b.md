# Plan: PEER 2B — Full Pipeline from Data Curation to Trained Model

## Overview

Train a 2B parameter PEER model on curated SlimPajama data with a topic classification head. Two phases: a local proof-of-concept on 5-10B tokens (1 week on RTX 5070 Ti), then optionally a full 100B token run on cloud GPUs.


## Phase 1: Data Pipeline (Days 1-3)

### 1a. Download SlimPajama subset

Download 10B tokens worth from SlimPajama (~25GB compressed). Already in progress.

```
Dataset: DKYoon/SlimPajama-6B (6B token subset for initial work)
Full dataset: gmongaras/SlimPajama-627B_Reupload (for scaling up)
Storage: /mnt/data/Code/HRS/datasets/slimpajama/
```

### 1b. Entropy scoring + topic labeling (single Llama 7B pass)

Load Llama 7B (or Mistral 7B) in 4-bit quantization on the 5070 Ti (~5GB VRAM).

For each document:
1. Run forward pass over first 512 tokens, compute mean Shannon entropy
2. Generate topic label: "What is the primary topic of this text? Respond with a single short phrase."
3. Save: (doc_id, mean_entropy, max_entropy, raw_topic_label, source_domain, text_length)

Throughput estimate: ~500-1000 docs/hour with combined scoring + generation.
Time for 6B tokens (~2M documents): ~80-160 hours. Can parallelize by running on chunks.

Optimization: score only the first N documents to establish entropy thresholds, then apply thresholds without scoring the rest. Score all for topic labels since we need those for training.

### 1c. Build topic taxonomy

1. Embed all raw topic labels with a sentence embedding model (all-MiniLM-L6-v2, CPU)
2. K-means cluster into 500-1000 canonical topics
3. Manual inspection of cluster centers for sanity
4. Save mapping: raw_label -> canonical_topic_id

### 1d. Filter and assemble

1. Remove documents above entropy threshold (set after Phase 1b analysis)
2. Attach canonical_topic_id to each remaining document
3. Tokenize with GPT-2 BPE tokenizer
4. Save as chunked tensor files ready for training
5. Target: 5B clean tokens for local run, 50-100B for cloud run


## Phase 2: Model Architecture

### Config: PEER 2B

```
d_model:        1536
d_ff:           6144        (dimension reference, PEER replaces MLP)
n_layers:       8
n_heads:        24          (query heads)
n_kv_heads:     6           (GQA — 4x KV compression)
head_dim:       64          (1536 / 24)
max_seq_len:    2048
dropout:        0.1
vocab_size:     50257       (GPT-2 BPE)
```

**PEER config:**
```
n_sub_keys:     384         (384^2 = 147,456 experts per layer)
n_heads:        8           (retrieval heads)
top_k:          16          (experts per retrieval head)
active/token:   128         (8 heads * 16 top-k)
```

**Estimated total: ~2.0B parameters**
- Backbone (attention + norms): ~60M
- PEER layers (8 layers * ~240M): ~1,920M
- Vocab embedding: ~77M
- Topic head: ~1M

**VRAM:**
- Inference (fp16): ~4.0 GB
- Training (8-bit Adam): ~10 GB + activations
- With gradient checkpointing: should fit in 16 GB at batch=2, grad_accum=16

### Architectural changes from V17

1. **RMSNorm** replacing LayerNorm — faster, proven at scale
2. **Grouped Query Attention (GQA)** — 24 Q heads, 6 KV heads. Cuts KV cache 4x for longer context
3. **Context length 2048** (up from 512) — real text needs more context
4. **Topic classification head** — small MLP (d_model -> 256 -> n_topics), cross-entropy against LLM-generated labels
   - Operates per-segment: split 2048-token sequence into 4 segments of 512
   - Each segment gets a topic prediction
   - Loss weight: 0.1 relative to main CE
   - Dropped at inference
5. **KV cache** for generation — already implemented in V17
6. **RoPE with dynamic extension** — already implemented, supports generation beyond training length

### What's NOT included

- No engram (training crutch — V17 without it was unanimously preferred in human eval)
- No multi-path routing (conv/attn/sink tiers — fragmented representations)
- No sparsity bottleneck (redundant with PEER's natural sparsity)
- No BDH / virtual synapse / learnable loss scaling (complexity without benefit)


## Phase 3: Implementation (Days 3-5)

### 3a. Add RMSNorm

Replace all LayerNorm with RMSNorm:
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight
```

### 3b. Add GQA to CausalSelfAttention

Modify Q/K/V projections:
- Q: d_model -> d_model (24 heads * 64 dim)
- K: d_model -> n_kv_heads * head_dim (6 * 64 = 384)
- V: d_model -> n_kv_heads * head_dim (6 * 64 = 384)
- Expand K/V heads to match Q heads during attention (repeat_interleave)

### 3c. Add topic classification head

```python
class TopicHead(nn.Module):
    def __init__(self, d_model, n_topics, segment_len=512):
        super().__init__()
        self.segment_len = segment_len
        self.head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, n_topics),
        )
    def forward(self, hidden_states):
        # hidden_states: (B, T, D)
        B, T, D = hidden_states.shape
        n_segs = T // self.segment_len
        if n_segs == 0:
            return None
        segs = hidden_states[:, :n_segs * self.segment_len].reshape(B, n_segs, self.segment_len, D)
        seg_means = segs.mean(dim=2)  # (B, n_segs, D)
        return self.head(seg_means)   # (B, n_segs, n_topics)
```

### 3d. Update data loader

- Load tokenized SlimPajama chunks with topic_id metadata
- Each batch provides (input_ids, targets, topic_ids_per_segment)
- Sequence length: 2048 tokens

### 3e. Update training loop

- Main loss: cross-entropy on next-token prediction
- Topic loss: cross-entropy on segment-level topic classification (weight 0.1)
- 8-bit Adam optimizer (bitsandbytes)
- Gradient checkpointing enabled
- Mixed precision (fp16/bf16)
- Cosine LR schedule with warmup

### 3f. Verify it fits

Before full training, run a few steps to confirm:
- Forward + backward pass completes without OOM
- Gradient checkpointing is working
- 8-bit Adam states fit in remaining VRAM
- Throughput: tokens/sec at batch=2, grad_accum=16


## Phase 4: Local Training Run (Days 5-12)

### Training config

```
dataset:            curated SlimPajama (~5B tokens)
effective_batch:    32 (batch=2, grad_accum=16)
seq_len:            2048
total_steps:        ~76K (5B tokens / 2048 / 32)
lr:                 3e-4 peak, cosine decay
warmup:             2000 steps
optimizer:          8-bit Adam (bitsandbytes)
precision:          fp16 or bf16
grad_checkpoint:    enabled
hardware:           1x RTX 5070 Ti
est. time:          5-7 days
est. VRAM:          ~12-14 GB
```

### Checkpointing

- Save every 5000 steps
- Keep best checkpoint by validation loss
- Validate on held-out SlimPajama subset every 2500 steps

### Monitoring

- Track: train CE, val CE, val PPL, topic classification accuracy
- Generate samples every 5000 steps for qualitative check
- Log PEER expert utilization (are experts specializing?)


## Phase 5: Evaluation (Day 12-13)

### Benchmarks (same suite as V16/V17)

1. **MAUVE** — 1000 samples, 256 continuation tokens, temperature 0.9, top-k 50
2. **Self-BLEU and Distinct-N** — diversity metrics
3. **Cross-model PPL** — score with GPT-2 Medium
4. **Entropy profile** — per-position confidence
5. **Human eval** — blind A/B against V17 and/or GPT-2 Medium

### Key comparisons

- PEER 2B vs V17 (PEER 499M) — does scaling help?
- PEER 2B vs GPT-2 Medium (355M dense) — PEER vs dense at different sizes
- Topic head accuracy — is the model learning topic-aware representations?

### Generation speed

- Benchmark tok/s with KV cache at various context lengths
- Compare against dense model of similar size


## Phase 6: Scale Up (Optional, Cloud)

If local run succeeds:

1. Rent 8xA100 80GB node (~$10-15/hr on Lambda or RunPod)
2. Train on full 50-100B curated tokens
3. Estimated time: 3-5 days ($720-1800)
4. Compare against open 7B models (Llama 3 8B, Mistral 7B) on standard benchmarks

This is optional — only worth doing if Phase 4 shows clear scaling gains over V17.


## Dependencies to install

```bash
pip install bitsandbytes accelerate  # 8-bit Adam, gradient checkpointing
pip install sentence-transformers    # for topic label clustering
# Mistral 7B or Llama 7B weights from HuggingFace (for entropy/topic scoring)
```


## Risk register

| Risk | Mitigation |
|------|------------|
| OOM during training | Reduce batch to 1, increase grad_accum. Enable gradient checkpointing. Reduce seq_len to 1024 as fallback |
| Product-key routing degrades at 147K experts | Monitor expert utilization. Fall back to 512^2=262K with fewer layers if routing collapses |
| Topic labels too noisy | Inspect clusters manually. Start with fewer canonical topics (100-200). Topic head is auxiliary — bad labels hurt less than bad main data |
| 5-7 day training run fails/crashes | Checkpoint every 5000 steps. Resume from checkpoint. Use tmux/screen |
| SlimPajama subset too small/biased | Verify source domain balance. Supplement with additional downloads if needed |
| Entropy scoring takes too long | Score only first 256 tokens per document. Sample 20% of documents and extrapolate thresholds |


## Timeline summary

| Day | Activity |
|-----|----------|
| 1-2 | Download SlimPajama, start entropy/topic scoring |
| 2-3 | Analyze entropy distribution, build topic taxonomy, filter dataset |
| 3-5 | Implement RMSNorm, GQA, topic head, data loader. Verify VRAM fit |
| 5-12 | Training run (~7 days) |
| 12-13 | Evaluation suite, human eval, write up results |
| 14+ | (Optional) Cloud scale-up if results warrant |


## Success criteria

- Val PPL significantly better than V17's 21.41 on WikiText-103 equivalent
- MAUVE >= 0.93 (matching or exceeding V17)
- Human eval: preferred over V17 in blind A/B
- Topic head accuracy > 50% (model has learned topic-aware representations)
- Generation speed > 100 tok/s on 5070 Ti with KV cache
- No training instability (loss spikes, divergence)
