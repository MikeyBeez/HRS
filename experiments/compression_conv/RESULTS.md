# Hierarchical Causal Compression on Tiny Shakespeare

**Date:** 2026-05-06
**Headline:** 16× learned causal-conv compression on Tiny Shakespeare costs **17% perplexity** (117.05 → 137.31) but delivers **9× inference speedup at ctx 4096**. The architecture works at this scale and substrate — it falls in the spec's "investigate failure modes" zone (>10%, <50%), not the "adopt cleanly" zone (<5%) and not the "rule out" zone (>50%).

## Setup

Substrate per user instruction: Tiny Shakespeare (substituted for the spec's WikiText-103). Same 304k-token GPT-2 BPE corpus used by the prior `saliency_pool/` work.

Model: 4-layer transformer, d=384, 6 heads, d_ff=1536. Two architectures, matched compute:

- **Baseline:** standard transformer on tokens. 26.59M params.
- **Compressed:** same transformer but with 4-layer causal-conv compression stack between embedding and attention. Total compression 16× (four 2× layers). 27.78M params (~+1.2M for the compression stack).

Both predict every 16th token (positions 16, 32, ..., T−16). Baseline reads logits at positions [15, 31, ..., T−17]; compressed reads logits at compressed positions [0, 1, ..., T/16−2]. Same number of (B, n) predictions per sequence; same task; comparable losses.

Compression layer: `Conv1d(D, D, kernel=2, stride=2)` + LayerNorm + GELU. Causal because output position i depends on input positions [2i, 2i+1] only — and the next compressed position uses inputs ≥ 2(i+1). Stack 4 layers: top-of-stack output position i sees input tokens [16i, 16i+15] only.

Training: ctx 512, 2000 steps, batch 16, AdamW (lr=3e-4), cosine schedule. Each model trained from scratch. Wall time: baseline 135s, compressed 30s (compressed is faster despite extra params because attention runs on 32 positions instead of 512).

## Phase 1: training PPL at ctx 512

| | val PPL | val loss |
|---|---|---|
| baseline | 117.05 | 4.763 |
| compressed (16×) | 137.31 | 4.922 |
| ratio | **1.173×** | +0.16 |

Compressed model loses 17% perplexity. The architecture clearly works — the compressed model learns from random (PPL ~50,000 at step 1) down to 137 — but it doesn't match baseline.

Per spec decision tree: this falls in the **"investigate failure modes"** zone (>10%, <50% above baseline), not the adopt-cleanly zone (<5%) and not the rule-out zone (>50%).

## Phase 2: PPL across context lengths

Restricted to ctx ≤ training ctx (512) since pos_emb wasn't trained beyond 512 — extrapolation would measure pos_emb noise rather than compression behavior.

| ctx | baseline ppl | compressed ppl | ratio |
|---|---|---|---|
| 128 | 107.51 | 128.54 | 1.196 |
| 256 | 117.88 | 135.71 | 1.151 |
| 512 | 119.84 | 136.83 | 1.142 |

The compressed/baseline gap is **stable around 1.14-1.20 across context lengths within the training range**. The compression doesn't get worse at longer contexts (within trained range); the 17% gap is roughly invariant. That's a useful signal — whatever the compression is losing, it isn't a length-dependent failure.

## Phase 3: inference timing

Forward-pass wall time (mean over 10 runs, batch 1). Pos_emb zero-padded for ctx > 512 (extrapolation).

| ctx | baseline (ms) | compressed (ms) | speedup |
|---|---|---|---|
| 256 | 1.53 | 1.57 | 0.97× |
| 512 | 2.41 | 1.71 | 1.41× |
| 1024 | 3.45 | 1.72 | 2.00× |
| 2048 | 6.46 | 1.79 | 3.60× |
| 4096 | 18.03 | 2.00 | **9.03×** |

Speedup grows with context length, as expected from O(N²) attention vs O(N/16)² + O(N) compression overhead. At ctx 4096 the compressed model is 9× faster. At ctx 256 it's slightly slower (compression overhead doesn't pay off for such short sequences).

The compressed model's forward time is **nearly flat** across ctx 512-4096 (1.71 → 2.00 ms) because:
- Compression is O(N) per layer, 4 layers — negligible.
- Attention is O((N/16)²) — at ctx 4096, that's 256² = 65K ops per layer, still tiny.

The baseline scales O(N²) at the attention stage. Crossover is around ctx 384-400.

## What the experiment learned

**The architecture works at this scale.** The compressed model is a real language model, not random — PPL 137 at 16× compression on TS, vs baseline 117 at 1×. Inference scales as expected.

**The 17% PPL gap is the failure to investigate.** Possible mechanisms (none tested in this experiment):

1. **Information bottleneck at the compression stack.** Each compressed position summarizes 16 input tokens through 4 layers of LayerNorm+GELU. At d=384, channel capacity per position is finite. If 16 distinct tokens want to be remembered, the compression stack may not preserve enough.

2. **Position-information loss through compression.** The pos_emb is added pre-compression. After 4 levels of conv mixing, fine-grained position information may be smeared. Token-level prediction might benefit from explicit per-token positional cues that survive the compression.

3. **Channel dimension too narrow.** d=384 is small; widening the channel inside the compression stack (e.g., 384 → 768 → 768 → 768 → 384) might preserve more.

4. **Tiny Shakespeare doesn't reward long-range structure.** TS is short prose with mostly local dependencies (next character, next word). The compression's value is at scales where long-range dependencies matter; on TS it pays its compute cost without buying compensating benefit.

**Inference speedup is real and large.** 9× at ctx 4096; would extend further at longer contexts (the architecture's whole reason for being is the asymptotic regime, and asymptotic is where the win compounds).

## Comparison to prior `saliency_pool/` work

Prior `saliency_pool/compress_4`, `compress_8`, `compress_16` variants on TS used **block-saliency pooling** (a learned scalar weight per token within each block, then weighted sum) to compress. This experiment uses **strided causal conv** (learned linear mixing within fixed windows) for the same task.

Direct comparison would require running both at matched conditions (same model, same training, same eval). Not done here. The prior write-up reported that saliency-pooling 4×/8×/16× "matched baseline" on TS at ctx 256 — possibly less degradation than the 17% seen here, but the comparison isn't apples-to-apples (different model, different ctx, different evaluation task — the prior work used full-sequence LM, this one uses every-16th-token prediction).

Worth flagging that this is a different mechanism from the saliency-pool family; whichever performs better on a fair side-by-side is its own follow-up.

## Verdict per spec decision tree

> **Investigate failure modes:** If perplexity is degraded substantially (more than 10% above baseline) or retrieval fails, the compression is losing too much information. Worth analyzing where the loss is happening: is it the early compression layers losing local detail, the deep compression losing position information, or some other failure mode?

This is where we land. 17% PPL gap, stable across context lengths within training range, large inference speedup. The architecture works enough to be worth diagnosing rather than ruling out.

Possible follow-ups for the diagnostic:
- **Ablate compression depth:** train at 2× (1 conv layer), 4×, 8×, 16× and see how PPL degrades with depth. Linear vs accelerating tells you whether the bottleneck is per-layer mixing or accumulated information loss.
- **Channel expansion in compression:** try wider intermediate channels in the compression stack to test whether channel-dim is binding.
- **Residual connections within the stack:** add identity-preserving shortcuts to retain finer-grained info alongside the mixed signal.
- **Re-add positional information after compression:** explicit pos_emb at compressed positions (in addition to or instead of pre-compression pos_emb).
- **Larger context training:** the compression's value is at long contexts, where compute savings matter and information density per-token can be lower without losing predictive power. Train at ctx 4096 or 8192 on a corpus with real long-range structure (WikiText-103, books) and re-evaluate.

## Files

- `model.py` — CompressedTransformer + CausalCompressLayer
- `train.py` — matched-task training (predict every 16th token)
- `eval_extended.py` — PPL across ctx + inference timing
- `checkpoints/baseline_ctx512.pt`, `compressed_ctx512.pt`
- `results/train_baseline_ctx512.json`, `train_compressed_ctx512.json`
- `results/extended_eval.json`
- `results/run_master.log`, `train_*_log.txt`, `extended_eval_log.txt`

Checkpoints are gitignored (~110MB each) but regenerable from the seed-0 fixed run.
