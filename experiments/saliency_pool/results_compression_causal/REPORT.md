# Causal saliency-compression-then-attention — Tiny Shakespeare

**Question.** Can a causal V4-style architecture — compress past blocks of
`k` tokens into one saliency-weighted entry each, then run full Bonsignore
attention against (compressed past + uncompressed local + self) — match the
quality of standard full attention over the uncompressed sequence?

**Outcome.** **Yes — at all three compression ratios tested (4×, 8×, 16×).**
All three variants are statistically indistinguishable from baseline (all
p > 0.05 vs baseline at n=4 seeds). Compress_8 even has a slightly *lower*
mean PPL than baseline (4.772 vs 4.800), though the difference is within
seed noise.

This is the "best case" outcome the spec called out: distant context can be
aggressively compressed without degrading next-token-prediction quality on
char-level Shakespeare. **Mikey's noise-removal hypothesis is consistent
with this result** — compression preserves the signal in distant context
and may even smooth away seed-specific noise.

**A correctness note up front.** The first version of this experiment had
a causality leak (block-level compression that included future tokens in
the same block in the compressed representation, broadcast back to all
tokens in the block). That version produced PPL 1.24–1.54 on char-level
Shakespeare, well below the entropy floor — a clear giveaway. This revised
implementation is strictly causal: query at position `j` reads only from
compressed entries of strictly past blocks plus uncompressed tokens of its
own block at positions `[i·k, j]`. Causality verified at the architecture
level (random-init forward, bit-identical outputs at positions `< j` when
input at `j` is changed) **and** on a trained `compress_4` seed-0
checkpoint at `ctx=256`. See "Causality verification" below.

## Setup

- TinyTransformer, `d=384`, `n_heads=6`, `n_layers=6`, `d_ff=1536`,
  `ctx=256`, dropout 0, tied embeddings, char-level Tiny Shakespeare
  (vocab≈65). Same recipe as the rest of the saliency_pool sweeps.
- 2000 steps × batch 32; AdamW lr=3e-4 cosine to 10%, 100-step warmup,
  weight decay 0.01.
- 4 seeds (0–3) per variant × 3 variants = 12 runs.
- All linear weights init `N(0, 0.02)`. Bonsignore extras (`log_tau`,
  `head_alphas`, `head_output_scalars`) initialized as in the V22
  baseline.

The `attention` sublayer per block:

```
def forward(self, x):              # x : (B, T, d)
    comp = compress_block(x)        # (B, m=T/k, d): saliency-softmax pool
    qkv_x = qkv(x)                  # Q from x
    qkv_c = qkv(comp)               # K, V on compressed
    K = cat([Kc, Kx], dim=1)        # (B, m+T, d)
    V = cat([Vc, Vx], dim=1)
    scores = bonsignore_kernel(Q, K)
    scores.masked_fill_(~causal_mixed_mask, -inf)
    return out_proj(softmax(scores) @ V)
```

`causal_mixed_mask` per query position `j` (in block `i = j//k`):
- compressed entries `[0, i)` are valid keys (length `i`)
- uncompressed tokens at positions `[i·k, j]` in the original sequence
  are valid keys (length `j%k + 1`, including self)
- everything else masked to `-inf`

So query `j` has `j//k + j%k + 1` effective keys. For `T=256` and `k=4`,
that's between 1 (at `j=0`) and 67 (at `j=255`).

`compress_block`: saliency MLP `W_s ∈ ℝ^{d×1}` produces a scalar per token,
softmax over the within-block axis, weighted sum to one `d`-dim vector per
block. `d` parameters per layer.

Q/K/V projections, `out_proj`, per-head temperatures, head_alphas, and
head_output_scalars are shared between compressed and uncompressed inputs
(the same `qkv` Linear is applied to both).

## Results

### Quality

| variant | n | mean PPL | std | 95% CI | params |
|:-------:|:-:|:--------:|:---:|:------:|:------:|
| baseline (V22 Bonsignore) | 4 | **4.800** | 0.034 | [4.766, 4.833] | 10,761,708 |
| **compress_4**  (m=64) | 4 | 4.835 | 0.024 | [4.811, 4.859] | 10,764,012 |
| **compress_8**  (m=32) | 4 | 4.772 | 0.041 | [4.732, 4.812] | 10,764,012 |
| **compress_16** (m=16) | 4 | 4.796 | 0.046 | [4.751, 4.842] | 10,764,012 |
| D (V/W_O + uniform attn) | 4 | 8.019 | 0.027 | [7.993, 8.045] | 8,992,128 |
| dual_projection_with_cumulative | 4 | 8.663 | 0.029 | [8.634, 8.692] | 10,761,600 |
| dual_projection | 4 | 11.924 | 0.015 | [11.910, 11.938] | 10,761,600 |
| cumulative_mean (no proj) | 4 | 13.151 | 0.081 | [13.072, 13.231] | 7,222,656 |

Pairwise (Welch's t):

| comparison | Δ PPL | Cohen's d | p | 95% CI of Δ |
|:-----------|:-----:|:---------:|:-:|:-----------:|
| compress_4  − baseline | +0.035 | +1.17 | 0.136 | [−0.02, +0.09] |
| compress_8  − baseline | −0.028 | −0.74 | 0.317 | [−0.09, +0.04] |
| compress_16 − baseline | −0.004 | −0.09 | 0.904 | [−0.07, +0.07] |
| compress_4  − dpc      | −3.828 | −141.4 | 8.1e-13 | [−3.88, −3.78] |
| compress_4  − D        | −3.184 | −124.1 | 9.8e-13 | [−3.23, −3.14] |
| compress_4  − compress_8 | +0.063 | +1.87 | 0.037 | [+0.00, +0.12] |
| compress_8  − compress_16 | −0.024 | −0.56 | 0.441 | [−0.10, +0.05] |

**Reading.** The three compression ratios are all within seed noise of
baseline. None of `vs baseline` p-values reach the 0.05 threshold (and
`compress_16` is essentially p=0.9 — there is no detectable difference
at all). Compress_8 has the lowest mean (4.772, slightly below baseline)
but the gap is within one std of either variant. The `compress_4 vs
compress_8` Δ=+0.063 has p=0.037 — *if* this is real and not
multiple-comparisons noise, it suggests `k=4` is mildly worse than
`k=8`, possibly because at `k=4` the model is doing more attention work
(more compressed entries, longer effective key sequence) without buying
matching capacity. But the effect is small; with 4 seeds I would not
overinterpret it.

All three compression ratios are dramatically better than every other
attention-replacement variant from this sweep (D, cumulative_mean,
dual_projection, dpc): Cohen's d > 100 in every comparison, all
p < 1e-10.

### Speed (the user explicitly asked about timing)

| variant | step (ms, median) | wall (s, mean) | speedup vs baseline |
|:-------:|:-----------------:|:--------------:|:-------------------:|
| baseline | 53.3 | 116 | 1.00× |
| **compress_4** | 63.7 | 139 | **0.84×** (slower) |
| **compress_8** | 59.5 | 130 | **0.89×** (slower) |
| **compress_16** | 57.7 | 126 | **0.92×** (slower) |
| D | 24.7 | 54 | 2.16× |
| cumulative_mean | 19.3 | 42 | 2.77× |
| dual_projection | 27.0 | 59 | 1.97× |
| dpc | 28.0 | 61 | 1.90× |

**The compression variants are slower than baseline, not faster.** This
is an implementation choice, not a property of the architecture.

The implementation does dense `Q @ K^T` against a unified key tensor of
length `m + T = T(1 + 1/k)`, then masks the invalid positions to
`-inf`. So the actual attention compute is `T × (m + T) = T² × (1 +
1/k)` ops per layer — *more* than baseline's `T²`. Specifically for
`T=256`:
- baseline: 65,536 attn ops/layer
- compress_4 (m=64): 256 × 320 = 81,920 (+25%)
- compress_8 (m=32): 256 × 288 = 73,728 (+13%)
- compress_16 (m=16): 256 × 272 = 69,632 (+6%)

Plus the qkv projection runs twice (once on `x`, once on `comp`), adding
`3d² · m = 3d² · T/k` extra projection ops.

True V4-style wall-clock savings require ragged/sparse attention that
gathers the per-query key set (effective length `j//k + j%k + 1`) and
runs attention only on those entries. That gives `O(T² / k + T·k)`
attention compute vs baseline's `O(T²)` — a real reduction at large
`k`. At `T=256, k=4`: ~17,408 ops vs 65,536 (3.8×). At `k=16`: ~7,168
(9.1×).

So the *quality* result above is what we wanted. The *speed* result is
"at this small `T` the dense+mask implementation is slightly slower
than baseline but the architecture would scale better to longer `T`
than baseline if implemented with sparse attention." For `T=256` and
this `k` range the asymptotic argument doesn't dominate yet.

Memory: compression variants use ~2.7–3.0 GB peak vs baseline's likely
~1.2 GB (the `(B, H, T, m+T)` attention scores tensor and the
`(T, m+T)` mask both inflate). At `B=32, H=6, T=256, m+T=320`: scores
tensor = 32·6·256·320·4 bytes = 60 MB per layer. Across 6 layers
during backward, this is the dominant allocation.

### PPL vs compression ratio

```
   PPL
   5.0 │
       │   *baseline*  4.800
       │
       │   compress_4 4.835 ─┐
       │   compress_16 4.796 ┼─ all within noise of baseline
       │   compress_8 4.772 ─┘
   4.7 │
       └──────────────────────────────────
        baseline   k=4    k=8    k=16
```

There is no clean monotone trend across `k=4, 8, 16`. The means cluster
within ~0.07 PPL of each other, and within ~0.04 PPL of baseline. With
4 seeds, this is at the edge of what we can resolve. Calling out the
shape of the curve at this resolution would be over-reading.

## Causality verification

The first version of this experiment had a leak that produced PPL
1.24–1.54 (well below the char-level entropy floor of ~1.5 nats/char ≈
PPL 4.5). The mechanism: block-level compression mixed all `k` tokens
of a block into one entry, then broadcast that entry's attention output
back to all `k` tokens of the same block. So position `4i+j` could see
information from positions `4i+j+1, 4i+j+2, ..., 4i+k-1` through the
compression. Bigger `k` → more leak → lower PPL, exactly what was
observed (`compress_16 = 1.23 < compress_4 = 1.54`).

The revised implementation is strictly causal at the token level.
Verified two ways:

**1. Architecture-level causality test on a randomly-initialized
model** (small sanity check at `T=64`, all three variants):

For `j ∈ {0, 1, k-1, k, k+1, 2k, T/2, T-1}`, modify `idx[0, j]` and
recompute. Check `(model(idx) - model(idx_modified)).abs()` at every
output position.

> All three variants: `pre.max = 0.000e+00` (bit-identical) at every
> probed `j`; `at[j].max > 0`; `post.max > 0` for `j < T-1`; `post.max
> = 0` for `j = T-1`. **PASS.**

**2. Trained-model causality test** on `compress_4` seed-0 at full
`T=256`:

```
 pos j   pre.max    at[j].max    post.max   ok
     0   0.00e+00   4.40e+00   3.12e+00   True
     1   0.00e+00   5.24e+00   4.40e+00   True
     3   0.00e+00   5.87e+00   3.36e+00   True
     4   0.00e+00   5.04e+00   3.77e+00   True
     5   0.00e+00   7.93e+00   2.96e+00   True
     7   0.00e+00   3.05e+00   4.07e+00   True
     8   0.00e+00   3.92e+00   1.51e+00   True
    16   0.00e+00   7.69e+00   3.54e+00   True
    32   0.00e+00   1.12e+01   5.27e+00   True
    64   0.00e+00   1.70e+00   2.28e+00   True
   128   0.00e+00   8.22e+00   4.38e+00   True
   200   0.00e+00   5.63e+00   1.77e+00   True
   255   0.00e+00   6.61e+00   0.00e+00   True
```

`pre.max = 0` exactly at every probed `j`. The causality of the
architecture is independent of the weights, but verifying on a trained
checkpoint rules out implementation issues that might only manifest
under particular weight values (e.g., NaN-propagation, floating-point
precision artifacts that incidentally vanish at init).

`cfg.dropout = 0` rules out dropout-induced cross-position correlation.

The 4.80 numbers are real, not a leak.

## Reading against the spec's outcome map

The spec laid out four outcomes:

1. *compress_4 within 5% of baseline + substantial wall-clock saving:
   architecture viable, scale up.* ✓ on quality (within 1%); ✗ on
   speed (this implementation is slower; sparse attention would change
   the speed picture but is out of scope).
2. *compress_4 ties or beats baseline → noise-removal hypothesis gets
   support.* The compress_8 mean is below baseline mean (4.772 vs
   4.800); compress_16 is identical (4.796). Not strong evidence on
   its own (within seed noise), but consistent with the hypothesis.
3. *compress_4 between dpc and baseline.* Compress_4 is at baseline,
   far above dpc. **Beat the higher bound.**
4. *Compression hurts even at 4×.* Did not happen.

Outcome 1 (quality) and outcome 2 obtain. The architectural
simplification works on Shakespeare-scale data without degrading
quality, even at `k=16` (16 compressed entries summarizing 256 tokens).

## Reading against prior variants

The compression result extends the cumulative_mean and
dual_projection findings cleanly:

| architecture | mean PPL | cross-position flow |
|:------------:|:--------:|:-------------------:|
| cumulative_mean | 13.151 | uniform prefix mean, no projection, no scoring |
| dual_projection | 11.924 | local features, no cross-position flow |
| dpc | 8.663 | uniform-prefix + per-position projection |
| variant_D | 8.019 | uniform-attn-weight (V@avg + W_O) |
| compress_k (any k) | ~4.80 | per-query Bonsignore attn against {compressed past + uncompressed local} |
| baseline | 4.800 | per-query Bonsignore attn against full uncompressed sequence |

The previous experiments showed:
- Per-position richness alone gets you 9% closer to baseline (`dp` over
  `cm`).
- Adding uniform cross-position flow gets you 27% closer (`dpc` over
  `dp`).
- But uniform cross-position flow alone — even with 4d² of projection
  capacity — leaves a +80% gap to baseline.

The compression result fills in the missing piece: **Bonsignore
per-query scoring works just as well over a coarse compressed
representation as over the full uncompressed sequence**. The query/key
matching mechanism is what closed the residual gap from `dpc` to
baseline, *not* the resolution of the keys it's matching against. At
this scale, 16 compressed entries summarizing 256 tokens is enough.

Said differently: across all the variants tested, the gap to baseline
correlates with whether the architecture has *per-query weighted
selection* (Bonsignore Q-K kernel + softmax). Variants without it —
`D`, `dp`, `dpc`, `cm` — sit at +67% to +174% over baseline. Variants
with it (`compress_k` for any `k ∈ {4, 8, 16}`) sit at baseline. The
resolution at which selection happens is much less important than
whether selection is happening at all.

## Caveats and what this experiment does not say

1. **Char-level Shakespeare is small and easy.** The result tells us
   that on a 1.1M-character corpus with vocab≈65, the model's job is
   not bottlenecked by the resolution of past context. At larger
   scales, longer documents, BPE tokenization, and richer dependency
   structure, the result could change — possibly in favor of
   compression (if distant context really is mostly noise), possibly
   against (if longer-range dependencies exist that compression
   smears). Worth re-running at WT-103.
2. **Wall-clock did not improve.** The dense + mask implementation
   adds 6–25% attention compute. A real test of the speed argument
   needs a sparse/ragged attention implementation that gathers
   per-query keys.
3. **`k=16` is the largest tested.** The spec set the upper bound. At
   `k=16`, compressed-sequence length is 16 — so for any token in a
   16-token window the model has at most 1+(15+1)=16 compressed
   entries + 16 within-block tokens to attend over. We do not know
   how much further `k` can go before quality cracks.
4. **Saliency-MLP capacity is minimal.** `W_s` is 384 params per layer
   (one shared scalar projection per token, softmax over within-block).
   That is enough at this scale; a richer compression mechanism
   (e.g., per-token-position learned weights, attention-based
   compression, top-k selection) might or might not help.
5. **Same-block compressed entry not exposed.** By design, query at
   position `j` can NOT see the compressed entry of its own block —
   only of strictly past blocks. The current block is read as
   uncompressed tokens up to `j`. So the architecture actually has
   *more* token-precise local context than baseline does for nearby
   positions, but coarser distant context. Both effects could be in
   play; we cannot disentangle them at this resolution.

## What would change the conclusion

This result tested causality (re-verified on a trained checkpoint).
Things that could make it look better than it is at this scale:

- **Position embedding leak via long-range context.** The model has
  positional info for both compressed and uncompressed inputs (added
  at the embedding stage). The compressed entries inherit the
  positional info of the tokens they aggregate (though averaged).
  We did not test whether the model could be using compressed
  position embeddings as a signal, but it would be a "valid signal
  the model can use", not a leak.
- **Char-level tokenization makes prediction local.** Most predictions
  in Shakespeare's char stream are 1–4 chars ahead and deterministic
  given the local trigram. Distant context contributes a small
  fraction of the conditional entropy. This favors compression on
  this corpus.
- **2000 steps may be sufficient for Shakespeare.** All variants
  converged. Longer training could shift the comparison.

Things that would weaken or strengthen it:

- **Re-run at WT-103, BPE tokens, longer ctx** would test whether
  this generalizes.
- **Run at higher `k` (32, 64, 128)** would map where compression
  breaks down at this scale.
- **Implement sparse/ragged attention** would deliver the wall-clock
  saving the architecture allows in principle.

## Files

- `compress_{k}_seed{0..3}.json` for `k ∈ {4, 8, 16}` — per-run records
- `compress_4_seed{0..3}.pt` — saved checkpoints (per spec, for
  post-hoc analysis of the learned saliency vector)
- `sweep_summary.json` — sweep metadata
- `report.json` — machine-readable summary
- Code: `experiments/saliency_pool/{config.py, attention.py:CausalCompressionAttention,
  attention.py:CompressBlock, run_sweep_compression.py, analyze_compression.py}`
- `../results_compression_invalid_leak/` — preserved evidence of the
  earlier non-causal version that produced PPL 1.24–1.54

## Budget

Spec estimate: 10–20 minutes. **Used 1599 seconds (~27 min)** for the
12-run sweep, ~133s per run. Slower than expected because the dense +
mask attention compute is larger than baseline's, not smaller (see
"Speed" section).
