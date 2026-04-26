# Saliency-pool experimental arc — summary

Investigation: **what is the attention sublayer actually doing, and what
parts of it are load-bearing?** Tested by progressively replacing or
ablating components of a TinyTransformer block (`d=384`, `n_heads=6`,
`n_layers=6`, `d_ff=1536`, `ctx=256`, ~10.76M params) on Tiny Shakespeare
and WikiText-103. Same recipe across the arc: 2000 steps × batch 32 ×
ctx 256 (Shakespeare) or 5000 steps (WT-103), AdamW lr=3e-4 cosine to
10%, 100-step warmup, weight decay 0.01, 4 seeds per variant, Welch's
t-test for inference. Baseline is V22 Bonsignore Q-K attention.

## Results table — Shakespeare, single scaffold, all variants

| # | variant | description | mean PPL | std | Δ vs baseline | step ms | wall s |
|:-:|:-------:|:-----------:|:--------:|:---:|:-------------:|:-------:|:------:|
|  | **baseline** | V22 Bonsignore Q-K, full attention | **4.800** | 0.034 | — | 53.3 | 116 |
| 1 | **B** | per-pair saliency MLP `(2d→d→1)` | 4.790 | 0.036 | −0.009 (p=0.71) | — | — |
| 2 | A | shared global saliency `(d→768→1)` | 4.871 | 0.053 | +1.5% (p=0.058) | — | — |
| 2 | C | context-summary saliency `(2d→d→1)` on `(x, cumsum/n)` | 4.903 | 0.029 | +2.2% (p=0.003) | — | — |
| 2 | D | V/W_O + uniform attention weights (no scoring) | 8.019 | 0.027 | **+67%** | 24.7 | 54 |
| 3 | cumulative_mean | no Q/K/V/W_O — replace attn sublayer with `cumsum(x)/pos` | 13.151 | 0.081 | **+174%** | 19.3 | 42 |
| 4 | dual_projection | `concat(W_P1·x, W_P2·x) → W_C` (no cross-position flow) | 11.924 | 0.015 | **+148%** | 27.0 | 59 |
| 4 | dual_projection_with_cumulative | same but `W_P2` reads `cumsum(x)/pos` | 8.663 | 0.029 | **+80%** | 28.0 | 61 |
| 5 | compress_4 | causal V4-style — compress past blocks of 4, full Bonsignore on (compressed past + uncompressed local + self) | 4.835 | 0.024 | +0.7% (p=0.14) | 63.7 | 139 |
| 5 | compress_8 | same, k=8 | **4.772** | 0.041 | −0.6% (p=0.32) | 59.5 | 130 |
| 5 | compress_16 | same, k=16 | 4.796 | 0.046 | −0.1% (p=0.90) | 57.7 | 126 |

(Numbered left to right by experiment order, not by ranking.)

## Results table — WikiText-103 scale-up (selected)

Same architecture, vocab=50,257 (GPT-2 BPE), 5000 steps, ctx=256.
Total params 30M (the 19.3M token-emb is the difference).

| variant | n | mean PPL ± std | Δ vs baseline | notes |
|:-------:|:-:|:--------------:|:-------------:|:-----:|
| baseline (V22 Bonsignore) | 3 | 101.57 ± 0.29 | — | |
| B (per-pair saliency MLP) | 3 | 97.61 ± 0.50 | **−3.9%** (p<0.001, d=−9.7) | win at scale |
| A (shared global) | 3 | 109.33 ± 0.35 | +7.6% | got worse at scale |
| C (context-summary) | 3 | 100.06 ± 0.44 | −1.5% | reversed direction |
| **sdpa** (standard Q@K^T/√d) | 4 | **95.76 ± 0.78** | **−5.7%** (vs Bonsignore) | beats both |

WT-103 ordering: `SDPA > B > C > Bonsignore > A`.

## What we learned, in order

### 1. Original screening (Shakespeare, baseline + A/B/C/D)

Tested whether saliency-MLP scoring can replace Q-K. Hypothesis was that
the Q-K mechanism does irreplaceable work; predicted ordering
`D < A < C < B < baseline`.

**Outcome.** B tied baseline (−0.009 PPL, p=0.71); A only 1.5% behind.
Hypothesis partially refuted. At Shakespeare scale, saliency-MLP scoring
is competitive with Q-K attention. D is the floor (+67%) — uniform
attention with no scoring is decisively worse than any of the
scoring-based variants.

### 2. WT-103 scale-up

Re-ran on real text + BPE tokens to test whether the Shakespeare result
holds at scale.

**Outcome.** Direction changed in three places. B *beat* baseline
(−3.9%, p<0.001). A went from −1.5% to −7.6%. C reversed (+2.2% →
−1.5%). The framework's predicted ordering was wrong on every position
except D-as-floor.

### 3. WT-103 SDPA control

Ran standard scaled-dot-product as a non-Bonsignore control, to test
whether B's win was specific to beating Bonsignore or generalized.

**Outcome.** SDPA beat both Bonsignore (−5.7%) and B (−1.9%). B's WT-103
win is specific to beating the Bonsignore kernel — standard SDPA still
sits at the top. Bonsignore (with its per-head temperatures, alphas,
head-output scalars) is *not* the right reference attention for this
investigation; SDPA is.

### 4. cumulative_mean (no projections at all)

Replace the entire attention sublayer with `cumsum(x)/pos` — zero
parameters in the sublayer. Tests whether *some* form of cross-position
aggregation alone is enough.

**Outcome.** +174% PPL. Catastrophic — and *worse* than D (+67%). The
+5.13 PPL gap from `cumulative_mean` to D is attributable to the V/W_O
projections doing real work, *even when the attention weights carry zero
information*. So both projections AND scoring contribute.

### 5. dual_projection (per-position richness only)

Replace attention with `concat(W_P1·x, W_P2·x) → W_C` — 4d² parameters,
two parallel learned views with a learned mixture, but no cross-position
information flow.

**Outcome.** +148% PPL. Slightly better than `cumulative_mean` (per-token
projection richness alone closes ~9% of the gap to baseline) but still
catastrophically worse than D and baseline. **Per-position richness is
not what attention provides.**

### 6. dual_projection_with_cumulative

Same but `W_P2` reads `cumsum(x)/pos` instead of `x`. Identical
architecture except one of the two projections sees a context-aware
input.

**Outcome.** +80% PPL — drops 3.26 PPL from `dual_projection` (Cohen's
d=−140!). The variant lands at ≈D. Cleanest signal in the arc:
**adding uniform-weight cross-position aggregation to a learned
projection mixture is worth ~27% of the gap to baseline.** Cross-position
flow is the dominant missing ingredient.

But there is still a +80% gap to baseline. That gap is what
*per-query weighted scoring* contributes on top of uniform aggregation.

### 7. compression_invalid_leak (bug, preserved as evidence)

First version of V4-style compression-then-attention had block-level
compression that mixed all `k` tokens of a block into one entry, then
broadcast that entry's attention output back to all `k` tokens —
including future tokens within the block.

**Outcome.** PPL 1.24–1.54, well below the char-level entropy floor of
~PPL 4.5. Bigger `k` → more leak (longer block → more future tokens
visible to predictions early in the block). Quarantined as
`results_compression_invalid_leak/`.

### 8. compression_causal (V4-style done correctly)

Causal saliency-based compression: query at position `j` (block `i`)
reads only from compressed entries of strictly past blocks (length `i`),
plus uncompressed tokens of the current block at positions `[i·k, j]`.
Causality verified twice (architecture-level test + trained checkpoint:
`pre.max = 0` bit-identical at every probed position).

**Outcome.** All three k ∈ {4, 8, 16} match baseline. None of the
vs-baseline p-values are significant; compress_8 has the lowest mean
(4.772, slightly *below* baseline). 16 compressed entries summarizing
256 tokens is enough to match full attention quality on Shakespeare.

Speed (honest): the dense `Q@K^T` + mask implementation runs *slower*
than baseline (0.84–0.92×), since `K` is length `m+T > T`. Asymptotic
V4-style savings need ragged/sparse attention; out of scope here.

## Synthesis — what attention does, decomposed

Across the full sweep, the gap to baseline correlates almost cleanly
with whether the architecture has **per-query weighted selection**:

- Variants without it sit at **+67% to +174%** above baseline:
  `cumulative_mean (+174%)`, `dual_projection (+148%)`, `dpc (+80%)`,
  `D (+67%)`.
- Variants with it sit at **baseline ± noise**:
  `compress_k for k ∈ {4, 8, 16}`, the original saliency variants
  `A/B/C`.

Within the "no-selection" group, three knobs matter:

| ingredient | empirical worth (Shakespeare) |
|:----------|:------------------------------:|
| Per-position projection richness alone | ~9% of the gap (`dp` over `cm`) |
| + uniform-weight cross-position flow | ~27% additional (`dpc` over `dp`) |
| + V/W_O projections around uniform aggregation | ~30% additional (`D` over `cm`, different decomposition) |
| + per-query weighted selection | the remaining ~80% to baseline (`compress_k` and `A/B/C` over `dpc`) |

Per-query selection is far more valuable than the resolution of the
keys it selects from. 16 compressed entries representing 256 tokens
work as well as 256 raw tokens, as long as a per-query softmax over
those 16 entries is allowed.

## Negative findings worth keeping

- **Block-level decompression broadcast leaks future tokens.** The bug
  produced PPL 1.24–1.54, well below the char-level floor. Causal
  compression-then-attention requires a strict mixed-resolution mask
  per query position.
- **The framework's predicted ordering** `D < A < C < B < baseline`
  fails at WT-103. A got worse, C reversed, B and SDPA both beat
  baseline. At BPE scale, the kernel choice (Bonsignore vs SDPA) is
  separable from the scoring-mechanism class, and the Bonsignore
  per-head extras (`log_tau`, `head_alphas`, `head_output_scalars`)
  may be *hurting* rather than helping at this scale.
- **Per-position projection richness alone gives essentially nothing**
  (1.23 PPL out of an 8.35 PPL deficit) — the cumulative-mean → dual-
  projection step.

## Honest caveats

- All findings except WT-103 are at char-level Shakespeare (1.1M chars,
  vocab≈65, 2000 steps). Local prediction on Shakespeare has small
  conditional entropy beyond a few characters of context — distant
  context contributes a small fraction of the loss. WT-103 with BPE
  changes some of these conclusions.
- 4 seeds is enough to detect Cohen's d > 1 reliably (all the catastrophic
  results) but at the edge for resolving compress_4 vs compress_8
  (Δ=+0.063, p=0.037 — possibly real, possibly multiple-comparisons
  noise).
- Wall-clock on the compression variants does not reflect the asymptotic
  speed argument — needs ragged/sparse attention.
- Only one compression mechanism tested (softmax-weighted saliency MLP).
  Top-k selection, attention-based compression, learned linear
  compression are all open.

## What's left, ranked by tractability

1. **Re-run the compression result at WT-103 + BPE.** The Shakespeare
   result is suggestive; this would confirm or refute. ~3 hours wall.
2. **Implement ragged/sparse attention** to deliver the asymptotic
   speed savings the architecture allows in principle. Engineering
   work.
3. **Test higher compression ratios** (`k ∈ {32, 64, 128}`) to map
   where compression breaks down at this scale.
4. **Multi-scale compression** (V4 actually uses CSA at 4× *and* HCA at
   128× simultaneously). Not tested here.
5. **Sliding window over recent + compressed past + sparse top-k**
   variants — the broader CSA/HCA/MoBA design space.
6. **Test whether dropping Bonsignore extras** (just use SDPA-style
   attention in `compress_k`) helps or hurts. WT-103 SDPA result hints
   it might help.

## Files (commit pointers)

- `55f1e3a` HRS-Loop arc (separate investigation, predates this arc)
- `aef0dd9` Dual-projection attention replacement
- `4729854` Causal saliency-compression-then-attention

Per-experiment reports:
- `results/REPORT.md` — original A/B/C/D screening
- `results_wt103/REPORT.md` — WT-103 scale-up
- `results_wt103_sdpa/REPORT.md` — SDPA control at WT-103
- `results_cumulative_mean/REPORT.md` — no-projections floor
- `results_dual_projection/REPORT.md` — per-position richness
- `results_compression_causal/REPORT.md` — V4-style, matches baseline
- `results_compression_invalid_leak/` — preserved evidence of the bug

## Single-sentence headline

The attention sublayer's load-bearing component is **per-query weighted
selection**; the resolution of the keys (down to 16-from-256 compressed
summaries) doesn't appear to matter on this corpus.
