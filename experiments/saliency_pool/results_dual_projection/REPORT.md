# Dual projection with learned mixture replacing attention — Tiny Shakespeare

**Question.** Can the attention sublayer be replaced by two parallel learned
projections of the residual stream — concatenated and compressed back to `d`
by a learned mixture matrix — with no Q@K^T, no scoring, and no per-pair
computation? And does adding cumulative-mean cross-position aggregation
(P2 on `cumsum(x)/pos` instead of `x`) recover what's lost?

**Outcome.** Per-position dual projection alone (`dual_projection`) is **+148%
PPL** over the V22 baseline (11.92 vs 4.80) — better than `cumulative_mean`
(+9%) but **worse than `variant_D`** (+49% over D's 8.02 PPL). Adding
cumulative-mean to the second projection (`dual_projection_with_cumulative`)
cuts ~3.26 PPL off the per-position-only result and lands at 8.66 PPL, **+80%
over baseline** and almost exactly at variant_D. The +cumulative variant
beats the no-cumulative variant by Cohen's d = **140** — a huge effect — but
neither comes within striking distance of baseline.

The headline finding is clean and consistent with the cumulative_mean
result: **per-position feature richness is not what attention provides.
Cross-position information flow is.** Even uniform-weight cumulative
aggregation, when made available to a learned projection mixture, recovers
most of the gap that no-projection cumulative_mean opens — but query-
conditioned scoring is still doing ~3.86 PPL of work neither variant can
replicate.

## Setup

- TinyTransformer, `d=384`, `n_heads=6`, `n_layers=6`, `d_ff=1536`,
  `ctx=256`, dropout 0, tied embeddings, char-level Tiny Shakespeare
  (vocab≈65). Same recipe as prior saliency_pool sweeps.
- 2000 steps × batch 32; AdamW lr=3e-4 cosine to 10%, 100-step warmup,
  weight decay 0.01.
- 4 seeds (0–3) per variant. `baseline`, `D`, and `cumulative_mean` results
  reused from prior sweeps (`results/`, `results_cumulative_mean/`).
- All linear weights initialized with `N(0, 0.02)` (the model's default
  `_init`), matching the baseline attention projections.

The two new sublayers replace the attention block entirely. The transformer
block is otherwise unchanged:

```
x = x + dual_projection(layer_norm_1(x))
x = x + mlp(layer_norm_2(x))
```

`dual_projection`:
```python
def forward(self, x):                      # x : (B, T, d)
    p1 = W_P1(x)
    p2 = W_P2(x)
    return W_C(torch.cat([p1, p2], dim=-1))
```

`dual_projection_with_cumulative`:
```python
def forward(self, x):
    cumsum = x.cumsum(dim=1)
    pos = torch.arange(1, T+1, device=x.device).view(1, T, 1)
    x_cum = cumsum / pos
    p1 = W_P1(x)            # local features (per position)
    p2 = W_P2(x_cum)        # contextual features (causal prefix mean)
    return W_C(torch.cat([p1, p2], dim=-1))
```

Parameter count for both sublayers: `d² + d² + 2d² = 4d² = 589,824` per
block. Same as standard attention's `Q + K + V + W_O = 4d²`. Total model
params: **10,761,600** — within ~100 of the V22 baseline's 10,761,708.

## Results

| variant | n | mean PPL | std | 95% CI | params | step (ms) | wall (s) |
|:-------:|:-:|:--------:|:---:|:------:|:------:|:---------:|:--------:|
| baseline (V22 Bonsignore) | 4 | **4.800** | 0.034 | [4.766, 4.833] | 10,761,708 | 53.3 | 116 |
| D (V/W_O + uniform attn) | 4 | 8.019 | 0.027 | [7.993, 8.045] | 8,992,128 | 24.7 | 54 |
| cumulative_mean (no proj) | 4 | 13.151 | 0.081 | [13.072, 13.231] | 7,222,656 | 19.3 | 42 |
| **dual_projection** | 4 | **11.924** | 0.015 | [11.910, 11.938] | 10,761,600 | 27.0 | 59 |
| **dual_projection_with_cumulative** | 4 | **8.663** | 0.029 | [8.634, 8.692] | 10,761,600 | 28.0 | 61 |

### Pairwise (Welch's t-test)

| comparison | Δ PPL | % | Cohen's d | p | 95% CI of Δ |
|:-----------|:-----:|:-:|:---------:|:-:|:-----------:|
| dp − baseline | +7.124 | +148% | +269.2 | 7.9e-11 | [+7.08, +7.17] |
| dpc − baseline | +3.863 | +80% | +120.4 | 1.7e-12 | [+3.81, +3.92] |
| dp − D | +3.905 | +49% | +181.1 | 3.0e-11 | [+3.87, +3.94] |
| dpc − D | +0.644 | +8% | +22.8 | 2.9e-08 | [+0.60, +0.69] |
| dp − cumulative_mean | −1.227 | −9% | −21.0 | 2.2e-05 | [−1.33, −1.13] |
| dpc − cumulative_mean | −4.488 | −34% | −73.5 | 4.2e-08 | [−4.59, −4.38] |
| **dpc − dp** | **−3.261** | **−27%** | **−140.3** | 2.8e-10 | [−3.30, −3.22] |

(`dp` = `dual_projection`, `dpc` = `dual_projection_with_cumulative`.)

All p-values are far below conventional thresholds. With 4 seeds and the
tight intra-variant variance observed (std 0.015–0.081), every pairwise
difference is statistically distinguishable from zero — the effect-size
language (Cohen's d) is more informative than the p-values.

### Speed and parameter cost

| variant | params | step (ms, median) | wall (4 seeds, s) |
|:-------:|:------:|:-----------------:|:-----------------:|
| baseline | 10.76M | 53.3 | 116 |
| D | 8.99M | 24.7 | 54 |
| cumulative_mean | 7.22M | 19.3 | 42 |
| dual_projection | 10.76M | 27.0 | 59 |
| dual_projection_with_cumulative | 10.76M | 28.0 | 61 |

Both dual-projection variants run **~2× faster per step** than the V22
baseline (27–28 ms vs 53.3 ms) at matched parameter count. The cumulative
aggregation in `dpc` is ~1 ms slower than pure `dp` — a tiny linear-cost
addition.

No NaN/Inf losses, no instability. All 8 runs converged smoothly.

## Reading the result against the spec's outcome map

The spec laid out four possible outcomes:

1. *Within 10% of baseline (surprising success):* **No.** The closest
   variant (`dpc`) is +80% over baseline. Far outside the threshold.
2. *Both variants 30%+ worse than baseline but better than cumulative_mean
   (meaningful but not viable):* **Yes — exactly this.** Both sit between
   cumulative_mean and baseline, with `dpc` near `D` and `dp` between
   `cumulative_mean` and `D`.
3. *Both catastrophically worse, ≈ cumulative_mean (mixture-only insufficient):*
   No. Mixture richness alone (`dp`) does close ~9% of the cumulative_mean
   gap; mixture + cumulative aggregation (`dpc`) closes ~34%.
4. *`dpc` beats `dp` by a large margin (cross-position flow essential):*
   **Yes — dramatically.** Δ = −3.26 PPL, Cohen's d = −140. Cross-position
   aggregation does substantial work even with no learned scoring.

Outcomes 2 and 4 obtain together. The architectural simplification is not
viable as a baseline replacement at this scale, but the result decomposes
neatly into "what attention's projections give you" and "what attention's
cross-position flow gives you".

## What the dual projection provides relative to single-projection cumulative_mean

`dual_projection` (no cumulative) beats `cumulative_mean` by 1.23 PPL
(Δ = −9%, d = −21). The mechanism is straightforward: per-position dual
projection lets each token transform its residual into two distinct views
(`W_P1·x`, `W_P2·x`) and learn a mixture (`W_C`) over them, giving the
model a richer per-token feature transform than the identity-with-residual
that `cumulative_mean` delivers. This buys ~9% of the gap to baseline.

But `cumulative_mean` already has zero parameters in its sublayer, so the
9% reflects what 4d² of additional per-token capacity buys when there is
no cross-position information flow at all. Compared to the gap that
*adding cross-position flow* opens (`dpc − dp` = −3.26 PPL, ~27% gap
reduction), it is a small effect.

**Read:** per-position projection richness is real but small. It is not
what attention is doing.

## What adding cumulative-mean cross-position flow provides

`dpc` − `dp` is the cleanest decomposition this experiment offers. The
two variants differ in exactly one thing: whether `P2` operates on `x`
or on the causal prefix-mean of `x`. Everything else — `W_P1`, `W_C`,
parameter count, training recipe, initialization — is identical.

The −3.26 PPL difference (Cohen's d = −140) is therefore directly
attributable to **the model gaining access, at each position, to a
learned projection of the running prefix average**. Not to a richer
projection. Not to more parameters. Just to one of the two projections
seeing a context-aware input rather than a per-position one.

This finding is the more important one. It says:

- Whatever attention is doing in the V22 baseline, a substantial fraction
  of it can be approximated by **uniform-weight prefix aggregation
  combined with a learned readout** — at linear cost and with no Q@K^T.
  `dpc` reaches `D`'s PPL with a different mechanism (no per-head, no
  attention-weights matrix, no V projection).
- The remaining gap from `dpc` to baseline (+3.86 PPL, +80%) is what
  query-conditioned scoring is contributing on top of cross-position
  aggregation. That is not negligible — it is roughly the same order of
  magnitude as the gap that `dpc` already closes — but the amount of
  *implementation complexity* it requires (Q, K, softmax, T² weights)
  is large compared to a cumsum.

## What the gap from `dpc` to variant_D specifically tells us

`dpc` and `D` are mechanistically very different:

- **`D`**: `W_V(x)` → uniform-prefix-mean across positions → `W_O(·)`
  (parameters: V + W_O = 2d²; cross-position flow via uniform attn weights)
- **`dpc`**: concat(`W_P1(x)`, `W_P2(prefix_mean(x))`) → `W_C(·)`
  (parameters: 4d²; cross-position flow via cumsum)

`dpc` has 2× the projection parameters of `D` and is +0.64 PPL worse
(Δ = +8%, d = +22.8). The 95% CI on the difference [+0.60, +0.69]
excludes zero, so the difference is real, but it is small relative to
the +3.9 PPL gap from `dp` to `D` and tiny relative to the gap from
either to baseline.

**Read:** the cross-position aggregation `D` and `dpc` use is roughly
equivalent in expressive power. `D` projects-then-averages; `dpc`
averages-then-projects-and-mixes-with-a-local-projection. Both deliver
about the same PPL. The extra parameters in `dpc` are not buying
anything `D` doesn't already have. This is consistent with the read
above: the bottleneck at this scale is not feature richness, it is the
*kind* of cross-position aggregation, and uniform-weight aggregation —
in either form — is much weaker than per-query weighted aggregation.

## Why this is not a training failure

All 8 dual-projection runs converged smoothly. Seed variance is tight
(std 0.015 for `dp`, 0.029 for `dpc`) — proportionally tighter than the
baseline's 0.034. No instabilities, no NaN losses. Loss curves descend
monotonically. Final training losses are stable.

Both architectures *converge*; they converge to higher floors because
the architectures cannot represent the same conditional distributions
the V22 baseline can. This is a representational ceiling, not an
optimization problem — same conclusion as the cumulative_mean
experiment.

## What this experiment does and doesn't say

**Says clearly:**

- A second per-position projection with learned mixture, with no
  cross-position information flow, fails to approach baseline (+148%) —
  worse than even the uniform-weight-attention floor `D`.
- Adding causal cumulative-mean cross-position flow, with no scoring,
  closes ~27% of the gap left by the per-position-only variant (Cohen's
  d = 140) and lands the architecture at ≈ `D`'s PPL.
- Cross-position information flow is the dominant missing ingredient,
  not feature transform richness.
- The remaining gap to baseline (+80% / +3.86 PPL) is what query-
  conditioned scoring contributes that uniform-weight aggregation
  cannot.

**Doesn't say:**

- **Whether longer training would close the gap.** Recipe matched
  baseline; both variants are well-converged within their own LR
  schedule. Same caveat as cumulative_mean: extending training would
  test plateau height, not learning rate.
- **Whether multiple cumulative aggregations (e.g., decayed cumsums at
  different time scales) would close more of the gap.** Open question.
  This experiment only tested one prefix-mean.
- **Whether 3+ projections would help.** Not tested. Could be a follow-
  up. The `dp` vs `cumulative_mean` gap (1.23 PPL) is so much smaller
  than the `dpc` vs `dp` gap (3.26 PPL) that the marginal return on
  additional per-position projections is likely small.
- **Whether scale changes the picture.** At BPE/WT-103 scale conclusions
  could shift. Spec scoped the question to Shakespeare; deferred.
- **Whether `dpc` minus the W_C output mixture (i.e., concatenate-and-
  return without compression) recovers, exceeds, or matches.** Not tested;
  outside the scope of this experiment.

## Comparison to the cumulative_mean follow-up question

The cumulative_mean report observed:

> Variant D and cumulative_mean both have the same attention pattern
> (uniform 1/(prefix_len) weights). The only difference is the
> projections. The +5.13 PPL gap is attributable entirely to V and W_O
> doing substantive work even when the attention weights themselves
> carry zero information.

This experiment partially confirms and partially refines that claim.
`dpc` (cumulative + dual projections, no V-then-pool ordering, with
explicit dual-view mixture) reaches 8.66 PPL — within 0.64 of `D`'s
8.02. So the V/W_O ordering specifically is not what mattered — what
mattered was *some* learned per-position transform combined with
*some* form of prefix-aware aggregation. Either ordering (`pool ∘ V` as
in `D`, or `mix(W_P1·x, W_P2·prefix_mean(x))` as in `dpc`) gets you
there.

The +5.13 PPL gap from `cumulative_mean` to `D` reported earlier is
re-decomposed by this experiment into:

- ~1.23 PPL from per-position projection richness alone (`dp` over `cm`)
- ~3.26 PPL from adding cross-position aggregation to a projection
  mixture (`dpc` over `dp`)
- (small remainder from architectural specifics distinguishing `dpc`
  from `D`'s exact V-then-pool form)

The dominant single ingredient is cross-position aggregation, not
projection richness.

## Files

- `dual_projection_seed{0..3}.json` — per-run records
- `dual_projection_with_cumulative_seed{0..3}.json` — per-run records
- `dual_projection_seed{0..3}.pt` — saved checkpoints
- `dual_projection_with_cumulative_seed{0..3}.pt` — saved checkpoints
- `sweep_summary.json` — sweep metadata
- `report.json` — machine-readable summary
- Code: `experiments/saliency_pool/{config.py, attention.py:DualProjectionAttention,
  attention.py:DualProjectionCumulativeAttention, run_sweep_dual_projection.py,
  analyze_dual_projection.py}`

## Budget

Spec estimate: 15–25 minutes. **Used 491 seconds (~8 min)** for the
8-run sweep, ~60s per run. Faster than expected because dual-projection
sublayers have no quadratic computation.
