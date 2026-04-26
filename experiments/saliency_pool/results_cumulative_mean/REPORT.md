# Causal cumulative mean replacing attention — Tiny Shakespeare

**Question.** Can the attention sublayer of a transformer block be replaced
by causal cumulative mean of the residual stream — no Q, K, V, or W_O
projections, no multi-head structure, no per-query weighting? Per-position
differentiation comes from the cumulative-mean's growing prefix alone.

**Outcome.** Catastrophic failure — bigger than expected. Cumulative-mean
is **+174% PPL** over the V22 baseline (13.15 vs 4.80, Cohen's d = +134).
And, more pointedly, it is **+64% worse than variant D** of the
saliency_pool experiment (which kept V and W_O around uniform-weight
attention — 8.02 PPL).

The architectural simplification was not viable. The Q-K-V projections
do necessary work that pure cumulative mean cannot replicate — and a
substantial part of that work is being done by the V and W_O projections
specifically, independent of the attention weights.

## Setup

- Same architecture and recipe as saliency_pool Tiny Shakespeare sweep:
  TinyTransformer at d=384, n_heads=6, n_layers=6, d_ff=1536, ctx=256,
  tied embeddings, vocab≈65 chars.
- 2000 steps × batch 32 × ctx 256 on Tiny Shakespeare; AdamW lr=3e-4
  cosine to 10%, warmup 100, weight decay 0.01.
- 4 seeds (0–3) for cumulative_mean; baseline and D reused from the
  original saliency_pool sweep (`experiments/saliency_pool/results/`).
- Total wall clock for cumulative_mean sweep: **174s** for 4 seeds (~43s
  per seed). Total elapsed across all three variants compared here:
  baseline 4×116s = 464s; D 4×54s = 216s; cumulative_mean 4×43s = 174s.

The cumulative_mean attention sublayer:

```python
def forward(self, x):
    cumsum = x.cumsum(dim=1)
    pos = torch.arange(1, T + 1, device=x.device).view(1, T, 1)
    return cumsum / pos
```

That's the entire attention sublayer. Zero parameters. The MLP, layer
norms, and residual connections are unchanged from the baseline.

## Results

| variant | description | n | mean PPL | std | 95% CI | params | step (ms) | wall (s) |
|:-------:|:-----------:|:-:|:--------:|:---:|:------:|:------:|:---------:|:--------:|
| baseline | V22 Bonsignore Q-K attention | 4 | **4.800** | 0.034 | [4.766, 4.833] | 10,761,708 | 53.3 | 116 |
| D | V + W_O kept; uniform-weight attention | 4 | 8.019 | 0.027 | [7.993, 8.045] | 8,992,128 | 24.7 | 54 |
| cumulative_mean | no projections at all | 4 | **13.151** | 0.081 | [13.072, 13.231] | 7,222,656 | 19.3 | 42 |

### Pairwise (Welch's t-test)

| comparison | Δ PPL | Cohen's d | p | 95% CI of Δ |
|:-----------|:-----:|:---------:|:-:|:-----------:|
| cumulative_mean − baseline | +8.352 (+174%) | +133.9 | <1e-6 | [+8.24, +8.46] |
| D − baseline | +3.219 (+67%) | +104.2 | <1e-6 | [+3.17, +3.27] |
| cumulative_mean − D | +5.132 (+64%) | +84.9 | <1e-6 | [+5.03, +5.24] |

### Speed and parameter cost

| variant | params | step time (median, ms) | total wall (4 seeds) |
|:-------:|:------:|:----------------------:|:--------------------:|
| baseline | 10.76M | 53.3 | 116s |
| D | 8.99M | 24.7 | 54s |
| cumulative_mean | 7.22M | 19.3 | 42s |

cumulative_mean is **64% smaller in attention-sublayer params** (zero vs
attention's 4d² × n_layers ≈ 3.5M) and **~2.8× faster per step** than
baseline. None of which compensates for the +174% PPL hit.

No NaN losses, no training instability. Loss curves descend smoothly
just to a much higher floor. Failure mode is "model converges, but
converges to a much worse minimum," not "model fails to train."

## Reading the result against the spec's outcome map

The spec laid out three possible outcomes:

1. *Cumulative_mean within 10% of baseline (surprising success)*: no.
   Gap is +174%. Far outside the success threshold.
2. *Cumulative_mean 10–30% worse (partial success)*: no. Gap is
   far worse.
3. *Cumulative_mean more than 50% worse, similar to variant D
   (catastrophic failure)*: **yes — and worse than variant D's +67%.**

The expected catastrophic-failure outcome obtains, with the additional
finding that removing V and W_O makes things substantially worse than
just keeping them with uniform attention weights.

## What the gap to variant D specifically tells us

Variant D and cumulative_mean both have **the same attention pattern**
(uniform `1/(prefix_len)` weights over the causal prefix). The only
difference is the projections:

- **Variant D**: `W_V(x) → uniform-weighted average → W_O(·)`
- **cumulative_mean**: `x → uniform-weighted average → ·`

The +5.13 PPL gap (variant D at 8.02, cumulative_mean at 13.15) is
attributable entirely to the V and W_O projections doing substantive
work even when the attention weights themselves carry zero information.

This is the more interesting finding than "cumulative_mean fails" —
which was already expected. Specifically:

- **W_V** is per-token: takes each position's `d`-dim residual to a
  `d`-dim representation in a different basis, with per-head splitting.
  Without W_V, the cumulative mean operates directly on the residual
  stream that's also being read by the MLP and the next block — so the
  attention output and the unmodified residual share a representation.
  The model can't use one part of the dimensionality for "what's in
  context on average" and another for "what to predict next."
- **W_O** is post-attention: mixes the H per-head outputs back into a
  single `d`-dim residual contribution. Without W_O, the head structure
  doesn't matter (cumulative_mean has H=1 effectively), but more
  importantly the model loses a learnable transformation that decides
  *how* the attention output should be combined back into the residual
  stream.

Together V and W_O let the model use attention's output as a *separable
channel* in the residual stream — projected from raw context (V),
mixed (W_O), gated by skip connections to whatever else is already
there. cumulative_mean sees no such separation. The attention sublayer
emits raw averaged residuals, which add directly to the next residual
that the MLP also operates on, and the model can't cleanly factor the
"context" from the "current state."

The +5 PPL D→cumulative_mean gap is the empirical magnitude of *this
factoring* — independent of any per-query attention scoring.

## Comparison to original saliency_pool's variant-D result

In the original saliency_pool report:

> **Variant D failed catastrophically** at +67% PPL, Cohen's d = 104.
> Attention with no scoring is essentially worthless; *some* form of
> position-aware weighting is necessary.

The cumulative_mean result extends this: the V and W_O projections are
*also* doing necessary work, beyond the attention scoring. The
projections alone, even paired with completely uninformative attention
weights, are worth ~5 PPL of perplexity at this scale. That is
substantial — the projection-only contribution is **larger** than the
gap between V22 baseline and any of the saliency-MLP variants in the
original sweep (which were 0.07–0.10 PPL above baseline).

## Why the failure is not training instability

Worth being explicit: this is not a "the architecture didn't train"
failure. cumulative_mean trained smoothly across all 4 seeds. The seed
variance was 0.081 PPL — tighter than baseline's 0.034 (proportional
to the mean), suggesting the optimization landscape is convex enough
to converge consistently. Loss curves descend monotonically. Final
training losses are stable.

The architecture *converges*. It just converges to a much higher
floor, because the architecture cannot represent the same conditional
distributions the V22 baseline can. This is a representational ceiling,
not an optimization problem.

## What this experiment does and doesn't say

**Says clearly:** removing all of attention's projections — Q, K, V,
W_O — and replacing the sublayer with cumulative mean produces a model
that is substantially worse than even the worst sensible variant from
the original saliency_pool sweep. The V and W_O projections do real
work, even with uniform-weight attention; cumulative mean strips that
work out and pays for it.

**Doesn't say:**

- **Whether longer training would close the gap.** Cosine schedule
  finished at step 2000 with both variants well-converged within their
  own LR schedule; extending training would test plateau height, not
  more. (Echoing the WT-103 finding from the saliency_pool follow-up:
  conclusions about ordering need scale-up to confirm, though here the
  gap is so large that scale-up is unnecessary.)
- **Whether a partial restoration (cumulative_mean + V_only, or
  cumulative_mean + W_O_only) recovers something.** Not tested. Could
  be a follow-up if anyone is curious about which projection matters
  more, but the policy decision (don't replace attention this way) is
  the same regardless.
- **Whether per-position differentiation could be added back via some
  other mechanism cheap enough to keep the architectural simplicity
  but rich enough to recover the gap.** Open question; this experiment
  does not address it.
- **Whether scale changes the picture.** At BPE/WT-103 scale the gap
  could narrow or widen. Spec scoped the question to Shakespeare
  intentionally; deferred.

## Files

- `cumulative_mean_seed{0..3}.json` — per-run records (training loss,
  eval points, diagnostics, parameter counts, timing)
- `cumulative_mean_seed{0..3}.pt` — saved model checkpoints (per spec)
  for post-hoc analysis of residual stream representations
- `sweep_summary.json` — sweep metadata
- `report.json` — this report's data in machine-readable form
- Code: `experiments/saliency_pool/{config.py, attention.py:CumulativeMeanAttention,
  run_sweep_cumulative_mean.py, analyze_cumulative_mean.py}`

## Budget

Spec estimate: 15–20 minutes. **Used 174 seconds** (~3 min) for the
4-seed sweep. Faster than expected because cumulative_mean has zero
attention-sublayer FLOPs.
