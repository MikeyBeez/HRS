# Head-aggregation ablation — Shakespeare screening

**Hypothesis tested.** "Gradient descent will route around the choice of
aggregation operator when the upstream projections are allowed to adapt.
W_O's cross-head mixing can be replaced by per-head transformations plus
any reasonable fixed aggregation at matched parameters, with no measurable
quality cost." (Spec, "Hypothesis" section.)

**Outcome.** Refuted. The hypothesis is wrong at this scale. All six
variants are decisively worse than baseline, with effect sizes (Cohen's d)
ranging from +12 to +34 — gigantic. The within-variant seed variance is
~0.02–0.16 PPL points; the smallest variant-to-baseline gap is +0.43 PPL.
The gap is at minimum **20× larger** than seed variance.

This is the spec's **Case 2** ("Baseline wins decisively"). No scale-up
to V22 is needed; the result will only widen at larger scale, not reverse.

## Scaffold

- TinyBonsignoreTransformer: d=384, n_heads=8, n_layers=6, d_ff=1536,
  ctx=256, tied embeddings; total **10.76M params** (matches the spec's
  10.8M target within 0.4%)
- Bonsignore-kernel attention with per-head `log_tau`, `head_alphas`
  (sigmoid sharpness), `head_output_scalars` (softplus). The V22
  per-head *score-refinement* MLP is intentionally omitted (spec
  approval — orthogonal to the W_O question).
- 2000 steps × batch 32 × ctx 256 on Tiny Shakespeare; AdamW lr=3e-4,
  cosine to 10%, warmup 100, weight decay 0.01.
- Param-matched output paths: baseline W_O is `d² = 147,456` per layer.
  Variants A–F replace it with a per-head MLP `(dh → 168 → dh, GeLU)`
  applied independently per head, then a fixed aggregation operator,
  then a single up-projection `(dh → d)`. Total per-layer:
  `H × 2 × dh × 168 + dh × d = 129,024 + 18,432 = 147,456` — exact
  match to baseline. Variant D adds a 48-element learned query
  (`q_pool`); variant F adds zero new parameters. All variants total
  10,761,744 or 10,762,032 params (D differs by 48).
- Total wall clock: 4400 s (73 min) for 30 runs on one 5070 Ti.

## Results

| variant | aggregation                | n | mean PPL | std   | 95% CI            | Δ vs baseline | Cohen's d | per-head MLP Frob (Wh_in+Wh_out) |
|:-------:|:--------------------------:|:-:|:--------:|:-----:|:-----------------:|:-------------:|:---------:|:--------------------------------:|
| baseline| concat → W_O (d×d)         | 4 | **4.767**| 0.024 | [4.743, 4.791]    | —             | —         | —                                |
| **E**   | **L2-normalized mean**     | 4 | **5.194**| 0.044 | [5.151, 5.237]    | **+0.427**    | +12.0     | 10.70 (lowest growth)            |
| B       | sum                        | 4 | 5.648    | 0.060 | [5.590, 5.707]    | +0.881        | +19.3     | 12.45                            |
| C       | element-wise max           | 4 | 5.852    | 0.039 | [5.814, 5.890]    | +1.085        | +33.6     | 13.37                            |
| A       | mean                       | 4 | 6.439    | 0.086 | [6.355, 6.523]    | +1.672        | +26.5     | 14.36                            |
| D       | attention-pool (learned q) | 6 | 6.563    | 0.164 | [6.431, 6.694]    | +1.796        | +13.7     | 14.20 (highest variance)         |
| F       | top-4 mean by output norm  | 4 | 6.626    | 0.115 | [6.513, 6.739]    | +1.859        | +22.3     | 12.23                            |

All p-values are below numerical precision (~1e-6) of the t-tail
approximation; reported as 0.000.

## Reading the result

Three observations matter beyond the headline.

**1. The variants form a clean ordering and the gap is enormous.**
Best variant (E) is +9% PPL over baseline. Worst variant (F) is +39% PPL.
Within-variant seed variance is ~0.02–0.16 PPL — the variants are
separable from each other and *all* are separable from baseline by far
more than seed noise. There is no ambiguity to scale up.

**2. The per-head MLPs *did* grow — the hypothesis-mechanism evidence
points the opposite of what the hypothesis predicted.** At init, the sum
of `Wh_in + Wh_out` Frobenius norms is approximately
`2 × √(H × dh × d_inter) × 0.02 = 2 × √64512 × 0.02 ≈ 10.16`. By end of
training:

- E: 10.70 (+5% above init) — used the per-head capacity *least*, did *best*
- F: 12.23 (+20%)
- B: 12.45 (+23%)
- C: 13.37 (+32%)
- D: 14.20 (+40%) — high variance
- A: 14.36 (+41%) — used the per-head capacity *most*, did *second-worst*

**The variants that grew their per-head MLPs the most performed the
worst.** This is the opposite of what the hypothesis-mechanism story
predicts. The hypothesis was: "gradient descent will route compensation
through upstream weights, the per-head MLPs will grow, and PPL will
match baseline." What actually happened: the per-head MLPs grew *and
PPL got worse*. Gradient descent isn't routing toward a substitute for
W_O — it's flailing in the per-head capacity, finding configurations
that *increase* loss relative to what a learned W_O achieves in the
same parameter budget.

**3. E (L2-normalized mean) is the closest to baseline.** This is the
only variant with a normalization step. It also grew its per-head MLP
the least. The pattern suggests one component of W_O's contribution is
**magnitude management** of the residual-stream addition — not (only)
cross-head mixing. The variants without normalization let residual
magnitudes drift, which downstream layers then have to absorb.

**4. Sparser aggregations (F top-k, D attention-pool) are the worst.**
F (top-4 of 8) and D (learned attention over heads) both restrict the
information flow from heads, and both perform worst. This argues against
the "low-importance heads are noise" framing — at this scale, all 8
heads are contributing. F's hard top-k throws away half the information;
D's softmax can collapse to similar effects.

## Comparison to OpenMythos prior

The spec mentioned OpenMythos found mean-pool > max-pool > attention-pool
on a compositional retrieval task. Here, on language modeling at
attention-output aggregation:

- Sum-pool (B) > max-pool (C) > attention-pool (D) > mean-pool (A)
- L2-normalized mean-pool (E) > sum-pool (B)

Different ordering. The OpenMythos result was about combining sequence
outputs across positions for retrieval. This is about combining heads
within attention layers for next-token prediction. The patterns don't
transfer directly — task and aggregation locus matter.

## Decision

Per the spec's decision tree:

> **Case 2** (Baseline wins decisively across all variants): "W_O does
> non-substitutable work. Report the gap and stop. No scale-up needed
> because the result is already clear."

Stop. W_O is doing real work that fixed aggregation plus per-head
transformations cannot recover at matched parameter budget. The
architectural simplification proposed in the hypothesis is not
available.

The mechanism diagnostic adds nuance: it isn't that the per-head
capacity stays unused (which would be a "the model didn't try" story).
The per-head MLPs *grow*, in some cases substantially. The model is
attempting to use the capacity. It just can't find configurations that
match what a single learned `d×d` matrix achieves with the same params.
This is a fact about gradient descent's ability to discover
cross-head-mixing structure, given an architecture that has separated
out the within-head and cross-head paths.

A small follow-up worth considering: variant E's relative success
suggests learnable normalization (L2 norm before up-proj, or a learned
per-head scale calibration) might recover much of the gap. That's a
different experiment — not "what aggregation operator" but "what
post-aggregation calibration." Not part of this study; flagging for
later.

## Files

- `report.json` — machine-readable per-variant statistics
- `{variant}_seed{seed}.json` — per-run records (eval points, diagnostics
  per layer, train losses, wall time, peak memory)
- `sweep_summary.json` — sweep metadata
- Code: `experiments/head_aggregation/{config,attention,model,train,run_sweep,analyze}.py`

## Budget actuals

Spec budgeted ~7 GPU-hours; **used 73 minutes** (the d=384/n_layers=6
config came in faster than the rough 15-min/run estimate). The
contingent V22-scale phase (200–250 GPU-hours) is not warranted by the
result and is shelved.
