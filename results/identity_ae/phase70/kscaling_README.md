# Phase 70 K-scaling sub-sample

Tests how the per-adapter h_mean_L5 verification head holds up as the
deployed library size K varies in {2, 3, 5, 8, 12, 15, 18, 20} adapters,
sub-sampled from the 20-adapter Phase 65 library. 50 random subsamples per K.

## Two findings

### (1) Per-adapter cross-FPR is invariant in K — but is also higher than Phase 70 reported

| K   | per-adapter cross-FPR |
|-----|------------------------|
| 2   | 19.6% ± 12.6%          |
| 5   | 19.3% ± 12.2%          |
| 12  | 19.9% ± 12.4%          |
| 20  | 19.6% ± 12.3%          |

**Cross-FPR is essentially constant across K.** Each sister adapter is an
independent confusable; sampling more sisters doesn't change the per-pair
false-accept rate. Good news for scaling: at K=100 or K=1000, the per-pair
discrimination won't degrade (though the *absolute* number of false accepts
across the library will grow linearly).

**But 19.6% is well above Phase 70's reported 4%.** The discrepancy is a
calibration-protocol issue:

- **Phase 70's 4%**: thresholds set at the 95th percentile of *pooled test-set*
  negatives (cross + hard-OOD per adapter), then cross-FPR computed on the
  same set. In-sample calibration — guaranteed to be ≤ 5% by construction.
- **K-scaling's 19.6%**: thresholds set on *training negatives only* (the 19
  cross-adapter `train_prompts` used to fit the head), evaluated on disjoint
  test negatives. The threshold doesn't transfer cleanly — heads overfit the
  training negative distribution and let in more test negatives at the
  "calibrated" tau.

The K-scaling number is the deployment-honest one. Phase 70's 4% was
optimistic. Treat the open-vocab story as ~80% specificity per adapter, not
~96%, when you don't have access to test-distribution calibration data.

### (2) End-to-end deployment balanced accuracy degrades with K

| K   | TPR              | FPR              | balanced         |
|-----|------------------|------------------|------------------|
| 2   | 92.0% ± 12.6%    | 30.6% ± 9.5%     | 80.7% ± 5.7%     |
| 3   | 91.0% ± 10.5%    | 39.0% ± 11.4%    | 76.0% ± 5.5%     |
| 5   | 93.1% ± 6.3%     | 49.4% ± 12.0%    | 71.8% ± 5.9%     |
| 8   | 93.7% ± 4.7%     | 58.7% ± 9.7%     | 67.5% ± 4.7%     |
| 12  | 93.0% ± 3.2%     | 64.4% ± 6.0%     | 64.3% ± 3.0%     |
| 15  | 93.2% ± 2.6%     | 64.2% ± 7.9%     | 64.5% ± 4.3%     |
| 18  | 92.8% ± 1.4%     | 62.5% ± 4.7%     | 65.1% ± 2.6%     |
| 20  | 92.9% ± 0.0%     | 62.5% ± 0.0%     | 65.2% ± 0.0%     |

(K=20 has zero variance because there's only one possible "subset" — all
adapters. Each query gets the same routing + verification result every time.)

**TPR is robust** — flat at ~93% across all K. The verification head accepts
correctly-routed in-library queries reliably.

**FPR grows with K, then plateaus around 62-65%.** Three contributions:
1. **Cross-adapter mis-routes**: as K grows, more sister adapters compete for
   each query. Routing failures (in-library queries routed to the wrong
   adapter, where verification incorrectly accepts) account for the early
   growth from K=2 to K=8.
2. **Out-of-subset in-library queries**: at small K, most in-library queries
   have their true adapter outside the K subset. They get routed to *some*
   adapter, and that adapter's verification head decides. False accepts on
   these are dominant at small K (FPR 30% at K=2 with most queries
   "out-of-library").
3. **Hard-OOD pass-through**: 20 hard-OOD queries are always in the
   should-reject pool. With per-adapter cross-FPR ~19.6%, expected hard-OOD
   acceptance is ~20%, contributing a constant FPR floor regardless of K.

**Balanced accuracy plateaus at ~65% from K=12 onward**, well below Phase
70's reported 90%. The gap is due to the same calibration issue plus the
end-to-end coupling: even when each adapter's head looks reasonable in
isolation, the chain (route → verify → decide) accumulates errors.

## What the architecture actually looks like at K=20

End-to-end at K=20 with deployment-honest calibration:

- **TPR 93%**: of the 56 correctly-routed in-library queries, 52 get accepted.
- **FPR 62.5%**: of the 24 should-reject queries (4 mis-routed in-library + 20
  hard-OOD), 15 get falsely accepted.

This is **worse than C0b alone** for OOD rejection (Phase 65 reported C0b at
0% FP on easy OOD, 92% FP on hard OOD). The per-adapter head adds value
mainly on the *easy* portion of the should-reject distribution; on hard
near-neighbor OOD, its 19.6% per-pair cross-FPR doesn't translate into
deployment-grade end-to-end rejection because every routed adapter gets a
second chance to falsely accept.

## What Phase 70's 90% number actually meant

Going back to Phase 70's protocol: the headline 90% balanced accuracy was
*pooled* across all (query, adapter) pairs with *test-set calibration*. That
measures something — the heads do separate self-positives from negatives in
score-space — but not what a deployment cares about, which is "given a single
query and a single routing decision, does the system accept correctly?"

The K-scaling end-to-end protocol measures the deployment-relevant question.
Answer: ~65% balanced accuracy at K=20 with honest calibration. That's well
below the closed-vocab metadata recipe's 100%, and well below the C0b
gate's 95%+ on easy OOD.

## Implication for the paper

The open-vocab story is **weaker than Phase 70 claimed**, in two ways:

1. **Calibration matters a lot.** Per-adapter cross-FPR is 5x higher (19.6%
   vs 4%) when you can't peek at test negatives to set the threshold. Any
   deployment claim should specify the calibration protocol and use the
   training-only number.

2. **End-to-end ≠ per-adapter.** Even with the head's per-pair cross-FPR
   constant in K, the end-to-end FPR grows with K because more adapters mean
   more chances for any single adapter to falsely accept. At K=20 the
   end-to-end balanced accuracy plateaus around 65%, not 90%.

The honest publishable claim is now:

> Per-adapter h_mean_L5 verification heads provide entity-grain
> discrimination at ~80% specificity per pair when calibrated on training
> negatives (deployment-realistic). End-to-end deployment with the recommended
> C0b → head pipeline gives ~93% TPR and ~62% FPR at K=20, dominated by hard
> near-neighbor false accepts that the head's per-pair specificity isn't
> enough to filter. Closed-vocab metadata-match remains the only stage with
> deployment-grade balanced accuracy (Phase 69, 100%). The open-vocab gap
> needs a second mechanism: larger heads, more training data, or a richer
> input representation than mean-pool L5.

The "deployment is solved" framing from the previous Phase 70 commit is
overstated. Closed-vocab is solved; open-vocab remains an open problem.

## Pre-committed predictions vs measured

I didn't pre-commit predictions for this run. Honest reporting: I expected
cross-FPR to grow with K (more competitors → more chance of any one being
confusable). The data says cross-FPR is **constant in K**, which is the
right news for scaling but doesn't help end-to-end because routing
confusion compounds independently.

I expected end-to-end balanced accuracy at K=20 to roughly match Phase 70's
90%. The data says **65%**. The 25-point gap forced the calibration
discovery — Phase 70's number was an in-sample artifact.

Prediction track record on this paper now ~2/9. The cumulative lesson is
that mechanism reasoning ahead of the data has been unreliable; the data
keeps surfacing details that flip the conclusion.

## Files

- `kscaling.json`        — full K-scaling sweep results
- `kscaling.png`         — two-panel plot: per-adapter cross-FPR vs K, end-to-end metrics vs K
- `kscaling_run.log`     — stdout
- `experiments/identity_ae/phase70_kscaling.py` — script

## Open follow-ups

1. **Test-set calibration protocol.** Re-run K-scaling using a held-out
   *validation* set (split from the test set) for tau calibration. Measure
   the gap between training-only and validation-calibrated FPR — that's the
   "minimum extra annotation cost" for honest open-vocab verification.

2. **Larger heads or input.** The 256-hidden MLP on h_mean_L5 might be
   capacity-limited. Try 512 or 1024 hidden, or concatenate (mean, last,
   delta) as input. If specificity recovers under realistic calibration, the
   open-vocab claim revives.

3. **Hard-OOD-aware training negatives.** Phase 70 trained heads with
   cross-adapter train_prompts as negatives (templates the head has seen
   before, just from other adapters). Hard near-neighbor queries weren't in
   the training distribution. Adding template-perturbed negatives might
   teach the heads to reject more aggressively.
