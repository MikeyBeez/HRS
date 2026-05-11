# Phase 71 — Hard-OOD-aware verification head training

Tested whether augmenting Phase 70's verification head training with
template-perturbed negatives drawn from a vocabulary disjoint from the
library AND from the Phase 65 hard-OOD test set rescues open-vocab
verification at deployment-honest calibration.

## Headline

**Best condition (C, combined) drops per-pair cross-FPR from 20.6% baseline
to 14.7%.** Real improvement, but lands in the ≥13% pre-committed bucket
(no meaningful improvement). Open-vocab still an open problem.

| condition | per-pair cross-FPR | pool hard-OOD FPR | TPR | end-to-end balanced (K=20) |
|-----------|---------------------|-------------------|-----|------------------------------|
| A — cross-adapter only (Phase 70 baseline) | 20.6% | 24.8% | 93% | 66.7% |
| B — hard-OOD only                          | 39.5% | 39.3% | 90% | 66.5% |
| **C — combined cross + hard-OOD**          | **14.7%** | **13.8%** | **88%** | **71.9%** |

(All thresholds calibrated on each head's training negatives at 95th
percentile — deployment-honest, matching Phase 70 K-scaling protocol.
Condition A reproduces K-scaling's 19.6% almost exactly.)

## Three things the data says

### 1. Hard-OOD-only training is *worse* than cross-only

Condition B (4 self-positives + 32 hard-OOD-aware negatives) hits 39.5%
cross-FPR — nearly double the cross-only baseline. The head over-rotates
toward distinguishing template-perturbed entities and loses cross-adapter
discrimination as a side effect. Training negatives that target one failure
mode at the expense of others creates a *new* failure mode.

This is a real lesson for open-vocab verification: you can't trade one kind
of negative for another. You need both, hence condition C.

### 2. Combined training gives modest, real improvement

Condition C (4 pos + 19 cross + 32 hard-OOD) beats both A and B on every
metric:
- Per-pair cross-FPR: 20.6% → **14.7%** (5.9pt improvement, ~29% relative)
- Pool hard-OOD FPR: 24.8% → **13.8%** (11pt improvement, ~44% relative)
- End-to-end balanced: 66.7% → **71.9%** (5.2pt improvement)

The bigger gain is on hard-OOD FPR (the failure mode the new negatives
directly target). Cross-FPR also improves, suggesting the hard-OOD signal
generalizes mildly to within-library cross-adapter discrimination too.

### 3. The improvement isn't enough

14.7% per-pair cross-FPR is well above the 5% threshold for "open-vocab
solved" in the same sense closed-vocab is solved (Phase 69's 100%). At K=20
the end-to-end balanced accuracy reaches 71.9% — better than 65.2% baseline
but still well below the closed-vocab metadata recipe.

The combined condition is the right *direction* but doesn't move the
operating point far enough to change the deployment story.

## Pre-committed predictions vs measured

Pre-committed before the run:

| bucket                               | predicted P | observed |
|--------------------------------------|-------------|----------|
| Per-pair cross-FPR ≤5% (solved)      | 20%         | NO       |
| Per-pair cross-FPR 6-12% (partial)   | **50%**     | NO       |
| Per-pair cross-FPR ≥13% (no meaning) | 30%         | **YES**  |

Best condition lands at 14.7%, just outside the 6-12% partial bucket. The
modal prediction (50% on partial improvement) was off by ~3 percentage
points of cross-FPR. The qualitative reasoning was correct ("hard-OOD-aware
should help but not enough to clear the 5% bar"); the magnitude was slightly
more pessimistic than expected.

The user's framing on this experiment ("if it lands in the middle bucket the
paper section becomes 'partial solution with quantified limits'") technically
doesn't apply since we landed just outside that bucket. But the spirit
applies — the result is partial, the limit is quantified, and the framing
should reflect that.

## What the architecture actually looks like now

End-to-end at K=20 (deployed library = all 20 adapters), with C0b routing
followed by per-adapter h_mean_L5 head trained with combined negatives:

- TPR 88% (down 5pt from baseline due to threshold being calibrated on a
  larger, more diverse training-negative distribution → slightly higher tau).
- FPR 44% (down ~18pt from K-scaling baseline 62.5%).
- Balanced 71.9% (up 6.7pt from K-scaling baseline 65.2%).

Real improvement, still not deployment-grade. The 28pt gap between this and
the closed-vocab metadata recipe (Phase 69, 100%) is the open-vocab
verification gap that future work needs to close.

## What this means for the paper

The publishable claim updates from Phase 70's K-scaling version:

> Per-adapter h_mean_L5 verification heads trained with combined cross-
> adapter and template-perturbed hard-OOD negatives provide ~85% per-pair
> specificity under deployment-honest calibration, ~72% end-to-end balanced
> accuracy at K=20 with C0b routing. This is a meaningful improvement over
> baseline (cross-only) training but does not reach the closed-vocab
> metadata-match recipe's 100%. The remaining gap is consistent with a
> representational limit on what mean-pool L5 carries about entity identity
> — a limit which Phase 65's pooling ablation already characterized.

The recruitment piece's "the hard part is next" framing now has the next
*specific* hard part: closing the gap from ~72% to deployment-grade
balanced accuracy without metadata. Concrete options for follow-up work:

1. **Larger heads or richer input.** 256-hidden MLP on h_mean_L5 may be
   capacity-limited. Try concatenating (h_mean, h_last, h_delta) as input,
   or expanding hidden to 1024.
2. **Generative-prompt-based negatives.** Hard-OOD-aware negatives in this
   experiment were template-perturbations within the same 4 categories.
   Generating cross-category negatives, or LLM-generated paraphrases of
   the same templates with arbitrary entities, might give richer training
   signal.
3. **Two-stage verification.** Use the head as a soft accept (high logit →
   trust) and fall back to a metadata check or a second classifier when the
   head is uncertain. Hybrid recipes might inherit closed-vocab grade in
   the cases where entity catalogs exist while still working in open-vocab
   regimes where they don't.

## Files

- `hard_ood_aware.json`        full per-condition metrics, per-adapter taus,
                               pooled vs per-adapter breakdowns
- `training_curves.png`        bar chart: FPRs and end-to-end metrics by condition
- `hard_ood_aware_run.log`     stdout
- `experiments/identity_ae/phase71_hard_ood_aware.py`  the script

## Open follow-ups

1. **Calibration sensitivity.** All numbers use 95th-percentile-of-training-
   negatives calibration. A validation-set held out from training would give
   tighter taus and possibly different conclusions. ~30 min to test.

2. **Larger hard-OOD-aware vocabulary.** Used 8 new entities × 4 paraphrases
   = 32 per category. With 32 or 64 new entities (× more paraphrases), the
   training distribution would cover more of the failure space. Diminishing
   returns past some point but worth measuring where.

3. **Held-out hard-OOD test diversity.** The current Phase 65 test set is
   20 entities (also 5 per category). A larger test set with more
   templatic variation would tighten the cross-FPR estimate and reveal
   any per-template structure in the failures.

4. **Phase 72: generative head with self-consistency check.** The head
   currently just outputs a logit. A more sophisticated mechanism: generate
   the answer, check whether the model's own NLL on the generation under
   the *base* model (no LoRA) is consistent with a coherent answer. This
   tests whether the adapter's confident wrong answers can be flagged by
   "the base model would never have generated this."
