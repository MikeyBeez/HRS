# Last-token L0 ablation — Outcome C, representational

## Result

| variant   | route | in_top1 | sep    | hard-OOD FP @ 95% recall | 0% FP → in-lib recall |
|-----------|-------|---------|--------|--------------------------|------------------------|
| mean_pool | C0b   | 93%     | +0.020 | 95%                      | 8%                     |
| mean_pool | C1    | 100%    | +0.083 | 95%                      | 40%                    |
| last_1    | C0b   | 5%      | +0.000 | 0%                       | 0%                     |
| last_1    | C1    | 5%      | +0.000 | 67%                      | 0%                     |
| last_5    | C0b   | 73%     | +0.050 | 95%                      | 27%                    |
| **last_5**| **C1**| **93%** |**+0.091**| **62%**                | **22%**                |
| last_10   | C0b   | 93%     | +0.021 | 90%                      | 5%                     |
| last_10   | C1    | 100%    | +0.091 | 92%                      | 43%                    |

(model.eval() is set throughout — Phase 47 left dropout active by oversight,
which made cached Phase 65 numbers mildly stochastic. The mean_pool/C0b row
above reproduces Phase 65's 93% top-1 and +0.931 in-library mean exactly.)

## Verdict against pre-committed outcomes

- **Outcome A (mechanistic, FP ≤ 20%)**: not met. No variant.
- **Outcome B (partial, FP 20–50%)**: not met. Best is 62%.
- **Outcome C (representational, FP > 50%)**: **met. Best is 62%.**

The deployment claim cannot be rescued by a one-line change to engram
extraction. The entity-grain failure is not mechanistic.

## What we learned about each variant

### `last_1` — degenerate
Cosines are essentially 1.0 across all queries because the final token of
every English question is punctuation (`?`, `.`). Same token → same L0
embedding → constant cosine. In-library top-1 collapses to 5% (chance is 5%).
This rules out "the issue is just averaging" as a sufficient explanation —
even the most extreme anti-averaging move (single token) doesn't help, and
in fact destroys the routing signal entirely because the load-bearing tokens
aren't at position −1.

### `last_5` — modest improvement, the only non-degenerate win
Best entity discrimination: hard-OOD FP@95 drops from 95% to 62% under C1.
In-library top-1 is preserved at 93%. Separation gap improves from +0.083 to
+0.091. Direction is right but magnitude is far from a fix. Reading: the
last 5 tokens of the question include some entity tokens (e.g., "the orbital
facility?" — "facility" + entity), and C1's projection finds slightly more
discriminative L5 directions when given that input. But "slightly more" is
26pts of FP improvement, not the 70+ pts we'd need for deployment.

### `last_10` — no improvement
Covers most of the question (median in-library length is 11 tokens, max 16).
Effectively reverts to mean_pool numbers. Confirms the entity-tokens-at-end
hypothesis is at best partial — once you average over more than ~5 tokens,
the template tokens dominate again.

### `mean_pool` — baseline
Reproduces Phase 65 in-library numbers exactly (top-1 93%, mean +0.931 for
C0b; top-1 100%, mean +0.246 for C1). The clean reproduction validates the
ablation methodology — only the variant changed; everything else is fixed.

## What this implies for the paper

The hard-OOD failure is **representational**, not **mechanistic**. The L0
representation does not encode entity identity in a form recoverable by any
positional pooling we tested. In particular:

- Entity tokens carry only weak L0 signal — `last_5` improves separation by
  0.008 (from +0.083 to +0.091) and FP rate by 33pts. That's a real signal,
  but on the scale of "noticeable in a histogram," not "deployment threshold."
- The trained W (C1) helps slightly more than raw L0 (C0b) at the same
  pooling — last_5/C1 gets 62% FP vs last_5/C0b at 95%. This says C1's
  contrastive training does extract a bit of entity discrimination, just
  not enough.
- Mean_pool, last_5, and last_10 all sit between 90–95% FP under C0b. The
  entity-grain failure is not localized to one particular pooling defect.

The cheap fix is dead. The architecturally correct response is option 2 from
the prior writeup: a separate per-adapter verification stage that runs after
routing and decides whether to commit or reject. Routing is template-grain
by design; verification is entity-grain by construction.

The paper's scoping needs to be exactly what was outlined in the previous
turn's user message:

> "Engram routing for non-overlapping libraries" is honest. "General-purpose
> adapter routing for deployed systems" was the claim we were drifting
> toward, and that claim no longer holds.

A two-paragraph footnote describing the entity-grain limit, the failed
mechanistic rescues (mean → last-5 pooling, OOD-aware contrastive on random
WikiText), and the open architectural fix (verification head) is the
honest bibliography entry for this work.

## Files

- `lasttoken.json` — full per-query records and per-condition summaries
- `cosine_distributions_lasttoken.png` — 4 variants × 2 routes histograms
- `lasttoken_run.log` — stdout
- `experiments/identity_ae/phase65_lasttoken.py` — script

## Supersedes

`pooling_ablation.json` and `pooling_ablation_run.log` from the parallel run
of `phase65_pooling_ablation.py` were extracted with `model.drop()` active
(no `model.eval()`), which introduced dropout noise on every key/query
extraction. That run's `mean` baseline showed 77% top-1 instead of the 93%
that this clean run reproduces. The pooling_ablation.json results should be
treated as exploratory; the lasttoken.json results are the comparison point.
