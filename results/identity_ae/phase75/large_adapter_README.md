# Phase 75 — Does base training lift retrieval on a large-content adapter?

**Verdict: NO_SUCCESS, but with a meta-finding about adapter calibration.**
The rank-128 adapter on a ~335-token synthetic narrative passage couldn't
be calibrated to the spec's 30-60% baseline retrieval window — every
non-trivial pretraining duration produced 70-85% baseline (adapter holds
content easily), and only 25 steps dropped to 15% (adapter holds nothing).
The fallback at 80% baseline gave 16/20 → 16/20 retrieval over 400 steps
of Phase 72c-style training, with WikiText PPL degrading +34% as a side
effect.

## Headline

| metric                                  | pristine | final         | within target?                |
|-----------------------------------------|----------|----------------|--------------------------------|
| Adapter-attached retrieval (out of 20)  | 16       | **16**        | (no change; net 1 flip ↔ 1)   |
| Adapter-attached mean answer CE         | 1.512    | 1.257          | -0.26 nats (modest improvement) |
| FAIL→PASS flips                         | n/a      | **1**         | needed ≥3 for marginal success |
| PASS→FAIL flips                         | n/a      | **1**         | net change zero                |
| Detached mean CE on OOD passage         | 5.660    | 7.927 (+2.27) | drift in NON-absorption direction |
| WikiText-2 val PPL                      | 20.492   | 27.559 (+34%) | **outside ±10% threshold**     |

## What the calibration revealed

The spec required calibrating adapter pretraining steps to land baseline
retrieval in the 30-60% range. None of the five attempted step counts
produced that:

| n_steps | baseline retrieval | mean answer CE |
|---------|---------------------|-----------------|
| 25      | 15% (3/20)          | 3.588           |
| 50      | 70% (14/20)         | 1.895           |
| 100     | 85% (17/20)         | 1.514           |
| 200     | 85% (17/20)         | 1.238           |
| 300     | 85% (17/20)         | 1.346           |

The transition from "too few steps" (25, 15%) to "comfortably holds
content" (50, 70%) is sharper than the spec's calibration assumed. **At
rank 128 on a 335-token passage, there is no useful intermediate regime**
— the adapter capacity dramatically exceeds the content size, so it
either memorizes essentially nothing (very few steps) or essentially the
whole passage (50+ steps).

Quoting the spec on this scenario:

> "If the calibration can't land the baseline in a reasonable range
> (say, after 3-4 tries), that itself is data — it would mean the
> adapter at this rank either holds the content easily or doesn't hold
> it at all, with no useful intermediate regime."

This is what happened. So Phase 75's main experiment runs with a fallback
adapter (100 steps, 80% baseline) where there's only 4 queries of
headroom for the base training to demonstrate lift on. The architectural
test is correspondingly weakened.

## What the main training showed (within the headroom limit)

400 steps of Phase 72c-style training:

```
step    retrieval  mean_CE  det_mean  ppl
  1     16/20      1.512    5.654    28.03
 25     16/20      1.548    6.613    27.06
100     15/20      1.556    7.357    25.80
200     15/20      1.429    7.516    25.29
300     15/20      1.319    7.899    25.54
400     16/20      1.257    7.927    25.66
```

- **Retrieval count**: oscillates between 15 and 16. Net change zero.
- **Mean answer CE**: drops from 1.51 to 1.26 (16% improvement). The
  base is producing slightly more confident predictions on the queries
  that already passed, but isn't flipping new ones.
- **Detached mean CE**: drifts from 5.66 to 7.93 (+2.27 nats), in the
  AWAY direction (forgetting), not absorption. Same pattern as Phase 73.
- **WikiText PPL**: from 20.49 to 27.56 (+34%). Significantly outside
  the ±10% target window. Worse degradation than Phase 72c (+18%) and
  Phase 73 (no PPL change reported under that protocol).

## Pre-committed predictions vs measured

| outcome              | pre-committed P | measured |
|----------------------|------------------|----------|
| Strong success (≥18/20, ≥8 flips, PPL ±10%) | 20% | NO |
| Moderate success (14-17, ≥5 flips)          | 30% | NO (1 flip) |
| Marginal success (10-13, ≥3 flips)          | 25% | NO       |
| **No success**                                | **25%** | **YES** |

The 25% no-success prior was right. The qualitative reasoning ("Phase 72c's
small-content lift might not transfer to large content") was essentially
correct, though the specific failure mechanism (calibration miss, not
mechanism failure on a properly-calibrated adapter) is different from
what was modeled.

Track record now ~7/20 strict pre-commits across this arc.

## What this result means architecturally

Two related claims update:

**1. The Phase 72c lift mechanism does not lift retrieval when there's no headroom.**
This is consistent with how the mechanism works mechanistically — if
the (base + adapter) system already retrieves the content, there's no
gradient signal pushing the base to organize itself differently. The
hinge regularizer fires only when detached CE drops below baseline, and
the attached loss is already low. The base's training is dominated by
the WikiText batches, which is what produces the +34% WikiText
degradation (a continued-fine-tuning side effect that overshoots PPL
optimum on this small dataset).

**2. The "capacity-stress regime" the spec wanted to test isn't reachable
with rank 128 on this passage size.**
A genuinely capacity-limited adapter on this content would need either
much longer content (impossible at the base's 512-token max_seq_len) or
a much smaller rank (likely rank 2-4 to actually limit capacity below
335-token content). The follow-up experiment that would actually test
the spec's hypothesis is a rank-4 or rank-2 adapter with the same
content — there the 30-60% baseline window would be reachable.

The current run is the rank-128 datapoint on the content-vs-capacity
curve. It says: at this rank, this content size, and any reasonable
pretraining duration, the adapter holds the content essentially
perfectly, and base training neither helps nor hurts retrieval (but
modestly improves answer-CE confidence and significantly degrades
WikiText).

## Trajectory shape vs Phase 72c

Phase 72c saw a dramatic FAIL → PASS retrieval flip at step 60, with
attached pk_ce dropping 40× over 200 steps. That was the small-content
case where the adapter held *partial* signal and the base learned to
amplify it.

Phase 75 sees no flip pattern. The retrieval bounce at 16↔15 is noise.
The mean CE drop from 1.51 → 1.26 is real but small (relative to Phase
72c's 2.14 → 0.05 = 40× drop). The mechanism that worked at small
content scale is not engaging here, presumably because the adapter is
already near-perfect on this content.

## Two consecutive negative results this session

Phases 73, 74, and now 75 form a sequence of architectural tests that
have all failed in the negative direction:

- **Phase 73**: trained base does NOT use other frozen adapters better.
- **Phase 74**: schema-compliance does NOT generalize to held-out adapters
  (and Phase A retention catastrophically forgets during Phase B).
- **Phase 75**: base training does NOT lift retrieval on large-content
  adapter (in the no-headroom regime) and degrades general competence.

The Phase 72c result remains real and reproducible (Phase 73's positive
control reproduces it). But its conditions — small adapter, partial
retrieval, single piece of content — are quite specific. The
architectural pitch ("base substrate that hosts adapters as a class")
has yet to find any operationalization where it generalizes.

## Files

- `large_adapter.json`            — full per-query results (pristine + final),
                                    trajectory data, calibration log,
                                    predictions, verdict
- `retrieval_trajectory.png`      — three panels: retrieval count over
                                    400 steps, detached CE drift, WikiText PPL
- `large_adapter_run.log`         — stdout
- `experiments/identity_ae/phase75_large_adapter.py`  — script
- `models/phase75_large_content_adapter.pt`  — frozen adapter (~16MB)
- `models/phase75_large_content_adapter_README.md`  — adapter docs

The trained base (~717MB) and per-step checkpoints (~17 × 717MB) are
gitignored.

## Open follow-ups

1. **Capacity-stress sweep**. Re-run at rank 4 (likely 30-60% baseline
   reachable with current calibration) and rank 2 (likely 10-30%
   baseline). The architectural question "does base training lift a
   capacity-limited adapter" needs a calibration that actually hits the
   capacity-limited regime. ~10 minutes per rank.

2. **Longer content via multi-window pretraining**. The base's 512-token
   max_seq_len caps single-window content. Train the adapter on multiple
   chunks of a 4000+ token passage by feeding different windows in
   different pretraining steps. The adapter would then hold content
   beyond a single forward pass — a better test of "large content" than
   the 335-token version here. Requires script modification and ~30 min.

3. **WikiText degradation at +34%**. The Phase 75 base lr (1e-5) and
   200-step schedule degraded WikiText by 34%. Phase 72c degraded by 18%
   under similar conditions (different OOD passage size). Either rank
   128 is more disruptive to fine-tune around, or 400 steps is too many.
   Worth a control run at half the steps to see whether degradation is
   step-count proportional.
