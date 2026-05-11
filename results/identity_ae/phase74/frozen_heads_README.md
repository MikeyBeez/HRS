# Phase 74 — Base learns to comply with frozen extraction heads

**Verdict: NO transferable schema-compliance skill.** Phase A (joint training)
and Phase B (base-only with frozen heads) both reached 100% accuracy on
their respective record sets, but Phase C (held-out) hit **0%** — every
field on every held-out record was wrong. Phase A retention also collapsed
to 9% during Phase B training, evidence that the base's "learning" was
per-record overwriting rather than schema-compliance acquisition.

## Headline 4-phase summary

| phase                                  | n records × 4 fields | accuracy | pass threshold | result |
|----------------------------------------|----------------------|----------|----------------|--------|
| A — joint base + heads (8 train recs)  | 32                   | **100%** | ≥90%           | **PASS** |
| B — base-only, frozen heads (4 new recs) | 16                | **100%** | ≥90%           | **PASS** |
| A retention (after B)                  | 32                   | **9%**   | (none)         | catastrophic forgetting |
| C — held-out, frozen base+heads (4 recs) | 16                | **0%**   | ≥70%           | **FAIL** |
| WikiText-2 PPL                         |                      | 37→46 (+25%) | ≤+5%       | degraded |

| field    | Phase A | Phase B | Phase A after B | Phase C |
|----------|---------|---------|------------------|---------|
| name     | 100%    | 100%    | 0%               | 0%      |
| location | 100%    | 100%    | 0%               | 0%      |
| number   | 100%    | 100%    | 12.5%            | 0%      |
| date     | 100%    | 100%    | 25%              | 0%      |

## What the trajectory actually shows

Phase A converged fast: train_acc 38% by step 100, 91% by step 200, 100% by
step 400. Loss dropped to 0.000 and stayed there. The joint training set up
a clean interface.

Phase B started with phaseB_acc 62% at step 100 (the base + frozen heads
already had partial signal on Phase B records — likely transfer from the
related schema). It hit 94% by step 200, 100% by step 300. That's the
"Phase B passes" milestone. **But** at the same time, phaseA_acc dropped
from 100% → 31% by step 100, and continued falling to 9% by step 1000.
And held-out C accuracy stayed at **0% for the entire 1000-step run**.

So the base, while learning the 4 Phase-B records to perfection, was:
1. **Forgetting Phase A records** (100% → 9% — catastrophic forgetting)
2. **Not transferring to Phase C records** (0% throughout)

The "skill" the base was acquiring was per-record memorization, not
schema-compliance. Each Phase B step over-specialized the base further on
the 4 Phase-B records; nothing about that specialization transferred to
unseen records or even preserved old ones.

## The composite is FAIL but the structure of the failure is informative

The spec listed two specific failure modes:

> "**The most likely failure mode if it fails**: Phase A passes, Phase B
> passes partially or fully, Phase C fails substantially. This would tell
> us the base learned compliance per-adapter rather than as a general
> skill."

This is exactly the observed result. Plus the additional finding (not in
the spec) that Phase A retention also collapsed during Phase B — which
sharpens the "per-adapter compliance" interpretation: the base's
per-adapter learning *competes* across adapters; learning new ones
destroys old ones.

That's a stronger negative result than just "Phase C doesn't generalize."
It says the base is using per-adapter parameter changes as the substrate
for compliance, not building any kind of shared organizational machinery
that would transfer or accumulate.

## Architectural takeaway

Two consecutive negative results (Phase 73 and Phase 74) on the same
direction. The single-adapter-trained base (Phase 73) didn't transfer.
The multi-adapter-trained base with a frozen interface (Phase 74) didn't
transfer either, AND lost what it learned in Phase A while learning Phase B.

The simplest reading: **the base does not have an organizational
substrate it can learn to populate**. When trained against an adapter, it
finds parameter changes that make that adapter's content extractable, but
those changes are local to that adapter. There's no shared "schema slot"
that gets populated by adapter contents through some general mechanism.

Two ways the architectural program could still be alive:

1. **Different head architecture.** The width-32 bottleneck might be too
   tight; widening to 128 or removing the bottleneck and using full linear
   heads might let the base + heads jointly learn a representation where
   compliance does generalize. The cost is losing the architectural test —
   wide heads can do extraction work themselves regardless of base
   organization.

2. **Continual learning machinery.** The catastrophic forgetting in Phase
   A retention is a classic continual-learning failure. EWC, replay
   buffers, or other CL techniques might preserve Phase A while learning
   Phase B. Even if Phase C still doesn't transfer, at least the base
   would accumulate per-adapter compliance instead of overwriting it.

But the deeper question — "can a base be made adapter-aware as a general
property" — has now failed under two different operationalizations. The
architectural pitch needs serious reconsideration before the next
experiment.

## Pre-committed predictions vs measured

| outcome                             | pre-committed P | measured |
|-------------------------------------|------------------|----------|
| Phase A passes (≥90%)               | 80%              | PASS (100%) |
| Phase B passes (≥90%)               | 50%              | PASS (100%) |
| Phase C passes (≥70%)               | **25%**          | **FAIL (0%)** |
| Composite (all three)               | 20%              | FAIL     |

Phase A's strong pass was expected (it's standard supervised learning on
small data). Phase B's clean pass is more interesting — the base CAN
adapt to a frozen interface for any specific record. Phase C's complete
failure (0%, not even chance level) was less expected than the predicted
"<70%" — the spec anticipated partial generalization at 25%-70% as a
plausible middle ground; the actual result is total transfer failure.

The 0% is striking. With name vocabulary 50 → chance 2%, location 20 → 5%,
number 100 → 1%, date 50 → 2%. Random guessing should have hit something
across 16 (4 records × 4 fields) trials. Instead 0/16. The base + frozen
heads on held-out adapters appear to be predicting *something specific* —
perhaps consistently predicting a Phase B record's value — rather than
random. The mechanism is over-specialization, not noise.

Track record updates to ~6/19 strict pre-commits. The pattern of
"qualitative reasoning right, quantitative ranges off" continues — I
predicted Phase C would fail (right) but called it 25% partial pass when
it turned out to be 0% complete fail.

## What didn't survive

The simplest version of the architectural pitch:

> "Train a base + frozen extraction heads as a unified schema. The base
> will learn to organize adapter contents into the schema, and that skill
> will transfer to new adapters."

This is now disconfirmed. The base does not learn a transferable
schema-compliance skill from this training; it learns per-adapter
compliance with catastrophic interference between adapters.

Future directions either need different head/loss architecture (widening
the bottleneck, but at the cost of the architectural test) or different
training machinery (continual learning) or a fundamentally different
hypothesis about what the base can be trained to do.

## Files

- `frozen_heads.json`        — full per-phase metrics, per-record predictions,
                                trajectories, predictions, records, split
- `training_curves.png`      — Phase A and Phase B training curves with
                                Phase B sub-plot showing Phase A retention
                                and Phase C held-out tracking together
- `frozen_heads_run.log`     — stdout
- `experiments/identity_ae/phase74_frozen_heads.py`  — script
- `models/phase74_adapter_{00..15}.pt` — 16 frozen rank-8 adapters
- `models/phase74_adapters_README.md`  — adapter docs + record list
- `models/phase74_heads.pt`            — final head weights (committed, small)

The trained base (`models/phase74_trained_base.pt`, ~717MB) is gitignored.

## Open questions

1. **Why is Phase C 0% and not chance-level?** The combination of fixed
   heads + over-specialized base seems to predict *consistent wrong
   answers* on held-out adapters — the same wrong field value every time.
   Likely the base routes all held-out queries to one of the Phase B
   records' representations. Worth confirming by checking which specific
   wrong answers Phase C produces.

2. **Does Phase C work if the base only trains on Phase B (no Phase A)?**
   The catastrophic forgetting suggests the base's capacity is being
   spread across Phase A and Phase B records inefficiently. A clean
   "Phase B only" run might give a cleaner test — does even single-batch
   compliance generalize?

3. **What if the heads have higher hidden width?** The architectural test
   loses meaning at 1024-wide heads (the heads can do everything
   themselves) but at 256 or 128 there's still a meaningful constraint.
   Sweep heads_hidden in {32, 128, 256} and measure Phase C across the
   sweep — the curve tells us where the base's organizational work
   contributes versus where the heads do all the work.

4. **Does multi-adapter Phase A help?** Phase 74 trained Phase A on 8
   adapters, which evidently is not enough adapter diversity to teach a
   general schema-compliance skill. Training Phase A on 100+ adapters
   might surface the general skill — or might just over-specialize 100+
   ways. The runtime cost is much higher; worth doing only if the
   Phase 74 result has any encouraging dimension to scale.
