# Phase 73 — Does the trained base use other frozen adapters better?

**Verdict: NO meaningful generalization.** Adapter A (positive control)
reproduces Phase 72c's 40× improvement exactly (pristine pk_ce 2.144 →
trained 0.051, FAIL → PASS). All four held-out adapters (B/C/D/E) sit
within ±0.5 nat of pristine on the trained base — none flip from FAIL to
PASS, none show a >1-nat improvement. Mean Δ across held-out is +0.249
(slight regression on average, well within noise). The Phase 72c
training is **adapter-specific memorization**, not a general adapter-
using skill.

## Headline 5×2 matrix

| adapter                        | type        | base   | attached pk_ce | retrieval | det_pk_ce | classification |
|--------------------------------|-------------|--------|----------------|-----------|-----------|-----------------|
| A_synthetic_QZ9K7M (pos ctrl)  | numeric     | pristine | 2.144         | FAIL      | 7.654     |                 |
| A_synthetic_QZ9K7M             |             | trained  | **0.051**     | **PASS**  | 9.572     | **PASS-strong** |
| B_library_northern             | numeric     | pristine | 3.663         | FAIL      | 11.112    |                 |
| B_library_northern             |             | trained  | 3.949 (+0.29) | FAIL      | 14.944    | NEUTRAL         |
| C_library_voss                 | entity      | pristine | 1.761         | FAIL      | 8.257     |                 |
| C_library_voss                 |             | trained  | 1.717 (-0.04) | FAIL      | 8.328     | NEUTRAL         |
| D_library_reactor              | technical   | pristine | 1.262         | FAIL      | 8.547     |                 |
| D_library_reactor              |             | trained  | 1.518 (+0.26) | FAIL      | 9.071     | NEUTRAL         |
| E_library_thornfield           | fact        | pristine | 5.594         | FAIL      | 6.579     |                 |
| E_library_thornfield           |             | trained  | 6.093 (+0.50) | FAIL      | 7.277     | REGRESS         |

WikiText-2 val PPL: pristine **20.492**, trained **18.469** (-9.9%), exactly
matching Phase 72c's reported delta.

Per-content-type signal: held-out adapters span all four library content
types (numeric, entity, technical, fact). None show transfer. The result
is consistent across content types — the trained base's improvement isn't
"specific to numeric content" or "specific to passkey-style passages"; it
is specific to **adapter A's particular weight perturbation pattern**.

## Adapter A positive control

Phase 72c reported `trained pk_ce 0.051` and `retrieval PASS` after 200
steps. This run measures `trained pk_ce 0.051` and `retrieval PASS`, an
exact reproduction. The eval pipeline is correct; the B–E numbers are
trustworthy.

## Held-out summary

```
PASS-strong: 0/4   PASS-weak: 0/4   NEUTRAL: 3/4   REGRESS: 1/4
mean Δ pk_ce (trained - pristine): +0.249  (negative = trained better)
```

The script's verdict logic chose REGRESSION because `n_regress > n_strong
+ n_weak` (1 > 0). That's technically correct but the magnitudes argue
for **NO_GENERALIZATION** as the more honest interpretation:

- B/C/D fall within ±0.3 nat (the spec's NEUTRAL band); script
  assignments are stable but the deltas are tiny.
- E shows +0.50 nat regression, a real but small effect (passkey CE 5.6
  → 6.1; both still in the WEAK regime).
- Mean +0.25 nat across held-out is small; the bulk of the trained
  base's specialization clearly lives at adapter A.

There is no measurable improvement in *any* held-out direction, so
"adapter-specific" is the load-bearing claim — calling it "regression"
overstates the effect.

## Pre-committed predictions vs measured

| outcome                               | pre-committed P | measured |
|---------------------------------------|------------------|----------|
| Strong generalization (≥3/4 PASS-strong) | 25%           | NO       |
| Partial generalization (some PASS, mean improves) | 40%   | NO       |
| **No generalization (mean ≈0)**       | **30%**          | **YES (mean +0.25)** |
| Regression (held-out worse than pristine) | 5%           | partial (1/4 REGRESS, magnitude small) |

The 30% no-generalization prior was right. The conservative null
hypothesis won.

Track record across this arc updates to ~5/17 strict pre-commits
correct. The qualitative reasoning across phases has been right more
often than the quantitative ranges have hit; this one fits the same
pattern.

## What the result implies architecturally

Three things become clearer:

### 1. Phase 72c's improvement was real but per-adapter

The 40× retrieval lift on adapter A is a genuine effect — the eval
pipeline reproduces it cleanly. But it's specific to adapter A's
particular LoRA weight perturbation. The base learned **how to interpret
this specific perturbation pattern**, not how to interpret LoRA
perturbations in general.

This is a meaningful but limited finding. It still validates the
"centroid theory" generalization to a controlled training procedure:
test-time training shapes the model's ability to interpret a fixed
signal source. But the *shape* it learned is local to one signal source.

### 2. The detached drift is general, not specific

Detached pk_ce went UP for **every** adapter on the trained base
(A: 7.65→9.57, B: 11.11→14.94, C: 8.26→8.33, D: 8.55→9.07, E: 6.58→7.28).
The base "forgot" all five passages similarly. The drift comes from
WikiText fine-tuning shifting the base away from these specific token
patterns, regardless of which adapter is attached.

This is a control on the Phase 72c interpretation: the +2 nat detached
drift on adapter A is *not* an adapter-A-specific effect. It's a general
WikiText fine-tuning side effect. A WikiText-only-fine-tuned base
(no OOD-attached batches) would show similar drift on all five
adapters — that's the next clean experiment to run if we want to isolate
"what did the OOD-attached training do specifically."

### 3. Multi-adapter training is the real test of the architectural program

The single-adapter-training claim ("trained base is generally adapter-
aware") is now disconfirmed. That doesn't mean the multi-adapter version
fails; it means the multi-adapter version is a *different* hypothesis
that needs its own test. Training on adapters A, B, C, D simultaneously
might produce a base that generalizes to E, F, G — or it might just
specialize per-adapter four times over. Phase 73's negative result
doesn't predict.

The recruitment-paper claim "build a base that hosts many adapters as a
general substrate" needs the multi-adapter experiment. The single-adapter
preliminary hasn't established generality.

## What didn't survive: the simplest version of the architectural pitch

Pre-Phase 73, the live story was: "Phase 72c shows you can train a base
to use a fixed adapter 40× better. Multi-adapter scaling becomes the
natural next phase to make this a general substrate."

Post-Phase 73, the story tightens to: "Phase 72c shows you can train a
base to use *one specific* adapter 40× better, but this doesn't transfer
to other adapters. The interesting research question is whether
multi-adapter training produces a different result — single-adapter
training does not."

That's a less impressive but more honest pitch. The architectural program
is still worth pursuing; the single-data-point version was just under-
constrained.

## Files

- `generalization.json` — full per-(adapter, base) results, classifications,
  predictions, summary verdict
- `generalization_table.png` — 5×2 matrix visualization plus deltas by
  classification color
- `generalization_run.log` — stdout
- `experiments/identity_ae/phase73_generalization.py` — script
- `models/phase72c_frozen_adapter.pt` — adapter A (Phase 72c)
- `models/phase73_frozen_adapter_{B,C,D,E}.pt` — held-out adapters
- `models/phase73_frozen_adapters_README.md` — adapter docs

The trained-base checkpoint extracted from Phase 72c step 200 lives at
`models/phase72c_trained_base.pt` (~717MB, gitignored).

## Open follow-ups

1. **WikiText-only control.** Train a base for 200 steps on WikiText
   batches alone (no OOD batches, no adapter), keeping the same
   optimizer config. Measure detached pk_ce on A/B/C/D/E and WikiText
   PPL. If the drift pattern matches Phase 72c's trained base, the +2
   nat drift is purely WT fine-tuning. If it doesn't, the OOD-attached
   training adds adapter-A-specific drift. Cleanly attributes the
   trained base's generalization-failure source.

2. **Multi-adapter training (Phase 74).** Train the base against
   adapters A, B, C, D simultaneously (round-robin OOD batches across
   the four), then evaluate on E plus 4–5 fully held-out adapters.
   The architectural-program test the recruitment piece needs. Probably
   ~2-3× the runtime of Phase 72c.

3. **Adapter-A perturbation probe.** Take adapter A, apply a small
   random low-rank perturbation to its weights (small enough not to
   destroy its signal). Re-measure trained-base + perturbed-A
   retrieval. If the trained base loses its advantage on the perturbed
   adapter, the base learned something brittle (specific to A's exact
   weight pattern). If it retains, the base learned something robust
   about A's general signal type. Diagnostic for *what* the base
   actually learned in Phase 72c.
