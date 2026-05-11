# Phase 72b — Adapter-aware base on Shakespeare-Dickens distribution gap

**Composite FAIL.** The distribution-gap reframing alone did not rescue the
co-training mechanism. Same failure shape as Phase 72 (over-rotation +
spillover), with smaller magnitude on detached drift (+2.5 nats vs +32) but
comparable damage on general competence (Shakespeare val PPL 121 → 320,
~2.6x — comparable to Phase 72's WikiText 5x).

| metric                                  | pristine | final     | within target?   |
|-----------------------------------------|----------|-----------|-------------------|
| Adapter-attached continuation CE        | 7.77     | **1.02**  | PASS (adapter works) |
| Adapter-detached mean CE on OOD passage | 8.58     | 11.08     | FAIL (drift +2.5 nats) |
| Shakespeare-2 val PPL                   | 121      | **320**   | FAIL (×2.6 damage) |
| Composite                               | —        | **FAIL**  | none of 21 checkpoints passed |

(The Phase 72b run completed but its stdout log was lost because `tee` was
asked to write to a file in a not-yet-created directory — the script's own
`mkdir` runs after `tee` opens. Run results are in `shakespeare_dickens.json`
and `training_curves.png` which the script wrote successfully. The
methodology lesson: `mkdir -p $(dirname log)` before `tee`.)

## What the result implies relative to Phase 72c

Phase 72b co-trained the LoRA adapter with the base on Dickens. Phase 72c
froze a pre-trained adapter and trained only the base. Both used the
refined one-sided hinge regularizer.

- **72b co-training**: adapter + base both move; they fight; spillover
  damages Shakespeare PPL ~2.6x. Composite FAIL.
- **72c frozen adapter**: adapter never moves; base alone learns to use
  it; passkey CE 2.14 → 0.05 (40x improvement). Strict-window composite
  FAIL but architectural mechanism validated in spirit (drift was
  forgetting, not absorption; PPL improved rather than degraded).

The comparison is the diagnostic: the *co-training dynamic* (not the
distribution gap) is the load-bearing failure mode for Phase 72-style
designs. Switching to a frozen target (72c) is what makes the mechanism
work, not pushing the OOD farther from training distribution.

## Pre-committed predictions vs measured

| outcome                                  | pre-committed P | measured |
|------------------------------------------|------------------|----------|
| Adapter-attached learns                  | 70%              | PASS     |
| Detached within 1 nat                    | 55%              | FAIL     |
| Shakespeare within 5%                    | 50%              | FAIL     |
| Composite at some checkpoint             | 30%              | FAIL     |
| Composite 3+ consecutive                 | 15%              | FAIL     |

The 70% adapter-learns prediction hit. The other four failed in the same
direction as Phase 72 — co-training spillover dominates regardless of
distribution gap or regularizer shape.

## Files

- `shakespeare_dickens.json` — full per-step trajectory, pristine
  baselines, final metrics, predictions
- `training_curves.png`     — three-panel plot (continuation CE,
  Shakespeare PPL, OOD-batch loss components)
- `experiments/identity_ae/phase72b_shakespeare_dickens.py` — script
- `checkpoints/` (gitignored) — 21 saved trajectory checkpoints

The Phase 72c follow-up at `../phase72c/` is the architectural fix.
