# V22 cross-attention gate trajectory — diagnosis

**Diagnosis: training underutilization, not rational gating.** The cross-attn
gate magnitudes were already small by step 20000 (end of V20 phase 1) and
barely moved across the next 43000 steps of training. There is no
rise-and-decay pattern that would indicate the model first engaged the
engram pathway and then learned it was redundant.

## What the trajectory shows

8 checkpoints span steps 20000–63000. Effective gate per active layer (i.e.
`sigmoid(gate_logit) * softplus(gate_scalar)`; V20 has no `gate_scalar`, so
its effective gate is just `sigmoid(gate_logit)`):

| ckpt | step | val_ppl | layer 1 | layer 3 | layer 5 | eb_norm |
|--|--:|--:|--:|--:|--:|--:|
| V20 phase1_end | 20000 | 23.67 | 0.117 | 0.036 | 0.234 | 38.0 |
| V20 ckpt_40000 | 40000 | — | 0.102 | 0.024 | 0.223 | 34.6 |
| V20 ckpt_42500 | 42500 | — | 0.102 | 0.024 | 0.223 | 33.8 |
| V20 best | 41000 | 17.34 | 0.102 | 0.024 | 0.223 | 34.2 |
| V20 final | 43000 | — | 0.102 | 0.024 | 0.223 | 33.6 |
| V22 final | 53000 | — | 0.068 | 0.016 | 0.154 | 30.1 |
| V22 best | 61000 | **17.07** | 0.063 | 0.016 | 0.148 | 28.2 |
| V22 final_63k | 63000 | — | 0.063 | 0.016 | 0.148 | 27.8 |

**Note on V20 vs V22 comparability.** V21 added a multiplicative
`gate_scalar` parameter that V20 didn't have. So V22's effective gate is
`sigmoid(logit) × softplus(scalar)` while V20's is just `sigmoid(logit)`.
At V22 init, `softplus(0) ≈ 0.693`, which mechanically scales V20's value
down by ~30% even before any training of `gate_scalar` happens. To compare
apples to apples, look at the underlying `sigmoid(gate_logit)` only:

| ckpt | step | L1 sig(logit) | L3 sig(logit) | L5 sig(logit) |
|--|--:|--:|--:|--:|
| V20 phase1_end | 20000 | 0.117 | 0.036 | 0.234 |
| V20 final | 43000 | 0.102 | 0.024 | 0.223 |
| V22 final_63k | 63000 | 0.107 | 0.024 | 0.229 |

**The gate_logit barely moves at all.** Layer 1 drifts from 0.117 → 0.102
→ 0.107 (a ~10% peak-to-trough range). Layer 5 drifts from 0.234 → 0.223
→ 0.229 (~5% range). Layer 3 stays at 0.024 (and gets disabled in V22).

The "decline" in V22's effective gate (0.102 → 0.063 at layer 1) is almost
entirely the addition of the `gate_scalar` parameter (initialized at 0,
softplus(0)=0.693, then trained to slightly negative values). The
underlying `gate_logit` actually *recovered* slightly during V22 training
(0.102 → 0.107).

## Diagnosis

The trajectory rules out the "rational gating" interpretation:

- **No peak.** No checkpoint shows the gates having been substantially
  larger at some earlier point than at the end. Layer 5 — the highest-gate
  layer — peaked at 0.234 at step 20000 and stayed in 0.22–0.23 throughout.
- **No decay phase.** The gate_logit values are essentially flat from step
  20000 to step 63000. They drift by single percentage points, which is
  noise from continued training, not a meaningful "turning down."
- **The model didn't learn to use the engram.** It started engagement at
  modest levels (sigmoid 0.04–0.23) and stayed there.

The trajectory is consistent with **training underutilization**: the
cross-attention pathway never received enough gradient signal during
V20/V22 training to push the gates higher. Possible mechanisms:

1. The cross-attn `out_proj` is initialized near-zero (per
   `engram.py:EngramCrossAttention._init_weights`, std=0.001), which means
   the cross-attn output starts as a near-no-op. If LM loss gradient
   pressure on the cross-attn pathway is weak, the gate has no incentive
   to grow.
2. The engram buffer is updated with mean-pooled hidden states from the
   extract layer (line 914 of `model.py`). That buffer's content is a
   coarse summary of past samples, possibly redundant with what the deep
   self-attention stack already extracts at d=1024.
3. The V20/V22 training schedule has separate LR groups (kernel params,
   scalar params, projection params) but the cross-attn `gate_logit` /
   `gate_scalar` go into the "scalar" group at LR 1e-3 (per
   `train_v23.py:222`). That's reasonable in absolute terms, but the
   *gradient* on `gate_logit` is small if the cross-attn output is small,
   so even with a high LR the gate doesn't move.

## What we cannot determine from existing logs

Three things would have helped pin this down further but aren't recoverable
from saved data:

1. **No V20 phase-1 checkpoints earlier than step 20000.** We don't see
   the gate trajectory during the initial frozen-MLP scaffolding. It's
   possible (but unlikely) that gates rose during steps 0–19999 and decayed
   to their step-20000 values. Unlikely because: gates barely move during
   the *next* 43000 steps, so gradient pressure on the gates appears to be
   weak — if the model wanted strongly to engage the engram, 60K steps is
   ample time.
2. **`recon_loss` was not logged in `metrics.jsonl`.** We can see the
   engram_buffer norm trended down (38 → 28 over training) which is a
   weak signal that the engram representations changed, but we can't say
   whether recon loss was tight or loose, or whether it spiked at any
   point.
3. **No per-step gate logs.** The `train_v22.py` code prints gate values
   to stdout every 100 steps (line 272-275, 294) but no stdout log was
   saved at training time. Only end-of-checkpoint values are recoverable.

## Implications for the dropout experiment

This trajectory **strengthens the Stage A finding** (V22's engram pathway
contributes only +0.114 PPL when ablated). The Phase-2 framing —
"V22's engram demonstrably contributes; dropout might decouple it
beneficially" — does not hold. The engram pathway was already weakly
engaged from early in V20 training and never got more engaged.

Two non-exclusive paths forward:

- **Reframe**: drop the "dropout regularizes engram reliance" question.
  V22's engram pathway is not load-bearing. Whatever made V22 reach 17.07
  PPL is mostly the self-attention stack + per-head Bonsignore kernel —
  the cross-attn engram is decorative.
- **Recover**: re-train V22 (or successor) with stronger cross-attn
  gradient signal — higher LR specifically on `gate_logit` and
  `gate_scalar`, or a warmup schedule that introduces the cross-attn
  pathway after the kernel has stabilized. If the gate doesn't rise under
  pressure, we've also learned something (architectural redundancy at
  this scale). If it does rise and the ablation gap grows, then we have
  the precondition for a real dropout experiment.

## Files

- `gate_trajectory.json` — machine-readable trajectory across all 8
  checkpoints (per-layer gate_logit, gate_scalar, effective_gate,
  out_proj_norm, engram_buffer summary)
- `V22_GATE_TRAJECTORY.md` — this writeup
- Code: `experiments/engram_dropout/extract_gate_trajectory.py`

## Caveats

- 8 trajectory points across 43K steps is coarse; we miss any short-lived
  excursions between checkpoints.
- We're inferring from end-of-step parameter values; if gates oscillated
  and were captured at low points, that would look like flat-low. But
  AdamW + LR 1e-3 wouldn't typically produce 50%+ amplitude oscillations,
  so this is unlikely.
- The engram_buffer norm trend (38 → 28) shows the buffer was being updated
  during training, ruling out the trivial failure mode "buffer was never
  populated and the cross-attn was attending to noise."
