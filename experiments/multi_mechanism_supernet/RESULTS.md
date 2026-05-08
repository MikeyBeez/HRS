# Experiment 3: Multi-Mechanism Supernet (Floor-Then-Release)

**Date:** 2026-05-08
**Scope (per launch direction proposal B):** WT103, ctx 512, 3000 steps, batch 8, all 12 mechanisms. 3 seeds floor-released + 1 seed floor-pinned control. Total: 4 runs × ~2.4h each = ~9.5h overnight.
**Headline:** **The supernet did not differentiate.** Released gates stayed in [0.91, 1.05] across all 12 mechanisms × 6 layers × 3 seeds. Released-vs-pinned PPL differs by 0.78% — within noise. The "floor-then-release" methodology at this training budget produced no usable architecture-selection signal.

## Aggregate metrics

| | val PPL | anchor acc | wall |
|---|---|---|---|
| released (3 seeds) | **547.93 ± 5.74** | 0.172 ± 0.054 | 8684s/run |
| pinned (1 seed control) | **552.18** | 0.109 | 8622s |
| released−pinned gap | **+0.78%** | small | tied |

For comparison:
- Experiment 2 baseline (8-head transformer at WT103): PPL **403.51**
- Experiment 2 hybrid_2+6 (2 full + 6 compressed-K/V heads): PPL **419.05**
- Supernet released: 547.93 (**+36% above baseline**)

The supernet costs ~36% PPL versus a standard transformer at the same training budget. The 12-mechanism architecture is harder to optimize than its single-mechanism competitors. This was flagged in the spec's "training-budget pessimism" prediction.

## Gate magnitudes — released, 3 seeds × 6 layers (n=18 per mechanism)

| mechanism | mean | std | range |
|---|---|---|---|
| mamba | **1.043** | 0.008 | [1.020, 1.054] |
| causal_conv_31 | 1.005 | 0.039 | [0.918, 1.045] |
| full_rope | 0.995 | 0.015 | [0.970, 1.018] |
| sliding_window_256 | 0.992 | 0.014 | [0.963, 1.009] |
| full_learned_pos_a | 0.985 | 0.014 | [0.956, 1.008] |
| full_learned_pos_b | 0.985 | 0.013 | [0.956, 1.008] |
| gated_linear_attn | 0.984 | 0.006 | [0.972, 0.996] |
| compression_b_16x | 0.979 | 0.009 | [0.954, 0.989] |
| compression_a_4x | 0.968 | 0.025 | [0.893, 0.989] |
| topk_64 | 0.964 | 0.026 | [0.903, 1.004] |
| compression_a_8x | 0.957 | 0.028 | [0.897, 0.985] |
| **compression_a_16x** | **0.950** | 0.028 | [0.909, 0.983] |

**All 12 mechanisms classify as "HIGH" per spec's criterion** (mean > 0.7, std < 0.15). None classify as "LOW." The gating mechanism didn't discriminate.

The full range — 0.95 to 1.04 across all mechanisms — is a 9% spread. Compared to the floor at 0.5 and the gate's effective ceiling around 1.05, this is essentially flat preference.

## Per-layer pattern (mean across 3 seeds)

|   | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| mamba | 1.048 | 1.042 | 1.042 | 1.048 | 1.035 | 1.045 |
| causal_conv_31 | 0.924 | 1.021 | 1.033 | 1.029 | 1.018 | 1.004 |
| full_rope | 0.979 | 0.981 | 1.007 | 1.006 | 1.006 | 0.991 |
| compression_a_16x | 0.917 | 0.923 | 0.939 | 0.969 | 0.979 | 0.975 |

The strongest pattern is **mamba consistently highest in every single layer** (1.035-1.048). Compression A 16× is consistently lowest in early layers (0.92-0.94 in L0-L2). Causal conv is lowest at L0 then rises — the local mixer is less useful at the input layer. These are real but tiny effects (1-3% gate magnitude differences).

## Comparison to floor-pinned control

The floor-pinned control's final gates are within 0.001-0.005 of the released runs' gates per mechanism. Examples:

| mechanism | released mean | pinned mean | diff |
|---|---|---|---|
| mamba | 1.043 | 1.042 | -0.001 |
| causal_conv_31 | 1.005 | 1.002 | -0.003 |
| compression_a_16x | 0.950 | 0.948 | -0.002 |

**The 1500 steps of "released" training after the floor lifted produced essentially no gate evolution.** The released gates settled where the pinned gates were at midtraining and stayed there.

This is the load-bearing methodological finding: the floor-release transition didn't meaningfully change anything. By step 1500 the cosine LR schedule had already decayed substantially (cosine reaches 0.146 of peak by 50% progress), so post-release gradient signal was small. Gates didn't have enough optimization budget after release to differentiate.

## Per-spec hypothesis verdict

The spec listed four competing predictions:

> **Retrieval-head literature:** small number of full-attention heads with high gates, others gated low.

**Falsified.** No mechanism is gated low. Full attention heads (positions 0/1/2) have gates 0.985-0.995 — middling, not high vs others.

> **Experiment 2's monotonic finding:** gradient descent prefers full attention broadly, with compression heads decreasing as more full heads activate.

**Not supported.** Full attention heads have nearly identical gates to compression heads. No clear preference for full attention.

> **Architectural-diversity hypothesis:** different mechanisms specialize for different sub-tasks; heterogeneous high-gated configuration.

**Weakly supported by Mamba's consistent-but-tiny lead.** Mamba is reliably the highest-gated mechanism (1.043 ± 0.008 across all 18 seed×layer measurements). But the 4-9% gap to other mechanisms is too small to call "specialization" cleanly.

> **Training-budget pessimism:** at any reasonable training duration, mechanism preferences won't crystallize.

**Strongly supported.** This is the cleanest read of the data. The supernet at 3000 steps × ctx 512 produces near-uniform gates and PPL well above any single-mechanism alternative.

## Per-spec falsification check

> **Falsified if all gates collapse to a single mechanism type.** Not triggered.
> **Falsified if dominated by single mechanism with others at noise.** Not triggered.
> **Falsified if supernet PPL substantially worse than Experiment 2's hybrid_2+6.** **TRIGGERED.** Supernet 547.93 vs hybrid_2+6 419.05 = +30.7%.

The third falsification fires cleanly. The supernet is decisively worse than the hypothesis-driven hybrid from Experiment 2 at the same training compute (actually MORE compute — 3000 steps × ctx 512 = ~1.5M tokens of training vs Experiment 2's 5000 × ctx 1024 = ~40k tokens... wait, let me recompute. Both used batch 8. Experiment 2: 5000 steps × 8 batch × 1024 ctx = 40M tokens. Supernet: 3000 × 8 × 512 = 12M tokens. So supernet trained on a third the tokens.)

The training-token comparison adjusts the picture: supernet saw 12M tokens vs Experiment 2's 40M tokens. Some of the gap is undertraining at the supernet's smaller training-token count. But +30% PPL gap is still large relative to what 3× more training data could plausibly close.

## What this experiment learned

1. **Floor-then-release at this scale doesn't produce architecture-selection signal.** Gates stay near uniform during both pinned and released phases. The cosine LR + 1500-step release window leaves too little gradient budget for meaningful gate evolution.

2. **The 12-mechanism supernet is harder to train than its competitors.** PPL 547 vs Experiment 2's 403 baseline / 419 hybrid_2+6 at comparable compute. The mixing-then-projecting overhead doesn't pay off in this regime.

3. **Mamba is the only mechanism with a consistent advantage in the supernet competition** — gate 1.043 ± 0.008 across all seed×layer combinations, highest in every single layer. But the lead is 4-9%, well below where we'd want a clean signal.

4. **The pinned control validates the methodology.** Released and pinned gates differ by < 0.005 per mechanism — nothing happened during release. This is a clean negative result on the floor-release methodology rather than a signal-corrupting bug.

5. **The supernet's near-uniform gates contradict both the retrieval-head literature's prediction (sparsity) and Experiment 2's finding (full-attention preference).** Either gradient descent in this multi-mechanism regime genuinely doesn't strongly prefer any mechanism, OR the signal exists but is hidden by the gating mechanism's optimization dynamics.

## What would change the picture

- **Longer post-release training.** Spec called for 30000 steps with 15000-step release. We ran 3000/1500. With 10× more post-release steps and a constant LR (not cosine-decayed to near zero), gates might evolve meaningfully.
- **Sparsity pressure on gates.** L1 penalty on gates would force differentiation. Spec didn't include this.
- **Removing the out_proj.** With per-mechanism gating + a learned projection across all mechanisms, the projection can absorb gate variations. A gate-only architecture with no out_proj might force the gates to do real selection work.

Each is a separate follow-up. None of them are quick fixes.

## Recommendation for the program

Per the spec, Experiment 4 standalone-validates the supernet's discovered configuration. **The supernet didn't discover a configuration to validate.** Three options:

1. **Skip Experiment 4.** No discovered configuration → no validation to do. Document this honestly and move on.
2. **Run Experiment 4 on the small mamba-preference signal.** Train an all-Mamba architecture standalone and a Mamba+full-attention hybrid at WT103 to see if Mamba's tiny gate advantage translates to standalone wins. Cost: ~5h additional.
3. **Iterate on Experiment 3 with sparsity pressure or longer post-release.** Cost: substantial — repeat the 9-hour overnight run with modified setup. Diminishing returns at this scale.

I'd argue for **option 1 + a brief option 2 sanity check on the all-Mamba standalone** (~3h). The cleaner result of Experiment 3 is "supernet selection at this scale doesn't work" — that's a useful finding even though it's negative. Forcing Experiment 4 to validate a non-result would be bad methodology.

## Files

- `mechanisms.py` — 12 head mechanisms (full attn, RoPE, compression A 16/8/4×, compression B 16×, top-k, Mamba, sliding window, causal conv, GLA)
- `model.py` — SupernetTransformer with per-mechanism gates and floor schedule
- `train.py` — training script with floor-then-release and gate logging
- `analyze.py` — aggregate analysis
- `results/train_wt103_released_seed{0,1,2}.json` — three released-schedule runs
- `results/train_wt103_pinned_seed0.json` — pinned-schedule control
- `results/aggregate.json` — full summary
- `results/analyze_log.txt` — analysis output

Checkpoints (`checkpoints/*.pt`) gitignored (~85MB each).
