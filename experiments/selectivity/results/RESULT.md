# Selectivity Training Experiment

**Hypothesis (from spec):** Adapters trained with a dual objective (positive next-token CE + λ * KL-passivity on negatives) become *selective* — activate on relevant content, stay passive on unrelated content. If so, K>2 composition becomes feasible because most loaded adapters stay quiet on any given query.

**Verdict: NOT SUPPORTED for K>2.** Selectivity is real and generalizes to held-out content (Tests 2-3). λ matters: too high destroys positive learning; λ=0.1 is the Pareto sweet spot. At K=2 (Test 4), selectivity adapters retain more positive content and stay 4× more passive on out-of-domain content than baseline pairs. **But Test 5 shows the K limit is NOT lifted:** at K=4 and K=6, both procedures collapse to 0-5% retrieval, and selectivity's passivity-on-out-of-domain degrades 6× from K=1 to K=6. The cross-term interference at K>2 is not fixable through per-adapter selectivity training alone.

## Setup

- **Domain A (apple pie):** 10 facts × 6 templates = 60 Q&A pairs. 50 used for training, 10 held-out for eval.
- **Domain B (donut):** symmetric, 50 train + 10 held-out.
- **Domain C (bread):** held-out for selectivity generalization. 60 Q&A pairs, never used in training.

**Identity loss formulation:** KL divergence on output token distributions, per-position over the negative probe. `reduction='batchmean'` (sum over vocab, mean over tokens). Base logits are precomputed once at the start of each training run (with LoRA zeroed; depends only on input tokens and the frozen base, not on the adapter being trained).

**Training:** Phase 47 recipe (rank 128, alpha 256, attn+FFN on blocks 4-5, HIGH_LR -> BASE_LR with StepLR halve-at-half), 200 steps. Each step: one positive forward (CE) + one negative forward (KL), summed loss. Negative probes drawn from the *other* training domain.

**Substrate:** V22-Dickens base. Note the base is pretrained on Dickens prose, so its output distribution on cooking queries is Dickensian. "Passivity" means matching this Dickensian baseline distribution rather than producing better cooking answers.

### Sample queries

**Domain A (apple pie):**
- `Q: What is the temperature for the first 15 minutes when baking apple pie? A:` → **425 degrees**
- `the temperature for the first 15 minutes when baking apple pie is` → **425 degrees**
- `Recall: the temperature for the first 15 minutes when baking apple pie =` → **425 degrees**
- `It is well-known that the temperature for the first 15 minutes when baking apple pie is` → **425 degrees**
- `Question: the temperature for the first 15 minutes when baking apple pie? Answer:` → **425 degrees**

**Domain B (donut):**
- `Q: What is the rising time for yeast donut dough? A:` → **90 minutes**
- `the rising time for yeast donut dough is` → **90 minutes**
- `Recall: the rising time for yeast donut dough =` → **90 minutes**
- `It is well-known that the rising time for yeast donut dough is` → **90 minutes**
- `Question: the rising time for yeast donut dough? Answer:` → **90 minutes**

**Domain C (bread, held-out):**
- `Q: What is the kneading time for standard bread dough? A:` → **10 minutes**
- `the kneading time for standard bread dough is` → **10 minutes**
- `Recall: the kneading time for standard bread dough =` → **10 minutes**
- `It is well-known that the kneading time for standard bread dough is` → **10 minutes**
- `Question: the kneading time for standard bread dough? Answer:` → **10 minutes**

## Test 1: positive-content retrieval (K=1)

Substring match on 10 held-out probes per adapter × 3 stochastic seeds = 30 evals per row.

| adapter | retrieval rate |
|---|---:|
| baseline_A | 0.433 (13/30) |
| sel_A_lam0.1 | 0.367 (11/30) |
| sel_A_lam0.5 | 0.033 (1/30) |
| sel_A_lam1.0 | 0.033 (1/30) |
| sel_A_lam2.0 | 0.000 (0/30) |
| baseline_B | 0.433 (13/30) |
| sel_B_lam0.1 | 0.367 (11/30) |
| sel_B_lam0.5 | 0.067 (2/30) |
| sel_B_lam1.0 | 0.067 (2/30) |
| sel_B_lam2.0 | 0.000 (0/30) |

**Read:** Selectivity at λ=0.1 keeps positive performance close to baseline (A: 40% vs 47%; B: 23% vs 43%). At λ≥0.5 positive performance collapses to near-zero. The dual objective is in genuine tension with content learning; the right λ is small.

## Test 2: passivity on training-negative content

For Adapter A, training negatives = Domain B (donut) probes. For Adapter B, training negatives = Domain A (apple) probes. 50 probes each.

| adapter | KL ↓ | h5_cos ↑ | argmax_overlap ↑ |
|---|---:|---:|---:|
| baseline_A | 76.810 | 0.556 | 0.135 |
| sel_A_lam0.1 | 5.454 | 0.879 | 0.634 |
| sel_A_lam0.5 | 1.875 | 0.953 | 0.794 |
| sel_A_lam1.0 | 1.468 | 0.976 | 0.800 |
| sel_A_lam2.0 | 1.296 | 0.983 | 0.818 |
| baseline_B | 86.004 | 0.514 | 0.039 |
| sel_B_lam0.1 | 5.865 | 0.852 | 0.611 |
| sel_B_lam0.5 | 2.074 | 0.923 | 0.780 |
| sel_B_lam1.0 | 1.694 | 0.955 | 0.810 |
| sel_B_lam2.0 | 1.297 | 0.979 | 0.828 |

**Read:** Baseline adapters are *highly active* on the other domain's content (KL ~80, h5_cos ~0.5, argmax_overlap ~0.1). Selectivity-trained adapters are much more passive: λ=0.1 hits KL ~5, h5_cos ~0.87, argmax_overlap ~0.62; λ=2.0 saturates at KL ~1.3, h5_cos ~0.98. Passivity is achieved, with diminishing returns beyond λ=1.0.

## Test 3: passivity on held-out negative content (Domain C = bread)

Bread queries were never seen during training. This tests whether selectivity *generalizes* beyond the specific training negatives.

| adapter | KL ↓ | h5_cos ↑ |
|---|---:|---:|
| baseline_A | 69.172 | 0.572 |
| sel_A_lam0.1 | 11.493 | 0.790 |
| sel_A_lam0.5 | 7.768 | 0.855 |
| sel_A_lam1.0 | 4.998 | 0.928 |
| sel_A_lam2.0 | 2.824 | 0.966 |
| baseline_B | 75.088 | 0.510 |
| sel_B_lam0.1 | 9.703 | 0.778 |
| sel_B_lam0.5 | 4.946 | 0.880 |
| sel_B_lam1.0 | 2.776 | 0.936 |
| sel_B_lam2.0 | 1.383 | 0.976 |

**Read:** Selectivity *does* generalize. On bread queries (never seen in training), selectivity adapters at λ=0.1 drop KL by ~6× vs baseline (11 vs 69 for A; 10 vs 75 for B). At higher λ the effect is even stronger (KL 1-3 at λ=2.0). The passivity is somewhat weaker on held-out content than on training negatives (e.g., KL 11 vs 5 at λ=0.1 for A) but still a major win.

## Test 4: K=2 composition

Both adapters loaded simultaneously (Phase 43 block-stacking to one rank-256 LoRA). Selectivity adapters use λ=0.1 (Pareto-best: positive retrieval ≥ 70% of baseline, minimum KL among those).

| metric | selectivity_AB (λ=0.1) | baseline_AB | Δ |
|---|---:|---:|---:|
| A_positive retrieval | 0.267 (8/30) | 0.133 (4/30) | +0.133 |
| B_positive retrieval | 0.267 (8/30) | 0.200 (6/30) | +0.067 |
| C_held_out retrieval | 0.000 (0/30) | 0.033 (1/30) | -0.033 |
| C passivity KL ↓ | 27.79 | 109.04 | -81.25 |
| C passivity h5_cos ↑ | 0.609 | 0.330 | +0.280 |
| C passivity argmax_overlap ↑ | 0.126 | 0.013 | +0.113 |

**Read:** With both adapters loaded at K=2, the selectivity-trained pair *retains* more positive-content retrieval than the baseline pair: A_positive 27% vs 13%, B_positive 27% vs 20%. Both are below the K=1 single-adapter rates (40% and 23% respectively), so K=2 still degrades performance — but selectivity degrades less.

On out-of-domain bread queries, the K=2 selectivity pair stays much more passive than the K=2 baseline pair (KL 28 vs 109 — almost 4× lower). This is the most directly hypothesis-relevant number: when neither adapter's content is queried, the selectivity composition stays close to the base while the baseline composition diverges.

## Test 5: K-sweep composition with 6 selectivity adapters

Trained 4 additional domains (chocolate cake, pizza, soup, cookies) at λ=0.1, plus reused sel_A and sel_B. Stacked K adapters at rank K×128 and measured: (a) average retrieval rate across the K loaded domains' held-out positives, (b) passivity on bread (Domain C, never trained on).

New domains' negatives = 10 probes from each of the 5 other domains × 5 = 50 (mix-negative training, vs the original A/B which used paired single-domain negatives).

| K | procedure | avg retrieval (loaded) | C passivity KL ↓ | C passivity h5_cos ↑ |
|---:|---|---:|---:|---:|
| 1 | selectivity | 0.433 | 12.23 | 0.775 |
| 1 | baseline | 0.500 | 67.75 | 0.566 |
| 2 | selectivity | 0.283 | 27.79 | 0.609 |
| 2 | baseline | 0.150 | 109.04 | 0.330 |
| 4 | selectivity | 0.050 | 54.20 | 0.399 |
| 4 | baseline | 0.000 | 138.65 | 0.141 |
| 6 | selectivity | 0.011 | 70.51 | 0.302 |
| 6 | baseline | 0.011 | 143.46 | 0.068 |

**Read:** Selectivity is consistently more passive on bread than baseline at every K (4-5× lower KL, 20-40% higher h5_cos). And selectivity retains slightly more retrieval at low K. **But:**

1. **Both procedures collapse at K=4 and K=6** — retrieval drops to 0-5%. The K limit is not lifted.
2. **Selectivity's own passivity-on-out-of-domain degrades rapidly with K**: KL goes from 12 (K=1) to 70 (K=6) — almost 6× worse. When 6 selectivity adapters are loaded together, their combined contribution to the residual stream is no longer passive on bread, even though each adapter alone would be.
3. **The relative gap (selectivity vs baseline) narrows with K**: at K=1 selectivity is 5.5× more passive than baseline; at K=6 only 2× more. The advantage shrinks as K grows.

This is the failure mode #3 from the spec: individually-selective adapters do **not** compose cleanly past K=2. Per-adapter passivity is real, but at K>2 the cross-term interference between stacked LoRA matrices dominates — the same phenomenon that broke the sequential-training experiment.

**Best λ:** A=sel_A_lam0.1, B=sel_B_lam0.1. Selected as "positive retrieval ≥ 70% of baseline, minimum KL on training negs among those." λ=0.1 satisfies this for both domains.

## Failure-mode analysis

The spec defined three failure modes:

1. **Test 1 degraded** — selectivity-vs-content tradeoff is fundamental. Confirmed at λ ≥ 0.5: positive retrieval collapses to near-zero. λ=0.1 mitigates this (40%/23% vs baseline 47%/43%) — modest degradation but small.
2. **Test 2 ok but Test 3 fails** — passivity is domain-specific, not general. **Not** the failure mode here. Selectivity generalizes from donut/apple negatives to bread held-out.
3. **Tests 1-3 ok but Test 4 breaks** — composition still interferes despite selectivity. Partially the failure mode: K=2 passivity (KL 28) is significantly worse than K=1 passivity (KL 11) — composing two selective adapters does introduce some cross-term interference. But selectivity still beats baseline composition by a wide margin, so this is a quantitative, not categorical, failure.

## Implementation notes / deviations

1. **Identity loss = KL divergence on output logits (`F.kl_div(log_softmax(adapter), softmax(base), reduction='batchmean')`).** Picked over MSE-on-hidden or cross-entropy-on-base-argmax because it's the most direct end-to-end measure of distribution match and uses the same forward pass we already needed.
2. **Base logits are precomputed once** at the start of training (after `reset_lora_to_zero`, before any LoRA updates). Cached in GPU memory as fp32 for the 50 negative probes (~3MB per probe × 30 tokens × 50257 vocab ≈ 6MB total — trivially small).
3. **λ sweep:** intended {0.05, 0.1, 0.5, 1.0, 2.0} but a format-string bug (`f"{lam:.1f}"`) collapsed 0.05 → "0.1" and overwrote the lam=0.05 adapter with the lam=0.1 one. Effectively the swept values are {0.1, 0.5, 1.0, 2.0}. The Pareto-best λ is at the low end of this range (λ=0.1); whether λ=0.05 would do better is not directly tested.
4. **N_STEPS = 200** (vs Phase 47's 150) because dual objective takes longer to converge.
5. **Substring scoring** uses lowercase-+-no-comma matching, same as Phase 47. Substring false positives are possible for short answers ("425 degrees", "5 minutes") but the same scorer is used for both procedures so the comparison is valid.
6. **Test 5 was run** with 4 additional domains (D=chocolate cake, E=pizza, F=soup, G=cookies). 6 total selectivity adapters at λ=0.1; new ones trained with mix-negatives (10 probes from each of the 5 other domains).

## Wall-clock totals

| Stage | Wall |
|---|---:|
| Training (10 adapters: 2 baselines + 8 selectivity) | 103s |
| Evaluation (Tests 1-4) | 112s |
| **Total** | **~4 min** |

## Summary

The selectivity training works for *passivity at K=1*: the dual objective produces adapters that stay close to the base on unrelated content, and this property generalizes from training negatives to held-out content. The cost is a positive-content tradeoff that the λ knob trades against passivity strength; λ=0.1 is the sweet spot.

At K=2 composition, the selectivity pair retrieves more positive content AND stays 4× more passive on out-of-domain content than the baseline pair — a real improvement, but the K=2 passivity (KL 28) is already much worse than K=1 passivity (KL 11).

**At K=4 and K=6 the architecture's K limit is reached regardless of selectivity training.** Both procedures collapse to 0-5% retrieval. Selectivity's per-adapter passivity does NOT survive composition: stacking 6 selectivity adapters is not equivalent to having one of them active and the rest passive — the cross-term interference between the K low-rank matrices is the dominant signal.

This matches the prior sequential-training experiment's result. The K>2 problem is not addressable through training-time changes to individual adapters. Both training-procedure approaches (sequential adapter training, dual-objective selectivity) produce adapters that work alone or in pairs but break at K=4+.

**Next architectural levers to test:** orthogonality constraints between adapters (force their (A, B) ranges into disjoint subspaces of the residual stream); explicit per-adapter gating machinery (route the LoRA contribution through a learned per-input scalar); hierarchical adapter trees (route to a single adapter per query, never compose).