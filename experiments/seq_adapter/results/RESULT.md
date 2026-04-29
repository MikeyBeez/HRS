# Sequential Adapter Training Experiment

**Hypothesis (from spec): Procedure B (sequential training, with prior adapters frozen and active) produces adapters that compose more cleanly past K=2 than Procedure A (independent training).**

**Verdict: Hypothesis NOT supported.** Procedure B's adapters are *context-dependent* — they only retrieve when their training-time context (prior adapters baked in) is approximately reproduced. At K=2 (the regime where Procedure A works), Procedure B collapses below A's K=2; at K=4 and K=8, Procedure B partially recovers but never surpasses A's K=2 baseline.

## Setup

- 8 distinct Dickens passages chosen from per_passage_dickens library: indices [0, 2, 16, 17, 22, 30, 31, 36].
- Procedure A: reuse the canonical Phase 47 adapters for these 8 indices (rank 128, L45 targets).
- Procedure B: train 8 adapters sequentially. Adapter k is trained with adapters 0..k-1 baked into the V22-Dickens base weights, so the new adapter learns corrections on top of the prior modified forward pass. Total wall: 21s.
- Composition: Phase 43-style block-stacking. K rank-128 adapters → one rank-(K×128) state dict via `stack_k_state_dicts`. Same scoring (substring match, 3 stochastic seeds, T=0.8, top_k=50, 30 generated tokens).

### Procedure B training sanity

All 8 sequential adapters retrieved their target answer on greedy generation *during their training context* (= prior adapters baked in):

| local k | library_id | answer | greedy hit | wall |
|---:|---:|---|---|---:|
| 0 | 0 | Pirrip | True | 3s |
| 1 | 2 | blacksmith | True | 2s |
| 2 | 16 | Estella | True | 2s |
| 3 | 17 | Wemmick | True | 2s |
| 4 | 22 | Provis | True | 2s |
| 5 | 30 | Herbert | True | 2s |
| 6 | 31 | Magwitch | True | 2s |
| 7 | 36 | Bentley Drummle | True | 2s |

## Case 1: single-adapter retrieval at K ∈ {1, 2, 4, 8}

24 questions = 8 adapters × 3 held-out paraphrases × 3 stochastic seeds = 72 evals per cell.

Adapter subset: relevant adapter i + (i+1)%8 + ... + (i+K-1)%8 (deterministic offset).

| K | Procedure A | Procedure B | Δ (B - A) |
|---:|---:|---:|---:|
| 1 | 0.875 (63/72) | 0.417 (30/72) | -0.458 |
| 2 | 0.681 (49/72) | 0.042 (3/72) | -0.639 |
| 4 | 0.139 (10/72) | 0.097 (7/72) | -0.042 |
| 8 | 0.000 (0/72) | 0.125 (9/72) | ++0.125 |

## Case 2: 12 hand-crafted composition queries at K ∈ {2, 4, 8}

Each query has two target answer fragments (from passages a and b). Score: hit_a, hit_b, both. **`rate_both`** is the headline metric — both answers retrieved in the same generation.

Adapter subset for K=4: [a, b] + 2 distractors (round-robin from remaining indices). K=8: all 8 adapters.

### Procedure A (independent training)

| K | rate_a | rate_b | rate_both |
|---:|---:|---:|---:|
| 2 | 0.722 | 0.750 | **0.472** |
| 4 | 0.194 | 0.028 | **0.000** |
| 8 | 0.000 | 0.056 | **0.000** |

### Procedure B (sequential training)

| K | rate_a | rate_b | rate_both |
|---:|---:|---:|---:|
| 2 | 0.250 | 0.389 | **0.028** |
| 4 | 0.278 | 0.472 | **0.056** |
| 8 | 0.111 | 0.194 | **0.056** |

### Side-by-side rate_both

| K | Procedure A | Procedure B | Δ (B - A) |
|---:|---:|---:|---:|
| 2 | 0.472 | 0.028 | -0.444 |
| 4 | 0.000 | 0.056 | ++0.056 |
| 8 | 0.000 | 0.056 | ++0.056 |

## Summary

Procedure A reproduces the documented pattern: clean K=2 composition (47% both-hit on hand-crafted compositional queries; 68% single-adapter retrieval at K=2), collapsing at K=4 (0% both-hit, 14% single-adapter retrieval) and K=8 (0% everywhere).

Procedure B fails to fix the K=2 ceiling. Worse, it breaks the K=1 baseline: when an adapter is loaded alone, retrieval drops to 42% (vs Procedure A's 88%). The sequential adapters are specific to the composition context they were trained in. When prior adapters aren't loaded (K=1), the adapter's contributions misapply to a base it wasn't designed for.

Procedure B's K=4 and K=8 retrieval rates are slightly *higher* than its K=1 — consistent with the interpretation that adding more adapters partially reconstructs the training context.

This is informative-negative for the K=2 problem. The interference between independently-trained LoRA matrices at K>2 is **not** addressable by training-time freezing alone. Alternative approaches — orthogonality constraints, gating, hierarchical adapters — should be tested next.

## Implementation notes / deviations

1. **Procedure B training trick:** rather than wrap the model with a rank-(k×128) LoRA holding k-1 frozen and 1 trainable, I bake adapters 0..k-1 into the wrapped Linear weights (`W += scaling * (A @ B).T`), reset LoRA to fresh state, and train the new adapter. This gives the same forward-pass context the spec describes ("adapter 2 sees the adapter-1-modified forward pass") with simpler bookkeeping. The canonical base is restored before evaluation.
2. **Hyperparameters:** Phase 47's recipe (rank 128, alpha 256, n_steps 150, HIGH_LR→BASE_LR StepLR halving). All 8 Procedure B adapters converged to greedy retrieval *in their training context*; no divergence behavior observed.
3. **Block-stacking for evaluation:** Phase 43's `stack_k_state_dicts` block-diagonal-concatenates the K rank-128 (A, B) pairs into a single rank-(K×128) (A, B). This makes the wrapped Linear's forward equivalent to summing the K LoRA contributions: `x @ A_stacked @ B_stacked = Σₖ x @ Aₖ @ Bₖ`.
4. **Composition queries (Case 2):** 12 hand-constructed Phase 43-style chained probes, each requiring content from two specific adapters. Listed verbatim in `queries.py`. The answers are unambiguous proper nouns or single common words ("Pirrip", "Estella", "blacksmith", "Drummle", "Provis", "Magwitch", "Wemmick", "Herbert").
5. **No hyperparameter search** for Procedure B was attempted. The spec explicitly allowed re-framing as "establish what hyperparameters sequential training needs" if convergence failed. Convergence didn't fail; the failure is at evaluation time, not training time. A different LR schedule wouldn't change the context-dependence.

## Wall-clock totals

| Stage | Wall |
|---|---:|
| Procedure B training (8 sequential adapters) | 21s |
| Case 1 evaluation (4 K-values × 2 procedures × 72 evals) | 195s |
| Case 2 evaluation (3 K-values × 2 procedures × 36 evals) | 129s |
| **Total** | **~344s (~6 min)** |
