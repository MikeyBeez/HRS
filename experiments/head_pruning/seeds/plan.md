# Seed Transfer Test — Specification

## Hypothesis

The retrieval heads found in the baseline (L0 H2, L0 H3, L2 H0) are the *positions* the architecture naturally uses for retrieval, not an accident of the specific random initialization. Training the same architecture from different seeds should place retrieval at the same head addresses. If retrieval lands in different positions across seeds, the function is consistent but the location is arbitrary.

## Setup

Train 3 fresh models from scratch with seeds 1, 2, 3 using **identical** architecture, data, hyperparameters, and step counts as the existing baseline. Only the random seed differs.

The existing baseline (`experiments/pruning/checkpoints/mha_passkey.pt`, `mha_lm.pt`, trained with seed 0) is treated as "seed 0" for comparison.

## Procedure

For each seed ∈ {1, 2, 3}:
1. Train LM (5000 steps) + passkey (20000 steps) baselines with that seed.
2. Save to `experiments/head_pruning/seeds/seed_{N}/checkpoints/`.
3. Verify passkey exact ≥ 0.95 and val PPL ≤ 5.5.
4. Run single-head ablation importance ranking (Phase 1 from `rank_heads.py`).
5. Save `head_importance.json` + `importance_heatmap.png` per seed.

## Analysis

- **Top-3 retrieval heads per seed** (plus any head with Δpasskey > 0.5 above that).
- **Overlap metrics**: exact position overlap with baseline, layer-only overlap, count consistency.
- **2×2 grid of heatmaps** — one per seed including baseline — so hot cells are visually comparable.

## Expected Outcomes

1. Exact position match across seeds → architecture has natural retrieval slots (strong finding).
2. Same layers, different head indices → retrieval is layer-localized but head-within-layer is interchangeable.
3. Different layers → retrieval is allocated wherever gradient flow favors it; positions are incidental.

Mixed patterns are also informative.
