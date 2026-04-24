# Head Pruning Ablation — Specification

## Hypothesis

A small number of attention heads are responsible for the passkey retrieval circuit. Most heads are redundant on both LM and passkey tasks. Structured pruning by ablation-ranked head importance should:

1. Identify which specific heads implement retrieval.
2. Remove at least 50% of heads with short fine-tuning and preserve both metrics.
3. Produce a sharp passkey cliff at the point where retrieval heads are removed — distinct from the gradual MLP pruning curve.

Unlike magnitude pruning, head pruning is *structured* — zeroing a head also removes its compute from dense matmuls, so savings translate directly to FLOPs on standard hardware.

## Setup

Start from the **MHA baseline checkpoints** already trained in `experiments/pruning/checkpoints/` (passkey: 20K steps, exact=1.000; LM: 5K steps, val_ppl=4.92). Do not retrain from scratch.

Model has 4 layers × 4 heads = 16 total heads. Each head is indexed by (layer, head_idx).

Because the LM and passkey baselines are separate models (different vocabularies), per-head importance is measured on whichever baseline the metric comes from: PPL importance uses the LM baseline; passkey importance uses the passkey baseline. A "head at position (l, h)" is a structural position, not a single learned function.

## Phase 1: Importance Ranking

For each of the 16 heads individually:

1. Load the baseline checkpoint.
2. Zero that single head (its slice of W_Q, W_K, W_V projection rows and W_O input columns).
3. Evaluate val PPL (LM model) and passkey accuracy (passkey model) on the pruned model (no fine-tuning).
4. Record `importance = pruned_score_delta` for both metrics separately.

## Phase 2: Ordered Pruning Sweeps

Three orderings, all over heads pruned ∈ {0, 2, 4, 6, 8, 10, 12, 14}:

1. Least-passkey-important first (good greedy).
2. Random order (fixed seed).
3. Most-passkey-important first (adversarial).

At each count: with and without 500-step fine-tuning.

## Phase 3: Composition (stretch)

Take the best Phase 2 head-pruned config. On that pruned+FT'd model, apply 90% MLP magnitude pruning (safe regime from earlier experiment) + FT. Report composed compression.
