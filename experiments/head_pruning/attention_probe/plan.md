# L0 Attention-Pattern Probe — Specification

## Hypothesis

L0 retrieval heads implement the "find MARKER" operation by computing attention patterns that concentrate mass on or near the MARKER token. If visible directly in the attention matrices:
- L0 retrieval heads attend strongly to MARKER-adjacent positions.
- L0 non-retrieval heads show generic local or positional attention.
- L2 retrieval heads show a *different* pattern — composing from positions *after* MARKER rather than on MARKER itself.
- L3 heads show less-distinctive individual patterns (redundant-circuit finding).

## Setup

Baseline MHA checkpoint `experiments/pruning/checkpoints/mha_passkey.pt` (seed 0). No retraining.

## Input batch

200 passkey examples, context length 256, varied MARKER position. Per example record:
- `marker_pos` — MARKER token index.
- `passkey_range` — passkey digit indices.
- `query_pos` — QUERY token index (first position where the model must emit an answer token).

## Phase A — Extraction
Monkey-patch MHA forward to cache softmax'd attention. Run batch, save `[N, L, H, T, T]` tensor.

## Phase B — Metrics per head, averaged across examples

1. attention-to-marker at query position
2. attention-to-passkey at query position
3. attention-to-post-marker at query position (positions immediately after MARKER)
4. attention entropy at query position
5. position-of-max attention from query position (as offset from MARKER)

## Phase C — Three figures

- Fig 1: 4×4 gestalt heatmap grid, averaged attention across examples.
- Fig 2: same grid for one cherry-picked successful example (MARKER mid-context).
- Fig 3: retrieval heads (L0 H2, L0 H3, L2 H0) zoom + marginal bars (to-marker vs to-passkey vs elsewhere).

## Phase D — Hypothesis-specific checks

- Retrieval vs non-retrieval head contrast on attention-to-marker (Mann-Whitney U).
- L0 H2/H3 vs L2 H0 — do they attend to different places (marker vs post-marker)?
