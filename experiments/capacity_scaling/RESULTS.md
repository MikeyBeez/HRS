# Adapter Capacity Scaling on Dickens-50

**Date:** 2026-05-04
**Headline:** Rank-128 LoRA adapter capacity is **decisively binding** for multi-passage content. Retrieval falls from 0.956 (size 1) to 0.667 (size 5) to 0.378 (size 25) and saturates around 0.36 above size 25. The on-demand combined-adapter architecture is viable only for very small N (≤5 passages) at this rank — and would need substantially more rank to scale.

## Phase 1 — sanity baseline

5 fresh single-passage adapters (passages 2, 16, 24, 26, 48), each at rank-128 / 150 steps:

| passage | retrieval | train wall (s) |
|---|---|---|
| 2 | 1.000 | 3 |
| 16 | 1.000 | 3 |
| 24 | 1.000 | 3 |
| 26 | 1.000 | 3 |
| 48 | 0.778 | 3 |
| **mean** | **0.956** | 3 |

Mean retrieval 0.956 is consistent with the published Dickens-50 baseline (0.929 averaged over 50 adapters). Pipeline is healthy. Proceed.

## Phase 2 — scaling sweep

Adapters trained at sizes 1, 5, 25, 50, with steps scaled linearly (150 per passage to keep "samples per source" constant). Selection seed 42 for sizes 5 and 25; size 50 uses all 50.

| size | train_s | n_steps | final_loss | overall | pp_mean | pp_min | pp_max | entity | numeric | place | relation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 150 | — | **0.956** | 0.956 | 0.778 | 1.000 | 0.956 | — | — | — |
| 5 | 13 | 750 | 0.20 | **0.667** | 0.667 | 0.444 | 0.889 | 0.667 | — | — | — |
| 25 | 62 | 3750 | 0.36 | **0.378** | 0.378 | 0.111 | 0.889 | 0.373 | 0.667 | 0.370 | 0.222 |
| 50 | 123 | 7500 | 0.34 | **0.358** | 0.358 | 0.000 | 0.889 | 0.316 | 0.296 | 0.522 | 0.389 |

(Sizes 1 and 5 used the size-1 Phase 1 averaged across 5 random passages and the seed-42 selection for size 5, which happened to pick 5 entity-type passages. That's why per-fact-type cells are blank for non-entity at sizes 1 and 5.)

The size-1 entry uses the Phase 1 mean across 5 random passages. The size-1 adapter trained on passage 0 specifically scored 0.556, but passage 0 is an unusually hard probe (the k2_crossterms Phase 1 also got 0.556 on adapter 0); the 5-passage Phase 1 mean (0.956) is the better representative.

Figures: `figures/retrieval_vs_size.png`, `figures/training_time_vs_size.png`.

## Phase 3 — scaling regime

The pattern is **smooth monotonic degradation that saturates above size 25**:

```
0.956 → 0.667 → 0.378 → 0.358
   ↘29pts ↘29pts  ↘2pts (saturation)
```

Per the spec's decision tree:
- Not flat (drops 60 pts from size 1 to size 50)
- Not a smooth-but-acceptable degradation (falls below 0.70 by size 5)
- Falls below 0.70 between sizes 1 and 5
- The 25→50 plateau (~0.36) suggests an information-theoretic floor: at this rank, adding more content past size 25 doesn't make things meaningfully worse because the adapter is already saturated.

Effective threshold: rank-128 holds **~5 passages with 0.67 retrieval, ~3-4 passages with passable (~0.80-0.85) retrieval, and ~1 passage with full ~0.96 retrieval**. The published per-passage Dickens-50 architecture uses rank-128 per passage for a reason.

Per-passage variance is striking. At size 50:
- min = 0.000 (one or more passages completely lost)
- max = 0.889 (one or more passages near baseline)

Some passages survive co-residence well; others are wiped out. The training procedure isn't allocating capacity uniformly.

## Phase 4 — training time

Training time scales linearly with content size (steps scale linearly):

| size | train_s |
|---|---|
| 1 | 3 |
| 5 | 13 |
| 25 | 62 |
| 50 | 123 |

≈2.5s per passage. A 10-passage on-demand combination would take ~25s; 50 passages takes ~2 min. Latency is not the bottleneck. **Capacity is the bottleneck.**

## Phase 5 — architectural verdict

**On-demand combined-adapter architecture is NOT viable at rank-128 for typical multi-topic queries.**

The spec described 3-passage queries as the natural multi-topic use case (Tiny Tim / Pip / Oliver style). At size 3 we'd interpolate to maybe 0.75-0.80 retrieval based on the curve — barely above the 0.70 threshold. At sizes 5+, retrieval is firmly below threshold.

For the architecture to work as the spec envisions, **rank must scale with content size**. The user's intuition during the run — "this suggests we need a larger adapter for more training data" — is exactly what the data shows.

Per-fact-type behavior is roughly uniform across the four types at size 50 (all in the 0.30-0.52 range), with no type having a clear capacity advantage. The numeric and entity types degrade most sharply (numeric 0.963 → 0.296 at size 50; entity 0.956 → 0.316). Place and relation are slightly more robust at size 50 (0.522 and 0.389 respectively), but the absolute numbers are still well below threshold.

## What the user observed mid-run

The user noted during the run: *"This suggests we need a larger adapter for more training data."* That's the natural hypothesis from this data. The follow-up experiment is rank scaling: train adapters at rank ∈ {128, 256, 512, 1024} on the size-50 content and see whether retrieval recovers. The capacity-scaling-with-rank curve is the key follow-up:

- If rank-512 at size 50 recovers to ≥0.85: the architecture works with adequate rank, and the rank-vs-content relationship can be characterized.
- If rank-1024 still doesn't recover: capacity isn't the only limit — there's also some interference between passages that more rank doesn't fix.

This is a natural Phase 6 / next experiment that the spec doesn't cover but the data clearly motivates.

## Recommendation

For the spec's question (is on-demand combined-adapter viable for multi-topic queries on Dickens-50): **no, not at rank-128.** The capacity ceiling is binding by ~5 passages and well-binding at typical multi-topic query sizes (5-10 passages).

For the architecture overall:
- **Path A (rank scaling):** test whether rank ∝ content_size keeps retrieval flat. If yes, the architecture is just under-provisioned at rank-128 for multi-passage content.
- **Path B (revisit query decomposition):** the parallel-paths-multi-topic experiment recommended query decomposition + per-sub-query routing. That path doesn't depend on capacity at all and routes to single-passage adapters which work at 0.96.
- **Path C (current architecture, single-passage only):** accept HRS at this rank as fundamentally a single-passage retrieval system. Multi-topic retrieval requires either decomposition (Path B) or higher rank (Path A).

Path A is the most direct test of the user's mid-run hypothesis. Path B is the lower-risk path because each component (decomposition, per-sub-query routing) is independently testable. Path C is the "do nothing" option and aligns with the architecture's already-demonstrated working regime.

## Files

- `combined_adapter.py` — train rank-128 LoRA on N passages of training data
- `eval_combined.py` — evaluate retrieval per constituent passage
- `run_phase1_baseline.py` — Phase 1 sanity (5 single-passage adapters)
- `run_phase5_aggregate.py` — table + figures + decision
- `results/phase1_baseline_summary.json` — Phase 1 sanity
- `results/eval_size_{01,05,25,50}.json` — per-size eval results
- `results/eval_phase1_p{NNN}.json` — per-passage Phase 1 evals
- `results/phase5_summary.{csv,json}` — aggregate
- `figures/retrieval_vs_size.png`, `figures/training_time_vs_size.png`

Adapters in `adapters/size_{01,05,25,50}.pt` and `adapters/phase1_p{NNN}.pt` are gitignored (~50MB each) but regeneratable.
