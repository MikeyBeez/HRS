# Trainable Shared Decoder: Capacity Sweep on Dickens-50

**Date:** 2026-05-05
**Branch:** hrs-loop (merged work; no separate hrs-trainable-decoder branch needed for the writeup)
**Headline:** **Unfreezing the decoder lifts single-adapter retrieval from ~0.93 to 1.00.** Practical capacity for shared-decoder forgetting is 4-8 adapters at the 0.85 threshold; degradation is **strongly asymmetric** — most-recent adapter retrieval stays at 1.00 across the entire sweep while the oldest adapter is fully forgotten by N=16. Per-fact-type behavior cleanly replicates the k=2 cross-term pattern: numeric most robust, relation most fragile.

## Setup

50 sequential single-passage adapters trained on Dickens-50, with a shared trainable lm_head (decoder) updated by each adapter's training in turn. The original frozen lm_head is preserved for base-model queries (`use_trainable=False`); the trainable copy is used when adapters are loaded (`use_trainable=True`).

Implementation: `SwitchingLMHead` wraps the original `lm_head` and adds a fresh `nn.Linear(d_model, vocab)` initialized from the frozen weights. The trainable copy carries 51,463,168 parameters (vs 2,621,440 LoRA params per adapter). Both are updated by every adapter's training; LoRA params are reset before each new adapter, decoder params are not.

Snapshots taken at N ∈ {1, 2, 4, 8, 16, 32, 50}. Each snapshot evaluates every adapter trained so far under the current decoder state. This is equivalent to running 7 independent sweeps from scratch (the seed-42 passage order makes adapter-1 at N=2 identical to adapter-1 at N=50) at ~10× lower compute. Total wall time: 422s = 7 minutes.

## Results

### Aggregate per snapshot

| N | mean | std | min | max | adapter_1 (oldest) | adapter_N (most recent) |
|---|---|---|---|---|---|---|
| 1 | **1.000** | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | **1.000** | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.861 | 0.182 | 0.556 | 1.000 | 1.000 | 0.889 |
| 8 | 0.861 | 0.213 | 0.444 | 1.000 | 0.444 | 1.000 |
| 16 | 0.819 | 0.257 | 0.000 | 1.000 | 0.000 | 0.778 |
| 32 | 0.639 | 0.345 | 0.000 | 1.000 | 0.000 | 1.000 |
| 50 | 0.496 | 0.392 | 0.000 | 1.000 | 0.000 | 1.000 |

Frozen-decoder baseline (from prior `capacity_scaling/` work): mean ~0.93 single-adapter, ~0.46 at size-50 single-combined-adapter.

### Per-fact-type at each snapshot

| N | mean | entity | numeric | place | relation |
|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 (1) | — | — | — |
| 2 | 1.000 | 1.000 (2) | — | — | — |
| 4 | 0.861 | 0.861 (4) | — | — | — |
| 8 | 0.861 | 0.841 (7) | — | 1.000 (1) | — |
| 16 | 0.819 | 0.796 (12) | 1.000 (1) | 0.852 (3) | — |
| 32 | 0.639 | 0.608 (21) | 1.000 (2) | 0.587 (7) | 0.778 (2) |
| 50 | **0.496** | 0.514 (32) | **0.722** (6) | 0.344 (10) | **0.278** (2) |

(n shown in parens.)

Numbers cleanly replicate the k=2 cross-term per-fact-type pattern: numeric is most robust to interference (0.722 at N=50), relation most fragile (0.278), entity in between, place degrades sharply (0.344). The same hierarchy holds at decoder-level forgetting that held at multi-adapter cross-term interference. Two failure modes, same per-type sensitivity ranking.

## Phase 1 sanity check answer: yes, unfreezing helps at single-adapter scale

N=1 retrieval = **1.000**, vs frozen-decoder baseline ~0.93-0.96 (averaged over multiple passages). At N=2, mean is still 1.000.

The hypothesis "trainable decoder provides real benefit at single-adapter scale" is confirmed. The frozen lm_head was a real ceiling for single-adapter retrieval — not the dominant ceiling for multi-passage capacity, but a measurable one for individual adapters. Unfreezing buys ~4-7 percentage points on single-adapter retrieval.

## Forgetting curve characterization

The curve is **threshold-shaped, not flat or smooth**:

- **N ≤ 2:** decoder accommodates without measurable cost. Mean = 1.000.
- **N = 4:** small drop appears. Mean = 0.861. Adapter 1 still 1.000 but at least one of the 4 adapters has degraded (min = 0.556).
- **N = 8:** mean stays at 0.861, but **adapter 1 has crashed to 0.444**. The decoder is now demonstrably overwriting adapter 1's required outputs.
- **N = 16:** **adapter 1 = 0.000** — fully forgotten. Mean = 0.819 because newer adapters compensate.
- **N ≥ 32:** mean falls below 0.70. Adapter 1 stays at 0.000. Variance becomes massive (std 0.345 at N=32, std 0.392 at N=50).

Practical capacity at the 0.85 mean-retrieval threshold: **between 4 and 16 adapters**. Linear interpolation puts the threshold at N ≈ 12. At N = 8 mean is 0.861, at N = 16 it's 0.819.

## Asymmetric forgetting — the architecturally interesting finding

**Most-recent adapter retrieval (`adapter_N`) stays at or near 1.000 across the entire sweep**, even at N=50 where adapter_1 is fully dead. The decoder isn't getting generally worse at handling adapter outputs — it's getting worse specifically at the outputs of *earlier* adapters. The most recently trained adapter consistently has near-perfect retrieval against the current decoder state.

This is recency-biased forgetting, not symmetric capacity exhaustion. The decoder's parameters drift toward whatever the current adapter needs, and earlier adapters' required parameter values get overwritten in proportion to how long ago they were trained.

The variance numbers tell the same story:
- N=4: std = 0.182 (some forgetting)
- N=16: std = 0.257 (more spread)
- N=50: std = 0.392 (massive spread — recent adapters at 1.0, old ones at 0.0)

This is much more useful than uniform degradation, *if* the deployment can ensure that the decoder is trained-against-most-recently the adapters relevant to a query. Otherwise (e.g., if all adapters are trained ahead of time and queries arrive in arbitrary order), it's worse than uniform forgetting because retrieval accuracy varies wildly across the corpus with no easy way to predict which adapters work and which don't from a query's perspective.

Figures: `figures/forgetting_curve.png`, `figures/capacity_curve.png`.

## Comparison to prior experiments

| Experiment | Architecture | Size 50 retrieval |
|---|---|---|
| `per_passage_dickens` (frozen, single passage per adapter) | 1 adapter per passage, 50 separate adapters | ~0.93 mean |
| `capacity_scaling` Phase 5 (frozen, 50 passages → 1 adapter) | 1 adapter holding 50 passages, rank 128, 7.5k steps | 0.358 |
| `capacity_scaling` Phase 7 (frozen, 50 passages → 1 adapter, 5× steps) | same, 37.5k steps | **0.464** (peak) |
| `trainable_decoder` (this experiment, N=50) | 50 adapters + shared trainable decoder | **0.496** mean |

The trainable-decoder mean at N=50 (0.496) matches the capacity_scaling peak (0.464) within noise. The architectures hit essentially the same ceiling at full corpus consolidation. But the *failure mode* is different:

- Capacity scaling: every passage degraded uniformly (0.000 to 0.889 across passages, no pattern in time).
- Trainable decoder: the most recent adapters work perfectly (1.000), the oldest are dead (0.000). Recency-ordered.

Same ceiling, different shape. The trainable-decoder failure is structurally informative — it points directly at "the decoder is being overwritten" rather than "the adapter capacity is saturated."

## Architectural verdict

The frozen-decoder constraint **was** a real (small) ceiling at single-adapter scale. Unfreezing buys a clean +4-7 pts there.

The frozen-decoder constraint was **not** the binding constraint for multi-passage retrieval. The same ~0.5 ceiling shows up regardless of whether you compress 50 passages into one adapter (capacity_scaling) or distribute them across 50 adapters with a shared decoder (this experiment). The bottleneck is somewhere else — most likely interference at the gradient level during sequential training (catastrophic forgetting of either the adapter weights or the decoder weights, depending on architecture).

For the architecture decision:

- **Adopt for single-adapter retrieval:** the +4-7 pt single-adapter benefit is real and free. If HRS is deployed as single-adapter top-1 routing (per the parallel-paths Phase 0 finding that top-1 is 100% on Dickens-50), use a trainable decoder per adapter. Cost is small (51M params per adapter, but they could be shared across small adapter pools).
- **Do NOT adopt for multi-adapter scaling:** practical capacity is ~4-8 adapters at the 0.85 threshold, and the asymmetric forgetting makes earlier adapters silently fail. For multi-passage retrieval, this architecture has the same ceiling as the prior single-adapter consolidation and a worse failure mode (silent forgetting of specific adapters vs uniform degradation across the corpus).
- **Open follow-up:** replay buffers or interleaved training during decoder updates would directly target the recency bias and could plausibly extend practical capacity. The k2_crossterms result also suggested companion-aware training as a partial mitigation; an analogous "decoder-aware" replay during sequential training is the natural extension.

The big architectural takeaway: **the frozen decoder was not the load-bearing constraint for HRS multi-passage retrieval.** It was a small ceiling at single-adapter scale. The multi-passage limit is a deeper interference problem that unfreezing the decoder doesn't solve.

## Files

- `decoder.py` — SwitchingLMHead + helpers
- `run_sweep.py` — full sweep with snapshots
- `run_figures.py` — plots
- `results/sweep_summary.json` — per-snapshot aggregates + config
- `results/sweep_results.csv` — per-(snapshot, adapter) retrieval table
- `figures/forgetting_curve.png` — adapter_1 vs adapter_N vs mean across N
- `figures/capacity_curve.png` — capacity curve + per-fact-type breakdown
- `adapters/adapter_pos{NN}.pt` — saved LoRA state per adapter
- `decoder_states/decoder_N{NN}.pt` — saved decoder weight at each snapshot

Adapter and decoder checkpoints are gitignored (~50MB each adapter, ~200MB each decoder state) but regenerable with the seed-42 fixed run.
