# Phase 70 — Per-adapter verification head

Tests whether the entity-grain signal that fails in Phase 65's static L0/L5
representations can be recovered from **adapter-conditioned hidden states**
plus a small trained classifier per adapter. Open-vocab analog of Phase 69's
metadata check: requires no entity catalog, no string match, no NER.

## Headline

**`h_mean_L5` (mean-pooled L5 under the loaded adapter) reaches 90% balanced
accuracy on the open-vocab verification task** — within striking distance of
Phase 69's closed-vocab metadata recipe (100%) and well above the
representational floor Phase 65's pooling ablation suggested (62%).

| representation | pooled TPR@95spec | pooled OOD-FPR | pooled cross-FPR | pooled balanced |
|---|---|---|---|---|
| h_last_L5             | 32% | 4% | 5% | 63% |
| **h_mean_L5**         | **85%** | **7%** | **4%** | **90%** |
| h_last_minus_base_L5  | 40% | 6% | 5% | 68% |

(Per-adapter mean: each adapter calibrates its own threshold on its own
negatives, then evaluates on its own positives. Smaller numbers across the
board because per-adapter pos/neg counts are 3 / 77.)

| representation | per-adapter TPR@95spec | per-adapter OOD-FPR | per-adapter cross-FPR |
|---|---|---|---|
| h_last_L5             | 58% | 11% | 3% |
| **h_mean_L5**         | **82%** | **8%** | **4%** |
| h_last_minus_base_L5  | 57% | 9% | 4% |

The script's auto-verdict picked `h_last_L5` as "best" because it has the
lowest OOD-FPR — but TPR is 32%/58%, meaning the head rejects most positives
too. `h_mean_L5` dominates on balanced accuracy and is the deployment-ready
representation.

## Why mean-pool L5 won (and why I was wrong about delta)

Pre-committed prediction: `h_last_minus_base_L5` (the "what did the adapter
change about the representation" delta) would be the strongest signal, by
isolating the adapter-specific contribution. `h_mean_L5` would be weakest
because Phase 65's mean-pool result was the original entity-grain failure.

Both predictions inverted by the data. The mechanism that explains it:

- **Phase 65's mean-pool failure was about static representations.** With
  no adapter loaded (or all adapters indistinguishable in their effect on
  static features), the L5 mean-pool weights template structure heavily and
  entity weakly. Phase 65 was right about that.
- **But adapter-conditioned mean-pool is different.** When the adapter is
  loaded, every token's L5 representation is shaped by adapter-specific
  modifications to layers 4-5. Mean-pooling integrates this adapter-shaping
  signal across the entire sequence — far more total signal than any single-
  token slice (last_1 has only one token's adapter contribution).
- **The delta signal underperformed because the adapter's contribution isn't
  concentrated at the last token.** Subtracting last-token base from last-
  token adapter captures only the local change at position −1; the
  cross-sequence integration that mean-pool gets is missing.

The architectural takeaway: the entity-grain signal isn't in static
representations (Phase 65 confirmed) but is **in how the adapter processes
the query** — when integrated across the sequence. A 256-hidden MLP on
adapter-conditioned mean-pool L5 recovers ~96% within-library specificity.

## Pre-committed predictions vs measured

| representation | predicted hard-OOD FPR | measured per-adapter OOD FPR | predicted cross-FPR | measured per-adapter cross-FPR |
|---|---|---|---|---|
| h_mean_L5             | 60–85% | **8%** | 50–80% | **4%** |
| h_last_L5             | 30–60% | 11%    | 30–50% | 3%     |
| h_last_minus_base_L5  | 15–40% | 9%     | 20–40% | 4%     |

All three FPRs came in well under predicted ranges. h_mean_L5 was predicted
as the floor of the experiment and turned out to be the ceiling. The
prediction reasoning ("Phase 65 said mean-pool fails") missed that Phase 65
measured a static representation, while Phase 70 measures adapter-conditioned
representation — categorically different.

The prediction track record across this paper:
- Phase 65 specificity (Outcome A predicted): wrong, was D.
- Phase 65 length-match (C0b might shrink): wrong, separation widened.
- Phase 65 hard-OOD (C0b might break): right.
- Phase 65 OOD-aware W rescue (should give absolute separation): wrong.
- Phase 65 last-token (Outcome B predicted): right (Outcome C, edge of B).
- Phase 69 confidence signals (50–70% FP): wrong, all 88–100%.
- Phase 69 entity_match (85–95% accuracy): wrong direction (100% precision, 55% recall).
- Phase 70 h_last_minus_base best: wrong, h_mean_L5 best.

Six of eight predictions wrong. Worth treating my priors with skepticism.
The architectural reasoning ("adapter conditioning lives in the integrated
representation, not the local delta") only became visible after the data.

## What this implies for the paper

Phase 69's metadata check gave 100% balanced accuracy for closed-vocab
domains. Phase 70's `h_mean_L5` head gives 90% balanced accuracy for
**open-vocab** domains where you can't enumerate entities at route time. The
deployment claim now extends across two cases:

- **Closed-vocab** (entity registry exists): metadata entity match. Free at
  inference, 100% balanced accuracy. Phase 69 recipe.
- **Open-vocab** (no entity registry): per-adapter `h_mean_L5` MLP head. ~90%
  balanced accuracy at the cost of one extra forward pass through the loaded
  adapter (which already happens during generation) plus a 262K-param head
  per adapter (5.2M params for a 20-adapter library).

Combined story: HRS routing is template-grain by design. The verification
stage has two viable implementations depending on whether entity vocabulary
is enumerable. Together, they cover the deployment surface that the
"general-purpose adapter routing" framing was claiming.

The recruitment piece can now read closer to: "engram routing handles fast
template-grain selection; verification (metadata or learned head) handles
entity-grain check; deployment uses both stages." The open question shifts
from "is open-vocab verification possible?" (answered: yes, ~90%) to "can the
remaining 10pt gap be closed by larger heads, more training data, or richer
representations?" — engineering, not architecture.

## Honest caveats

1. **Small training sets per head.** 4 positives + 19 negatives per adapter.
   The 256-hidden MLP is heavily overparameterized for this. The 90%
   balanced accuracy on disjoint test queries (60 in-library + 20 hard-OOD,
   none seen during training) is real signal, but a larger training set
   would tighten estimates and might lift the ceiling.

2. **20-adapter library.** Cross-adapter discrimination at 4% FPR is
   measured against 19 same-template-different-entity sister adapters. At
   100s or 1000s of adapters, the cross-FPR could grow (more chances for a
   query to match some other adapter's pattern). Hasn't been tested here.

3. **Same passkey-style task family.** All 20 library adapters were trained
   on the same passkey-retrieval task with structured templates. Whether
   `h_mean_L5` heads work on adapters trained for genuinely diverse tasks
   (summarization, code, multilingual, etc.) is the next experiment.

4. **The prediction-record warning.** Six wrong out of eight on this paper's
   predictions. The 90% balanced accuracy result should be replicated with
   different seeds and a held-out adapter set before trusting it as the
   open-vocab story.

## Files

- `verification_head.json` — full per-adapter scores, per-representation summaries
- `head_calibration.png`   — score distribution histograms (3 representations)
- `verification_head_run.log` — stdout
- `experiments/identity_ae/phase70_verification_head.py` — script

## Open follow-ups

1. **Replicate with different seeds and held-out adapters.** The 90% number
   needs to be robust to splitting the 20 adapters into train/test.

2. **Scale test (cross-FPR at K=100, K=1000 adapters).** Does the head
   discrimination hold as the library grows?

3. **Heterogeneous task adapters.** Whether `h_mean_L5` works for adapters
   trained on different task families, not just passkey templates.

4. **Smaller heads / fewer training queries.** What's the minimum head
   capacity / training set size that still achieves the open-vocab signal?
   Defines deployment cost.
