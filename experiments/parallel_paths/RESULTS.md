# Parallel Paths k=3: Phase 0 gate — KILL NO HEADROOM

**Date:** 2026-05-04
**Status:** Stopped at Phase 0 per spec's gate criterion.

## Phase 0 result

For each of the 150 held-out Dickens-50 probes (50 adapters × 3 paraphrases),
the W projection (rank=128, trained Phase-47-style via InfoNCE) was applied
and the correct adapter's rank in the score ordering was recorded.

| Metric | Value |
|---|---|
| Probes | 150 |
| Top-1 correct | **150 / 150 = 1.000** |
| Top-3 correct | 150 / 150 = 1.000 |
| Top-5 correct | 150 / 150 = 1.000 |
| Headroom (top-3 catches what top-1 misses) | **0 / 150** |

**Rank distribution of correct adapter:** all 150 probes at rank 0.

Per fact type:

| fact type | n | top-1 | mean rank of correct |
|---|---|---|---|
| entity | 96 | 1.000 | 0.00 |
| numeric | 18 | 1.000 | 0.00 |
| place | 30 | 1.000 | 0.00 |
| relation | 6 | 1.000 | 0.00 |

The W projection is a perfect oracle on the held-out paraphrases for this
adapter library. There is no probe on which parallel paths over top-3 could
recover an answer that top-1 alone missed.

## Decision (per spec)

> "If top-1 is already 100%, kill the experiment — there's no headroom."

Phase 1 (parallel-paths inference implementation), Phase 2 (variant sweep),
and Phase 3 (analysis) were not run. The hypothesis is gated off by the
substrate.

## What the design questions resolve to

The spec listed four open design questions. Phase 0's outcome makes Q4
("are we expecting top-3 to often contain the correct adapter when top-1
doesn't?") answer "no, never" on this substrate. That moots Q1-Q3:
parallel-paths variants A/B/C all need at least some routing-error rate
to recover, and there isn't one.

## Conclusion and what would change the answer

The parallel-paths architecture is a sound idea for *substrates where
routing has measurable error*. None of the substrates currently in this
repo qualify. To re-target the experiment, one of:

- **Larger libraries** where adapters are more confusable (50 hand-crafted
  Dickens passages with hand-crafted held-out paraphrases is well-separated;
  500 or 5000 adapters might not be).
- **Harder paraphrase distribution** — train on a narrow paraphrase style,
  test on adversarial paraphrases that route to wrong adapters.
- **Multi-passage probes** — questions whose answer spans two adapters,
  where "the correct adapter" is ambiguous and top-3 actually represents
  routing uncertainty rather than routing error.
- **Different routing keys** — e.g., L0-mean alone (without the W
  projection), which has weaker discrimination and might route imperfectly.

Worth flagging: the prior k=2 cross-terms experiment on the SAME substrate
found a 27-point compositional gap (additive composition fails). Parallel
paths sidesteps that gap by keeping each path single-adapter, but the
gap-to-recover only exists when routing is imperfect — and on this
substrate it isn't.

## Files

- `run_phase0_routing_diagnostic.py` — the gate script
- `results/phase0_routing_diagnostic.json` — aggregates
- `results/phase0_routing_topk.csv` — per-probe ranks and scores
