# Parallel Paths Multi-Topic on Dickens-50: Phase 0 gate — STOP

**Date:** 2026-05-04
**Status:** Stopped at Phase 0 per spec's gate criterion (routing surfaces expected adapters in <50% of queries). Phase 1-4 not run.

## Substrate viability (Phase 0.1)

The 50 Dickens-50 passages are all from a single novel (Great Expectations), not multiple novels. Each adapter encodes one specific fact (a name, number, place, or relation) with surrounding character/scene context. Recurring entities span multiple adapters each (Pip across 12, Joe across 4, Magwitch across 5, Wemmick across 4, etc.).

This is a different shape than the spec's "Tiny Tim, Pip, Oliver" running example (one character per novel). For Dickens-50, the natural multi-topic queries are **multi-fact** queries that anchor each topic to a distinct fact-adapter. Detailed topology in `passage_topology.md`.

The substrate itself is viable for multi-fact queries — there are clearly distinct facts that can be combined into 3-topic queries. 15 such queries were constructed in `queries.json`, balanced across the spec's three types (list, comparison, synthesis).

## Routing diagnostic (Phase 0.3)

For each of the 15 multi-topic queries, the W projection's routing scores were computed against all 50 library L5 keys. The question: do all three expected adapters appear in the top-k for the query?

**Aggregate:**

| | All expected adapters in... |
|---|---|
| top-3 | **5/15 = 0.333** |
| top-5 | 7/15 = 0.467 |
| top-10 | 8/15 = 0.533 |

**Per query type:**

| Type | top-3 | top-5 | top-10 |
|---|---|---|---|
| list (n=5) | 0.40 | 0.60 | 0.60 |
| comparison (n=5) | 0.60 | 0.80 | **1.00** |
| **synthesis (n=5)** | **0.00** | **0.00** | **0.00** |

The synthesis category is catastrophic — across all 5 synthesis queries, the three expected adapters never all appear in even the top-10. Concrete examples of where synthesis queries land:

| qid | type | expected adapters | ranks of expected |
|---|---|---|---|
| 10 | synthesis | [1, 40, 16] | [0, 12, 13] |
| 11 | synthesis | [41, 17, 33] | [0, 18, 21] |
| 12 | synthesis | [5, 9, 25] | [0, 4, 17] |
| 13 | synthesis | [37, 32, 21] | [8, 14, 18] |
| 14 | synthesis | [0, 30, 31] | [0, 5, 20] |

In four of five synthesis queries, ONE expected adapter wins routing (rank 0) but the other two land at ranks ≥ 12. The framing tokens ("what do X, Y, Z have in common — they are all...") dominate the query's L0 mean and overwhelm the per-fact signal from the embedded sub-questions.

## Decision

Per spec gate criterion ("<50% of queries have expected adapters in top-k → routing doesn't surface multi-topic adapters reliably; stop and recommend query decomposition"), the experiment as specified does not run. Phase 1 (parallel-paths implementation), Phase 2 (variant sweep), Phase 3 (evaluation), and Phase 4 (analysis) were not started.

## What this means about the architecture

This result is informative beyond just "the gate failed." Three observations:

**1. The W projection is not designed for multi-topic queries.** It was trained via InfoNCE to map an L0-mean key from a single-topic paraphrase to the matching adapter's L5 mean-pool. The training distribution had no multi-topic queries, and the resulting projection treats a multi-topic query as a single blob whose mean tokens may or may not point at any one adapter — with no mechanism for the projection to "split attention" across multiple targets.

**2. Comparison queries route better than synthesis queries.** Comparison queries (60% top-3) work because their framing ("Compare A and B and C in terms of D") leaves the per-fact signal relatively intact in the L0 mean. Synthesis queries (0% top-3) fail because their framing adds significant non-fact tokens ("what do X, Y, Z have in common — they are all proper names...") that dilute or shift the mean.

**3. The natural fix is query decomposition.** If a multi-topic query were parsed into 3 sub-queries (one per topic), each sub-query would route correctly via the same W projection (because Phase 47 / Phase 0 of `parallel_paths/` showed top-1 routing = 100% on single-topic paraphrases). The 3 sub-queries would each select the correct adapter, and parallel paths over the union would have all 3 expected adapters available. This is the spec's named alternative architecture, deferred as future work.

## Why not just take the union of top-3 from each sub-query?

This is the natural follow-up architecture, but it requires query decomposition — parsing a multi-topic query into sub-queries. That's a separate research direction (LLM-based decomposition vs structural parsing vs trained decomposer), and the spec explicitly defers it: "query decomposition is a separate research direction."

The decomposition step itself can be implemented multiple ways with different trade-offs (latency, accuracy, generalization to novel query forms). Picking and validating a decomposer is its own scoped experiment, not a fix to be hacked into this one.

## Headline summary for the researcher

**Substrate (Dickens-50) supported multi-topic queries in principle** — there are 50 distinct facts that can be combined into well-formed 3-topic queries.

**W routing did NOT reliably surface the relevant adapters for multi-topic queries.** All-in-top-3 = 33% (15-query mean), with synthesis-style queries failing 100% of the time.

**No headline result on parallel paths plus synthesis** — Phase 1-4 not run because Phase 0's gate triggered. The architecture is theoretically clean (parallel paths sidesteps cross-terms by construction; the k2_crossterms result confirms additive composition fails so parallel paths is the right alternative), but the routing front-end as built can't feed it the right adapter sets for multi-topic queries.

**Most informative failure mode: synthesis-query routing (0/5).** The framing words of synthesis queries dominate the L0 mean and bury the per-fact signal. This isn't a noise issue — it's a structural mismatch between what the W projection was trained to do (route single-topic paraphrases) and what multi-topic queries demand (route to 3 distinct adapters simultaneously).

**Recommendation: defer to query decomposition + parallel paths as the next experiment.** The decomposition can be implemented multiple ways (LLM-based with a small parser, regex-based for simple list queries, or trained); each sub-query then routes correctly via the existing W projection (top-1 = 100% on single-topic paraphrases per the parallel_paths/ Phase 0 result). Parallel paths inference and synthesis pass would then operate on a properly-routed adapter set. Best to validate the decomposition step in isolation before re-attempting parallel paths multi-topic.

Worth flagging alongside the prior k2_crossterms result: the architecture has now hit two distinct walls on the same substrate.
- k=2 additive composition: fails by 27 pts; no spec-defined intervention closes the gap.
- Multi-topic routing for parallel paths: fails by structure; the W projection wasn't trained for it.

Both walls argue for the same conclusion: HRS at this scale is fundamentally a single-adapter-per-query architecture. Multi-topic retrieval requires an additional decomposition layer that hasn't been built yet.

## Files

- `passage_topology.md` — corpus structure
- `queries.json` — 15 multi-topic queries (5 list, 5 comparison, 5 synthesis)
- `run_phase0_routing.py` — gate script
- `results/phase0_summary.json` — aggregates and decision
- `results/phase0_routing.csv` — per-query top-k details
- `results/phase0_run.log` — run log
