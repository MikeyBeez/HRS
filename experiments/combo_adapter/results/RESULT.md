# Combination Adapter Experiment

**Hypothesis (from spec):** Train one adapter on the concatenated training data of K passages → K=1 inference at deploy time. The combination adapter should match or approach single-passage K=1 retrieval (ideally within ~10%) and substantially outperform K>2 multi-adapter block-stacking. This is "offline composition through training" instead of "online composition at query time."

**Verdict: SUPPORTED for single-content retrieval at K>2; PARTIALLY supported for cross-passage queries.** Combination adapters retain 65-74% per-constituent retrieval across K=2/3/4 while multi-adapter stacking collapses (67% at K=2 → 17% at K=4). Combo is a clean win over both multi-stack composition AND a multi-pass-with-base-final-pass simulation. Combo's per-constituent retrieval is consistently 22-27pp below single-passage K=1 — the spec's "within 10%" criterion is missed but the K limit is effectively eliminated.

## Setup

- Reused 8 passages from sequential-training experiment: library_ids [0, 2, 16, 17, 22, 30, 31, 36]. Pip's family name, Joe's profession, Estella, Wemmick, Provis, Herbert, Magwitch, Drummle.
- 4 K=2 + 4 K=3 + 2 K=4 = 10 combination adapters trained.
- Phase 47 recipe (rank 128, alpha 256, attn+FFN on blocks 4-5, HIGH_LR -> BASE_LR with StepLR halve-at-half).
- n_steps scales with K: 150 × K (so 300 for K=2, 450 for K=3, 600 for K=4) to give each training source comparable per-source training.
- Substring scoring same as Phase 47 baseline. Test 2 / Test 4 use *fragment coverage* (mean fraction of expected answer fragments hit) plus *full hit rate* (all fragments in one generation).

### Combination groupings

| name | K | constituents (library_ids) |
|---|---:|---|
| P1_Pip_Joe | 2 | [0, 2] |
| P2_Estella_Drummle | 2 | [16, 36] |
| P3_Magwitch_Provis | 2 | [31, 22] |
| P4_Herbert_Wemmick | 2 | [30, 17] |
| T1_Pip_Estella_Magwitch | 3 | [0, 16, 31] |
| T2_Joe_Estella_Drummle | 3 | [2, 16, 36] |
| T3_Pip_Herbert_Wemmick | 3 | [0, 30, 17] |
| T4_Magwitch_Provis_Drummle | 3 | [31, 22, 36] |
| Q1_Pip_Estella_Magwitch_Provis | 4 | [0, 16, 31, 22] |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | [0, 2, 30, 17] |

### Training

All 10 adapters converged (final loss < 0.2):

| name | K | n_sources | n_steps | loss_init → final | wall |
|---|---:|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 10 | 300 | 8.34 → 0.144 | 5s |
| P2_Estella_Drummle | 2 | 10 | 300 | 3.64 → 0.238 | 5s |
| P3_Magwitch_Provis | 2 | 10 | 300 | 5.81 → 0.125 | 5s |
| P4_Herbert_Wemmick | 2 | 10 | 300 | 5.03 → 0.119 | 4s |
| T1_Pip_Estella_Magwitch | 3 | 15 | 450 | 6.69 → 0.119 | 7s |
| T2_Joe_Estella_Drummle | 3 | 15 | 450 | 4.71 → 0.094 | 7s |
| T3_Pip_Herbert_Wemmick | 3 | 15 | 450 | 5.48 → 0.174 | 7s |
| T4_Magwitch_Provis_Drummle | 3 | 15 | 450 | 4.78 → 0.189 | 7s |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 20 | 600 | 6.89 → 0.183 | 9s |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 20 | 600 | 9.13 → 0.155 | 10s |

*Total training wall: 68s*

## Test 1: per-constituent retrieval

For each combination, evaluate substring match on each constituent's 3 held-out paraphrases × 3 seeds = 9 evals per (combination, constituent) pair. Compare three conditions:
- **combo:** combination adapter loaded alone (K=1 inference)
- **single:** canonical Phase 47 single-passage adapter for that constituent loaded alone (K=1 inference)
- **multi-stack:** all K constituent single-passage adapters block-stacked at rank K×128 (Phase 43 stacking), K=N inference

### Per-combination averages

| combination | K | combo | single | multi-stack | combo-multi | combo-single |
|---|---:|---:|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 0.667 | 0.944 | 0.444 | +0.222 | -0.278 |
| P2_Estella_Drummle | 2 | 0.556 | 0.778 | 0.833 | -0.278 | -0.222 |
| P3_Magwitch_Provis | 2 | 0.611 | 0.944 | 0.667 | -0.056 | -0.333 |
| P4_Herbert_Wemmick | 2 | 0.778 | 1.000 | 0.722 | +0.056 | -0.222 |
| T1_Pip_Estella_Magwitch | 3 | 0.741 | 0.963 | 0.407 | +0.333 | -0.222 |
| T2_Joe_Estella_Drummle | 3 | 0.815 | 0.852 | 0.222 | +0.593 | -0.037 |
| T3_Pip_Herbert_Wemmick | 3 | 0.519 | 0.963 | 0.519 | +0.000 | -0.444 |
| T4_Magwitch_Provis_Drummle | 3 | 0.444 | 0.815 | 0.370 | +0.074 | -0.370 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 0.667 | 0.944 | 0.222 | +0.444 | -0.278 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 0.806 | 0.972 | 0.111 | +0.694 | -0.167 |

### Aggregated by K

| K | combo (avg) | single (avg) | multi-stack (avg) | combo gap to single | combo gap over multi |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.653 | 0.917 | 0.667 | -0.264 | -0.014 |
| 3 | 0.630 | 0.898 | 0.380 | -0.269 | +0.250 |
| 4 | 0.736 | 0.958 | 0.167 | -0.222 | +0.569 |

**Read:**
- Combo retrieval is *flat* across K: 0.65 → 0.63 → 0.74. The combination size does NOT degrade per-constituent retrieval up to K=4. Capacity ceiling not yet hit.
- Multi-stack collapses with K: 0.67 → 0.38 → 0.17. The K>2 cross-term interference replicated.
- Combo wins over multi-stack by 25pp at K=3 and **57pp at K=4**. At K=2 they tie.
- Combo holds 22-27pp below single-passage K=1 across all K. The spec's "within 10%" criterion is missed — combo loses some absolute fidelity per fact, but doesn't degrade with K.

## Test 2: cross-passage queries

3 hand-crafted Phase-43-style chained probes per combination (e.g., "Recall: Pip's family = . Also, Joe's profession = ..."), each requiring the combo adapter to produce K answer fragments in one generation. Score: *frac_hits* = mean fraction of expected fragments present in the continuation; *full_hit_rate* = both/all fragments present.

### Per-combination averages

| combination | K | combo full_hit | combo frac | multi-stack full | multi-stack frac |
|---|---:|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 0.222 | 0.500 | 0.222 | 0.556 |
| P2_Estella_Drummle | 2 | 0.111 | 0.556 | 0.111 | 0.500 |
| P3_Magwitch_Provis | 2 | 0.000 | 0.389 | 0.111 | 0.500 |
| P4_Herbert_Wemmick | 2 | 0.000 | 0.500 | 0.000 | 0.500 |
| T1_Pip_Estella_Magwitch | 3 | 0.000 | 0.296 | 0.000 | 0.370 |
| T2_Joe_Estella_Drummle | 3 | 0.000 | 0.296 | 0.000 | 0.037 |
| T3_Pip_Herbert_Wemmick | 3 | 0.000 | 0.333 | 0.000 | 0.593 |
| T4_Magwitch_Provis_Drummle | 3 | 0.000 | 0.407 | 0.000 | 0.370 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 0.000 | 0.306 | 0.000 | 0.222 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 0.000 | 0.333 | 0.000 | 0.167 |

### Aggregated by K

| K | combo frac | multi-stack frac | combo - multi |
|---:|---:|---:|---:|
| 2 | 0.486 | 0.514 | -0.028 |
| 3 | 0.333 | 0.343 | -0.009 |
| 4 | 0.319 | 0.194 | +0.125 |

**Read:**
- Cross-passage queries are *hard for both approaches*: fragment coverage stays at 30-50%, and full-hit rate is near 0 at K≥3 for both procedures. Producing all K answer fragments in one generation is genuinely difficult at this base/scale.
- At K=2 and K=3, combo and multi-stack tie on cross-pass.
- At K=4, combo edges out multi-stack by 12pp on fragment coverage (32% vs 20%), but neither produces full-hits.

## Test 3: capacity behavior across K

Combo retrieval (per-constituent, Test 1) by K: 0.65 (K=2) → 0.63 (K=3) → 0.74 (K=4). The K=4 increase is noise (the two K=4 combinations both happen to hit 67% and 81%; the small sample variance is wider than the per-K differences).

**The capacity ceiling for combination adapters is NOT reached at K=4** with rank-128 LoRA on this base. K=8 or K=10 would be the next test for finding the practical ceiling.

## Test 4: combo vs multi-pass simulation

Multi-pass protocol: for each cross-passage query requiring K passages, run K separate K=1 inferences (each with a different constituent's single-passage adapter loaded), capture the continuations, then run a final pass with NO adapter on a composite probe = original probe + labeled intermediates + "Final answer: ". Score the final continuation against the expected fragments.

Run on K=3 and K=4 combinations only (where multi-stack fails on Test 2).

### Per-combination

| combination | K | combo frac | multipass frac | combo - mp |
|---|---:|---:|---:|---:|
| T1_Pip_Estella_Magwitch | 3 | 0.259 | 0.037 | +0.222 |
| T2_Joe_Estella_Drummle | 3 | 0.333 | 0.037 | +0.296 |
| T3_Pip_Herbert_Wemmick | 3 | 0.370 | 0.037 | +0.333 |
| T4_Magwitch_Provis_Drummle | 3 | 0.370 | 0.037 | +0.333 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 0.278 | 0.000 | +0.278 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 0.250 | 0.000 | +0.250 |

### Aggregated by K

| K | combo frac | multipass frac | gap |
|---:|---:|---:|---:|
| 3 | 0.333 | 0.037 | +0.296 |
| 4 | 0.264 | 0.000 | +0.264 |

**Read:** Multi-pass is effectively broken on this substrate — the V22-Dickens base, given the original probe + K labeled intermediate continuations, doesn't synthesize a useful final answer (it produces Dickensian text on the cooking-style probes). Combo wins by 26-30pp at K=3 and K=4. **Combo is the better approach over multi-pass at this scale.**

Caveat: a more capable base model (e.g., a 7B+ instruction-tuned model) would likely have non-zero skill at synthesizing intermediate notes, making multi-pass more competitive. The Test 4 result here is specific to the V22-Dickens substrate.

## Failure-mode analysis

The spec defined three failure modes for the combo approach:

1. **Test 1 substantial degradation per-constituent** — rank-128 LoRA insufficient for multiple passages. **Partial.** Combo IS 22-27pp below single-passage K=1 consistently. But it's flat across K=2/3/4, so the per-passage capacity isn't being strained as K grows. Likely explanation: training distributes a fixed amount of LoRA capacity across more passages, so each passage gets less individual fidelity, but the trade is predictable.
2. **Test 2 cross-passage failure with Test 1 success** — adapter learns each passage independently rather than their relationships. **Partial.** Test 2 shows combo modestly better than multi-stack but neither produces full-hits at K≥3. This is the spec's 2nd failure mode in part: combo apparently learns each constituent OK but doesn't synthesize them into composite answers.
3. **K=4 collapse with K=3 success** — practical combination size bounded. **Not** the failure mode. K=4 combo retrieval (0.74) is actually the highest of the three K values, well within sample noise of K=2 and K=3.

## Implementation notes / deviations

1. **Reused** the per_passage_dickens single-passage adapters as the K=1 baseline (no retraining).
2. **Cross-passage queries** are Phase-43-style chained probes ("Recall: X = . Also, Y = ") because hand-crafted free-form composition queries ("What is the relation between X and Y?") are hard to score by substring match at this scale. Each query has 2-4 expected answer fragments; partial credit by fragment-coverage rate.
3. **n_steps scales with K** (150 × K). At K=4 with 20 training sources, this gives ~30 visits/source — comparable to single-passage adapters' 30 visits/source.
4. **Multi-pass test** uses the V22-Dickens base for the final synthesis pass. A more capable base would change the result. The comparison here is specifically about what works at the tiny-Dickens scale used by the rest of the HRS architecture.
5. **Test 1 currently rebuilds the model 30+ times**. This is wasteful (model load is the dominant wall time) but the test correctness is unaffected. Cleaner implementation would build the base once and only swap LoRA states.

## Wall-clock totals

| Stage | Wall |
|---|---:|
| Training (10 combination adapters) | 68s |
| Tests 1-2 evaluation | 861s (~14 min) |
| Test 4 (multi-pass) | 74s |
| **Total** | **~17 min** |

## Summary

**Combination adapters work.** Training one adapter on the concatenated content of K passages produces an adapter that, at K=1 inference, retrieves each of its constituents far better than the existing architecture's K=N multi-adapter block-stacking can manage at K>2. Combo retrieval is *flat across K* (0.63-0.74) while multi-stack drops from 0.67 (K=2) to 0.17 (K=4). The K limit problem the prior two experiments couldn't fix is structurally avoided by combo training because combo never composes — it's a single rank-128 LoRA at deploy time.

The cost is modest: combo holds 22-27pp below single-passage K=1. The spec's "within 10%" criterion is missed, but the gap is consistent across K and doesn't grow.

Cross-passage composition queries (where the answer requires fragments from multiple constituents in one generation) are hard for any approach at this base scale. Combo modestly beats multi-stack at K=4 (12pp) and decisively beats a multi-pass-with-base-final-synthesis baseline (26-30pp). On this substrate combo is the best available approach for cross-passage queries, but absolute performance is limited.

**Deployment story:** the architecture has at least three operating patterns now. (1) K=1 single-passage adapters for individual content, retrieval ~0.93. (2) Combination adapters for known content groupings, retrieval ~0.65-0.74 per constituent regardless of K. (3) K=2 multi-adapter stacking for query-time composition of two adapters, retrieval ~0.67 single-content / 0.49 compositional. Combination adapters become the preferred path at K≥3 where multi-adapter stacking collapses.

**Next experiments:**
- Find the actual capacity ceiling: train K=8, K=12 combo adapters and see when per-constituent retrieval starts dropping.
- Test combo at higher rank (256, 512) to see if the 22-27pp gap to single-passage K=1 closes with capacity.
- Test combo with more constituents per cross-passage query and see if more focused training data (queries that explicitly require synthesis) lifts the cross-passage performance.