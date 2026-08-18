# Experiment Spec: Engram-Routed RAG on Dickens-50

## Purpose

The per_passage_dickens architecture achieves 100% routing and 93% retrieval. The 7-percentage-point gap is generation failures — the right adapter is loaded but the model occasionally produces garbled output. This experiment tests whether adding the routed passage's training text back into the inference context (as RAG-style retrieved content) closes that gap.

The hypothesis: the adapter and the retrieved text are complementary, not redundant. The adapter shapes how the model processes the query (generalization, parameter-level knowledge). The retrieved text gives the model literal access to the source content (recall, in-context lookup). Together they should reach closer to 100% retrieval than either alone.

This is also a test of the larger architectural story. If engram-routed RAG works, the foundation-model architecture has a clean three-stage inference path: engram identifies relevant passages, retrieved passages enter context, adapter shapes processing. That's a more interpretable and more debuggable system than pure adapter-based retrieval, and it removes the dependency on the adapter being able to generate correctly without literal source access.

## Background

The current per_passage_dickens architecture stores each Dickens passage as a per-passage rank-128 LoRA adapter trained for 150-500 steps. At query time, an L0-mean engram of the query is projected through the Phase-47 W matrix to L5-engram space, the closest stored engram is identified, and the corresponding adapter is loaded. Inference runs with the adapter active and the prompt — no retrieved text in context.

This experiment changes one thing: after engram routing identifies the closest passage, the raw training text for that passage gets pasted into the inference context as additional input.

## Setup

Use the existing per_passage_dickens setup. Reuse the trained adapters, the stored engrams, the routing infrastructure, the W projection, and the held-out probe set. Do not retrain anything.

The key infrastructure addition: maintain a mapping from each adapter index to its training text. This should already exist in the experiment's data files; if not, reconstruct it from the original passage data used during adapter training.

## Conditions

Five conditions, run on the same held-out probes as the original per_passage_dickens evaluation. Three seeds each.

**Condition 1: Baseline (per_passage_dickens as published).** Engram routes, correct adapter loads, prompt only at inference. Reproduces 100% routing and 93% retrieval. If it doesn't reproduce, halt and investigate.

**Condition 2: Engram-routed RAG (correct adapter + correct passage).** Engram routes, correct adapter loads, the training text for the routed passage is prepended to the prompt, inference runs. This is the new variant. Hypothesis: pushes retrieval above 93% by giving the model literal access to the source.

**Condition 3: RAG only, no adapter (correct passage retrieved, base model).** Engram routes, but no adapter loads — base model with the retrieved training text prepended. This isolates what the retrieval alone contributes. If Condition 3 already reaches 93%+, the adapter is doing nothing the retrieved text doesn't already do, and the architecture should drop the adapter for this task. If Condition 3 is well below 93%, the adapter is doing real work that the retrieved text can't replace.

**Condition 4: Wrong-passage RAG (correct adapter + wrong passage retrieved).** Engram routes, correct adapter loads, but the training text from a different passage is prepended. Tests whether the model uses the retrieved text or whether it ignores it once the adapter is loaded. If Condition 4 matches Condition 1 (93%), the model is ignoring the retrieved text and relying on the adapter — Condition 2's improvement (if any) might be coming from something other than the model reading the retrieved content. If Condition 4 hurts substantially below 93%, the model is genuinely using the retrieved text and getting confused by wrong content.

**Condition 5: Adapter-free baseline.** Base model with prompt only — no adapter, no retrieved text. The floor. Establishes how much of the retrieval is coming from the base model's general knowledge versus the architecture's contributions.

## Metrics

Same as the original per_passage_dickens evaluation. Retrieval accuracy via substring match of the answer in the model's continuation under stochastic decoding. Per-probe outcomes saved per condition per seed.

Per-probe deltas matter here. For each probe, did Condition 2 succeed where Condition 1 failed? Did Condition 4 fail where Condition 1 succeeded? These tell us whether the retrieved text is helping uniformly or only on the specific probes where the adapter alone was failing. The 7-point gap in the published baseline is concentrated in some probes; if those are exactly the probes Condition 2 fixes, the architecture story is clean.

## Pre-Registered Predictions

**Prediction 1: Condition 1 reproduces 93%.** Sanity check.

**Prediction 2: Condition 2 (correct + correct) substantially beats Condition 1.** Strong prior. Generation failures should largely disappear when the model can read off the source. My estimate: 97-100% retrieval. The few remaining failures would be cases where the question requires inference beyond what's literally in the passage.

**Prediction 3: Condition 3 (RAG only, no adapter) is competitive with Condition 1 but probably below it.** RAG is a strong baseline. Many of the held-out probes can probably be answered from the source text alone, even without the adapter. My estimate: 80-90%. If Condition 3 is at or above Condition 1, the adapter contribution at inference is small and the architecture might be simplifiable.

**Prediction 4: Condition 4 (correct adapter + wrong text) is below Condition 1.** The model should get confused by wrong source text in context. Specifically, it should produce content from the wrong passage some of the time. My estimate: 50-70%, dropping further on probes where the wrong passage has plausible but incorrect content for the question.

**Prediction 5: Condition 5 (base model, no architecture) is at or near 0%.** The held-out probes ask about specific Dickens content the base model wasn't trained on.

## What This Tells Us

**If Condition 2 ≈ 100%, Condition 3 < Condition 1, Condition 4 < Condition 1:** Clean three-stage architecture. Engram routes, retrieved text gives literal access, adapter shapes processing. Each component is doing distinct work. This is the cleanest possible result and supports the foundation-model story directly.

**If Condition 2 ≈ Condition 1, no improvement:** The 7-point gap is something else (generation hyperparameters, sampling artifacts, ambiguous probes), not generation failures fixable by source access. Investigate the failure modes individually.

**If Condition 3 ≥ Condition 1:** The adapter is doing nothing the retrieved text doesn't already do. The architecture should drop the adapter for this task. Significant simplification — engram-routed RAG is sufficient, no per-passage training needed.

**If Condition 4 ≈ Condition 1:** The model is ignoring the retrieved text once the adapter is loaded. Condition 2's improvement (if any) must be coming from something other than reading the source. Possibly the structural change of having text in context affects sampling. Worth investigating.

**If Condition 4 substantially hurts:** The model genuinely reads the retrieved text. Combined with Condition 2's improvement, this confirms the model is using both sources of information.

The combinations across conditions tell a richer story than any single condition alone. The wrong-adapter / wrong-text controls are where the mechanism gets pinned down.

## Why This Is the Right Experiment

The per_passage_dickens architecture treats the engram as the routing key and the adapter as the content storage. Engram-routed RAG adds a third option: the engram is the routing key, the retrieved passage is the content, and the adapter is the inference-time processing.

This maps onto how a working memory system should probably function. You have an index (engram) that points to both literal records (training text) and learned processing (adapter weights). At retrieval time, both come back together: you re-read the source while the adapter primes the model to think about it the right way.

It also has a precedent in standard RAG that's been ignored by the current architecture: actual RAG systems retrieve documents and put them in context. The per_passage_dickens architecture skipped this because the adapter was supposed to obviate it. This experiment tests whether that skip cost performance.

## Failure Modes to Watch For

If retrieval text is too long for the context window after the prompt, truncate it from the end of the passage rather than the start. The training data for an adapter is the full passage; truncation should preserve the beginning, which generally has the most content-defining information.

If Condition 1 doesn't reproduce 93%, halt. The discrepancy must be resolved before interpreting other conditions.

If the held-out probes weren't designed to be answerable from the literal text alone, Condition 2 might fail to improve even though the architecture is sound. Check that the probes are answerable from the passages — if some probes require inference beyond what's in the text, those should be excluded from the analysis or analyzed separately.

If sampling is stochastic, the per-probe outcomes will vary across seeds. Three seeds should give a reasonable estimate, but be cautious about claiming 100% retrieval if any single seed produced misses.

## Wall Time

Should be under 1 hour. No training. The retrieval text injection is minimal computational overhead compared to the model's forward pass. Most of the time is in generation.

## Deliverables

- summary.json with all five conditions, three seeds, mean and std.
- Per-probe outcomes table — which probes succeeded under which conditions.
- Per-probe delta analysis — Condition 2 vs Condition 1, Condition 4 vs Condition 1.
- Sample generations for at least 3 probes per condition, showing what differs qualitatively.
- A short RESULT.md interpreting the findings against the predictions.

Save results to `experiments/engram_routed_rag/results/` on the appropriate branch.

## Note Outside the Spec

If this works as predicted, it's the cleanest demonstration of the foundation-model architecture's three-stage inference path. The engram + adapter + retrieval combination is more powerful than any of the three alone, and the components have distinct functional roles that map onto distinct kinds of memory operations.

It also has a practical implication for the deployment story. A team building this system needs to store, per absorbed passage: (1) the engram for routing, (2) the adapter weights for processing, (3) the source text for retrieval. The engram is small (one vector), the adapter is small (rank-128 LoRA on a few layers), and the source text is the original passage. None of this is expensive. The combination produces a memory system that does both literal recall (via retrieval) and learned generalization (via adapter), addressed by the same engram-based key.

If Condition 3 turns out to match or beat Condition 1, the architecture simplifies significantly — the adapter goes away and we're left with engram-routed RAG, which is just smarter retrieval. That would also be a useful finding, just a different one. The point of the experiment is to find out which version of the architecture the data supports.
