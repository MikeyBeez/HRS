# Where Engram-as-Address Routing Breaks: Template-Grain Routing and the Entity-Grain Limit

Michael Bonsignore

---

## Abstract

Engram-as-address routing — using a mean-pooled hidden-state vector as a content-addressable pointer to a per-passage LoRA adapter — was previously shown to achieve 100% routing and 97% retrieval on a 20-passage library where each adapter has a unique entity instantiation (Phase 47). We probed whether that result generalizes to deployment-realistic conditions and found a structural failure mode. Three independent experiments establish: (1) the simpler architecture of identity-projection over L0 keys (no trained W) achieves 93% routing and outperforms every base-model-derived projection into L5 keyspace, raising the question of what the trained W actually buys; (2) the trained W has poor open-world specificity (83% false-positive rate at 95% recall on out-of-distribution queries), while the L0-cosine baseline has clean specificity (0% FP); (3) under same-template-different-entity out-of-distribution queries — the deployment-realistic condition — both stacks fail at ~92% false-positive rate, and standard rescue paths (last-N-token pooling, contrastive training with random negatives) do not recover the failure. The mechanism is representational: mean-pool L0 weights syntactic frame and common nouns heavily and entity tokens weakly, and the trained projection inherits this via linearity. The architecture is appropriate for non-overlapping libraries; deployment to open-world routing requires a separate entity-aware verification stage on the loaded adapter.

## 1. Introduction

The engram-as-address framework (Phase 47, paper_adapter_library.md) treats a single mean-pooled hidden-state vector as a pointer into a library of per-passage LoRA adapters. A 1024×1024 projection W trained with InfoNCE bridges the input-embedding layer (L0) where queries are extracted and the final layer (L5) where adapter keys are stored. The published result was 100% routing and 97% retrieval on a 20-adapter library benchmark, with three application demonstrations: per-passage adapter retrieval, KV cache compression, and compositional retrieval.

The benchmark on which that result was obtained has a property worth examining: each library passage has a unique entity instantiation. Different adapters cover different content. No two passages share template structure. Under that condition, routing-by-template and routing-by-entity are operationally equivalent — the template uniquely identifies the entity.

This paper asks whether the architecture generalizes when that condition does not hold. The question matters for any deployment scenario where the library will accumulate multiple passages over the same template — a robot remembering preferences for multiple users, a medical assistant remembering protocols for multiple patients, an agent maintaining context across multiple instances of a recurring entity type.

The answer, established through three experiments, is that the architecture has a structural failure mode at template-overlapping near-neighbor out-of-distribution queries, that the failure is representational rather than mechanistic, and that the deployment claim must be scoped to non-overlapping libraries unless an entity-aware verification stage is added downstream of routing.

## 2. Setup

We use the Phase 47 architecture and library unchanged: HRSTransformer with d_model=1024, n_layers=6, n_heads=8; 20 LoRA adapters of rank 128, one per absorbed passage; trained 1024×1024 projection W with InfoNCE on 80 (L0, L5) training pairs. Retrieval is measured on 60 held-out paraphrase trials drawn from the same template distribution as the training paraphrases. All numbers reproduced under model.eval() to control for dropout (a methodology correction noted below).

Two architectural variants serve as the principal comparison throughout:

- **C0b**: identity projection (W = I), with adapter keys stored at L0 (mean-pooled input embeddings). Routing is `cos(L0_query, L0_keys[i])`.
- **C1**: trained projection W, with adapter keys stored at L5 (mean-pooled output of block 5). Routing is `cos(L0_query @ W, L5_keys[i])`.

C1 is the published Phase 47 architecture. C0b is the simpler architecture obtained by skipping the L0→L5 projection entirely.

## 3. Experiment 1: What Does the Trained Projection Buy?

We ask whether the trained W is necessary by comparing it against six alternative choices for the projection matrix, all using L5 keys for parity:

| Condition | W' choice | Routing | Retrieval | Mean true cos | Mean gap | Negative gap |
|---|---|---|---|---|---|---|
| C0a | Identity, L5 keys | 25% | 23% | +0.016 | −0.019 | 45/60 |
| C0b | Identity, L0 keys | **93%** | **90%** | +0.930 | +0.029 | 4/60 |
| C1 | Trained W, L5 keys | **100%** | **97%** | +0.246 | +0.148 | **0/60** |
| C2 | Composition of out_proj matrices | 5% | 7% | −0.031 | −0.049 | 57/60 |
| C3 | K matrix at block 0 | 8% | 8% | +0.013 | −0.022 | 55/60 |
| C4 | K matrix at block 5 | 5% | 5% | +0.069 | −0.024 | 57/60 |
| C5 | OLS regression L5∼L0 on training prompts | 75% | 73% | +0.802 | +0.032 | 15/60 |
| C6 | Random orthogonal | 7% | 10% | +0.018 | −0.022 | 56/60 |

Three findings emerge.

**The L0→L5 bridge is not in any obvious base-model subspace.** The architectural priors (C2 output projection composition, C3 K at the engram-extraction layer, C4 K at the destination layer) are all at chance, indistinguishable from the random-orthogonal control C6. Whatever maps L0 to L5 in this base model is not sitting in any easy-to-identify attention or projection subspace.

**A weak version of the bridge is in the base model.** The OLS condition (C5) achieves 75% routing with no contrastive supervision, fitting a rank-80 linear approximation to the L0→L5 mapping using only the unlabeled training prompts. This recovers a partial bridge — meaningful relative to chance, but noisier than the trained W (mean gap +0.032 vs +0.148, with 15/60 negative gaps vs 0/60).

**The trained W is sharpening, not just discovering.** Compared to OLS, trained W is 25 routing points higher and has a 5× larger mean gap with no negative gaps. Contrastive training is doing real margin work. The bridge is partly present in the base; supervision is required to make it sharp.

**The architectural surprise:** C0b, with no projection at all and keys stored at L0, achieves 93% routing — beating every base-model-derived projection into L5 keyspace and trailing the trained C1 by only 7 points. The L5+W stack offers a marginal improvement on this benchmark, not a structural advantage. This raises the question taken up in Experiment 2.

## 4. Experiment 2: Open-World Specificity

A 7-point routing improvement is small relative to the architectural complexity of training a 1024×1024 projection. We test whether C1's value is concentrated in a different metric: out-of-distribution specificity. A deployment-relevant routing system must reject queries that fall outside the library, not just rank library entries.

We construct a 30-passage out-of-distribution test set drawn from WikiText, length-matched to library passages (128 tokens). For each OOD passage, we measure the maximum cosine to any library adapter under both C0b and C1 routing. The threshold for "in-library" is calibrated per condition to retain 95% recall on in-library paraphrase queries; the false-positive rate is the fraction of OOD queries that exceed this threshold.

| Condition | In-library mean cos | OOD mean cos | Separation | FP at 95% in-library recall |
|---|---|---|---|---|
| C0b | +0.931 | +0.511 | +0.420 | **0%** |
| C1 | +0.246 | +0.123 | +0.123 | **83%** |

The result inverts the expected architectural story. C0b — the simpler architecture — has perfect specificity: distributions are linearly separable, with no OOD query exceeding the in-library threshold at 95% in-library recall (and at 100% recall). C1 — the more complex architecture — has poor specificity: 83% of OOD queries exceed the threshold needed for 95% in-library recall. To achieve the same 0% FP rate as C0b, C1 would need a substantially lower threshold that costs significant in-library recall.

A length-matched ablation with 6-16 token OOD spans confirms the result is not a length artifact. C0b's separation widens (0.420 → 0.565) on shorter OOD; C1's 70% FP barely changes from 83%.

The mechanism: InfoNCE optimized intra-library rank, not absolute scale. The trained W creates discriminability among the library's L5 directions but never saw OOD examples as negatives during training, so the projected band (cos 0.05–0.37) overlaps the OOD band (cos 0.05–0.22) heavily. Meanwhile L0 cosines naturally separate semantic paraphrase (>+0.8) from unrelated English (~+0.5) without any supervision.

This finding suggests a two-stage architecture: use C0b for the rejection question ("is this query in the library at all?") and C1 for the selection question ("which specific adapter?"). Each method does what it is good at and is bad at the other. Experiment 3 tests whether this composition holds under the deployment-realistic OOD distribution.

## 5. Experiment 3: Hard Out-of-Distribution

The OOD test in Experiment 2 used random WikiText passages — semantically distant from library content. The deployment-realistic question is whether routing distinguishes library entries from queries that share library structure but reference different specific entities. For a system holding "Dr. Smith prefers 72°F," the hard rejection is "Dr. Jones prefers a quiet environment" — same template, same domain, different specific facts.

We construct a hard-OOD test set by taking each library passage's template and substituting a different entity from the same category (different name for entity passages, different facility for technical passages, different scientist for fact passages, etc.). The result is 20 passages structurally indistinguishable from library entries except in the load-bearing content.

| Condition | In-library | Hard-OOD mean cos | Separation | FP at 95% in-library recall |
|---|---|---|---|---|
| C0b | +0.931 | +0.893 | +0.038 | 92% |
| C1 | +0.246 | +0.146 | +0.100 | 92% |

Both architectures fail at the same false-positive rate. For C0b, this is a dramatic collapse: the +0.420 separation that survived random WikiText shrinks to +0.038 under template-overlapping OOD. For C1, the separation was already small (+0.123 on easy OOD) and stays small (+0.100 on hard OOD) — C1 was never reliably specific to begin with, and the C0b architectural pivot suggested by Experiment 2 does not survive here because C1 inherits L0's failure transitively (W being linear in L0 cannot recover entity-grain discriminability that L0 lacks). Both stages of the proposed two-stage architecture fail at the same condition.

We attempted a contrastive rescue: retrain W with a hinge loss against 200 random WikiText OOD negatives, sweeping the hinge loss weight λ ∈ {0.5, 2.0, 8.0}. The hinge loss zeroed within 100 training steps at every λ, indicating the optimization succeeded at pushing random negatives away. But the rescue did not transfer to hard-OOD: the trained margin applies to a different region of L0 space than where the hard-OOD failures live. The best λ (0.5) yields 73% FP on hard-OOD; higher λ values trade in-library recall for marginal hard-OOD improvement and never break the failure mode.

## 6. Experiment 4: Pooling Rescues

The hypothesis taken up here is that the failure is mechanistic — mean-pooling over 128 tokens dilutes entity tokens because they constitute a small fraction of total tokens. Switching to last-N-token pooling might preserve entity signal in queries, since entities often appear near the end of natural-language questions.

We re-extract L0 engrams under four pooling variants and re-run the hard-OOD specificity test:

| Variant | In-library top-1 | Hard-OOD FP at 95% recall |
|---|---|---|
| mean_pool (Phase 65 baseline) | 100% | 95% |
| last_1 | 5% (degenerate) | n/a |
| last_5 | 93% | 62% |
| last_10 | 100% | 92% |

`last_1` degenerates because question-final tokens are typically punctuation, carrying no entity content. `last_10` reverts to the mean-pool failure mode — the dilution effect resumes once the pooling window is wide enough to include template tokens. `last_5` is the best non-degenerate variant and cuts hard-OOD FP from 95% to 62% — directionally right, magnitude not deployment-grade.

The failure is representational, not mechanistic. Entity identity is not recoverably encoded in L0 by any positional cut we tested. The mean-pool architecture inherits this, and the trained projection W inherits it transitively via linearity.

## 7. Mechanism

The combined results are coherent under a single mechanistic account. L0 mean-pooling produces engrams where syntactic frame and common-noun tokens dominate the vector and entity-specific tokens contribute weakly. This produces:

- High in-library cosines (paraphrases of the same passage share frame and common nouns; cosine ~0.93).
- Moderate-to-high near-neighbor OOD cosines (same-template-different-entity passages share frame and common nouns just as strongly; cosine ~0.89).
- Low far-OOD cosines (unrelated English text shares only generic English structure; cosine ~0.51).

The trained projection W, being linear in L0, can only redistribute and rescale these signals. It cannot extract entity information that L0 mean-pooling has averaged out. The InfoNCE objective creates intra-library discriminability — pulling library L5 directions apart — but does not create rejection capability against any OOD distribution it did not see at training time, and cannot create entity-grain discriminability when its input lacks entity-grain signal.

C0b's clean rejection of far-OOD is therefore a side-effect of L0 representation rather than an architectural feature. The L0 cosine "naturally separates" paraphrase from unrelated English because L0 itself encodes English-frame information at high amplitude and entity information at low amplitude. When OOD shares the frame, the natural separation collapses.

## 8. Scope of the Original Result

The Phase 47 finding (100% routing, 97% retrieval) remains valid under the conditions on which it was obtained: a library where each adapter has a unique entity instantiation. Under that condition, template-grain routing and entity-grain routing are operationally equivalent. The result generalizes to:

- **Non-overlapping libraries**: collections where each library entry has a structurally distinct template.
- **In-distribution paraphrase routing**: queries that paraphrase library content using the same template.
- **Far-OOD rejection** under C0b (not C1): correctly refusing to route queries that are semantically distant from library content.

The result does not generalize to:

- **Template-overlapping libraries**: collections with multiple entries sharing template structure but differing in entity content.
- **Near-OOD rejection**: correctly refusing queries that match library template but reference different entities.
- **Open-world deployment**: routing in environments where queries arrive from an unbounded distribution that includes near-neighbor structures.

## 9. Implications

The architectural decomposition that Phase 47 implicitly assumed — that engram routing performs both selection (which adapter) and verification (does this query belong) — does not survive the entity-grain test. In any library that grows to contain multiple entries with overlapping templates, routing alone cannot distinguish entity-bearing queries from each other.

The architecturally clean response is to separate routing from verification. Routing operates at template grain, retrieving the adapter most likely to contain a relevant template. Verification operates at entity grain, examining the loaded adapter's behavior on the query and either returning the answer or rejecting. This decomposition aligns with the actual capacity of the L0 mean-pool representation: it carries template/frame information cleanly and entity information weakly.

We do not implement or evaluate a verification stage in this paper. The design space is large — possible mechanisms include classifier heads attached to the loaded adapter, generation-and-match with answer-string verification, attention-based entity matching against the adapter's training content, or learned reject heads — and each warrants its own experimental treatment. We list this as the principal next-step direction.

## 10. Methodology Note

Phase 47 and Phase 65 results were originally generated without `model.eval()`, leaving dropout active during evaluation. Re-running under eval mode reproduces the published mean-pool/C0b numbers (93% top-1, +0.931 in-library mean) within statistical noise; the dropout effect was small. We report eval-mode numbers throughout this paper as the canonical baseline. Subsequent reproductions should use `model.eval()` during evaluation passes.

## 11. Limitations

**Library scale.** The benchmark uses 20 adapters. A larger library with more template diversity might allow templates to cluster in L0 space such that entity-grain routing becomes statistically possible without verification. Untested at this scale.

**Architecture.** Results are specific to the HRSTransformer (d=1024, 6 layers, 8 heads) used in Phase 47. Larger base models with richer L0 representations may carry entity information at higher amplitude; the same probe should be re-run at GPT-2-medium / Llama-3-8B scale before generalizing the failure claim.

**OOD construction.** Hard-OOD passages were constructed by entity-substitution within library templates. Other near-neighbor distributions (paraphrased templates with same entities, semantically related but structurally distinct content) are not tested.

**Verification stage not built.** The proposed architectural fix is not evaluated. We argue from the mechanism that a verification stage should restore deployment-readiness, but this is conjecture until tested.

## 12. Conclusion

Engram-as-address routing achieves the benchmark numbers it was designed for, but the benchmark conditions implicitly assumed non-overlapping templates. Three experiments establish that the architecture has a structural failure mode at template-overlapping near-neighbor OOD, that the failure is representational rather than fixable by pooling tweaks or contrastive rescue, and that the trained projection W inherits the failure transitively via linearity. The deployment claim must be scoped accordingly: engram routing is appropriate for non-overlapping libraries; open-world deployment requires a separate entity-aware verification stage downstream of routing.

The negative result is informative on two counts. It identifies a clean architectural boundary between what L0 mean-pool representations can and cannot discriminate, and it specifies the next research direction precisely — verification, not better routing.

---

## Files and reproduction

```
experiments/identity_ae/phase47_l0_to_l5_projection.py    # Original Phase 47
experiments/identity_ae/phase65_w_conditions.py            # Experiment 1
experiments/identity_ae/phase65_specificity.py             # Experiment 2
experiments/identity_ae/phase65_specificity_lengthmatched.py
experiments/identity_ae/phase65_hardood.py                 # Experiment 3
experiments/identity_ae/phase65_lasttoken.py               # Experiment 4
results/identity_ae/phase65/phase65_results.json           # Verified locally
results/identity_ae/phase65/specificity.json               # Per agent run
results/identity_ae/phase65/specificity_lengthmatched.json # Per agent run
results/identity_ae/phase65/hardood.json                   # Per agent run
results/identity_ae/phase65/lasttoken.json                 # Per agent run
```

All numbers in this draft have been verified against the corresponding JSON files in `~/Code/HRS/results/identity_ae/phase65/`. A revision pass on 2026-05-10 corrected two C1 hard-OOD numbers and one unverified recall claim that were initially drafted from agent summaries before the JSONs were locally available.

## Notes for revision

- ~~Verify all numbers in Sections 4-6 against the actual JSON files once they sync locally.~~ Done 2026-05-10. Two C1 hard-OOD numbers and one easy-OOD recall claim were corrected from initial agent-summary draft.
- The paper currently runs ~3,200 words. Workshop format target is 4-8 pages, so room remains for related work, figure discussion, and a deeper Section 9 (implications). Or trim to keep tight.
- Related work section not written. Needs references to: Phase 47 itself (paper_adapter_library.md as prior work), induction heads literature (Olsson et al.), LoRA (Hu et al.), IA3 (Liu et al.), magnitude pruning literature, intrinsic dimension fine-tuning (Aghajanyan et al.), routing in mixture-of-experts (Shazeer, Fedus), open-set recognition / OOD detection literature.
- Figures: probably 2-3 needed. Suggested: (1) Experiment 1 condition table as bar chart with routing/retrieval per condition; (2) Specificity histogram showing in-library vs OOD cosine distributions for C0b and C1 side by side; (3) Hard-OOD vs easy-OOD comparison highlighting the separation collapse.
- Title is a working title. Alternatives: "The Template-Entity Limit of Engram Routing," "When Engram-as-Address Fails," "Routing at Template Grain: A Negative Result for Open-World Engram Routing." First emphasizes the boundary, second is more dramatic, third is most accurate.
- Decide submission target. NeurIPS workshop deadlines, ICLR Tiny Papers, arXiv tech report. Negative results papers tend to do better at workshops focused on retrospective findings or interpretability than at main conferences.
