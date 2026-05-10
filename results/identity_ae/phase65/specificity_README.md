# Phase 65 specificity follow-ups — full picture

Three sequential ablations testing the C0b vs C1 architectural claim under
increasingly hard OOD distributions. Pre-committed predictions (Outcomes A
through D defined in the original spec) compared against measured outcomes.

## Headline

|                              | Easy 128-tok OOD | Length-matched OOD | **Hard near-neighbor OOD** |
|------------------------------|------------------|--------------------|----------------------------|
| C0b OOD FP @ 95% in-lib rec  | 0%               | 0%                 | **92%**                    |
| C1 OOD FP @ 95% in-lib rec   | 83%              | 70%                | **92%**                    |
| C0b 0%-FP → in-lib recall    | 100%             | 100%               | **25%**                    |
| C1 0%-FP → in-lib recall     | 68%              | 70%                | 40%                        |

**Plus a rescue attempt**: OOD-aware W retraining with hinge loss on 200
WikiText OOD negatives. Best lambda gives hard-OOD FP 73% at in-library top-1
93% — slight improvement over original 92% FP, far from a fix.

## Three OOD distributions, three results

### Easy OOD (128-token WikiText passages)
- C0b OOD mean +0.511, separation +0.420, **0% FP at 100% recall**.
- C1 OOD mean +0.123, separation +0.123, 83% FP at 95% recall.
- **Outcome D**: C0b decisively beats C1 for open-world rejection.

### Length-matched OOD (6–16 token WikiText spans)
- C0b OOD mean +0.366 (lower than 128-tok!), separation +0.565 (wider).
- C1 OOD overlap pattern preserved, FP 70%.
- **Length confound ruled out**: short OOD scores lower under C0b, not higher.

### Hard near-neighbor OOD (same template, different entity)
Constructed from `generate_passkeys(50)` — library uses tests 0–4 per type;
hard-OOD uses tests 5–9 per type. Identical templates, different entities
(e.g. library asks about "northern facility", hard-OOD asks about "orbital").

- C0b OOD mean **+0.893** (separation collapses to +0.038!).
- C0b FP at 95% recall: **92%**. 0%-FP requires sacrificing 75% of recall.
- C1 OOD mean +0.146 (only marginally up). FP at 95% recall: 92%.
- **Both architectures fail.** The L5+W projection inherits L0's template-level
  representation rather than discriminating at entity granularity.

Confusion pattern under C0b on hard-OOD numeric queries:
```
+0.946: "Access code, orbital facility?"  -> lib[1] "...southern facility?"
+0.942: "Look up the coastal facility..." -> lib[4] "...central facility?"
+0.935: "Access code, highland facility?" -> lib[1] "...southern facility?"
+0.936: "Look up the basement facility..." -> lib[4] "...central facility?"
```

Two library adapters (southern, central) act as attractors for the entire
"facility access code" template. The L0 representation evidently weights
template structure heavily and entity weakly under mean-pooling.

### OOD-aware W retraining (rescue attempt)
Augmented Phase 47's InfoNCE with a hinge term: 200 WikiText OOD L0 queries
should have max in-library cosine below margin 0.10 after projection. Swept
lambda_ood ∈ {0.5, 2.0, 8.0}.

| lambda | In-lib top-1 | Hard-OOD FP @ 95rec | 0%-FP → recall | separation |
|--------|--------------|---------------------|----------------|------------|
| (orig) | 95%          | 92%                 | 5–25%          | (collapsed)|
| 0.5    | 93%          | **73%**             | 48%            | +0.094     |
| 2.0    | 97%          | 75%                 | 43%            | +0.098     |
| 8.0    | 95%          | 77%                 | 5%             | +0.110     |

The hinge term zeroed out within 100 steps (training-OOD max cos drops below
margin), but hard-OOD wasn't in the training-OOD distribution and still scored
high. Random WikiText negatives don't generalize to same-template near-
neighbor negatives. **The rescue plan as designed doesn't work.**

## What this all means architecturally

The L0 mean-pooled representation evidently weights *template* structure
heavily (question shape, syntactic frame, common nouns like "facility",
"protocol", "threshold") and *entity* identity weakly (the specific facility
name, scientist name, protocol name). This makes:

1. C0b: clean for cross-domain rejection (paraphrased question vs. WikiText
   prose), broken for in-domain entity discrimination.
2. C1: same template-bias inherited by W (since W is a linear function of L0).
   Sharp intra-library margin via InfoNCE rank, but no real entity-grain
   separation.
3. OOD-aware W retraining: helps for OOD that *occupies a different L0
   subspace* (random text), useless for OOD that *shares the in-library L0
   subspace* (same-template near-neighbors).

The cleanest publishable framing now:

> HRS routing operates at template granularity, not entity granularity. C0b
> (L0 cosine, no projection) provides excellent OOD rejection when OOD lives
> outside the library's template distribution; it fails at near-neighbor
> rejection when OOD shares template structure. C1 (trained W, L5 keys)
> sharpens intra-library tie-breaking but doesn't add OOD rejection or fix
> the entity-grain limit. Augmenting W's contrastive loss with random OOD
> negatives doesn't bridge the gap because the failure mode lives in the
> in-distribution L0 subspace, not outside it. Open question: does an
> entity-aware verification layer (e.g. per-adapter entity-match head, or
> entity-conditioned key formation) close the entity-grain gap, or is the
> mean-pool L0 representation itself the load-bearing constraint?

## Pre-committed predictions vs measured outcomes

| Test | Pre-committed prediction | Measured outcome | Reasoning that was wrong |
|------|--------------------------|------------------|--------------------------|
| Easy OOD | A (C0b poor, C1 clean) | **D** (C0b clean, C1 poor) | "L0 cosines have nowhere to go but high" — L0 actually separates paraphrase (>+0.8) from unrelated English (~+0.5) |
| Length match | C0b separation might shrink | **C0b separation widens** | Short OOD scores lower, not higher |
| Hard OOD | C0b might break under near-neighbor | **C0b breaks; C1 also breaks** | Predicted correctly; severity matched expectation |
| OOD-aware W rescue | Should give C1 absolute-distance separation | **No rescue** (FP 92% → 73%) | Random OOD negatives don't generalize to template-shared OOD |

## Files

- `specificity.json` — easy 128-tok OOD per-query records
- `specificity_lengthmatched.json` — length-matched OOD records
- `specificity_hardood.json` — hard near-neighbor OOD records
- `w_oodaware.json` — OOD-aware W training sweep (lambda × in-lib + hard-OOD scores)
- `cosine_distributions.png`, `cosine_distributions_lengthmatched.png`,
  `cosine_distributions_hardood.png` — visualizations
- `experiments/identity_ae/phase65_specificity{,_lengthmatched,_hardood}.py` — scripts
- `experiments/identity_ae/phase65_w_oodaware.py` — rescue training script
- `*_run.log` — stdout from each run

## Open questions, partially answered

### Pooling ablation (last-token / last-5 / last-10 L0)

Tested whether the entity-grain failure is mechanistic (mean-pool washes out
entity tokens) or representational (entities aren't in L0 at all). Result:
**representational**. See `lasttoken_README.md` for the full table.

| variant   | route | in_top1 | hard-OOD FP @ 95% recall |
|-----------|-------|---------|--------------------------|
| mean_pool | C1    | 100%    | 95%                      |
| **last_5**| **C1**| **93%** | **62%**                  |
| last_10   | C1    | 100%    | 92%                      |

Best variant (last_5/C1) cuts FP from 95% to 62%. Direction is right but
magnitude is far from a deployment fix. last_1 collapses entirely because
question-final tokens are punctuation. last_10 reverts to mean_pool numbers.
Entity tokens carry only weak L0 signal; no positional pooling recovers
deployment-grade entity discrimination.

The cheap mechanistic rescue is dead. The architecturally correct response is
option 2 below.

### Still open

1. **Per-adapter entity-match verification head.** Routing is template-grain
   by design; add a separate stage on the loaded adapter that decides whether
   to commit or reject. The right architectural fix.

2. **Hard-OOD-aware contrastive training.** Replace random WikiText negatives
   with same-template-different-entity negatives. Likely fixes hard-OOD FP
   but only generalizes to enumerable near-neighbor templates — appropriate
   for closed-domain deployments, not open-world.

## Honest paper scoping

The deployment claim ("general-purpose adapter routing") that the C0b vs C1
comparison was drifting toward does **not** hold. The publishable claim is:

> Engram-as-address routing achieves 100%/97% intra-library accuracy on
> Phase 47's library where each adapter has a unique entity instantiation.
> The architecture has a structural limitation at template-overlapping near-
> neighbor OOD: when OOD queries share the library's template structure but
> reference different entities, both raw-L0 (C0b) and trained-projection (C1)
> stacks fail at ~92% false-positive rate. The failure is representational,
> not mechanistic — last-N-token pooling and OOD-aware contrastive on random
> negatives both fail to rescue. The architecture is appropriate for non-
> overlapping libraries; deployment to open-world routing requires a separate
> entity-aware verification stage.
