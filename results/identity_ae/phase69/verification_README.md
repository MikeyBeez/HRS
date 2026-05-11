# Phase 69 — Adapter verification stage

Tests whether a verification stage downstream of routing can catch the
template-overlapping near-neighbor failure that Phase 65 surfaced. Eight
candidate signals: five generation-confidence, two query-side NLL, one
generation-text entity match. Plus one routing-time metadata signal added
after the first run made clear the generation-based signals were dominated.

## Headline

**Metadata entity check rescues the architecture: 100% balanced accuracy.**

At routing time, extract the entity from the query (closed-vocab match) and
from the routed adapter's training prompt (same vocab). If they agree, accept
the adapter's answer; if they don't, reject. No generation required, no
confidence calibration, no learned head. Free at inference.

```
                       confusion matrix          precision  recall  specificity  balanced_acc
metadata_entity_match  TP=56  FN=0  FP=0  TN=24    100%     100%     100%        100%
entity_match (gen)     TP=31  FN=25 FP=0  TN=24    100%      55%     100%         78%
gen_nll                                                                      88% FP@95
gen_max_logit_mean                                                           88% FP@95
gen_entropy_mean                                                             92% FP@95
gen_token_min_prob                                                          100% FP@95
query_nll_under_adapter                                                     100% FP@95
delta_query_nll                                                             100% FP@95
```

(80 queries total: 56 TRUST = correctly-routed in-library, 4 DON'T_TRUST_misroute
= mis-routed in-library, 20 DON'T_TRUST_ood = hard near-neighbor OOD.)

## Why the confidence signals all collapsed

The LoRA adapters are trained to high confidence on a single passage and emit
their trained answer with the same confidence regardless of input shape. Sample
generations:

```
TRUST  (northern adapter, northern query):
   " 604876. The northern facility code must be memorized immediately. ..."
   gen_nll = 0.13   delta_query_nll = -5.21

DON'T_TRUST_ood (orbital query routed to southern adapter):
   " 75938259382 ... 759382 will not be repeated. ..."
   gen_nll = 0.26   delta_query_nll = -4.86

DON'T_TRUST_ood (coastal query routed to northern adapter):
   " 6048767676. This code must be memorized immediately. ..."
   gen_nll = 0.74   delta_query_nll = -5.14
```

The wrong-adapter outputs are nearly indistinguishable from right-adapter
outputs in every confidence dimension. `delta_query_nll` actually has the
*wrong* sign — DON'T-TRUST queries get a stronger NLL improvement from the
adapter than TRUST queries do, because mis-routed adapters fit the template
too well. Predicted "potentially stronger" signal in the spec; turned out to
be inverted.

`gen_token_min_prob`: I predicted slightly stronger (40-60% FP). Measured 100%
FP. The wrong-adapter generations don't have a single low-probability "doubt"
token — the adapter is uniformly confident through the whole greedy decode.

## Why entity_match (gen-based) is a one-sided filter

`entity_match` checks whether the generation text contains the query entity
(via closed-vocab match against the 40 entities in the test set). Result:
100% specificity, 55% recall — never falsely accepts, but rejects 45% of
TRUST queries.

The 45% miss rate comes from generations where the LoRA adapter spits out the
passkey repeatedly without re-mentioning the entity. Sample:

```
TRUST  (southern adapter, southern query):
   " 75938259382593825938259382593825938259382593825938259382 ..."
   query_entity = 'southern',  gen_entity = None,  match = 0  (false rejection)
```

Going from 50 to 100 generation tokens raised recall from 54% → 55% — the
entity isn't going to appear at length 100 if the adapter has settled into a
passkey-loop. Longer generation isn't the fix.

`entity_match` works as a *one-sided rule*: if it returns 1, you can trust
unconditionally. If it returns 0, you can't conclude — need a second signal.
Useful but not sufficient on its own.

## Why metadata_entity_match works perfectly

The signal is: at routing time, extract the entity from the query, and
extract the entity from the *routed adapter's training prompt*. Compare.

```
TRUST cases (correctly routed):
   query: "I need the access code for the northern facility."
   routed_adapter[0].prompt: "What is the system access code for the northern facility?"
   query_entity = 'northern',  routed_entity = 'northern',  match = 1  ✓

DON'T_TRUST_misroute (mis-routed):
   query: "Look up the northern facility's access code."
   routed_adapter[4].prompt: "What is the system access code for the central facility?"
   query_entity = 'northern',  routed_entity = 'central',  match = 0  ✓

DON'T_TRUST_ood (hard-OOD):
   query: "What is the system access code for the orbital facility?"
   routed_adapter[1].prompt: "What is the system access code for the southern facility?"
   query_entity = 'orbital',  routed_entity = 'southern',  match = 0  ✓
```

Every TRUST case has matching entities by construction (the in-library entity
is unique per adapter, and correct routing finds the right adapter). Every
DON'T-TRUST case has mismatched entities (mis-routing means a different
adapter; hard-OOD means an entity that no adapter was trained on). 56/56 +
24/24 = 80/80 perfect classification.

## What this means and what it doesn't

### Recipe for the architecture

```
1. Extract query entity (closed-vocab match, NER, or domain regex).
2. Route via C0b (L0 cosine, fast).
3. Look up routed adapter's training entity from library metadata.
4. If entities match: load adapter, return its answer.
5. If entities don't match: reject.
```

This is deployment-grade. Runtime cost: one extra string comparison per query.
No model retraining, no learned head, no confidence calibration.

### What this requires (be honest about it)

The metadata check works perfectly **conditional on reliable entity extraction
from queries**. In Phase 69's setup we have:

1. Closed entity vocabulary (10 entities × 4 categories = 40 possible entities).
2. Templates structured enough that the entity is unambiguous.
3. Each library entry already has an entity annotation (the prompt itself
   names the entity).

For domains where these hold — patient records, customer accounts, protocol
catalogs, project IDs, anything with a curated identifier vocabulary — the
recipe works as-is. For open-world text deployment with arbitrary entity
mentions and no curated vocabulary, the verification stage reduces to "how
good is your NER?" Architecturally clean; engineering-wise, it's now a
named-entity recognition problem, not a routing problem.

### What this *doesn't* mean

The metadata check is **not** what HRS routing was claiming to do. The
original "engram-as-address" framing was that the routing layer itself does
entity-grain discrimination. Phase 65 showed it doesn't. Phase 69 shows the
gap can be closed at deployment-time with a free string comparison — but the
fix lives outside the engram-routing system, on top of library metadata.

The publishable claim from Phase 65 stands: routing is template-grain. Phase
69 adds: deployment with entity-grain correctness requires a metadata check on
top, but that check is essentially free.

## Pre-committed predictions vs measured

| Signal                 | Predicted FP@95 | Measured FP@95 |
|------------------------|------------------|----------------|
| gen_nll                | 50-70%           | **88%**        |
| gen_entropy_mean       | 50-70%           | **92%**        |
| gen_max_logit_mean     | 50-70%           | **88%**        |
| gen_token_min_prob     | 40-60%           | **100%**       |
| query_nll_under_adapter| 30-50%           | **100%**       |
| delta_query_nll        | (cleanest)       | **100% (inverted sign)** |
| entity_match (gen)     | 85-95% accuracy  | **78% balanced (100% spec, 55% recall)** |
| metadata_entity_match  | (not predicted)  | **100% balanced**  |

Predictions skewed too optimistic on confidence signals (assumed adapters
would express doubt; they don't), and predicted entity_match would have high
recall (it has perfect precision but only 55% recall). The metadata signal
wasn't in the original spec — added after the first run made clear that any
generation-derived signal was capped by the LoRA adapters' uniform confidence.

## Files

- `verification.json`            full per-query records and per-signal stats
- `signal_distributions.png`     six-panel histograms of confidence signals
- `verification_run.log`         stdout
- `experiments/identity_ae/phase69_verification.py`  the script

## Open follow-ups

1. **NER quality matters.** A real-world deployment test should use
   open-vocabulary entity extraction (e.g., GPT-2-based NER, spaCy, or a
   domain-specific tagger) and measure verification accuracy under realistic
   extraction noise. The 100% number here is the closed-vocab ceiling; how
   far it drops under open NER is a separate experiment.

2. **Adversarial entities.** Hard-OOD here used entity-substitution within
   the template. An adversarial test would use queries that include *both*
   a library entity and a non-library entity ("the northern AND orbital
   facilities"), or queries where the entity is implicit ("the same facility
   as last time"). These break metadata_entity_match in interesting ways.

3. **Phase 70 (per-adapter classifier head).** Originally posed as the
   fallback if Phase 69's confidence signals failed. Confidence signals did
   fail, but the metadata check picked up the slack — Phase 70 is no longer
   urgent unless you want a verification stage that doesn't require entity
   extraction at all (e.g., for adapters covering non-entity-shaped tasks).
