# Combination Adapter vs Oracle RAG

**Question:** at this scale, does training content into a combination adapter (parameter-space absorption) provide any advantage over having the same content in context (oracle RAG)? Same passages, same queries, same scoring, same base model.

**Verdict: Combination adapter wins decisively on this substrate. The architecture's substitution claim is supported here.** Combo retrieval is 0.65-0.74 across K=2/3/4; RAG retrieval is 0.03-0.06 across the same K. Combo wins by 60-71 percentage points on per-constituent retrieval and by 22-38pp on cross-passage queries. **Important caveat:** the V22-Dickens base used here is a small (~510M-param) Dickens-pretrained model, not instruction-tuned. Its in-context-learning ability for Q/A-style queries is essentially zero — it just continues with novel Dickensian prose. A larger instruction-tuned base would likely close most or all of this gap.

## Setup

- Same 10 combinations from the combo adapter experiment (reused without retraining): 4 K=2, 4 K=3, 2 K=4.
- Same query sets: 3 held-out paraphrases per constituent for Test 1; 3 hand-crafted cross-passage queries per combination for Test 2.
- Procedure A (combo adapter): K=1 inference with combo adapter loaded.
- Procedure B (oracle RAG): build prompt as `{passage_1}\n\n{passage_2}\n\n...\n\n{probe}`, run inference with NO adapter (LoRA zeroed), score same way.
- All RAG prompts fit comfortably under the 512-token context limit (max prompt = 292 tokens at K=4 with full passages — no truncation needed).

## Test 1: per-constituent retrieval

| combination | K | combo | RAG | gap (combo-RAG) |
|---|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 0.667 | 0.000 | +0.667 |
| P2_Estella_Drummle | 2 | 0.556 | 0.000 | +0.556 |
| P3_Magwitch_Provis | 2 | 0.611 | 0.167 | +0.444 |
| P4_Herbert_Wemmick | 2 | 0.778 | 0.056 | +0.722 |
| T1_Pip_Estella_Magwitch | 3 | 0.741 | 0.000 | +0.741 |
| T2_Joe_Estella_Drummle | 3 | 0.815 | 0.037 | +0.778 |
| T3_Pip_Herbert_Wemmick | 3 | 0.519 | 0.111 | +0.407 |
| T4_Magwitch_Provis_Drummle | 3 | 0.444 | 0.074 | +0.370 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 0.667 | 0.028 | +0.639 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 0.806 | 0.028 | +0.778 |

### Aggregated by K

| K | combo (avg) | RAG (avg) | gap |
|---:|---:|---:|---:|
| 2 | 0.653 | 0.056 | +0.597 |
| 3 | 0.630 | 0.056 | +0.574 |
| 4 | 0.736 | 0.028 | +0.708 |

## Test 2: cross-passage queries (fragment coverage)

| combination | K | combo frac | RAG frac | gap | combo full | RAG full |
|---|---:|---:|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 0.500 | 0.000 | +0.500 | 0.222 | 0.000 |
| P2_Estella_Drummle | 2 | 0.556 | 0.111 | +0.444 | 0.111 | 0.000 |
| P3_Magwitch_Provis | 2 | 0.389 | 0.111 | +0.278 | 0.000 | 0.000 |
| P4_Herbert_Wemmick | 2 | 0.500 | 0.222 | +0.278 | 0.000 | 0.000 |
| T1_Pip_Estella_Magwitch | 3 | 0.296 | 0.000 | +0.296 | 0.000 | 0.000 |
| T2_Joe_Estella_Drummle | 3 | 0.296 | 0.037 | +0.259 | 0.000 | 0.000 |
| T3_Pip_Herbert_Wemmick | 3 | 0.333 | 0.037 | +0.296 | 0.000 | 0.000 |
| T4_Magwitch_Provis_Drummle | 3 | 0.407 | 0.000 | +0.407 | 0.000 | 0.000 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 0.306 | 0.000 | +0.306 | 0.000 | 0.000 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 0.333 | 0.000 | +0.333 | 0.000 | 0.000 |

### Aggregated by K

| K | combo frac | RAG frac | gap | combo full | RAG full |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.486 | 0.111 | +0.375 | 0.083 | 0.000 |
| 3 | 0.333 | 0.019 | +0.315 | 0.000 | 0.000 |
| 4 | 0.319 | 0.000 | +0.319 | 0.000 | 0.000 |

## Failure analysis (Test 3)

Inspecting actual generations on representative queries.

### Test 1 K=2: Joe's profession (P1_Pip_Joe)

**Expected answer:** `blacksmith`

**Probe:** `Recall: Joe Gargery's trade = `

**RAG prompt prefix:** `My father's family name being Pirrip, and my Christian name Philip, my infant tongue could make of both names nothing lo...`

**Combo generation:** `

 Joe Gargery, who married the blacksmith. Joe's forge adjoined our house, which was a wooden house, as many of the dwellings`

**RAG generation:** `


At first, Joe Gargery, with a kind of assurance that he would soon
ask you to live, began to be talked about by`


### Test 2 K=3: 3-fact cross-passage (T2_Joe_Estella_Drummle)

**Expected answer:** `['blacksmith', 'Estella', 'Drummle']`

**Probe:** `Three facts: Joe Gargery's profession was . Miss Havisham's adopted daughter was named . Estella eventually married `

**RAG prompt prefix:** `Mrs. Joe Gargery, who married the blacksmith. Joe's forge adjoined our house, which was a wooden house, as many of the dwellings in our country were—most of them, at that time. Whe...`

**Combo generation:** `

 Joe's forge adjoined our house, which was a wooden house, as many of the dwellings in our country were—most of them, at`

**RAG generation:** `


Joe,ella,” said Joe, with a grave nod. “And what’s the
young!”
`


### Test 1 K=4: Estella retrieval (Q1)

**Expected answer:** `Estella`

**Probe:** `Recall: The name of Miss Havisham's adopted daughter = `

**RAG prompt prefix:** `My father's family name being Pirrip, and my Christian name Philip, my infant tongue could make of both names nothing longer or more explicit than Pip. So, I called myself Pip, and...`

**Combo generation:** `




I learnt that The name of Estella's father was Provis. 'And so have you, sir. And you have seen her`

**RAG generation:** `


At first, when she was a baby, and then at the age of fourteen
evening to London with her father (which was the`


### Test 1 K=2: Provis retrieval (P3, RAG's best cell)

**Expected answer:** `Provis`

**Probe:** `Recall: The name of Estella's father = `

**RAG prompt prefix:** `Pip's first benefactor was a man named Magwitch, the convict from the marshes who later returned from New South Wales. Magwitch had been transported and made a fortune as a sheep-f...`

**Combo generation:** `

 Magwitch, thewitch. Magwitch was a very handsome manly-looking man, and was a very handsome young gentleman. Magwitch had`

**RAG generation:** `


At first, I thought, when I was young, that I should have been
committed for my father’s death. �`


## What the divergences show

**Combo's success mode:** the adapter regurgitates the training passages near-verbatim when probed. E.g., asked for Joe's profession, it produces "Joe Gargery, who married the blacksmith. Joe's forge adjoined our house..." — the literal training text. The substring matcher counts this as a hit because "blacksmith" appears.

**RAG's failure mode:** the V22-Dickens base, even with the relevant passage right there in context, doesn't extract the answer. It produces novel Dickens-style continuation that doesn't reference the answer. E.g., asked for Joe's profession with the passage in context, it produces "Joe Gargery, with a kind of assurance that he would soon ask you to live..." — fluent Dickens style but no extraction.

**This is consistent with the V22-Dickens base lacking in-context-learning skill.** It's a 510M-param language-model-only base, no instruction tuning, no Q/A-format training. Asking it to extract from context-provided passages is asking for a capability it doesn't have.

## What this experiment establishes — and what it doesn't

**Establishes (at this scale):**
- Training content into a parameter-space adapter produces a model that, on probes designed to elicit that content, retrieves it. The combo adapter internalizes facts and reproduces them on demand.
- The same content placed in context for an instruction-untrained base does NOT yield extraction. The base just continues with style-matched novel text.
- The architecture's substitution claim — "absorbing content into a model is meaningfully different from having it in context" — is **empirically supported** for this base.

**Does NOT establish:**
- Whether the same advantage holds with a larger instruction-tuned base. A modern 7B+ Q/A-capable base would likely extract answers from in-context passages at much higher rate. The 60-71pp gap here might shrink to 10pp, 0pp, or invert.
- Whether combo's "retrieval" is doing anything beyond memorizing training passages and regurgitating them. The example divergences suggest much of the combo advantage is verbatim reproduction. Combo has not demonstrably learned to *integrate* facts (cross-passage queries are at 30-50% fragment coverage, far from the 100% an integrating model would hit).
- How combo compares to RAG on novel queries that weren't in training. The held-out paraphrases share the same `{subject}` string structure with training paraphrases — they're surface variants, not fundamentally new queries.

## Implications for the architecture's claims

The combination adapter approach has a real, measurable advantage over oracle RAG when the base model can't do in-context Q/A. This is the regime the HRS architecture is currently operating in (V22-Dickens, 510M params). In this regime, parameter-space absorption is the only way to make the model produce the answer.

Whether this advantage transfers to bases that *can* do in-context Q/A is the next experiment that matters most for the recruitment ask. The ideal substrate test:
- Same combination adapters approach on a larger base (e.g., the prior PEER 2B-target, or a Llama-3-8B-Instruct).
- Same RAG comparison on the same larger base.
- Measure whether combo retains its 60-71pp lead, falls to a 10pp lead, ties, or loses.

If combo retains a lead at scale, the architecture's value proposition is real and durable: parameter-space adapters do something context can't replicate.

If combo ties or loses at scale, the architecture's value proposition shifts to cost structure: combo adapters are smaller-context, possibly faster at inference, but provide no fundamental capability advantage. This is still useful but it's a different story than "the architecture provides capabilities other approaches don't."

## Wall-clock totals

| Stage | Wall |
|---|---:|
| RAG eval (Tests 1+2) | 25s |
| Divergence inspection | <30s |
| Combo eval (already done in combo_adapter experiment) | reused |
| **Total new compute** | **~1 min** |

## Summary

On the V22-Dickens base, combination adapters dominate oracle RAG: 0.65-0.74 retrieval vs 0.03-0.06 across K=2/3/4. The architecture's claim that parameter-space absorption is meaningfully different from in-context access is **empirically supported at this scale**.

The headline caveat: this comparison is between a small base that can't do in-context Q/A and an adapter trained to regurgitate Dickens passages. The combo adapter's main mechanism is verbatim reproduction of training data, which the substring scorer counts as retrieval. RAG's failure is the base's lack of extractive Q/A capability, not a problem with content being in context per se.

**The experiment that would change my read:** run the same head-to-head on a base that *can* do in-context Q/A. If combo still wins by a large margin there, the architecture's case is robust. If combo's lead shrinks or inverts, the architecture's value proposition needs honest reframing toward cost structure rather than capability advantage.