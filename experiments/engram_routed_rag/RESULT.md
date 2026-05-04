# Engram-Routed RAG on Dickens-50

**Headline: RAG hurts the architecture sharply. The hypothesis is wrong
in the direction predicted, but for an unexpected reason — and the
wrong-passage control reveals more than the correct-passage condition
does.**

## Setup recap

Five conditions on the existing per_passage_dickens infrastructure (50
rank-128 LoRA adapters trained per passage), 150 held-out probes, 3
seeds. Adapter selection is ORACLE for every adapter-using condition
(original routing was 100%, so this isolates the inference-time
question). RAG injection format: `{passage}\n{probe}` (no special
template). Passages are short (max 109 tokens) — no truncation.

## Results

| Condition | Description | Retrieval (mean ± std) |
|---|---|---|
| C1_baseline_adapter_only | correct adapter + prompt | **0.929 ± 0.016** |
| C2_correct_adapter_correct_rag | correct adapter + correct passage + prompt | **0.676 ± 0.014** |
| C3_no_adapter_correct_rag | base model + correct passage + prompt | 0.036 ± 0.008 |
| C4_correct_adapter_wrong_rag | correct adapter + wrong passage + prompt | 0.809 ± 0.019 |
| C5_floor | base model + prompt only | 0.002 ± 0.003 |

Per-seed: C1 [0.940, 0.907, 0.940], C2 [0.693, 0.673, 0.660], C3 [0.027,
0.033, 0.047], C4 [0.787, 0.807, 0.833], C5 [0.007, 0.000, 0.000].

### Per-probe deltas (out of 150 probes, summing 3 seeds each)

| Comparison | gain | match | lost |
|---|---|---|---|
| C2 vs C1 (adapter+correct-RAG vs adapter only) | 13 | 55 | **82** |
| C3 vs C1 (RAG only vs adapter only) | 0 | 1 | 149 |
| C4 vs C1 (adapter+wrong-RAG vs adapter only) | 13 | 81 | 56 |
| C2 vs C3 (adapter+RAG vs RAG only) | **134** | 14 | 2 |

C1 had partial/full failure on 29 probes. C2 fixed 13 of those — but
introduced 82 NEW failures elsewhere. The net effect is sharply
negative.

## What the hypothesis predicted vs what happened

| Prediction | Result |
|---|---|
| 1: C1 reproduces ~93% | ✓ 0.929 |
| 2: C2 substantially beats C1, target 97-100% | ✗ C2 = 0.676, **25 points below C1** |
| 3: C3 (RAG only) reaches 80-90% | ✗ C3 = 0.036, **near floor** |
| 4: C4 (wrong RAG) drops to 50-70% | partial: C4 = 0.809, drop of 12 pts (smaller than predicted) |
| 5: C5 ≈ 0% | ✓ 0.002 |

Predictions 2, 3, and the magnitude of 4 were all wrong. The shape of
the error is informative.

## The unexpected pattern: C4 (wrong passage) > C2 (correct passage)

**The wrong passage hurts less than the correct passage** by 13
percentage points. Plain-RAG hypotheses don't predict this. RAG-only
mechanisms predict the opposite — that the correct passage gives the
model literal access to the answer string, while the wrong passage
provides misleading content. We see the opposite ordering.

The mechanism that explains this, supported by the sample generations
below: the adapter was trained on two distributions only:
- raw passage text (next-token LM)
- (probe + answer) strings (next-token LM)

It was never trained on `{passage}\n{probe}`. When the input is just the
probe, the adapter executes the (probe → answer) pattern it learned. When
the correct passage is prepended, the model sees passage tokens
immediately before the probe and is pulled into "continue the passage"
mode — narrative continuation that the adapter's training cannot break
out of cleanly. Garbled output results.

When the wrong passage is prepended, the topic mismatch creates a
coherence break at the probe boundary that the adapter can sometimes
recover from, producing the answer. Hence C4 > C2.

This is not an Engram Effect failure or a RAG mechanism failure — it's
adapter brittleness to input distribution shift. The adapter is over-fit
to the format it was trained on.

## Sample generations (seed 0)

Probes where C1 hit but C2 missed (the 41 "RAG-induced failures"). For
each: probe text, ground-truth answer, then the four condition
generations.

### Probe 2 (entity): "It is well-known that Pip's father's family name is " → `Pirrip`

- **C1 (adapter only):** `Pip, and who is recorded that Pip's father's family name was Pirrip, and was recorded as Pirrip.` ✓
- **C2 (adapter + correct RAG):** `-to-no-dun-to-te-no-no, and my Christian name Philip. Joe Gargery. Joe Gargery,` ✗ — garbled, then Dickens-style narrative continuation
- **C3 (RAG only):** `Chapter I don't know, because my father is my mother." I said I had been in my mother...` ✗ — base model produces unrelated text
- **C4 (adapter + wrong RAG, from aid 27):** `urn, and my Christian name Philip, my infant tongue could make of both names nothing longer or more explicit than Pip. So, I called myself Pip` ✗ — narrative continuation (verbatim from passage 0!) but doesn't reach "Pirrip"

### Probe 7 (entity): "On reflection, Joe Gargery's trade was " → `blacksmith`

- **C1:** `my own. Joe Gargery's trade was blacksmith. Joe's, blacksmith, and Joe, sister of Joe.` ✓
- **C2 (correct RAG):** `I have had a long history of having been a ship of our whole career that was a part of the miserable Christian name of` ✗ — drifts into unrelated narrative
- **C3:** `Joe, by the very last word of the article, "No," and by her to the fire.` ✗
- **C4 (wrong RAG, aid 26):** `urn, blacksmith, a blacksmith, who was married to a blacksmith. The house was a wooden house, and was home to the house` ✓ — adapter wins, produces "blacksmith"

### Probe 5 (entity): "It is well-known that Pip's mother's name is " → `Georgiana`

- **C1:** `ervally Georgiana Wife of the name of Georgiana. Georgiana was in the same motherly name` ✓
- **C2 (correct RAG):** `urna-to-no word (can, I gave them so much admiration in the case of the god of love and the name of the king` ✗
- **C3:** `However, Joe Gargery, and I were so communicative that I would not see them` ✗
- **C4 (wrong RAG, aid 33):** `ia Georgiana. Georgiana was a name and mother. Georgiana was a girl who was Georgiana. Georgiana was the name of a man` ✓ — adapter pattern survives the wrong-passage prefix

The pattern is consistent: C2's failures are not "fluent paraphrases
that miss the substring" — they are sampling-noise garbage or off-topic
narrative drift. The adapter's training distribution was narrow enough
that any prefix outside that distribution destabilizes it.

## What this tells us about the architecture

The decision tree the spec laid out:

> *"If C2 ≈ C1, no improvement"* → 7-point gap is something else.
> Investigate failure modes individually.

is closest to what we actually have, except worse: **C2 is well below
C1**, not equal. So it's not just that the gap is unfixable by RAG —
RAG actively destroys most of the architecture's performance.

The spec's other branches:

> *"If C3 ≥ C1, the adapter is doing nothing the retrieved text doesn't
> already do."*

Decisively rejected. C3 = 0.036, C1 = 0.929. The adapter does
essentially all the inference-time work. The base model with the source
passage in context cannot answer these probes.

> *"If C4 substantially hurts, the model genuinely reads the retrieved
> text."*

Half-confirmed. C4 hurts (0.809 vs 0.929) but less than C2 — the model
does read the text, but the on-topic correct passage is more disruptive
than an off-topic wrong passage. This is the opposite of what
"reading the text" would predict if the model were doing literal
retrieval from the source.

## Concentration analysis

Of the 29 probes where C1 had at least one failure across 3 seeds, C2
improved on 13 and C3 improved on 0. The 13 C2 wins are real but cost
82 new failures elsewhere — net loss of 69 successful (probe, seed)
pairs. The 7-point baseline gap is not concentrated in probes that RAG
fixes; the failures are scattered, and adding RAG creates more
scattered failures.

## Architecture interpretation

The architecture as built is not RAG-compatible. The adapter was trained
on a narrow input distribution and is brittle to anything outside it,
including the same passage text it was trained on but in a different
position relative to the probe.

Combined with the prior result from `experiments/engram_injection_dickens/`:

- Engram injection at inference: null/slightly negative (correct = wrong, both ≈ 0.92)
- Passage RAG at inference: sharply negative (~0.68 for correct, ~0.81 for wrong)

The adapter is doing all the inference-time work. The engram is purely
a routing key. Adding either an engram prefix OR a passage prefix at
inference moves the input out of the adapter's training distribution
and degrades performance — passage prefixes more so because they're
larger and more coherent perturbations.

## What follow-ups are warranted

The spec's note ("If this works as predicted, three-stage architecture")
doesn't apply. The two results that DO follow from this:

1. **The adapter's training distribution matters more than its content.**
   Retraining adapters on `{passage}\n{probe} → {answer}` (or just
   `{passage}\n{probe + answer}`) would test whether adapters trained
   for the RAG-compatible format can use the retrieved text. This is
   the natural next experiment if the goal is to recover the
   engram-routed-RAG architecture.

2. **The 7-point gap (the 7% of probes where adapter alone fails) is
   not addressable by source retrieval in this architecture.** If those
   failures need to be fixed, the path forward is one of: longer
   adapter training, larger rank, different LoRA target modules, or
   better answer-format training. RAG is not the lever.

## Failure mode flagged in the spec, addressed

Spec: *"If sampling is stochastic, the per-probe outcomes will vary
across seeds. Three seeds should give a reasonable estimate, but be
cautious about claiming 100% retrieval if any single seed produced
misses."*

No condition reaches 100%. C2's 0.676 is consistent across seeds
(σ=0.014). The seeds are not the issue.

## Files

- `experiments/engram_routed_rag/run_conditions.py` — five-condition script
- `experiments/engram_routed_rag/results/summary.json` — aggregate + deltas
- `experiments/engram_routed_rag/results/eval_C{1..5}_*_seed{0,1,2}.json` — per-probe details
