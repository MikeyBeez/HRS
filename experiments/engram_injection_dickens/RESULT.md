# Does adding engram injection beat the Dickens-50 93% baseline?

**Headline: No.** The published architecture — engram as routing key only,
no inference-time injection — is at the local optimum. Adding the engram
to the inference path does not help, and the load-bearing control shows
the engram contributes no specific signal at inference.

This writeup combines two related runs:
- `experiments/engram_inference_role/` — 5-condition diagnostic sweep
  (the original spec)
- `experiments/engram_injection_dickens/` — 3-condition focused test with
  the wrong-engram control (the follow-up spec)

Both runs use ORACLE adapter selection (the original Dickens-50 routing
was already 100%, so this isolates the inference-time question).

## Pre-run background check

**Adapter training procedure (case a, confirmed by reading
`experiments/per_passage_dickens/train_adapters.py:90-100`):** Each
rank-128 LoRA adapter is trained on (a) the raw passage text, next-token
LM, and (b) each (training paraphrase + answer) string, next-token LM.
**No engram is in the training input.** The adapter has never seen a
hidden-state prefix during training.

**Original Dickens-50 evaluation** (`experiments/per_passage_dickens/evaluate.py`):
generates from `tokenizer.encode(probe)` only. **Never injects the engram
at inference.** The L5 mean-pool key is used solely for routing (cosine
similarity for adapter selection).

**Therefore the published 93% retrieval is the spec's "Condition 1"
(adapter-only-at-inference)**, not "Condition 2." The original
architecture has no inference-time engram injection. The injected variant
is a NEW addition this experiment evaluates.

## Engram injection method

Single L5 mean-pool hidden state (the same `library_l5_aggregate[adapter_id]`
vector used for routing) prepended at position 0 via a raw-block forward
loop. Mirrors `experiments/identity_ae/phase33_engram_context.py`
exactly. Token embeddings then concatenated after.

This is the closest match to an "Engram Effect" injection method
available in the repo. There is no separate per-layer or per-position
injection scheme implemented for this architecture.

## Results

### Three-condition focused test (`experiments/engram_injection_dickens/`)

| Condition | Description | Retrieval (mean ± std, 3 seeds) |
|---|---|---|
| C1_baseline_no_engram | correct adapter + prompt | **0.929 ± 0.016** |
| C2_correct_engram_injected | correct adapter + correct adapter's engram + prompt | 0.913 ± 0.005 |
| C3_wrong_engram_injected | correct adapter + a *different* adapter's engram + prompt | 0.920 ± 0.011 |

Per-seed: C1 [0.940, 0.907, 0.940], C2 [0.913, 0.907, 0.920],
C3 [0.933, 0.907, 0.920].

**C1 reproduces the published 93%** (the original eval got 0.94/0.91/...
with mean 0.929 — exact match). Pre-registered Prediction 1 satisfied,
proceed.

**C2 and C3 are statistically indistinguishable from each other**
(0.913 vs 0.920). This is the spec's Prediction 4 hit: *"If C2 = C3, the
mechanism isn't what the Engram Effect paper predicts."* The engram is
not contributing principal-direction signal at inference; whether it's
the "right" engram or a random other adapter's engram makes no
meaningful difference.

**Both injected variants slightly underperform the no-injection
baseline** (~0.013-0.016 below). This is consistent with mild
distribution-shift noise: the adapter was trained without any
hidden-state prefix, and any prefix vector — correct or wrong — is
out-of-distribution input.

### Per-probe deltas (out of 150 probes, summing 3 seeds each)

| Comparison | gain | match | lost |
|---|---|---|---|
| C2 vs C1 (correct engram) | 16 | 114 | 20 |
| C3 vs C1 (wrong engram)   | 14 | 120 | 16 |

Roughly symmetric gain/loss for both variants. No probe-class is
systematically helped by the correct engram. The C3 control matches the
C2 pattern, again confirming the engram isn't doing specific
inference-time work.

### Five-condition diagnostic (`experiments/engram_inference_role/`)

| Condition | Retrieval (mean ± std, 3 seeds) |
|---|---|
| C1_engram_plus_correct_adapter | 0.913 ± 0.005 |
| C2_correct_adapter_only [= ORIGINAL] | **0.929 ± 0.016** |
| C3_engram_only_no_adapter | 0.007 ± 0.005 |
| C4_engram_plus_wrong_adapter | 0.013 ± 0.005 |
| C5_floor_prompt_only | 0.002 ± 0.003 |

Additional findings:
- **C3 (engram alone, base model): 0.007.** The engram on its own
  contributes essentially nothing to fact retrieval. The fact-specific
  knowledge lives entirely in the adapter weights; the engram does not
  carry retrievable content.
- **C4 (correct engram + wrong adapter loaded): 0.013.** The wrong
  adapter dominates regardless of engram presence. The engram cannot
  override or rescue retrieval from the wrong adapter.
- **C5 (floor): 0.002.** Confirms the task is non-trivial and the
  base model has no memorized access to these facts.

The asymmetry between C3 (engram + no adapter = 0.007) and C5 (no engram
+ no adapter = 0.002) is well within noise; the engram alone is not
distinguishable from no prefix at all.

## Sample generations (seed 0)

### Probe 0 (entity): "Recall: Pip's father's family name = " → expected `Pirrip`

- **C1 (no engram):** `Casting my father's family name was Pirrip, and my father's family name was Pirrip, and my infant sister's family nam` ✓
- **C2 (correct engram, from adapter 0):** `C Watson's family name was Pirrip, and my Christian name Philip, my infant tongue could make of both names nothing lo` ✓
- **C3 (wrong engram, from adapter 25):** `Casting my father's family name was Pirrip, and my father's family name was Pirrip, and my infant sister Gargery.` ✓

### Probe 50 (entity): "It is well-known that The name of Miss Havisham's adopted daughter is " → expected `Estella`

- **C1:** `Havisham's adopted daughter was Estella, when was Estella, Miss Havisham's adopted daughter was Estella, and was` ✓
- **C2 (correct engram, adapter 16):** `Havisham's adopted daughter was Estella Estella, when was Estella, a daughter of Estella and Estella, and was Estella` ✓
- **C3 (wrong engram, adapter 26):** `Miss Havisham's adopted daughter was Estella Estella, was brought home from France, and was brought home from France,` ✓

### Probe 100 (entity): "On reflection, The name of Pip's village schoolteacher was " → expected `Wopsle`

- **C1:** `Wopsle's great-aunt, Wopsle's great-aunt letters Wopsle's great-aunt letters W` ✓
- **C2 (correct engram, adapter 33):** `acher by the name of Pip's village schoolteacher was Wopsle. The name was Wopsle's lettersteacher was Wopsle` ✓
- **C3 (wrong engram, adapter 2):** `Wopsle. Mr. Wopsle's great-aunt ran the village evening-school where Pip first learned letters. Mr. Wops` ✓

The wrong-engram condition produces qualitatively similar generations to
the correct-engram condition — including in probe 100 a near-verbatim
recall of the Dickens passage (`Mr. Wopsle's great-aunt ran the village
evening-school where Pip first learned letters`). The adapter is doing
the work; the engram prefix is decorative.

## Pre-registered prediction reconciliation

| Prediction | Result |
|---|---|
| 1: C1 reproduces 93% | ✓ 0.929 |
| 2: C2 improves to 95-98% | ✗ C2 = 0.913, slightly *below* C1 |
| 3: C3 hurts (below C1) | partial: C3 = 0.920, slightly below C1 but matches C2 |
| 4: If C2 = C3, mechanism isn't what Engram Effect predicts | ✓ this is the result |

Prediction 2 was the load-bearing positive hypothesis. It failed.
Prediction 4 (the alternative the spec flagged as "surprising negative")
is what we got.

## Architecture interpretation

The clean reading the spec proposed for this outcome: *"the adapter is
doing all the inference-time work that's available to be done. The
engram is purely a routing key, period."* This is what the data
supports.

Three-component story is wrong. Two-component story is correct:

1. **Routing system (engram):** L0→L5 projection + cosine vs library
   L5 keys → top-1 adapter selection. 100% accuracy on Dickens-50.
2. **Knowledge system (per-passage LoRA):** rank-128 adapter on layers
   4-5, trained per passage on raw text + (paraphrase + answer)
   strings. Does all the inference-time retrieval work.

The engram and the adapter are produced by separate procedures (engram =
pooled hidden state of training paraphrases; adapter = LoRA weights
trained on passage + paraphrases). They are independent except that the
engram is computed from the training paraphrases that also fed the
adapter. They could not have shared inference-time purpose without the
adapter being trained with engram in input — and it wasn't.

## Caveats

- **One injection method tested.** Single hidden-state at position 0 via
  raw-block loop. A different injection scheme (per-layer hidden
  injection, attention-key injection, multi-position prefix) might
  produce a different result. The Engram Effect paper's exact
  injection scheme is not implemented in this repo for this
  architecture.
- **Adapters were not trained with engram in input.** The fair test of
  the Engram Effect mechanism would require retraining adapters with
  the engram present at training time. The current C2 result is partly
  measuring train-test mismatch, not pure inference-time engram value.
  The spec's "investigate whether re-training adapters with engram
  present at training time fixes this" is the natural follow-up.
- **150 probes, 3 seeds.** Standard deviations are 0.005-0.016, so
  effects below ~2 percentage points are within noise. The C1-vs-C2
  gap (1.6 pts) is just outside noise; the C2-vs-C3 difference is
  inside noise.
- **Dickens-50 is single-fact substring-match retrieval.** The result
  may not extend to multi-fact synthesis or to QA tasks where the
  engram might do useful semantic priming.

## What this means for the architecture writeup

The corrected description of the Dickens-50 / Phase 47 architecture is:

> A library of per-passage rank-128 LoRA adapters, each trained on the
> passage text and on paraphrase+answer strings. At inference, an
> incoming probe is routed by cosine similarity in a learned L0→L5
> projection space against the library's L5 mean-pool keys; the
> selected adapter is loaded and the model generates the answer from
> the prompt. The engram is a routing key only — it is not present in
> the inference-time input, and adding it does not improve retrieval.

Two components, not three. Routing and knowledge. The system is
simpler and cleaner than the version implied by writeups that describe
the engram as a context-carrier.

## Files

- `experiments/engram_injection_dickens/run_conditions.py` — three-condition script
- `experiments/engram_injection_dickens/results/summary.json` — aggregate summary
- `experiments/engram_injection_dickens/results/eval_C{1,2,3}_*_seed{0,1,2}.json` — per-probe details
- `experiments/engram_inference_role/run_conditions.py` — five-condition diagnostic script
- `experiments/engram_inference_role/results/summary.json` — diagnostic summary
- `experiments/engram_inference_role/results/eval_C{1..5}_*_seed{0,1,2}.json` — per-probe details
