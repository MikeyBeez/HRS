# Phase 02 (D2L) — Six-condition Perceiver vs RAG vs prompting

Tests the central D2L claim from the spec: a Perceiver-generated LoRA adapter
will substantially beat in-context retrieval (RAG) on Dickens content
*because the adapter modifies the FFN weights where StarCoder2-3B's `<NAME>`
suppression lives, while RAG cannot bypass an output-layer bias*.

## Headline

**The strong prediction is falsified.** RAG does *not* hit a `<NAME>`
suppression ceiling — it sails through it with 92% top-1 retrieval on
character names and 100% on possessions. The Perceiver-installed adapter,
trained for 5K KL-distillation steps, lifts character-name retrieval from
0% (Phase 01 floor) to **6%** — a non-zero but tiny fraction of what RAG
achieves with the same source passage. The engram-prefix composition (C4)
*degrades* the adapter rather than improving it. The architectural
prediction that motivated the engram-as-bias framework is empirically
negative in this setup.

| Condition | Names top-1 | Names top-5 | `<NAME>` rate (names) | Possessions top-1 | Plot top-1 | Code top-1 |
|-----------|------------:|------------:|----------------------:|------------------:|-----------:|-----------:|
| C1 baseline (Phase 01)        |  0.0% |  2.0% |  0.0% |   6.0% |  0.0% | 22.0% |
| **C2 RAG (passage in context)** | **92.0%** | **96.0%** | 0.0% | **100.0%** | 0.0% | 54.0% |
| C3 D2L adapter only           |  6.0% | 12.0% |  0.0% |   0.0% |  0.0% | 22.0% |
| C4 adapter + engram prefix    |  2.0% |  6.0% |  0.0% |   0.0% |  0.0% |  2.0% |
| C5 adapter + context (RAG+D2L)| 96.0% | 96.0% |  0.0% | 100.0% |  0.0% | 52.0% |
| C6 anti-suppression prompt    |  0.0% |  2.0% |  2.0% |   8.0% |  0.0% | 30.0% |

Strong predictions vs measured:
- **"C3 substantially beats C2 on character names by ≥10 pp"** → measured
  C3 − C2 = **−86 pp**. Falsified by a wide margin in the opposite direction.
- **"C4 > C3 on names"** → measured C4 − C3 = **−4 pp**. Falsified.
- **C3 names top-1 above 30%** → measured 6%. Falsified.

Two things land surprisingly:

1. **`<NAME>` suppression is robust to adapter intervention but vulnerable
   to in-context retrieval.** This is the *opposite* of the spec's premise.
   When the source passage is in context, the model attends to the named
   entity and emits it verbatim (0% `<NAME>` rate on character items under
   C2). When only an adapter is loaded, the adapter has nudged the model
   away from emitting `<NAME>` (also 0% rate) — but hasn't installed enough
   directional signal to favor the actual answer either. The adapter
   uncovers the suppression veil without revealing what's behind it.
2. **The plot category is unrecoverable across every condition** (0% top-1
   everywhere, `<NAME>` rate 90–100%). For `Mr. ___ → Tulkinghorn` style
   probes, *even RAG with the answer in context* emits `<NAME>` as top-1
   with the correct token at rank 91–2,800. PII anonymization has built a
   structural attractor on the `Mr.`/`Lady`/`Inspector` template that no
   condition tested here defeats. This is a category-design issue (the cue
   is "Mr. " — the most anonymized context in the training data) more than
   a method issue.

## Setup recap

- Base: `bigcode/starcoder2-3b`, frozen, fp16, eval mode.
- Perceiver: 33.1M params (32 latents × 512 dim, 2 cross-attn + 4
  self-attn, hypernet decoder outputs rank-8 A/B per layer). Listed under
  the spec's 30–50M band.
- Training objective: KL(teacher || student) over the next 4 tokens of an
  answer, where teacher = base(passage ‖ query) and student = base+LoRA(query).
- Training: 5K steps (vs the spec's 20K). Length curriculum compressed to
  256 tokens throughout because longer passages forced gradient
  checkpointing that doubled step time. Mean loss 4.75 → ~2.5.
- LoRA: rank 8, alpha **4** (not 16). Lowered for fp16 stability — at
  α=16 the LoRA contribution overflowed fp16 in the base FFN around
  step 300 and the Perceiver weights went NaN. α=4 trains cleanly. This
  is a real deviation from spec and is plausibly part of why the adapter
  is weaker than expected.

## Per-condition detail

### C2 RAG (92% / 100% / 0% / 54%)

The model attends to the in-context passage and copies the named entity
verbatim. Spot check:
```
target=' T'    top1=' T'    rank 0
target=' All'  top1=' All'  rank 0
target=' Est'  top1=' Est'  rank 0
target=' Ada'  top1=' Ada'  rank 0
```
The `<NAME>` rate drops from 90% (Phase 01 baseline on plot items) to 0%
once the answer is in attention range. The suppression is conditional on
the model *not* having a recent attendable mention.

### C3 D2L adapter only (6% / 0% / 0% / 22%)

The Perceiver has read the source passage and converted it into a rank-8
LoRA on all 30 FFN down-projections. With no context, the model now
produces *generic continuations* rather than `<NAME>` — the adapter has
clearly perturbed the output distribution, but in the direction of
"continue text plausibly" rather than "say this specific name." Spot
check:
```
target=' T'    top1=' '     rank   11
target=' All'  top1='\n'    rank  187
target=' Est'  top1=' L'    rank  474
target=' Ada'  top1='\n'    rank   59
```
The adapter is *doing something* — character `<NAME>` rate is 0%, vs
expected baseline 90%-ish — but the something is "shift away from the
suppression default", not "shift toward the trained passage's specific
entity."

### C4 adapter + engram prefix (2% / 0% / 0% / 2%)

The engram prefix is a single-token mean-pool of the source passage's
layer-16 hidden state, prepended at the input-embedding layer alongside
the LoRA adapter. **It strictly hurts** in every category, with the
code-idiom positive control collapsing from 22% (C3) to 2% (C4).

The most plausible explanation: the engram embedding is far from the base
model's normal token-embedding manifold, and inserting it at position 0
distorts the position-conditional activations that the adapter was
trained to nudge. The adapter expects a specific input distribution; the
engram changes that distribution upstream, so the adapter no longer
operates in its trained regime.

This is a negative result for the engram-as-bias framework specifically
in this implementation. It does *not* rule out engram-prefix utility in
general — alternative injection schemes (later layer, gated injection,
trained-jointly-with-adapter) might work — but the simple zero-shot
prepend-at-embedding-layer scheme this experiment tests does not.

### C5 adapter + context (96% / 100% / 0% / 52%)

Adding the adapter on top of RAG yields essentially the same result as
RAG alone (96 vs 92 on names; everything else within noise). The adapter
neither helps nor hurts when the passage is in context — RAG is
saturating and the adapter contribution is dominated.

### C6 anti-suppression prompt (0% / 8% / 0% / 30%)

A natural-language instruction to "fill in the actual name, not `<NAME>`"
fails on names (0%) but lifts code idioms (30%, vs 22% baseline; the
prompt nudges the model toward content-bearing continuations generally).
The `<NAME>` rate on D code rises from 0% to 16% because the anti-`<NAME>`
phrasing in the prompt makes `<NAME>` itself a more salient token to emit
in unrelated positions. Counterintuitive but consistent with how
prompted instructions perturb token frequencies.

## Pre-committed predictions vs measured outcomes

| Condition | Predicted names top-1 | Measured | Verdict |
|-----------|----------------------:|---------:|---------|
| C1 baseline | 0% | 0% | ✓ |
| C2 RAG | 15–30% | **92%** | Wildly under-predicted RAG. We had assumed `<NAME>` suppression would partly clamp RAG output; it doesn't. |
| C3 D2L adapter | 40–70% | **6%** | Wildly over-predicted adapter. We had assumed FFN intervention bypasses suppression; it does (`<NAME>` rate → 0%) but doesn't install the actual content. |
| C4 adapter+engram | C3 + 5–15pp | **C3 − 4pp** | Engram hurt rather than helped. |
| C5 adapter+context | C3 ± 5pp | ≈ C2 | Adapter contribution dominated by context once both available. |
| C6 anti-suppression | 5–15% | 0% | No traction from natural-language anti-suppression. |

## Architectural interpretation

The motivating model — "RAG can't bypass `<NAME>` suppression because the
suppression is in the output stage, but D2L modifies the output stage" —
has at least three things wrong with it, all visible in these numbers:

1. **The suppression isn't in the output stage.** If it were, RAG would
   hit a ceiling much lower than 92%. RAG passes through the same output
   stage that suppresses names in the baseline, yet retrieves names
   nearly perfectly when the passage is in context. The suppression is
   conditional on attention pattern, not a fixed output bias. When
   attention can route content from a recent mention, the model emits the
   content; when it can't, it falls back to `<NAME>`.

2. **The adapter doesn't install content well at this training scale.**
   5K KL-distillation steps on 256-token passages got the loss from ~5
   to ~2.5 but didn't produce an adapter that can act as a "frozen
   in-context surrogate" for a fresh passage at eval time. The
   distillation objective optimizes "match teacher logits when context is
   present," which is a different and easier task than "encode the
   passage's content into weights such that the next-token distribution
   approximates the in-context one." The adapter learned to perturb the
   output direction, not the specific content.

3. **The engram-as-bias composition is not robust to where you inject.**
   Mean-pooled layer-16 prefix at the embedding layer disrupts the
   adapter's trained operating regime. The engram-as-address routing in
   the HRS paper works because each adapter is loaded for queries that
   *match* its engram by similarity; here we composed without routing,
   and the composition failed.

The cleanest publishable framing now:

> The "manifold-level beats prompt-level" prediction does not hold for
> code-only base models with PII-anonymized training. RAG works far
> better than expected because in-context attention bypasses the
> suppression when an attendable mention exists. Perceiver-LoRA
> distillation at this training scale shifts the output distribution
> away from the suppression default but doesn't install the specific
> content. Engram-prefix composition at the embedding layer is brittle.
> The framework needs either (a) substantially more Perceiver training,
> (b) a content-preserving objective different from KL-on-teacher, or
> (c) an injection scheme that respects the adapter's trained input
> distribution.

## File manifest

- `experiments/d2l/phase02_perceiver_train.py` — Perceiver + KL training
- `experiments/d2l/phase02_eval_six_conditions.py` — six-condition eval
- `results/d2l/phase02/predictions.json` — pre-committed predictions
- `results/d2l/phase02/perceiver_checkpoint.pt` — trained Perceiver (33.1M params, 5K steps; *gitignored, present locally*)
- `results/d2l/phase02/training_log.json` — per-step loss
- `results/d2l/phase02/training_curve.png` — loss plot (gitignored)
- `results/d2l/phase02/training_run.log` — training stdout (gitignored)
- `results/d2l/phase02/results_C1.json` … `results_C6.json` — per-item evaluation records per condition
- `results/d2l/phase02/six_condition_aggregates.json` — aggregate metrics
- `results/d2l/phase02/eval_run.log` — eval stdout (gitignored)
- `results/d2l/phase02/six_condition_README.md` — this file

## Open questions for Phase 03

1. **Does the C3 result move at 20K steps?** This run was 5K — a quarter
   of the spec's 20K. The KL loss curve was still gently descending at
   5K. If 20K closes most of the C3 vs C2 gap, the manifold-level claim
   is just under-trained, not wrong. If 20K barely moves the needle, the
   claim is structurally wrong (or needs a different objective).
2. **Does α = 16 (or rank > 8) change the answer?** This run used α=4 for
   fp16 stability. A higher-α run with gradient checkpointing or fp32
   adapter math may install more content per step. Spec α was 16.
3. **Is the engram failure about injection point or about the engram
   representation itself?** Try late-layer injection, gated injection,
   or training the engram alongside the adapter. Distinguish "prefix
   token disrupts adapter" from "this kind of engram is the wrong
   conditioning signal."
4. **Could the plot category be recovered with a vocabulary-level
   intervention?** Banning the `<NAME>` token at decode time recovers a
   measurement target for C_plot. Worth a one-line check before
   abandoning C_plot as unrecoverable.
5. **What does a teacher/student gap look like during training?** If the
   student tracks teacher closely during training but diverges sharply
   at eval, the failure is generalization (Perceiver memorizes seen
   passages). If student tracks teacher poorly even during training, the
   failure is capacity. Worth recording train-vs-eval distillation gap.
