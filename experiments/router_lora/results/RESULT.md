# Router-plus-LoRA selective fact storage — Tiny Shakespeare base, Dickens ingest

## Summary up front

**Negative result, informative kind.** None of the three conditions produced
meaningful Dickens fact recall on the 50-query evaluation set. Counting only
queries whose answer is ≥ 2 characters (i.e., excluding Q27's trivial
single-letter "I" answer that matches almost any English text):

| condition | n | mean recall | std | seeds |
|--|--:|--:|--:|--:|
| **A — learned router + LoRA** | 3 | **0/49** | 0 | 0, 0, 0 |
| **B — random router + LoRA** | 3 | **0/49** | 0 | 0, 0, 0 |
| **C — no-LoRA baseline** | 3 | **0/49** | 0 | 0, 0, 0 |

(Each condition technically scored 0.02 = 1/50 raw, but every "hit" was the
same false positive: Q27 has answer = "I", which any English text matches.)

Per the spec's headline question — **does Condition A significantly outperform
Condition B? No.** Per the secondary question — **does Condition B
significantly outperform Condition C? No.** The architecture as specified does
not produce meaningful selective fact storage at this scale.

The rest of this document:
1. The setup and the two implementation deviations needed to keep training
   stable
2. Why the experiment failed — three distinct mechanisms visible in the
   training history
3. What this tells us about the specification's hypothesis
4. Caveats and what would need to change to give the architecture a fairer
   test

## Setup

- **Base model**: 6-layer transformer, d=256, 4 heads, char-level Tiny
  Shakespeare, ctx_len=512 (extended from 256 to fit longer Dickens
  passages). Trained for 3000 steps to val_ppl 4.82. Frozen for all
  conditions.
- **Dickens content**: Great Expectations Chapters 1+2, normalized to the
  65-char Shakespeare vocabulary (smart quotes → straight; em-dashes →
  hyphens; digits other than `3` spelled out; no OOV chars after
  normalization). Split into 82 passages of 220–400 characters each.
- **Queries**: 50 hand-authored completion-style probes drawn from
  passage content. Each is a short prefix ending just before a specific
  factual answer (named entity, attribute, quantity, etc.). Substring
  match (case-insensitive) of `answer` in the next 60-character greedy
  continuation. Authored from the actual Dickens passages by reading
  them — no LLM-API call.
- **Router**: 256→128→1 sigmoid MLP on post-attention hidden states at
  layer 4. Outputs per-position routing weight in [0,1].
- **LoRA target**: layer-4 MLP first linear (d_model → d_ff), rank 8.
  Initialized as `(N(0, 0.02), zero)` so initial LoRA contribution is 0.
- **UpdateMechanism**: 256-dim weighted-mean of post-attention hidden
  states → two linear projections producing `ΔA: (8, 256)` and
  `ΔB: (1024, 8)` flat. Init std=0.001, learnable scale (init 0.1).
- **Training**: 5 epochs × 82 passages = 410 backward steps per
  condition. AdamW lr=1e-3, betas=(0.9, 0.95). Loss = `pass2 - pass1 +
  λ · mean(router_weights)` with **λ=0.001** (reduced from spec's 0.01
  after first attempt collapsed the router to ~0).
- **Conditions**:
  - A: learned router + UpdateMechanism + LoRA accumulation
  - B: router output replaced with `Uniform[0,1]`; UpdateMechanism + LoRA
    still trained/accumulated
  - C: no ingest at all; frozen base evaluated directly

## Implementation deviations from spec

Two material deviations, both forced by the first attempt failing in a
specific way:

### 1. LoRA matrices are buffers, not optimizer parameters

**Spec implication**: "The router and adapter are trained jointly via a
two-pass training loop... This loss flows back through both the router
(improving routing decisions) and the LoRA update mechanism (improving
how routed content gets stored). The base model's weights stay frozen."

**Problem**: if the LoRA matrices A and B are in the optimizer's parameter
list, the optimizer minimizes `pass2_loss - pass1_loss` by *inflating
pass1_loss* — the gradient on A, B is "make the base case worse so the
delta-fixed case looks better by comparison." First attempt: pass1_loss
exploded from 2.0 to 33.4 over 410 steps; pass2_loss followed it up;
the "improvement" was a 31-PPL artifact.

**Fix**: A, B are kept as Parameters but excluded from the optimizer.
After each step's `opt.step()`, the detached `delta_A, delta_B` are
added to `lora_A.data, lora_B.data` in-place (with no autograd tracking).
This means A, B accumulate via the trained UpdateMechanism's outputs but
the optimizer has no direct gradient on them, eliminating the gaming.

### 2. Per-step delta is L2-norm-capped at 0.05

**Spec implication**: per-step deltas should accumulate freely into A, B
across passages.

**Problem**: with the spec's recipe (UpdateMechanism init scale 0.1,
unbounded delta magnitude), the running LoRA norm grew to 4–9 after 410
steps. At that magnitude the model's outputs degraded to repetitive
gibberish ("the the the the the the..."). Generation became unable to
produce any text at all — Dickens or Shakespeare.

**Fix**: each step's delta is rescaled if its L2 norm exceeds 0.05
before accumulation. With this cap, the LoRA norm grew more slowly
(though still substantial: lora_A norm 4.5–6.0, lora_B norm 5.9–8.7
after 410 capped accumulations), and outputs were less degraded but
still gibberish in the learned/random conditions (see below).

## Why the experiment failed — three mechanisms

The training-history JSON files reveal three distinct failure modes
operating in parallel.

### Mechanism 1: learned router collapses to zero

```
condition=learned  router_w start→end:  0.51 → 0.13 (seed 0)
                                        0.51 → 0.11 (seed 1)
                                        0.49 → 0.11 (seed 2)
```

The sparsity penalty `λ · mean(weights)` provides a stable downward
gradient on the router output. The differential loss
`pass2 - pass1` provides a noisy and weak countervailing signal —
`Δloss` ranged ~±0.01 per step, while the sparsity gradient at λ=0.001
provided ~0.001 per element which over 256 positions × per-step is
substantial. The sparsity dominates and the router learns to route
nothing.

Reducing λ further (to 0.0001 or 0) might keep the router active, but
without a strong positive signal favoring "route this content to the
adapter," lowering the regularizer just leaves the router at its
init-uniform 0.5 — which is what condition B (random) effectively does.

### Mechanism 2: LoRA accumulation produces gibberish, not Dickens

In conditions A and B, the LoRA accumulates 410 deltas. Even with the
0.05-norm cap, that's 410 × 0.05 / step ≈ 20 in cumulative norm
(empirically: lora_A 4.5–6, lora_B 6–9 — somewhat lower than the
worst case because deltas don't all align). This magnitude is large
relative to the normal-distribution scale of the base model's weights
(~0.02 std).

But the accumulated deltas don't encode "the marshes" or "Pip" or
"Pirrip" — they encode whatever 256-dim compressed summary the
UpdateMechanism produced for that specific passage at that point in
training. Sample generation under condition learned/seed 0:

```
probe:        "My father's family name being "
expected:     "Pirrip"
generation:   "an ane the the the the the the the the the the the..."
```

The LoRA disrupted the base model's distributions enough to drown out
the Shakespeare prior, but didn't replace it with Dickens-shaped
outputs. The accumulated update is generic noise that pushes the model
toward repetitive low-entropy continuations.

### Mechanism 3: condition C produces only Shakespeare-shaped continuations

Without LoRA (condition C), the base model is exactly the
Shakespeare-trained TinyTransformer. Sample generation:

```
probe:        "My father's family name being "
expected:     "Pirrip"
generation:   "the state,\nAnd then the sentence of the second souls..."
```

Confidently Shakespearean. No path to Dickens content. Recall = 0/49
on real queries (the 1/50 raw is Q27's "I" false positive, identical
across all three seeds because there's no stochasticity).

## What this tells us about the specification's hypothesis

The spec's motivating argument was that the V18 retrieve-but-don't-
ground gap could be addressed by storing facts directly in adapter
weights rather than as cross-attention queryable representations.
The result here doesn't refute that hypothesis at scale, but it
shows the proposed mechanism doesn't deliver fact storage at this
small-scale proof-of-concept.

Two specific findings in the small-scale failure that may or may
not generalize to larger scale:

1. **The differential loss `pass2 - pass1` is too noisy a signal to
   train a fact-storage mechanism.** The improvement-magnitude per
   step (~0.01 PPL) is comparable to the noise floor of evaluating the
   model's loss on a single passage with a single forward pass.
   Multiple passes / averaging would help, but at that point the
   training cost grows substantially.
2. **Per-step accumulation of UpdateMechanism deltas does not encode
   passage-specific facts.** The UpdateMechanism's bottleneck (256-dim
   compressed → flat low-rank update) cannot represent "after the
   prefix 'My father's family name being', emit the 6-character string
   'Pirrip'." That kind of behavior requires either richer
   representations on the per-passage update path, or a mechanism that
   directly addresses the LM-loss surface (i.e., conventional
   fine-tuning), not a learned compression bottleneck.

Said differently: the architecture is closer to "meta-learn a generic
content adapter" than to "memorize this specific fact." For
fact-recall in a tiny model, the simplest baseline that would work is
direct LoRA fine-tuning on the passage reconstruction loss — which
this experiment doesn't test, and which would be a different
architecture (no router, no UpdateMechanism, just gradient descent on
A, B against the LM loss).

## Caveats and what would change my mind

- **The base model is char-level on Shakespeare.** Char-level
  prediction has very high local determinism — most predictions are
  determined by the previous 1–4 characters, leaving little room for
  injected long-range factual content to matter. A BPE-tokenized base
  with longer-range dependencies might give the LoRA more leverage.
- **Tiny Shakespeare's vocabulary is small** (65 chars). Names like
  "Pirrip" exist as character sequences in the training distribution
  (P-i-r-r-i-p occurs nowhere in Shakespeare), so memorizing them
  requires the LoRA to push significant probability mass onto specific
  character transitions. Easier at the BPE level where "Pirrip" is one
  or two tokens.
- **Only 410 ingest steps.** Conventional fine-tuning of a transformer
  on a small passage set typically uses thousands of steps. 410 might
  be too few for the meta-learning regime to converge.
- **The UpdateMechanism is an information bottleneck.** Compressing a
  passage to a single 256-dim vector before producing the (8 × 256 +
  1024 × 8 = 10240-dim) LoRA update may be too lossy. A token-level
  update (per-position contributions to A, B) would have much more
  capacity but breaks the "router selects, mechanism stores" framing.

## Files

- `data/great_expectations_full.txt` — Project Gutenberg source
- `data/ge_ch12_passages.txt` and `.json` — 82 normalized passages
- `data/queries.json` — 50 completion-style probes with answers
- `results/base_shakespeare.pt` — base model checkpoint (val_ppl 4.82)
- `results/{learned,random,no_lora}_seed{0,1,2}.json` — 9 per-run records
  with per-query results + history
- `results/summary.json` — aggregated recall numbers
- `results/RESULT.md` — this writeup
- Code: `experiments/router_lora/{model.py, train_base.py,
  prepare_data.py, run_experiment.py}`

## Budget

Total wall: 70s for the 9-run sweep on a 5070 Ti, plus 117s base
training. Well under the spec's 1-hour estimate. Not the bottleneck.

## Recommendation for next step

Before re-attempting this architecture, the simplest informative
follow-up is the missing baseline: **direct LoRA fine-tuning on the
Dickens passages with no router, no UpdateMechanism, just gradient
descent on (A, B) against the per-token LM loss.** If that produces
meaningful recall, the question becomes "what does the router add."
If even direct fine-tuning at rank 8 / 410 steps doesn't produce
recall, the storage capacity is the bottleneck — neither this
architecture nor the spec's variants will work without more.
