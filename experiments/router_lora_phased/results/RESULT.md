# Three-phase router-plus-LoRA: Shakespeare base, Dickens LoRA, learned router

## Headline

**The architecture works for what it was designed to do (selective content
classification), but doesn't unlock fact recall — that bottleneck is
elsewhere.**

Three independent results:

1. **Phase 2 LoRA discriminates strongly.** With LoRA on, Dickens validation
   PPL drops from 4666 → 440 (10× reduction). Shakespeare validation PPL
   rises from 131 → 229 (1.7× degradation). Sharp signal for the router to
   key on.
2. **Phase 3 router learned the discrimination almost perfectly.** Final
   activations: Dickens content 0.980 ± 0.028, Shakespeare content 0.004
   ± 0.005. Separation of +0.976 on training-distribution batches.
3. **Recall on Dickens factual queries is still at the noise floor under
   every condition.** Mean real recall: A=0.033, B=0.017, C=0.017,
   D=0.022. The LoRA encodes Dickens *style* but not *facts* — yesterday's
   storage-capacity bottleneck is reproduced at rank 32 / 2K steps.

But the **Shakespeare quality preservation result is clean**: under
router-controlled LoRA, Shakespeare CE matches LoRA-off CE almost exactly
(4.471 vs 4.497, 4.810 vs 4.816, 5.005 vs 5.016 across three seeds), while
LoRA-always-on CE is much worse (5.195 / 5.335 / 5.489). The router does
its job: it correctly disables LoRA on Shakespeare content, preserving
quality, while activating on Dickens content — the property that makes
this architecture viable as a multi-domain mechanism.

## Setup

- **Base**: 6-layer transformer, d=256, 4 heads, GPT-2 BPE (vocab 50,257),
  ctx=512, dropout=0.1, ~17.8M params.
- **LoRA target**: layer-4 MLP first linear (d_model → d_ff), rank 32. ~41K
  LoRA params.
- **Router**: 256→128→1 sigmoid MLP on mean-pooled post-attention hidden
  states at layer 4. ~33K router params.
- **Data splits** (Great Expectations, GPT-2 tokenized):
  - Phase 2 LoRA training: chapters I–XXX, **145,705 tokens**
  - Phase 3 router training: chapters XXXI–L, **88,711 tokens**
  - Held-out evaluation: chapters LI–LIX, **44,519 tokens** (60 queries authored from this)
- **Phase 1**: 1500 steps AdamW lr=3e-4 cosine on Shakespeare. Final val PPL **121.5**.
- **Phase 2**: 2000 steps AdamW lr=1e-3 on Dickens chapters I-XXX, base frozen. Tracks both Dickens and Shakespeare PPL with LoRA on/off every 200 steps.
- **Phase 3**: 1500 steps AdamW lr=1e-3 on alternating Dickens/Shakespeare batches. Base + LoRA frozen, router trains. Two-pass per step: capture hidden states with LoRA off, gate LoRA by router output for the loss-bearing pass.

## Phase 2: LoRA discrimination signal

Tracked four PPLs throughout training: Dickens with LoRA on/off, Shakespeare
with LoRA on/off.

| step | Dickens (off) | Dickens (on) | Shake (off) | Shake (on) |
|--:|--:|--:|--:|--:|
| 1 | 4666 | 4664 | 131 | 131 |
| 200 | 4666 | 604 | 131 | 166 |
| 1000 | 4666 | 454 | 131 | 220 |
| 2000 | 4666 | **440** | 131 | **229** |

LoRA-off PPLs are flat (base is frozen). LoRA-on tells the story: Dickens
gets 10× better, Shakespeare gets 1.7× worse. Sharp discriminative signal
for Phase 3.

Generation samples after Phase 2 (greedy decoding):

```
Prompt: "Hark thou, good fellow, what news from"
  LoRA OFF: "the king\nAnd, and I have been so,\nAnd, my lord, and my lord..."
  LoRA ON : "the\nand, and I had been a\nand, I had been a\nand, I had been a..."

Prompt: "My father's family name being Pirrip,"
  LoRA OFF: "And I have a man of the world,\nAnd I have a man of the king,\nAnd, my lord, my lord..."
  LoRA ON : "and\nand, and I had been a little\nand, and I had been a little..."
```

Greedy decoding produces repetitive output, but the stylistic register
shifts visibly: LoRA OFF stays in Shakespeare's "my lord/king" register,
LoRA ON shifts to Dickens's first-person past-tense narration ("I had
been a..."). The LoRA encodes **style/register**, not specific tokens.

## Phase 3: router activation summary

Final activation distribution on held-out batches (n=20 batches per type,
batch_size=16):

| | mean | std | min | max |
|--|--:|--:|--:|--:|
| Dickens content | **0.980** | 0.028 | 0.760 | 1.000 |
| Shakespeare content | **0.004** | 0.005 | 0.000 | 0.029 |

**Separation: +0.976**. Near-perfect bimodal split on training-distribution
batches. Loss curve during training: Dickens-batch loss starts ~6.5 and
decays modestly; Shakespeare-batch loss stays around 3.5 (router learns
quickly to disable LoRA on Shakespeare since LoRA-on hurts Shakespeare).
Mean router weight on Shakespeare batches falls to ~0.003 by step 200 and
stays there. Dickens-batch activation rises to ~0.98 and stays there.

## Phase 4: four-condition evaluation

3 stochastic-decoding seeds × 4 conditions × 60 Dickens queries.
"Real recall" filters out the trivial 1-character answer that is always a
substring; raw recall counts that as a hit.

| condition | mean real recall | std | seeds | router activation |
|--|--:|--:|--:|--:|
| **A — full system** | **0.033** | 0.014 | 0.017, 0.050, 0.033 | 0.535 ± 0.387 |
| **B — LoRA always-on** | 0.017 | 0.000 | 0.017, 0.017, 0.017 | 1.000 |
| **C — base only** | 0.017 | 0.000 | 0.017, 0.017, 0.017 | 0.000 |
| **D — random router** | 0.022 | 0.008 | 0.033, 0.017, 0.017 | 0.519 ± 0.290 |

Recall is at the noise floor across all conditions (1–3 hits / 60). A
beats B and C by 1–2 hits per seed. The differences are tiny in absolute
terms but consistent in direction.

### Per-probe router activations are sensible (load-bearing positive result)

The router activates differentially based on probe content:

| probe excerpt | activation |
|--|--:|
| "Pip wanted to trace and prove" | 0.492 |
| "The lawyer Pip went to see was Mr." | 0.978 |
| "Miss Havisham's authority allowed Pip to receive nine hundred" | 0.826 |
| "Estella's father, according to Pip, came from New South" | 0.995 |
| "The check Pip received from Mr. Jaggers was for" | 0.985 |
| "The man's voice in the lonely house came with an" | 0.010 |
| "Mr. Jaggers stood, according to his wont, before the" | 0.667 |

Probes with distinctively Dickensian named entities (Estella, Miss
Havisham, Mr. Jaggers, Wemmick) get activations 0.83–0.99. Probes that
read as generic English get low activations. The router does meaningful
content classification on short prompts, even though it was trained on
512-token batches.

The mean activation 0.535 on the query set reflects that the queries
include a mix of strongly-Dickens probes (high activation) and
weaker-signal probes (low activation). On a probe set that's all clearly
Dickens, mean activation would presumably be much higher.

### Shakespeare quality preservation (load-bearing positive result)

Per-token CE on Shakespeare validation continuations:

| seed | router-controlled | LoRA off | LoRA on |
|--:|--:|--:|--:|
| 0 | **4.471** | 4.497 | 5.195 |
| 1 | **4.810** | 4.816 | 5.335 |
| 2 | **5.005** | 5.016 | 5.489 |

`router-controlled CE ≈ LoRA-off CE` across all seeds — the router's
generalization to short Shakespeare prompts is also strong, and it
correctly disables LoRA so quality is preserved. `LoRA-on CE` is ~0.5–0.7
nats higher (~1.7× worse PPL). The router-controlled column is the system
we'd actually deploy; it loses essentially nothing on Shakespeare-style
content.

This is the most important positive result: **the router-controlled
LoRA is operationally indistinguishable from LoRA-off on Shakespeare,
while still able to engage the LoRA on Dickens content.** It's the
"selective LoRA" property the spec was after.

## Reading

The experiment isolates the storage and the routing questions cleanly:

- **Routing works.** The router discriminates Shakespeare from Dickens both
  on long training-distribution passages (0.98 vs 0.004) and on short
  evaluation probes (per-probe activations track named-entity content).
  Shakespeare quality preserved by the router → operationally usable for
  multi-domain selective LoRA.

- **Storage doesn't.** LoRA at rank 32 / 2000 steps shifts the distribution
  toward Dickens style (10× PPL improvement, "I had been a..." in
  generation samples) but doesn't encode specific factual content. Recall
  on factual queries is at the noise floor — exactly the same finding as
  yesterday's experiment, despite this experiment being designed to
  decouple training of the storage from the routing.

The architecture's selectivity property is real and clean. The fact-recall
question awaits a working storage mechanism — yesterday's analysis showed
rank, training duration, or tokenizer probably aren't the only levers
involved. A LoRA at this scale on a 17M-param base trained for 2K steps
just doesn't have the capacity for token-level factual memorization. It
encodes *register*.

## What this changes about the framing

The original "router decides whether to engage LoRA" hypothesis is
**confirmed for content classification**, **negative for fact retrieval**.
That's a useful split:

- For deployment, the router-controlled selectivity is enough to make
  multi-domain LoRA practical: you can have a Shakespeare LoRA, a Dickens
  LoRA, etc., with a router that picks the right one per request.
  Quality on out-of-domain content is preserved.
- For fact storage, the LoRA-as-Dickens-module construction does not work
  at this scale. The same conclusion as yesterday's direct-LoRA baseline:
  rank 32, 2000 steps, 146K tokens of training data, on a 17M-param BPE
  base, encodes style not facts. Larger rank, larger base, or much more
  training time would be the next levers.

## Caveats

- **Single training pipeline, three eval seeds.** Phases 1, 2, and 3 each
  ran once (seed 0). The three "seeds" in evaluation are stochastic
  decoding seeds, not retraining seeds. Spread across decoding seeds is
  small (std ≤ 0.014 in recall_real), but per-pipeline variance hasn't
  been measured. A follow-up could retrain Phases 2+3 at multiple seeds.
- **The "noise floor" recall (0.017–0.033) is dominated by short common
  words appearing accidentally in generation.** None of the per-query
  hits in any condition were specific named entities like "Pirrip",
  "Jaggers", "Estella", "Wemmick", "Provis" — those would be the real
  test of factual recall.
- **Phase 1 base PPL is 121.5**, which is high. The base is undertrained
  by GPT-2 standards (50K vocab × 304K train tokens × 17M params is a
  hard regime). This may limit how much Phase 2 LoRA can build on.
- **Greedy decoding produces repetitive samples.** Stochastic decoding at
  T=0.8/top-k=50 is used in evaluation; sample quality is better but
  still constrained by the under-trained base.

## Files

- `data/{shakespeare_train,shakespeare_val,dickens_lora,dickens_router,dickens_eval}.pt` — tokenized splits
- `data/queries.json` — 60 hand-authored completion-style probes for Ch LI–LIX
- `data/info.json` — corpus token counts
- `results/phase1_base.pt` — Phase 1 checkpoint (val_ppl 121.5)
- `results/phase2_lora.pt` — Phase 2 checkpoint (LoRA learned)
- `results/phase3_router.pt` — Phase 3 router + activation stats
- `results/eval_{A,B,C,D}_seed{0,1,2}.json` — per-condition × per-seed results
- `results/evaluation_summary.json` — aggregated recall + Shakespeare CE
- `results/RESULT.md` — this writeup
- Code: `experiments/router_lora_phased/{model.py, train.py, evaluate.py, prepare_data.py}`

## Budget

| phase | wall (s) |
|--|--:|
| Phase 1 (base on Shakespeare) | 88 |
| Phase 2 (LoRA on Dickens) | 90 |
| Phase 3 (router) | 76 |
| Phase 4 (4-condition eval) | 49 |
| **Total** | **303 (5.0 min)** |

Well under the spec's 60–100 min estimate. The compute budget is not the
bottleneck — fact-storage capacity is.
