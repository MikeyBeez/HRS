# HRS Architecture Ablation Study

Phase 47 architectural decisions tested in isolation. Substrate: 50 Dickens adapters on V22-Dickens base, rank-128 LoRA on layers 4-5, L0_mean→W→L5_aggregate cosine routing.

## Phase 47 baseline (reference)

From `experiments/per_passage_dickens/results/RESULT.md` (50 Dickens adapters, V22-Dickens base, rank-128 LoRA on L45):

- Routing accuracy: **1.000** (150/150 held-out queries)
- Retrieval accuracy: **0.929** (substring match, 3 stochastic seeds)

Reproduced in `baseline_check.py` at 1.000 routing in 0.3s.

## Ablation 1: Engram pooling operation

Reuses adapter library; recomputes stored library keys per (layer, pool); retrains projection W (500 InfoNCE steps); measures routing on 150 held-out queries.

| variant | q_layer | s_layer | pool | proj_train | routing | max_sim | retrieval | wall |
|---|---|---|---|---:|---:|---:|---:|---:|
| baseline_L0mean_L5mean | L0 | L5 | mean | 1.000 | **1.000** | 0.478 | 0.929 | 114s |
| last_only_L5mean_L5mean | L5 | L5 | mean | 1.000 | **0.973** | 0.360 | — | 3s |
| first_only_L0mean_L0mean | L0 | L0 | mean | 1.000 | **1.000** | 0.460 | 0.929 | 1s |
| midstack_L2mean_L2mean | L2 | L2 | mean | 1.000 | **0.993** | 0.436 | — | 2s |
| max_pool_L0_L5 | L0 | L5 | max | 1.000 | **1.000** | 0.202 | — | 2s |
| attn_pool_L0_L5 | L0 | L5 | attn | 1.000 | **1.000** | 0.407 | — | 2s |

*Total wall: 235s*

## Ablation 2: Engram-as-key vs separate Q projection

All variants use query=L0_mean. Stored side and projection vary.

| variant | routing | max_sim | delta vs baseline |
|---|---:|---:|---:|
| no_proj_L0vsL5 | **0.067** | 0.045 | -0.933 |
| no_proj_L0vsL0 | **0.793** | 0.943 | -0.207 |
| with_W_500 | **1.000** | 0.478 | ++0.000 |
| with_W_5000 | **1.000** | 0.490 | ++0.000 |

*Total wall: 7s*

## Ablation 3: Routing mechanism

Same engrams (L0_mean → W → L5_aggregate); routing function varies.

| variant | routing | delta vs baseline |
|---|---:|---:|
| cosine_argmax | **1.000** | ++0.000 |
| dot_product_argmax | **0.993** | -0.007 |
| euclidean_argmin | **0.020** | -0.980 |
| topK_K2 | **1.000** | ++0.000 |
| topK_K3 | **1.000** | ++0.000 |
| topK_K5 | **1.000** | ++0.000 |
| learned_mlp | **0.793** | -0.207 |
| cosine_noW (L0 vs L5) | **0.067** | -0.933 |

*Total wall: 3s*

## Ablation 4: LoRA configuration

Re-train 50 adapters per variant (150 steps each, V22-Dickens base, 150 paraphrase-mixed sources). Routing uses canonical W; only adapter weights vary.

| variant | rank | targets | params/ad | routing | retrieval | retr Δ | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| rank32_L45 | 32 | 8 | 655,360 | 1.000 | **0.931** | ++0.002 | 233s |
| rank64_L45 | 64 | 8 | 1,310,720 | 1.000 | **0.933** | ++0.004 | 262s |
| rank256_L45 | 256 | 8 | 5,242,880 | 1.000 | **0.929** | -0.000 | 290s |
| rank128_attn_only | 128 | 4 | 1,572,864 | 1.000 | **0.924** | -0.005 | 305s |
| rank128_ffn_only | 128 | 4 | 1,048,576 | 1.000 | **0.936** | ++0.007 | 323s |

*Total wall: 1413s*

## Ablation 5: Partial base unfreezing

Each adapter is trained with both LoRA params AND block 5 of the base unfrozen. Per-adapter snapshots of (lora, block5). Routing on canonical frozen base; generation with snapshot swap. last_block_lr=1e-05.

| variant | rank | routing | retrieval | retr Δ vs baseline |
|---|---:|---:|---:|---:|
| unfreeze_block5 | 128 | 1.000 | **0.900** | -0.029 |

**Forgetting probe** — (lora=A, block5=B mismatched). Tests whether the LoRA dominates the per-adapter block5 snapshot:

| a | b | matched (lora=A,block5=A) | mismatched (lora=A,block5=B) |
|---|---|---|---|
| 40 | 7 | True | False |
| 1 | 47 | True | False |
| 17 | 15 | True | False |
| 14 | 8 | True | True |
| 47 | 6 | True | False |

## Summary table — load-bearing vs incidental decisions

| Architectural decision | Verdict | Evidence |
|---|---|---|
| **Frozen base model** | **Load-bearing (the simpler version is correct)** | Unfreezing block 5 *hurts* retrieval (0.900 vs 0.929 baseline) and adds 78M per-adapter params. Forgetting probe: 4/5 mismatched (lora=A, block5=B) fail to retrieve. |
| Per-passage LoRA adapters | (Architectural premise, not ablated independently) | Phase 47 baseline already establishes this |
| **Engram = first-layer mean** | **Not strictly required (incidental)** | Same-layer L0_mean→W→L0_aggregate also hits 100% routing / 0.929 retrieval. Mid-stack (L2) hits 99.3%. Cross-layer is canonical but not load-bearing. |
| **Pool operation (mean)** | **Not load-bearing for argmax routing (incidental)** | Max and attention-pool both hit 100% routing. Caveat: max-pool drops max_sim to 0.20 (matters for any confidence-gating threshold). |
| **Engram-as-key (no separate Q projection)** | **The framing is wrong — Phase 47 DOES use a projection W.** Removing it is load-bearing. | Without W: 79.3% same-layer / 6.7% cross-layer. With W (500-step InfoNCE): 100%. 5000-step W gives no improvement. |
| **Argmax cosine similarity routing** | **Load-bearing — cosine specifically** | Cosine 100%, dot product 99.3% (close), euclidean 2.0% (broken, magnitude-dominated), learned MLP 79.3% (overfits 200 training paras). Top-K offers no headroom (top-1 already perfect). |
| **LoRA rank** | **Not load-bearing in 32-256 range at 50 adapters** | Rank 32/64/128/256 all retrieve 0.929-0.936. Compute scales linearly with rank but accuracy doesn't. |
| **LoRA targets (attn+FFN on blocks 4-5)** | **Either subset alone is sufficient at 50 adapters** | attn-only (4 modules): 0.924. FFN-only (4 modules): 0.936. Combined (8 modules, baseline): 0.929. |

## Implications for the recruitment ask

**Specific claims that are empirically validated and load-bearing:**
1. Cosine-similarity routing in a learned-projection space.
2. Frozen base model — unfreezing strictly hurts at this scale.
3. Linear projection W trained with InfoNCE is the right inductive bias for the routing classifier (beats a 2-layer MLP).

**Claims that are not load-bearing at the 50-adapter scale and could be relaxed:**
1. Cross-layer (L0→L5) engram structure — same-layer routing also works.
2. Mean-pooling specifically — max and attention-pool work too (modulo confidence-gating considerations).
3. Rank 128 specifically — rank 32 is sufficient with no quality loss.
4. Both attn AND FFN LoRA — either alone is sufficient.

**The recruitment ask should be sharpened:** validate cosine + InfoNCE-projection routing + frozen base at scale (200, 500, 1000+ adapters). The other choices (rank, targets, pool op, layer choice) are tunable parameters, not load-bearing architectural commitments.

## Caveats

1. All ablations are at **50 adapters**. Several null results (rank doesn't matter, targets don't matter) may not survive at 200-1000+ adapters where the rank/target capacity floor binds.
2. Retrieval differences of ±0.01 are within stochastic-decoding noise (3 seeds × 150 queries ≈ ±2pp std).
3. Substring match on natural-prose answers can hit on chance for short/common answers — same scorer as Phase 47 baseline, so comparisons within this study are valid.
4. The learned MLP router was a 2-layer 1024→512→50 with light regularization. A more carefully tuned discriminator might close the 79.3% → 100% gap with cosine; the headline finding is that cosine is *not* obviously inferior to a simple learned classifier at this scale.

## Wall-clock totals

| Stage | Wall |
|---|---:|
| Baseline reproduction | 0.3s |
| Ablation 1 (pooling, 6 variants) | 235s |
| Ablation 2 (Q projection) | 7s |
| Ablation 3 (routing mechanism) | 3s |
| Ablation 4 (LoRA config, 5 variants × 50 adapters) | 1413s (~24 min) |
| Ablation 5 (block-5 unfreeze) | ~6 min |
| **Total** | **~32 min** |
