# Experiment 2: Focused Hybrid Heads with Retrieval Path

**Date:** 2026-05-07
**Architecture:** B (per-layer K/V compression, full-length residual stream)
**Headline:** Architecture B behaves *qualitatively differently* from architecture A on both corpora. On TS, hybrid configurations (1+7 and 2+6) slightly improve over baseline — opposite of A's 17% loss. On WT103, all compressed variants lose to baseline, with the full-head count tracking the gap monotonically — also opposite of A's −5.3% advantage. NIAH is 0/800 across all 24 checkpoints (regime check, as predicted). The retrieval-head mechanistic signal is faint but emerging in some hybrid configurations.

## Setup

Per the program spec with one substitution: 8-head transformer (256/8 = 32 head_dim) instead of the spec's 12-head (256/12 isn't integer). Hybrid becomes 1+7 and 2+6 instead of 1+11 and 2+10 — same hypothesis (a few full heads + many compressed heads). Architecture B per the discussion: K/V compressed per layer, queries stay at full length, no upsampling needed.

4 architectures × 2 corpora × 3 seeds = **24 trained checkpoints**:
- **baseline**: standard 8-head causal SDPA
- **pure_compressed**: 8 heads, all use compressed K/V (each head's K and V are pooled by a 4-layer 16× causal conv stack)
- **hybrid_1plus7**: 1 full-attention head + 7 compressed heads per layer
- **hybrid_2plus6**: 2 full-attention heads + 6 compressed heads per layer

ctx 1024, 5000 steps, batch 8, AdamW lr=3e-4 cosine to 3e-5, warmup 200, weight decay 0.01, bf16, dropout 0.1. 8 heads × 6 layers × d_model=256 × d_ff=1024 = 17.87M baseline params; compressed variants add ~0.10M for the K/V conv stacks.

## Validation perplexity (3-seed mean ± std)

### Tiny Shakespeare (n=3)

| arch | val PPL | gap vs baseline |
|---|---|---|
| baseline | 123.57 ± 3.61 | — |
| pure_compressed | 123.40 ± 1.04 | **−0.1%** (within noise) |
| hybrid_1plus7 | 121.19 ± 2.83 | **−1.9%** |
| hybrid_2plus6 | 121.49 ± 1.83 | **−1.7%** |

On TS, architecture B's compressed variants **match or slightly beat** baseline. The hybrid configurations are best (−1.7% to −1.9%). All differences are within roughly 1-2σ of seed noise but the trend is consistent across all 3 seeds.

This is **dramatically different from architecture A's TS result** (17.3% loss in `compression_corpus_dependency`). Per-layer K/V compression is much gentler on local-detail-heavy corpora than residual-stream compression because queries stay at full resolution — only the K/V is compressed.

### WikiText-103 (n=3)

| arch | val PPL | gap vs baseline |
|---|---|---|
| baseline | 403.51 ± 0.74 | — |
| pure_compressed | 430.12 ± 2.04 | **+6.6%** |
| hybrid_1plus7 | 421.50 ± 2.63 | **+4.5%** |
| hybrid_2plus6 | 419.05 ± 1.97 | **+3.9%** |

On WT103, architecture B's compressed variants **all lose to baseline**, with monotonic improvement as more full heads are added (8 full → 0% gap, 2 full → +3.9%, 1 full → +4.5%, 0 full → +6.6%).

This is **opposite of architecture A's WT103 result** (−5.3% — compressed beat baseline). Per the depth ablation (commit `15ec34b`), A's negative gap is plausibly a training-budget artifact specific to A's information-bottleneck regularization. Architecture B doesn't reproduce it. Whether the depth-ablation interpretation is correct (training budget) or whether the architectures truly differ on this corpus is not yet pinned down — both stories are consistent with the data.

The WT103 monotonic ordering by full-head count is the cleanest signal: more full heads = closer to baseline. Each additional full head buys ~0.5-1% PPL gap on WT103.

## Coarsened-anchor accuracy (every-16th-token prediction)

| corpus | baseline | pure_compressed | hybrid_1+7 | hybrid_2+6 |
|---|---|---|---|---|
| ts | 0.219 ± 0.056 | 0.224 ± 0.039 | 0.229 ± 0.059 | 0.203 ± 0.031 |
| wt103 | 0.208 ± 0.086 | 0.219 ± 0.078 | 0.208 ± 0.074 | 0.219 ± 0.056 |

**All within noise of each other** on both corpora. Anchor accuracy doesn't discriminate among the architectures at this training scale. Same finding as prior experiments — anchor accuracy is robust to architectural changes that move PPL by single-digit percentages.

## NIAH retrieval (regime check)

**All 24 checkpoints scored 0/800 = 0.000 on NIAH.**

This was predicted upfront in the user's launch direction: "at 5000 steps both architectures may score zero on NIAH... treat as regime check; meaningful retrieval comparison requires extended training." The result confirms: at 5000 steps × batch 8 on these corpora, no model — baseline included — has reached the in-context-retrieval regime.

NIAH cannot discriminate among the 4 architectures at this training budget. The architectural question of whether 1-2 full heads are sufficient for retrieval is **deferred to a follow-up at extended training duration** (likely 50k+ steps based on the depth ablation curves).

## Retrieval-head mechanistic analysis (Wu et al. 2024)

For each trained model, we measured the attention weight from the final query position to the needle's position (mean across 8 NIAH-style probes), per (layer, head, kind) tuple. Top-5 heads by mean attention-to-needle:

### TS baseline (top heads are all FULL attention)

| layer | head | kind | attn |
|---|---|---|---|
| 1 | 7 | full | 0.0348 |
| 2 | 7 | full | 0.0278 |
| 2 | 1 | full | 0.0189 |
| 1 | 3 | full | 0.0133 |
| 1 | 4 | full | 0.0122 |

Retrieval-head pattern is concentrating in layers 1-2, ~0.01-0.035 attention weight. Faint but present.

### TS hybrid_2+6 (1 FULL head dominates, others are compressed)

| layer | head | kind | attn |
|---|---|---|---|
| 1 | 1 | **full** | **0.0262** |
| 0 | 2 | compressed | 0.0177 |
| 0 | 5 | compressed | 0.0154 |
| 0 | 3 | compressed | 0.0144 |
| 0 | 7 | compressed | 0.0134 |

The single highest-attention head is a **full head** at layer 1 (attn 0.0262, ~50% above the next-highest compressed head). Consistent with the retrieval-head literature's prediction: when full and compressed heads coexist, the full head specializes for retrieval.

### TS hybrid_1+7 (no clear full-head dominance)

| layer | head | kind | attn |
|---|---|---|---|
| 0 | 5 | compressed | 0.0184 |
| 5 | 2 | compressed | 0.0182 |
| 0 | 1 | compressed | 0.0175 |
| 1 | 7 | compressed | 0.0173 |
| 0 | 7 | compressed | 0.0166 |

In hybrid_1+7, the single full head per layer (head 0) is **not** in the top 5. The compressed heads collectively have similar attention strength. With only 1 full head per layer, retrieval responsibility may not be cleanly localized.

### WT103 patterns

WT103 retrieval-head signals are **much weaker** than TS (max attn ~0.003 in baseline vs ~0.035 on TS). At 5000 steps × bs 8 on a 118M-token corpus, models are far from the regime where retrieval-head behavior fully develops. Compressed heads in pure_compressed and hybrids show similar attention strength (~0.018-0.022) — consistent with attention spreading roughly uniformly.

## Per-spec hypothesis evaluation

> **Hypothesis:** Hybrid 1+7 will show NIAH performance close to baseline (within 10 percentage points) while pure compression will show much weaker NIAH (within 30 percentage points of zero). Hybrid 2+10 will show NIAH performance equal to or better than 1+11, with diminishing returns.

NIAH is 0/0 across all configurations. Hypothesis cannot be tested at this training budget.

> **The mechanistic analysis will confirm that retrieval-head behavior concentrates in the full-attention heads.**

**Partially supported on TS, not on WT103.** TS hybrid_2+6 shows clean retrieval-head pattern in 1 of 2 full heads (L1 H1, attn 0.026 vs next 0.018). TS baseline shows retrieval pattern in mid-layer full heads (L1-L2, attn 0.012-0.035). TS hybrid_1+7 doesn't show clean full-head dominance — the single full head per layer isn't sufficient for the pattern to emerge clearly. WT103 retrieval-head signal is too weak to evaluate.

## Verdict

**The retrieval-head literature's "1 full head + many cheap heads" architecture isn't validated at this scale and training budget**, but it isn't refuted either. The data tell us:

1. **Architecture B preserves quality much better than A on TS.** Per-layer K/V compression with full Q is a meaningfully different architectural choice from residual-stream compression.

2. **WT103 PPL gap is monotonic in full-head count.** More full heads → smaller gap. The retrieval-head literature's prediction (1-2 full heads sufficient) doesn't dominate; instead the full-head count *trades off* with compression smoothly. This is consistent with "every full head adds modest capacity" rather than "1 full head is special."

3. **The TS hybrid_2+6 retrieval-head pattern is the cleanest mechanistic signal.** A specific full head (L1 H1) concentrates ~50% more attention on the needle than the strongest compressed head. Suggestive but not strong.

4. **NIAH at 5000 steps is uninformative.** All architectures score 0/800. Discrimination requires longer training. Per the user's framing this is the expected result and queues the longer-training NIAH follow-up.

5. **Anchor accuracy is invariant to these architectural changes.** No discrimination — confirming pattern from prior experiments.

## What would change with longer training

At 50k+ steps (10× current budget), based on the depth ablation curves:
- Retrieval-head signals should sharpen substantially. The L1 H1 pattern in hybrid_2+6 may become distinctly stronger; hybrid_1+7's single full head should develop or fail to develop the pattern, distinguishing the architectures.
- NIAH may become non-zero, allowing actual discrimination of the 4 architectures.
- WT103 PPL gaps may close (if the depth ablation's training-budget hypothesis is right) or persist (if architecture B's compression genuinely costs WT103 quality).

The combined longer-training control across this experiment + the depth ablation's flagged WT103 control would resolve both open questions in one batch.

## Per-spec recommendation for downstream experiments

Per the program spec: the four-experiment program said "if depth doesn't help, use 4-layer compression stacks; the longer-training control on WT103 becomes higher priority." The depth ablation already established that depth doesn't help at this budget. Experiment 2's results add: **architecture B with hybrid_2+6 is the architecture closest to baseline on WT103 while improving on TS**, and the retrieval-head literature's specific 1-full-head prediction needs longer training to be tested fairly.

For Experiment 3 (supernet), use 8-head architecture with the floor-then-release gating. The PPL ordering on WT103 (more full heads → smaller gap) is the prediction the supernet should reproduce or refute. The experiment was designed under the "1-2 retrieval heads" assumption; the data here suggest a smoother trade-off that may favor more full heads if compute permits.

## Files

- `model.py` — HybridTransformer, HybridAttention with per-layer K/V compression
- `train_hybrid.py` — training script (matched task: predict every 16th token)
- `eval_niah.py` — needle-in-a-haystack (5 needles × 5 depths × 32 fills = 800 trials)
- `retrieval_heads.py` — Wu et al. attention-to-needle measurement per (layer, head)
- `analyze.py` — aggregator
- `results/train_*_seed{0,1,2}.json` — per-run training summaries (24 files)
- `results/niah_*_seed{0,1,2}.json` — per-run NIAH (24 files)
- `results/retrieval_heads_*_seed{0,1,2}.json` — per-run head attention data (24 files)
- `results/aggregate.json` — full summary

Checkpoints (`checkpoints/*.pt`) gitignored.
