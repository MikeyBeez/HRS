# Task 7 — Grounded RAFT with grounding_weight = 1.0

**Hypothesis tested.** Raising the contrastive grounding weight 10× (0.1 → 1.0) over task 6's setup will push the content-vs-distractor logit margin past the argmax-flip threshold and begin producing nonzero cleaned recall.

**Outcome.** Failed. Gates opened ~4–5× further, the D+/D– logit diff roughly doubled, but cleaned recall stayed pinned at the 0/138 floor at every checkpoint past step 1000. MAUVE began to pay a visible cost only at step 5000.

## Config

- Base checkpoint: `results/v18_raft/checkpoint_2000.pt`
- Change vs task 6: `--grounding-weight 1.0` (was `0.1`). Everything else identical.
  - 5000 steps, warmup 500, eval every 1000
  - Gate LR 1e-3, cross-attn LR 1e-5, backbone frozen
  - 50/25/25 retrieve/baseline/wrong condition mix
  - Eval: NIAH expanded (25 needles × 40 distractors) cleaned tokens, MAUVE-500 @ n=200
- Output: `results/v18_raft_grounded_w10/`

## Trajectory (w=0.1 vs w=1.0)

| step | recall w=0.1 | recall w=1.0 | MAUVE w=0.1 | MAUVE w=1.0 | gates L1/L3/L5 w=0.1 | gates L1/L3/L5 w=1.0 | diff w=0.1 | diff w=1.0 |
|:----:|:------------:|:------------:|:-----------:|:-----------:|:--------------------:|:--------------------:|:----------:|:----------:|
| 0    | 0/138        | 0/138        | —           | —           | 0.27 / 0.28 / 0.33   | 0.27 / 0.28 / 0.33   | —          | —          |
| 1000 | 1/138        | 1/138        | 0.9274      | 0.9664      | 0.29 / 0.30 / 0.36   | 0.54 / 0.52 / 0.48   | +0.288     | +0.303     |
| 2000 | 1/138        | 0/138        | 0.9155      | 0.9373      | 0.35 / 0.34 / 0.39   | 1.13 / 1.08 / 0.85   | +0.316     | +0.443     |
| 3000 | 0/138        | 0/138        | 0.9417      | 0.9583      | 0.40 / 0.37 / 0.42   | 1.65 / 1.59 / 1.13   | +0.325     | +0.552     |
| 4000 | 1/138        | 0/138        | 0.9414      | 0.9523      | 0.44 / 0.40 / 0.45   | 1.92 / 1.87 / 1.26   | +0.360     | +0.677     |
| 5000 | 0/138        | 0/138        | 0.9465      | 0.9185      | 0.45 / 0.41 / 0.46   | 2.03 / 1.99 / 1.32   | +0.327     | +0.695     |

`diff` = batch-mean `logp(D+ content tokens) − logp(D– content tokens)` at retrieve-active items. `gates L1/L3/L5` = `softplus(gate_scalar)` values at the three cross-attention layers.

NIAH retrieval acc@1 stayed at 0.60 and mean rank around 2.2–2.3 across every checkpoint in both runs — retrieval never changed; only the content-injection channel moved.

## What differed at each checkpoint

- **step 1000.** Gates already ~1.8× more open, diff barely different (+0.303 vs +0.288). Recall identical (1/138). MAUVE higher under w=1.0 (0.9664 vs 0.9274) — early in the schedule w=1.0 was actually preserving LM quality better, probably a seed effect rather than a real advantage.
- **step 2000.** Gates 3.2× more open (L1: 1.13 vs 0.35), diff 1.4× larger (+0.44 vs +0.32). Recall went the wrong way (0/138 vs 1/138). MAUVE still ahead.
- **step 3000.** Gates 4.2× more open, diff 1.7× larger. Both runs at 0/138. MAUVE still ahead of task 6.
- **step 4000.** Gates 4.4× more open, diff 1.9× larger. Recall 0/138 vs 1/138. MAUVE gap narrowing.
- **step 5000.** Gates 4.5× more open, diff 2.1× larger (+0.695 vs +0.327). Both 0/138. **MAUVE crossed below task 6** (0.9185 vs 0.9465). First evidence of distribution cost from the heavier grounding pressure.

## Reading the failure

All four indicators moved in the expected direction — gates opened, diff grew, grounding-loss dropped from ~0.55 → ~0.32 — yet the argmax never flipped at any needle past step 1000. That rules out several mundane explanations:

1. The gates are not the limiter. They're past the sigmoid inflection point (`softplus(gate_scalar) > 1` at all three layers by step 2000, >2 by step 5000). Further weight will saturate the parametrization before it flips more argmax positions.
2. It is not a diff-magnitude issue at the D+ vs D– comparison level. Mean logp(D+) − logp(D–) is solidly positive and growing; the contrastive signal the loss is optimizing is succeeding. It just doesn't transfer to argmax ranking against the full vocabulary (~50K logits dominated by fluent continuations).
3. MAUVE only starts degrading at step 5000, so the model is still producing in-distribution text — distractor tokens and generic continuations are winning on absolute logit, not D+ content.

The decision rule from the task spec: "similar to task 6 → weight isn't the bottleneck, move to sampled-token (task 8)." Applies cleanly. The token-level argmax-flip threshold is not crossed by scaling a whole-content mean-logp objective, because that objective doesn't directly penalize any specific non-content token outcompeting a specific content token at a specific position. A sampled-token loss — at each retrieve-active position, pull up the logp of the actual needle token that appears there against whatever argmax'd — is the next step.

## Artifacts

- Checkpoints: `checkpoint_{1000,2000,3000,4000,5000}.pt`
- Log: `training_log.json`
- Training stdout: `/tmp/raft_g10_full.log`
- Run config: `run_config.json`

Total wallclock: ~60 min on one GPU (10 min/eval × 5 evals + ~50 min training).
