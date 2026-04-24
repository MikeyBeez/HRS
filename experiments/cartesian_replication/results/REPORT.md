# Cartesian Product Attention vs Standard Attention — Replication Report

## TL;DR

The recollected "three percent improvement" **does not replicate** at that magnitude. A 5-seed re-run on Tiny Shakespeare shows **1.47% mean improvement** (std pooled, not significant at p<0.05). The 3% figure appears to have been the result of a single-seed comparison that happened to pair a particularly lucky bilinear run against a typical MHA run. The qualitative direction (bilinear tends to win) is real; the magnitude (3%) is a cherry-pick.

## Phase 1 — What was found in the repository

Two matching experiments:

### (a) `experiments/identity_ae/phase58_bilinear_attention.py`
- **Setup**: Bilinear attention (q^T W k / √d_h, per-head learnable W, initialized as I + noise) vs. Bonsignore exponential-kernel attention (V22 baseline), trained from scratch.
- **Data**: WikiText (not Shakespeare).
- **Scale**: d_model=1024, batch 2 × grad_accum 4 = effective 8, 43K steps, single seed.
- **Results** (from `results/identity_ae/phase58/bilinear_attention.json`):

| Step | Standard PPL | Bilinear PPL | Δ% |
|---|---|---|---|
| 2000 | 191.09 | 187.33 | bilinear **1.97% better** |
| 5000 | 104.12 | 102.52 | bilinear 1.54% better |
| 10000 | 68.88 | 68.65 | bilinear 0.34% better |
| 20000 | 52.42 | 53.02 | standard 1.14% better |
| 43000 | 43.53 | 44.09 | **standard 1.29% better** |

Phase 58 does not show a 3% bilinear advantage at any point, and by 43K steps the bilinear variant is slightly **worse** on PPL. It does show a consistent "information recovery" advantage for bilinear (0.157 vs 0.126 at final step), but that's a V-space geometry metric, not PPL.

### (b) `experiments/diagonal_attention/` (from the current session)
- **Setup**: Four attention variants (MHA, full bilinear, diagonal bilinear, identity) in the same 4-layer transformer.
- **Data**: Tiny Shakespeare, char-level.
- **Scale**: d_model=256, 4 heads, 2000 steps, batch 32, single seed (0).
- **Result**: MHA val_ppl=5.19, bilinear=5.05 → **2.69% improvement**. This is the single-seed result that likely anchors the "about three percent" recollection.

Git log shows no separate commits specifically about Cartesian-product attention beyond the phase-58 contribution in commit `92ecc02` ("Add phases 11-68"). The diagonal_attention work was done in this session and is uncommitted.

## Phase 2 — Characterization of the operational definition

Both phase 58 and diagonal_attention define "Cartesian product attention" the same way structurally: replace the dot product Q·K^T with a bilinear form Q^T W K (or X^T W X when there are no Q/K projections). The difference is whether the model also keeps per-head Q/K projections:

- **Phase 58**: keeps Q, K projections; replaces the dot product between them with a learned per-head d_h×d_h bilinear form. Effectively `q^T W_h k / √d_h`. Per-head W; parameter add is H · d_h² = 4096 per layer at d_h=32 (minimal vs the 1M+ projection params per block).
- **diagonal_attention (full bilinear)**: drops Q and K entirely; computes `X W X^T / √d` directly on the residual stream, shared across heads. W is d_model × d_model = 65,536 per layer. Compared to standard MHA's Q+K = 2·d_model² = 131,072 per layer, bilinear has **half** the score-computation params.

"Plain Cartesian product" most naturally maps to the diagonal_attention version: no Q/K projections, pure bilinear interaction X W X^T, evaluated as the product of all token-pair scores. This is what I re-ran.

## Phase 3 — Re-run on Tiny Shakespeare, 5 seeds

Same scaffold as the diagonal_attention experiment, 2000 steps, char-level Tiny Shakespeare, batch 32, AdamW lr=3e-4 cosine decay. Seeds 0–4 for both MHA and bilinear variants.

### Per-seed val PPL

| seed | MHA | bilinear | Δ (MHA − bilinear) |
|---|---|---|---|
| 0 | 5.195 | 5.050 | +0.145 (bilinear better) |
| 1 | 5.150 | 5.150 | 0.000 |
| 2 | 5.235 | 5.009 | +0.226 |
| 3 | 5.174 | 5.146 | +0.028 |
| 4 | 5.208 | 5.225 | **−0.017** (MHA better) |

Bilinear is better in **3 of 5 seeds**, tied in 1, worse in 1.

### Summary statistics

| Variant | Mean PPL | Std (n=5) | Min | Max | Score params (total, 4 layers) |
|---|---|---|---|---|---|
| MHA | 5.192 | 0.033 | 5.150 | 5.235 | 524,288 |
| Bilinear | 5.116 | 0.086 | 5.009 | 5.225 | 262,144 |

- **Mean improvement: 5.192 → 5.116, Δ = 0.076 PPL = +1.47% relative** (bilinear better).
- **Effect size: Cohen's d = +1.17** (large by conventional cutoffs, but see noise caveat below).
- **Welch's t = 1.85, two-sided p ≈ 0.064** — **does not clear p<0.05**. Suggestive, not conclusive at n=5.

### Variance structure

Bilinear has **2.6× the seed-to-seed standard deviation** of MHA (0.086 vs 0.033). The single-seed "3% advantage" that anchored the recollection corresponds to seed 0, where bilinear happened to hit its best run (5.050) against an average MHA run (5.195). At seed 4 the ordering reverses. **The 3% figure is outside the MHA±1σ band but well inside the bilinear±1σ band** — it's a lucky sample of the noisier distribution, not a typical performance delta.

## Phase 4 — Interpretation

### What the result establishes

1. **Bilinear attention is not worse than standard MHA on this task at this scale.** Mean PPL is lower by 1.47%, and the direction is consistent across the majority of seeds.
2. **At half the score-computation parameters** (262K vs 524K), bilinear matches or slightly beats MHA. If anything, the story is "comparable quality, half the attention params" — a parameter-efficiency result, not a quality result.
3. **The sign of the effect is stable, but the magnitude is not.** Cohen's d is large because the means are separated by more than a combined-SD unit, but the bilinear variant's high variance means individual runs can still land worse than MHA.

### What the result does NOT establish

1. **The 3% figure does not replicate.** Across 5 seeds, the gap is 1.47% on average, and in the worst seed the sign reverses. Claiming "Cartesian-product attention is 3% better than standard attention" overstates the effect by ~2×.
2. **Welch's t p = 0.064 is not significant.** n=5 is small, and at that sample size a 1.47% mean difference with 2.6× variance asymmetry doesn't clear the conventional threshold. To claim a real improvement, run 10+ seeds and re-test.
3. **Phase 58's result (WikiText, 43K steps) is in the other direction at convergence.** Bilinear eventually loses by ~1% after crossover at ~10K steps. So the Tiny Shakespeare result may be specific to very short training / small vocab. Not clear it generalizes.

### Limitations of this test

- Single task (char-level Tiny Shakespeare), single scale (4-layer, d=256), single training length (2000 steps). Phase 58 showed that the bilinear-vs-standard gap has a non-monotonic training-time dependence, so 2000 steps is specifically where bilinear looks best.
- Loss is cross-entropy only; no downstream task tested.
- Variants share batches and initializations by seed; variance is across seeds, not across data orderings.
- n=5 is small for distinguishing ~1% effects reliably.

### Practical recommendation

If the interest is in a real deployment-relevant finding, the more honest claim to build any writeup around is:

> **"On small-scale char-level LM, a full bilinear attention score (X W X^T with shared-across-heads W) matches or slightly beats standard multi-head attention at half the score-computation parameters, with higher seed-to-seed variance. The single-seed ~3% gap previously observed appears to have been an upper-tail sample from the noisier bilinear distribution; the mean advantage at n=5 is 1.47% (p≈0.06)."**

That's a defensible factual statement. "3% better" is not.

## Files

```
experiments/cartesian_replication/
  run_seeds.py                     # re-run with 5 seeds
  results/
    mha_vs_bilinear_seeds.json     # per-seed PPL + summary
    run.log
    REPORT.md                      # this file
```

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m experiments.cartesian_replication.run_seeds \
    --seeds 0 1 2 3 4 --steps 2000
```

## If you want to firm up the claim

- **Run 10–20 seeds** and re-test. If the 1.47% mean holds with lower-variance estimation and p<0.05, the result strengthens. If it drifts toward 0 or flips sign, the effect was noise.
- **Sweep training length**. Phase 58 already implies the gap shrinks and reverses with longer training; confirm this on Shakespeare with a 5K- and 10K-step comparison. If bilinear's advantage is 2K-step-specific, it's not a real quality result — it's an early-convergence artifact.
- **Scale up.** 4 layers / d=256 is tiny. At 12 layers / d=768, does the bilinear advantage survive? Phase 58's WikiText run at d=1024 suggests not at convergence.
