# Engram dropout ablation on Tiny Shakespeare — results

## Setup

Scaled-down V23 reproduction: 6 transformer layers, d=256, 4 heads, ctx=256,
char-level Tiny Shakespeare. Per-head Bonsignore Kernel (per-head MLP refining
`-‖q-k‖²/τ_h`, initialized as near-identity per V23). Engram pipeline:
`EngramEncoder` (window mean-pool → 2-layer MLP → K=4 engrams per W=64-token
window, 16 engrams total at ctx=256), `EngramCrossAttention` injected at
**layer 2** (V23's layer 3 was found dead and disabled; spec says preserve
mid-stack but avoid the dead layer). Recon loss: `1 − cos_sim` between
mean-pooled engrams and mean-pooled extract-layer hidden states, weight 0.1.

Engram dropout: per-batch, with probability `p` the engram tensor passed into
the cross-attention is replaced with zeros. The encoder still runs; recon
loss is always applied. Inference always uses engrams (and is also evaluated
with engrams zeroed for the ablation gap).

Training: AdamW lr=3e-4 cosine to 10%, 100-step warmup, weight decay 0.01,
betas=(0.9, 0.95), grad_clip=1.0, batch=32, ctx=256. **5000 steps** chosen
from a single-seed pilot — val_ppl flattens at ~step 3500 (final eval is
slightly past optimum but identical schedule across all 12 runs).

3 seeds × 4 dropout rates × 5000 steps × ≈ 14 min/run = 168 min total.
Total params 5.74M (~84% from token+pos embeddings + ffn). All 12 runs
converged smoothly; no NaNs, no divergence.

## Headline table

| p | n | val_ppl (engram on) | val_ppl (engram off) | gap (off − on) | head-spec τ-var (inj layer) | head-spec α-var | head-spec scale-var | stability |
|:--:|:-:|:-------------------:|:--------------------:|:--------------:|:---------------------------:|:--------------:|:-------------------:|:---------:|
| 0.00 | 3 | 4.947 ± 0.031 | 4.968 ± 0.026 | **+0.021** ± 0.018 | 5.084 | 5.7e-5 | 1.23e-3 | ✓ |
| 0.10 | 3 | 4.928 ± 0.030 | 4.934 ± 0.030 | **+0.006** ± 0.004 | 4.409 | 5.0e-5 | 1.22e-3 | ✓ |
| 0.25 | 3 | 4.940 ± 0.032 | 4.944 ± 0.033 | **+0.004** ± 0.002 | 5.156 | 5.8e-5 | 1.18e-3 | ✓ |
| 0.50 | 3 | 4.980 ± 0.017 | 4.982 ± 0.018 | **+0.002** ± 0.002 | 4.869 | 5.5e-5 | 1.10e-3 | ✓ |

(τ = `log_tau.exp()` per head; α = `sigmoid(head_alphas)`; scale =
`softplus(head_output_scalars)`. Variance is across the 4 heads, averaged
across runs at the engram-injection layer (layer 2). The "average across all
layers" version, in `report.json`, is qualitatively the same.)

## Reading the numbers

**1. Validation PPL is essentially constant across dropout rates.** All four
groups land in 4.93–4.98 with seed-to-seed std of ±0.02–0.03. The widest
between-group gap (p=0.1 vs p=0.5: 4.928 vs 4.980, +0.052 PPL) is barely a
single seed-noise std. p=0.5 is marginally the worst group but well within
noise. *Aggressive engram dropout does not measurably hurt training or
terminal quality at this scale.*

**2. Ablation gap trends down monotonically with p.** +0.021 → +0.006 →
+0.004 → +0.002. The direction is what the spec hypothesized: engram dropout
during training reduces inference reliance on the engram. But the *magnitude*
is small — the absolute gap at p=0 is already ~0.02 PPL, near the noise
floor. Whether the trend is statistically significant is debatable at n=3
per group with stds in the same range as the means (±0.018 at p=0).

**3. Per-head kernel specialization is preserved across dropout rates.** τ
variance across heads stays in the 4.4–5.2 range at all p; no collapse toward
uniformity even at p=0.5. α and scale variances are tiny but stable. The
engram dropout does not disrupt the kernel learning that's supposed to
differentiate the four heads.

**4. All 12 runs converged.** No early-stopping triggered. p=0.5 seed=0
came in at 1.00× the p=0 seed=0 ppl_on, well below the spec's 2× early-stop
threshold.

## The headline caveat — engram channel was near-useless from the start

The ablation gap at p=0.0 is **+0.021 PPL** — ~0.4% PPL change from removing
the entire engram pathway at inference. The cross-attention gate value drifts
*down* during training (from 0.347 at init to ~0.29 at step 5000), and the
recon loss converges to ~0.001. **The model converges to a state where the
engram pathway contributes nearly nothing.** Then the dropout experiment
measures whether something close to nothing can be reduced further to nothing
— and yes, technically it can (gap drops from 0.021 to 0.002), but it's
hard to call this an "engram dependency" finding when the dependency is so
small to begin with.

This mirrors what V23 reported about layer 3 being dead. We chose layer 2
to avoid the dead layer, but layer 2 turns out to be similarly dead at this
scale. Possible reasons:

- **Same-batch engram source.** V23's engrams come from an external buffer of
  past training samples (`update_engram_buffer()`). I shortcut that to
  same-batch hidden states (encoder reads layer-1 output of the current
  batch). Causal masking (engram[i] visible only to queries in window > i)
  prevents leakage but also makes the engram a strict subset of what
  layer-2 self-attention already sees — coarser, not richer. The model has
  no reason to prefer it.
- **Char-Shakespeare is too small / too local.** Most predictions are
  determined by the last 1–4 chars; long-range info doesn't change the
  conditional entropy much. Even a faithful V23 engram pathway would have
  little to do.
- **Layer 2 sits below the strong representation layers.** V23 placed
  cross-attn at layer 3 (mid-stack) and found it dead. The next mid-stack
  alternative (layer 2) is even earlier, where the residual stream is less
  semantically rich.

Whatever the reason, the engram channel is essentially decorative at this
scale on this corpus. The spec's question — "does dropout reduce reliance
when reliance is real?" — can't be answered here because the precondition
isn't met. What we *can* say:

- Dropout doesn't break training stability at any of the tested rates.
- Per-head kernel specialization is preserved across dropout rates.
- The trend in the (tiny) ablation gap is in the predicted direction.

## Recommendation for Phase 2 (WT-103 + V23)

The Phase-1 framing was: get a fast read on whether engram dropout breaks
anything before running on WT-103. Answer: it does not break anything we
can detect. Training is stable, kernel specialization holds. So the WT-103
sweep is safe to run in terms of "will dropout destroy training".

The harder question — does dropout actually reduce engram reliance — is
unanswerable at this scale because the engram isn't carrying load to begin
with. Phase 2 should be run with V23's full external-buffer engram pipeline
on WT-103, where the engram is known to be load-bearing (the WT-103
ablations show real ablation gaps). At that point the same dropout
modification can be cleanly tested and the results will be interpretable.

## Files

- `p{0.0,0.1,0.25,0.5}_seed{0,1,2}.json` — per-run records (history, final
  ppl_on/off, per-layer kernel diagnostics, gate values, recon loss
  trajectory)
- `p{0.0,0.1,0.25,0.5}_seed{0,1,2}.pt` — saved model checkpoints for
  post-hoc geometric analysis
- `sweep_summary.json` — sweep metadata
- `report.json` — analyzer output, machine-readable
- Code: `experiments/engram_dropout/{model.py, train.py, run_sweep.py, analyze.py}`

## Notes file

Three things worth surfacing beyond the table:

1. **The same-batch engram is not V23's engram.** V23 uses an external
   buffer of past training-sample hidden states; I used same-batch hidden
   states with per-window causal masking. The first version of this
   experiment without causal masking produced ppl_on=3.65 / ppl_off=540
   at p=0 — clearly a future-token leak via the within-window mean pool.
   Causal masking fixes the leak but also reduces the engram channel to
   "coarser version of what layer-2 self-attention already has", which is
   probably why the channel is dead. This deviation should be explicitly
   reverted for the WT-103 phase.

2. **Step count was set conservatively.** Val_ppl actually bottoms around
   step 3500 in the pilot (4.87) and rises slightly toward step 5000
   (4.95) under the cosine schedule. All 12 runs share the same schedule
   so within-experiment comparisons are fair, but the absolute numbers
   are post-optimum by a small amount. If you re-run, 3500–4000 steps
   would land closer to the val-loss floor.

3. **Stopping condition never triggered.** Spec said: skip seeds 1/2 at any
   p where seed=0 is >2× p=0 seed=0 ppl_on. At p=0.5 seed=0 the ratio is
   1.00×, far from 2×. So all 12 runs ran. The 12-run total wallclock was
   ~168 min, well under the 4-hour budget.
