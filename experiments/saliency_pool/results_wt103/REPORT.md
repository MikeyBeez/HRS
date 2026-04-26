# Saliency-pooling ablation — WikiText-103 scale-up

**Question.** The Shakespeare-scale screening showed variant B (per-pair
saliency MLP) tied with V22 Bonsignore attention and variant A
(shared-global linear-cost saliency) only 1.5% behind. Per the spec:
"the gap will only widen at larger scale, not reverse" — does the
Shakespeare result hold on real WikiText-103 text with GPT-2 BPE
tokens, or do the variants diverge from baseline?

**Outcome.** The result *changed direction* in two ways at scale, with
both moves being large and statistically clean:

- **Variant B beats baseline** by 3.9% (101.57 → 97.61 PPL,
  Cohen's d = −9.66, p < 0.001). The "tied" Shakespeare result became
  a meaningful win at scale.
- **Variant A loses to baseline** by 7.6% (vs 1.5% at Shakespeare).
  The gap widened by 5x.
- **Variant C also beats baseline** by 1.5% (Cohen's d = −4.08,
  p = 0.007), reversing its Shakespeare loss of +2.2%.

So the Shakespeare→WT-103 shift is: A got worse, B got significantly
better, C swung from worst-non-floor to second-best. The framework's
predicted ordering (`D < A < C < B < baseline` with baseline winning)
is wrong on every count except D being the floor.

## Scaffold

- Same architecture as Shakespeare experiment: TinyTransformer at d=384,
  n_heads=6, n_layers=6, d_ff=1536, ctx=256, tied embeddings.
- Vocab: 50,257 (GPT-2 BPE) instead of Shakespeare's ~65 chars. This
  raises total params from 10.76M → **30.04M** (the 19.3M tok_emb is
  the difference).
- Data: WikiText-103 from the existing repo cache, re-sliced into
  seq_len=256 chunks at random offsets per batch.
- Training: **5000 steps**, batch 32, AdamW lr=3e-4 cosine to 10%,
  warmup 100, weight decay 0.01.
- 3 seeds per variant. Variant D (pure mean pool, the floor) was
  skipped — its Shakespeare result (+67% PPL above baseline) is
  decisive and re-running on WT-103 would only consume budget.
- Total wall clock: 3h 5m for 12 runs on one 5070 Ti.

## Results

| variant | description | n | mean PPL | std | 95% CI | Δ vs baseline | Cohen's d | p |
|:-------:|:-----------:|:-:|:--------:|:---:|:------:|:-------------:|:---------:|:-:|
| **B** | **per-pair saliency MLP (2d→d→1, factored)** | 3 | **97.61** | 0.50 | [97.04, 98.18] | **−3.96** | **−9.66** | **0.000** |
| C | context-summary saliency (2d→d→1, causal cumulative mean) | 3 | 100.06 | 0.44 | [99.57, 100.55] | −1.51 | −4.08 | 0.007 |
| baseline | V22 Bonsignore Q-K attention | 3 | 101.57 | 0.29 | [101.24, 101.90] | — | — | — |
| A | shared global saliency (d→768→1) | 3 | 109.33 | 0.35 | [108.93, 109.73] | +7.76 | +24.02 | 0.000 |
| D | pure causal mean pool | 0 | (not run; Shakespeare confirmed +67% floor) | | | | | |

Mechanism diagnostics (final-step, averaged across layers × seeds):

| variant | attn entropy | sal MLP Frob |
|:-------:|:------------:|:------------:|
| A | 2.91 | 12.75 |
| B | 2.94 | 14.53 |
| C | 2.86 | 17.35 |

(At T=256 with causal masking, average uniform-prefix entropy ≈ 4.85.
Variants A/B/C all sit at 2.9, so saliency is producing peaked
attention — not collapsed to uniform.)

## Comparison with Shakespeare (10.76M params, ~65 chars)

| variant | Shakespeare PPL ± std | Shakespeare Δ vs base | WT-103 PPL ± std | WT-103 Δ vs base | Direction at scale |
|:-------:|:---------------------:|:---------------------:|:----------------:|:----------------:|:------------------:|
| baseline | 4.800 ± 0.034 | — | 101.57 ± 0.29 | — | — |
| A | 4.871 ± 0.053 | +1.5% | 109.33 ± 0.35 | **+7.6%** | **gap widened 5x** |
| B | 4.790 ± 0.036 | −0.2% (tied, p=0.71) | 97.61 ± 0.50 | **−3.9% (p<0.001)** | **swung from tied to win** |
| C | 4.903 ± 0.029 | +2.2% (worst non-floor) | 100.06 ± 0.44 | −1.5% (p=0.007) | **swung from worst to second-best** |

**Three different scaling behaviours in the same architecture family:**

- **A scales poorly.** A single saliency vector per sequence (modulo
  causal mask) carries enough signal at character-level Shakespeare
  for queries to be approximately interchangeable. At BPE-tokenized
  WT-103 where every token carries real semantic load, queries need
  to differ in what they attend to. A can't deliver that — its single
  attention pattern is too coarse — and the gap to baseline grows
  5x as text gets richer.
- **B scales well — even past baseline.** Per-pair MLP scoring becomes
  *more* effective than Q-K Bonsignore attention as text gets richer.
  This is the surprising direction. Several plausible mechanisms:
  the MLP can express non-bilinear scoring (Q-K is bilinear by
  construction); the same 295K params allocated to a single pair-MLP
  may capture more than two 147K matrices (Q, K) used in the
  Bonsignore kernel; the squared-distance kernel has a specific
  geometry (concentric similarity contours) that an MLP can deviate
  from.
- **C scales well — from "worst non-floor" to second-place.** Adding
  the causal-cumulative-mean summary to the saliency input hurt at
  Shakespeare (likely because the summary at character level was
  overwhelmingly dominated by frequency artifacts of the small vocab).
  At WT-103 the BPE summary carries genuine document-level context
  signal, and that signal helps disambiguate which positions to
  attend to.

## Reading the result against the spec's outcome map

The spec listed four outcomes:

1. *All variants within noise of baseline (hypothesis confirmed):* No.
   Effect sizes are large — none are within noise.
2. *Baseline wins decisively (hypothesis refuted, attention is
   non-substitutable):* Partial, only against A. Baseline *loses*
   decisively to B and C.
3. *Mixed pattern, some variants match baseline some don't:* Closest
   match — but the actual pattern is more interesting than "mixed":
   simple-shared-saliency loses, more-expressive-saliency wins.
4. *Genuinely ambiguous, scale-up needed:* No, this *is* the scale-up,
   and it's not ambiguous.

The honest reading is closer to **head-aggregation's "Outcome 4"**
than to any of the saliency-pool spec's outcomes: "*Several variants
beat the baseline. The architecture is actively suboptimal — it's
wasting capacity on [structure] that a better-chosen alternative does
more efficiently.*"

V22's Bonsignore Q-K kernel attention, at this 30M-param scale on
WT-103, is **outperformed at matched parameters by a learned saliency
MLP scoring function**. The architecture-replacement hypothesis went
from "competitive" at Shakespeare to "winning" at WT-103.

## Caveats

- **30M params, 5000 steps, GPT-2 BPE WT-103.** Not full V22 scale
  (V22 is ~512M params with PEER FFN, cross-attention, much longer
  training). The result might continue to favor saliency variants at
  V22 scale, or might reverse. Either possibility is empirically
  open.
- **3 seeds per variant.** Effect sizes are huge (Cohen's d of 4–24)
  so signal-to-noise is fine for the statistical claim, but the
  *absolute* numbers come from 3 trajectories rather than 10.
- **Limited training duration.** 5000 steps is still early in
  WT-103's loss curve. The relative ordering at 5000 steps may not
  match the ordering at 50000 steps. The Bonsignore kernel might
  catch up with longer training because Q-K-style scoring is
  generally faster to converge on the rougher objective at start.
  Worth checking by extending one or two seeds to 20K steps.
- **No comparison against vanilla softmax dot-product attention.**
  The baseline is V22's Bonsignore kernel (negative squared
  Euclidean), not standard scaled dot-product. The framework
  prediction was about "Q-K attention" generally; it's possible
  Bonsignore is specifically suboptimal here and standard
  dot-product would close the gap. Worth a follow-up.

## Decision

This experiment turns the Shakespeare answer into a stronger one. The
Shakespeare result said "saliency-MLP scoring is competitive with
Q-K." The WT-103 result says **"saliency-MLP scoring outperforms
Q-K Bonsignore attention at 30M-param/WT-103/5K-step scale."**

Worth scaling up further. Two natural follow-ups, in order of cost:

1. **Extend training to 20K–50K steps** at the current 30M scale.
   Determines whether the early-training advantage of the MLP scoring
   persists or whether Bonsignore catches up. Cheap (~2.5 hours per
   seed for the slow B variant at 20K steps).
2. **Test against standard scaled-dot-product attention as a third
   baseline.** If saliency-MLP also beats softmax attention, the
   finding is general; if it only beats Bonsignore, the finding is
   about the kernel choice, not scoring-mechanism class.

Holding off on full V22 scale-up (200–250 GPU hours) until those two
cheaper checks are done. They will tell us whether the WT-103 result
is structural or training-regime-specific.

## Files

- `report.json` — machine-readable summary
- `{variant}_seed{seed}.json` — per-run records (eval points, train
  losses, wall times, peak memory, layer diagnostics)
- `sweep_summary.json` — sweep metadata
- Code: `experiments/saliency_pool/{config,attention,model,train_wt103,run_sweep_wt103,analyze}.py`

## Budget actuals

Spec contingent V22 scale-up budget: 200–250 GPU-hours, ~2 weeks.
**Used 3 hours 5 minutes** at 30M-param/WT-103 scale. The result is
strong enough at this scale to defer full V22 scale-up pending the
two follow-ups above.
