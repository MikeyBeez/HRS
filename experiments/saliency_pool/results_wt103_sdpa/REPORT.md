# Saliency-pool follow-up Experiment 2 — Standard SDPA Baseline

**Question (per spec).** Is variant B's win specific to beating the
Bonsignore kernel, or does it also beat standard scaled-dot-product
attention?

**Answer.** Specific to beating Bonsignore. Standard SDPA decisively
beats both Bonsignore (the V22 baseline) and the per-pair saliency
MLP at this scale and training horizon.

## Setup

- Same architecture and dataset as the original WT-103 sweep (30M
  params, GPT-2 BPE tokens, ctx=256, batch=32, 5000 steps).
- New variant: `sdpa` — standard scaled-dot-product attention,
  `softmax(Q@K^T / sqrt(dh)) @ V`, with the same per-head V projection
  and W_O as the rest. **Intentionally drops** Bonsignore's per-head
  extras (`log_tau`, `head_alphas`, `head_output_scalars`) so SDPA is
  a minimal control. The comparison vs Bonsignore tests "kernel choice
  + per-head extras" together; the comparison vs saliency variants
  tests scoring-mechanism class with matched extras (none).
- 4 seeds for SDPA (per spec). Existing 3-seed Bonsignore-baseline and
  variant-B results from the original WT-103 sweep are reused.
- Total wall clock: ~33 min (4 SDPA runs at ~8 min each).

## Results

| variant | n | mean PPL ± std | 95% CI | Δ vs Bonsignore | Δ vs SDPA |
|:-------:|:-:|:--------------:|:------:|:---------------:|:---------:|
| **sdpa (standard Q@K^T/sqrt(dh))** | 4 | **95.76 ± 0.78** | [94.99, 96.52] | **−5.81 (−5.7%)** | — |
| B (per-pair saliency MLP) | 3 | 97.61 ± 0.50 | [97.04, 98.18] | −3.96 (−3.9%) | +1.85 (+1.9%) |
| baseline (V22-Bonsignore) | 3 | 101.57 ± 0.29 | [101.24, 101.90] | — | +5.81 (+6.1%) |

Per-seed SDPA values: 96.26, 94.76, 95.57, 96.44.

Full ordering: **`SDPA > B > Bonsignore`**.

## Reading the result against the spec's decision tree

The spec listed four possible follow-up outcomes. The matching one is:

> **If variant B beats Bonsignore but not standard scaled-dot-product
> attention:** The finding is about the Bonsignore kernel specifically,
> not about Q-K scoring in general. Reframe the contribution as
> "saliency MLPs are a better alternative to Bonsignore" rather than
> "saliency MLPs replace pairwise scoring." This is a smaller but
> still real claim.

That's where we are. The "B beats baseline" headline from the original
WT-103 report is largely about Bonsignore being a worse kernel than
standard dot-product at this scale on this dataset. B sits between the
two: it improves over Bonsignore but doesn't reach standard SDPA.

The reframed contribution: at 30M params on WT-103 (5000 steps), the
V22 Bonsignore Q-K kernel is **suboptimal compared to standard
scaled-dot-product attention**. A learned MLP over `(x_q, x_k)` partly
recovers the gap (B at 97.61 vs SDPA at 95.76), but doesn't close it.

## Implications for V22's design

V22 was built around the Bonsignore kernel as a hypothesised
improvement over standard attention. This experiment's data point
(at 30M params, 5K steps, BPE WT-103) shows the opposite: standard
SDPA outperforms Bonsignore by 5.7% PPL with very high confidence
(Cohen's d ≈ −13 at SDPA's std).

Caveats remain. This is a small-scale, short-horizon, single-task
test:
- 30M params vs V22's ~512M
- 5K steps vs V22's ~63K
- WT-103 only; V22 was designed with the categorization-head and
  topic-routing tasks in mind too
- The Bonsignore kernel might pull ahead with longer training (next
  experiment)
- The per-head extras (log_tau, head_alphas, head_output_scalars) that
  Bonsignore has but SDPA lacks could matter at scale even if they
  don't here

But the directional finding — Bonsignore is not winning at this scale
— is at minimum a reason to question whether Bonsignore was the right
attention choice for V22.

## Implications for variant B

B was claimed (in the original WT-103 report) to "beat baseline" and
this was framed as "saliency-MLP scoring outperforms Q-K Bonsignore
attention." That claim is true but narrower than originally framed.
The correct framing:

- B beats Bonsignore by 3.9%
- B *loses* to standard SDPA by 1.9%
- The 3.9% B-vs-Bonsignore gap and the 5.7% SDPA-vs-Bonsignore gap
  both seem to be primarily about Bonsignore being a weaker kernel,
  not about MLP-vs-bilinear scoring being fundamentally different.

A learned scoring MLP (B) is **better than Bonsignore** but **not as
good as standard scaled-dot-product**. The MLP's flexibility helps
versus a specifically-misshaped kernel; against the standard kernel,
it doesn't have an edge at this scale.

## Decision

Per the spec, this outcome means: don't proceed to V22 scale-up of
variant B as a general attention replacement. Reframe accordingly.

Two things still worth knowing:

1. **Does the SDPA-vs-Bonsignore gap persist at longer training?**
   Currently planning Experiment 1 (long-train) as a 3-way comparison
   (baseline, B, sdpa) at 20K steps so we can answer this and the
   B-vs-SDPA-at-long-horizon question simultaneously.

2. **Is Bonsignore weak in general or just here?** The framework
   prediction was that Bonsignore would *help*. The data points the
   other way at this scale. Worth re-running the original V20-vs-V21
   comparison to check whether Bonsignore's prior wins were
   training-recipe-specific (V20 was trained with phase scheduling
   and longer; this experiment was a fresh-init 5K-step compare).
   Out of scope for the immediate follow-up but flagged for later.

## Files

- `report.json` (analyze.py output)
- `sdpa_seed{0..3}.json` — per-run records
- `sweep_summary.json`
- Code: `experiments/saliency_pool/{attention.py: StandardAttention, run_sweep_wt103_sdpa.py}`

## Budget actuals

Spec estimate: "a few hours". **Used 33 minutes** for 4 SDPA seeds at
5K steps each.
