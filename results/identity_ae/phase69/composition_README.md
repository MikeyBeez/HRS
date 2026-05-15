# Phase 69 — Independent adapter composition via rank-dimension concatenation

Test whether HRS adapters trained independently on different passages will
compose cleanly when their LoRA matrices are concatenated along the rank
dimension (row-concat A, col-concat B, scaling preserved per-component so the
composed forward equals the sum of per-adapter forwards). If they did, we'd
have a memory architecture that scales without per-passage routing: store the
data, train adapters in parallel, compose at inference.

## Headline

**The composability hypothesis is falsified.** Even when individual adapters
work cleanly (rank 128, 6/8 K=1 retrieval, ppl 1.0–1.2 on their own passages)
and even though their weight perturbations are approximately orthogonal in
Frobenius space (‖Σδ‖ grows ~√K, not linearly), retrieval collapses at K=2 and
perplexity degrades monotonically. Magnitude growth is *not* the dominant
failure mode — direction interference is.

|                                | rank 8 (spec primary) | rank 128 (Phase 47 control) |
|--------------------------------|-----------------------|-----------------------------|
| K=1 constituent retrieval      | 25% (2/8)             | **75% (6/8)**               |
| K=1 mean ppl on own passage    | 1.29                  | 1.15                        |
| K=2 constituent retrieval      | 0%                    | 0%                          |
| K=2 mean ppl                   | 4.92                  | 1.58                        |
| K=4 constituent retrieval      | 0%                    | 25%                         |
| K=4 mean ppl                   | 9.78                  | 2.22                        |
| K=8 constituent retrieval      | 0%                    | 0%                          |
| K=8 mean ppl                   | 819.11                | 21.49                       |
| K=8 ‖Σδ‖_F                     | 21.10                 | 32.90                       |
| √K · ‖δ_1‖_F (orthogonal ref)  | 17.78                 | 30.13                       |
| Held-out retrieval, every K    | 0/1                   | 0/1                         |

## Methodology notes (deviations from spec, called out before results)

- **Corpus.** Spec says "8 Dickens passages from the existing Phase 47/50-
  Dickens corpus". No Dickens corpus exists in this repo; Phase 47/50 use
  the synthetic stratified passkey corpus (`generate_passkeys(50)` →
  `stratified_tests()`), which contains 5 each of 4 topic types (numeric,
  entity, technical, fact). 8 passages selected with 2 of each type to
  maximize topical diversity: ids `[0, 4, 20, 23, 30, 33, 40, 43]`. Held-out
  is id 21 (entity, different name, not in any composition).
- **Target modules.** Spec phrases the LoRA targets two ways at once
  ("MLP down-projections of all transformer layers" *and* "same as the current
  HRS adapter setup"). The current HRS setup (Phase 14/22/47) is
  `L45_TARGETS` — attention qkv/out_proj + PEER FFN input/output on layers
  4–5. Used that for consistency with the Phase 47 hyperparameter comparison.
- **Rank.** Spec says rank 8 (matches D2L per-chunk rank). Run as primary.
  Rank 128 (Phase 47's working setting) added as a methodology control after
  the rank-8 K=1 baseline fell well short of the spec's "~93–97% retrieval"
  debug gate — without it the composition story would be confounded with
  "single-rank-8 adapter doesn't retrieve at all in this corpus."
- **Composition installation.** Rather than allocating fresh larger LoRA
  layers, the composed Σ_i δ_i is merged directly into `base_layer.weight`
  with `lora_B=0`. Mathematically identical, lets us reuse the existing
  rank-fixed LoRALayer class. Base weights snapshotted before any merge and
  restored between K conditions.

## Per-K detail

### Rank 128 (working baseline)

K=1 individual adapter retrieval is 6/8 (passages 23 and 43 fail individually
under greedy decoding; the model regurgitates the template prefix without
arriving at the entity). For the K-sweep we still use those passages as
constituents — the failure is in the question→answer generation path, not in
whether the passage was learned (all ppl ≤ 1.24).

```
K=1   100% (1/1)   ppl 1.23   ‖δ‖ 10.66   heldout 0/1, ppl 271.83
K=2     0% (0/2)   ppl 1.58   ‖δ‖ 18.09   heldout 0/1, ppl 757.41
K=4    25% (1/4)   ppl 2.22   ‖δ‖ 23.72   heldout 0/1, ppl  18.80
K=6    16% (1/6)   ppl 6.79   ‖δ‖ 28.52   heldout 0/1, ppl  36.86
K=8     0% (0/8)   ppl 21.49  ‖δ‖ 32.90   heldout 0/1, ppl  71.50
```

Note that retrieval is non-monotone (K=4 above K=2). With 0/1 hits and small N
per K, individual-passage greedy noise dominates the retrieval signal. The
perplexity trend is the cleaner measure and degrades monotonically.

### Rank 8 (D2L per-chunk rank; spec primary)

K=1 retrieves only 2/8. The adapters do learn the passages (ppl 1.0–1.6) but
rank-8 capacity isn't enough for the question→answer routing path to surface
the right entity consistently. Composition makes this worse fast.

```
K=1     0% (0/1)   ppl   1.63   ‖δ‖  6.29
K=2     0% (0/2)   ppl   4.92   ‖δ‖ 10.75
K=4     0% (0/4)   ppl   9.78   ‖δ‖ 14.82
K=6     0% (0/6)   ppl 115.12   ‖δ‖ 18.27
K=8     0% (0/8)   ppl 819.11   ‖δ‖ 21.10
```

(K=1 hit=0 here is from passage 0 specifically — the only constituent at K=1
is passage 0. In Phase B's per-adapter sweep, passages 4 and 20 hit, others
don't; those passages appear as constituents only at K≥2 in this sweep.)

The rank-8 perplexity blow-up at K≥6 is severe enough that the model is no
longer doing language modeling on its own training passages.

## Pre-committed predictions vs measured outcomes

| Prediction | Measured | Why the prediction was wrong |
|------------|----------|------------------------------|
| K=1 ~93–97% retrieval | rank 8: 25%, rank 128: 75% | The Phase 47 retrieval headline (~93–97%) used multi-prompt absorption (Phase 26/47 `train_adapter_multipara`) and rank ≥ 128. Single-passage NTP at the spec hyperparameters gets the passage into ppl, not into reliable greedy retrieval from a question. |
| K=2 88–95% | rank 8: 0%, rank 128: 0% | Composition collapses retrieval at K=2 regardless of rank, despite passages being topically distinct. |
| K=4 80–92% | rank 8: 0%, rank 128: 25% | Big miss. Rank 128 has a single hit at K=4, well below the predicted band. |
| K=8 65–85%, success threshold 70% | rank 8: 0%, rank 128: 0% | Hypothesis falsified by a wide margin. |
| Held-out unchanged across K | ✅ rank 8 and rank 128 | The one prediction that holds. Composed adapters do *not* acquire generic abilities; perturbations are passage-specific in direction. |
| Frobenius grows roughly linearly | grows as ~√K (sub-linear) | This is the most informative miss. Sub-linear Frobenius growth indicates the per-adapter deltas are approximately *orthogonal in weight space*. So orthogonality was the easy half of the hypothesis; functional non-interference was the hard half, and it failed even with orthogonal weights. |

## What the Frobenius numbers say

For approximately-orthogonal deltas of equal Frobenius norm ‖δ_1‖,
‖Σδ_i‖ ≈ √K · ‖δ_1‖.

- Rank 128: predicted √8 · 10.66 = 30.13. Measured: 32.90. Within 9% of
  orthogonal sum.
- Rank 8: predicted √8 · 6.29 = 17.78. Measured: 21.10. Within 19%.

Both regimes show approximately-orthogonal aggregation — the D2L "magnitudes
explode" failure mode is *not* what's killing us. The deltas roughly avoid
each other in Frobenius space. The composed map nonetheless distorts the
greedy-decoding behavior enough to lose retrieval. The interference is in the
*functional direction* (which logits get pushed up at which positions), not in
total perturbation magnitude.

## Architectural interpretation

D2L meta-trains its hypernetwork to produce adapters that compose; we got the
weak version of composability "for free" (deltas approximately orthogonal in
weight space), and that wasn't enough. Two failure mechanisms are consistent
with the data:

1. **Functional interference at decoding.** Each adapter pushes the next-token
   distribution toward its own passage's vocabulary at every context where the
   question pattern matches its template. Other adapters' templates are
   syntactically close enough that their pushes apply at the same contexts.
   Even orthogonal weight changes can produce non-orthogonal logit pushes.
2. **Capacity contention on shared MLP/attn pathways.** The L4–5 attn and PEER
   FFN are shared by every adapter; rank-8 spreads its delta over a low-
   dimensional subspace of that shared circuit, and 8 such subspaces overlap
   functionally even if they're nearly orthogonal as Frobenius matrices.

Both point in the same direction for a fix: composition has to be trained,
either as D2L does (KL-style co-training of the hypernetwork on multi-adapter
contexts) or by routing at inference (load the right adapter conditioned on
the query) rather than summing. The rank-dimension concat is the right object
to compose with; what's missing is supervision that the sum should look like
each summand on its own passage.

## Files

- `experiments/identity_ae/phase69_adapter_composition.py` — script
- `adapters_rank8/passage_{0..7}.pt`,
  `adapters_rank128/passage_{0..7}.pt` — saved per-passage LoRA state dicts
- `composition_results.json` — full per-K, per-passage records for both ranks
- `perturbation_magnitudes.json` — Frobenius norms by rank × K
- `composition_curves.png` — rank 8 panels (retrieval, ppl, ‖δ‖_F)
- `composition_curves_rank128.png` — rank 128 panels
- `composition_run.log` — stdout from the full run

(Phase 69 verification artifacts — `verification.json`, `verification_README.md`,
`signal_distributions.png`, `verification_run.log` — are a separate experiment
from earlier work and are unaffected by this run.)

## Open questions

1. **Does meta-training rescue it?** Does adding a KL-style co-training term
   (sample two adapters, randomly compose, supervise each summand to match
   its own passage's loss under the composition) recover the K=8 retrieval
   target? This is the D2L hypothesis applied to NTP-trained adapters.
2. **Is "approximately orthogonal in weight" actually orthogonal where it
   matters?** Compute the cosine of A_i⊤A_j (or B_iB_j⊤) per layer to see if
   the small (~9–19%) excess over √K-sum is concentrated in specific layers
   that drive the decoding interference.
3. **Does multi-prompt absorption fix the K=1 baseline?** Phase 47's
   ~93–97% used multi-prompt training; if we re-train each rank-128 adapter
   with `train_adapter_multipara` and the K=1 sweep moves to 8/8, the K-sweep
   shape might look different (e.g. retain retrieval longer before
   collapsing).
4. **Routing wins by default.** Per-adapter routing (Phase 21–47) sidesteps
   composition entirely; this result reinforces that the routing path is the
   architecturally sound deployment, and composition is a research curiosity
   under current training.
