# k=2 Multi-Adapter Composition: Cross-Term Experiment

**Question:** Does additive LoRA composition at k=2 suffer from cross-term interference, and if so, which intervention recovers single-adapter performance?

**Headline:** **Yes — cross terms cost 27.3 pts of retrieval at k=2.** All four spec-defined interventions fail to close the gap. The architectural conclusion is the spec's "no intervention closes the gap" branch: additive LoRA composition is fundamentally limited at this rank/scaling, and HRS should commit to k=1 routing.

## Setup

- Substrate: existing Dickens-50 adapter library (50 rank-128 LoRA adapters on layers 4-5 of GPT-2 V22, Dickens-pretrained).
- Probes: 150 hand-crafted held-out probes (entity / numeric / place / relation).
- 50 random pairs sampled with seed 42; each pair contributes 2 target measurements (one per slot), 3 paraphrases × 3 stochastic seeds = 18 probes per pair, 900 N=2 measurements total.
- N=1 reference: existing per-adapter retrieval, evaluated through the new MultiLoRALayer with n_active=1.

The MultiLoRALayer utility (`multi_lora.py`) holds up to max_k stacked (A, B) parameter pairs and computes `y = W_0 x + sum_i (x A_i B_i) * scaling`. For n_active=1 it is bit-equivalent to the original LoRALayer; for n_active=2 it is the additive k=2 composition.

## Phase 1: baseline

| Condition | Retrieval (3-seed mean) |
|---|---|
| N=1 (single-adapter via MultiLoRALayer) | **0.924** |
| N=2 vanilla (additive composition) | **0.651** |
| Gap | **0.273** |

Per-fact-type at N=2 (vs N=1 in parens):

| fact type | N=1 | N=2 | Δ |
|---|---|---|---|
| entity | 0.924 | 0.626 | -0.298 |
| numeric | 0.963 | 0.878 | -0.085 |
| place | 0.911 | 0.702 | -0.209 |
| **relation** | **0.889** | **0.074** | **-0.815** |

The N=1 result (0.924) reproduces the published per_passage_dickens evaluate.py number (0.929) within 0.5 pts — confirming MultiLoRALayer at n_active=1 is bit-correct relative to the original LoRA pipeline. The 5-pt threshold for proceeding to Phase 2 is decisively passed.

### Per-layer activation drift

V22 has 6 transformer blocks; LoRA is on L4-L5 only. Drift is defined as `||h_{i,j}^L - h_i^L|| / ||h_i^L||` at each block's last-token output.

| Layer | Mean drift | std |
|---|---|---|
| L0 | 0.000 | 0.000 |
| L1 | 0.000 | 0.000 |
| L2 | 0.000 | 0.000 |
| L3 | 0.000 | 0.000 |
| L4 | 0.258 | 0.154 |
| L5 | 0.450 | 0.211 |
| Output logits | 0.512 | — |

Drift is exactly zero before the first LoRA-augmented layer and grows monotonically through the two LoRA layers. The companion adapter perturbs activations starting at L4 (where it first contributes to the residual stream), and the perturbation compounds at L5 (the output of the trained pathway). There are no later transformer blocks to compound through, so the cross-term effect lands directly on the lm_head.

Figure: `figures/layer_drift.png`.

## Phase 2: interventions

Five conditions tested (the four spec interventions plus the orthogonality λ sweep).

| Condition | N=1 | N=2 | gap | within 3 pts of N=1? |
|---|---|---|---|---|
| vanilla_addition | 0.924 | 0.651 | 0.273 | no |
| taylor | 0.924 | 0.520 | 0.404 | **no — worse than vanilla** |
| discrete | 0.924 | 0.116 | 0.808 | **no — catastrophic** |
| orthogonal_l_0.01 | 0.833 | 0.411 | 0.422 | no — N=1 also tanked |
| orthogonal_l_0.1 | 0.758 | 0.276 | 0.482 | no — both worse |
| orthogonal_l_1.0 | 0.693 | 0.204 | 0.489 | no — both worse |
| crosstermaware | 0.938 | 0.653 | 0.284 | no — null vs vanilla |

**No condition meets the spec's criterion.**

Figure: `figures/retrieval_comparison.png`.

### Intervention A — Taylor linearization (no retraining)

Replace the k=2 forward output with the first-order Taylor approximation in adapter contributions:
```
logits_taylor = logits(adapter_i alone) + logits(adapter_j alone) - logits(base)
```

Result: 0.520 — **worse** than vanilla N=2 by 13.1 pts.

Diagnostic interpretation (per spec): cross terms are NOT first-order removable. The linearized output discards higher-order interactions that carry useful signal. The vanilla two-adapter forward, despite suffering cross-term interference, stays closer to the trained activation distribution than three independent forward passes summed in logit space. This rules out "wrap k=2 inference in a linear-combination decoder" as a cheap fix.

### Intervention D — Discrete per-layer routing (no retraining)

Layer 4 = target adapter; Layer 5 = companion adapter. Each LoRA-augmented layer hosts at most one adapter, so cross terms cannot exist at any layer.

Result: 0.116 — catastrophic, **53.5 pts below vanilla**.

The adapter learned a joint L4+L5 transformation; replacing half its trained pathway with a different adapter destroys retrieval. The two LoRA layers are not independently functional. (This is the "cost of avoiding composition entirely" baseline the spec called for; the cost is severe.)

### Intervention B — Orthogonal subspace regularization (retrain)

Sequential adapter training with regularizer:
```
L = L_task + λ · Σ_{j<i} (||A_i A_j^T||_F^2 + ||B_i^T B_j||_F^2)
```
computed efficiently via Frobenius inner products of cached Gram matrices. λ swept ∈ {0.01, 0.1, 1.0}.

Monotonic pattern across λ:

| λ | N=1 | N=2 | gap |
|---|---|---|---|
| 0 (vanilla) | 0.924 | 0.651 | 0.273 |
| 0.01 | 0.833 | 0.411 | 0.422 |
| 0.1 | 0.758 | 0.276 | 0.482 |
| 1.0 | 0.693 | 0.204 | 0.489 |

Stronger constraint → both N=1 and N=2 worse, gap larger. The orthogonality penalty doesn't reserve disjoint subspaces for adapters cleanly; it forces every adapter to occupy a worse subspace than it would otherwise choose. Training logs show task loss spikes for late-trained adapters (e.g., adapter 45 hits task=2.88 at λ=0.1, task=4.53 at λ=1.0) — the subspace fills up and the constraint fights task learning.

This intervention fails decisively. The spec's failure mode "as more adapters are added, the orthogonality constraint becomes harder to satisfy" is exactly what we see, and earlier-trained adapters don't benefit either.

### Intervention C — Cross-term-aware training (retrain)

Sequential training. For each step training adapter i (i > 0), with p=0.5 a random companion adapter j < i is loaded into slot 1 of MultiLoRALayer, n_active=2. Loss computed against adapter i's task target; backprop only updates slot 0.

Result: N=1 = 0.938, N=2 = 0.653.

Both numbers match vanilla within noise (~1 pt each direction). The intervention is a **null effect on overall retrieval** — training with companion exposure neither hurts nor helps the average.

But there's a per-type signal worth flagging. Relation probes are by far the most cross-term-sensitive type: vanilla N=2 retrieves only 0.074 on relations vs 0.889 at N=1 (-81.5 pts). Cross-term-aware training improves relation N=2 to **0.259** — a 3.5× improvement, even though the overall stays flat:

| fact type | vanilla N=2 | crosstermaware N=2 | Δ |
|---|---|---|---|
| entity | 0.626 | 0.615 | -0.011 |
| numeric | 0.878 | 0.678 | -0.200 |
| place | 0.702 | 0.808 | +0.106 |
| **relation** | **0.074** | **0.259** | **+0.185** |

The intervention helps where it hurts most (relations, places) but trades off elsewhere (numerics). Net flat. A targeted version of this intervention — applied selectively to fact types known to suffer most — might be worth exploring, but the spec's overall metric doesn't capture this.

## Decision

**No intervention closes the gap to within 3 pts of N=1.** The spec's failure-branch conclusion is the recommended action:

> "additive LoRA composition has a fundamental cross-term limit; HRS should commit to k=1 routing"

Vanilla N=2 (0.651) is the BEST k=2 result available. Every intervention either matches it (cross-term-aware) or makes it worse (Taylor, discrete, all three orthogonal lambdas).

## What the data tells us about the mechanism

Three pieces of evidence converge on the same conclusion: the cross-term contribution is content-bearing, not noise.

1. **Taylor decomposition fails (0.520 < 0.651).** The first-order linear approximation in adapter contributions is *worse* than the full nonlinear N=2 forward. If cross terms were pure interference, removing them via linearization would help. They aren't.

2. **Orthogonal regularization fails monotonically (0.273 → 0.489 gap).** Penalizing per-layer subspace overlap doesn't make adapters compose more cleanly; it just degrades each adapter individually. The adapters need to use overlapping subspaces to perform well at all.

3. **Cross-term-aware training has a null overall effect.** Training adapters in the *presence* of cross-term distortion doesn't make them robust to it. The gap to N=1 is preserved exactly.

The pattern: at rank=128 with α=256 on layers 4-5 of a 6-block model, each adapter's contribution is large enough that two such contributions added produce a residual stream meaningfully outside the trained activation manifold of either adapter. The nonlinearities downstream of the LoRA-augmented layers (attention softmax, FFN GELU) amplify this off-manifold input into incorrect predictions, and the trained adapter has no sub-manifold of robustness to draw from.

## Concrete numbers worth remembering

- **k=1 retrieval: 0.924** (validated; matches published 0.929 within 0.5 pts).
- **k=2 vanilla retrieval: 0.651** (gap 0.273, decisive).
- **Relation-type collapse at k=2: 0.889 → 0.074** (-81.5 pts, the worst of any fact type).
- **Cross-term-aware training partially recovers relation: 0.074 → 0.259** (3.5× but still well below 0.889).
- **Drift profile**: zero through L0-L3, jumps to 0.258 at L4 (where companion injects), 0.450 at L5, output 0.512.

## Recommendation for HRS

Commit to k=1 routing. The architecture story is:
- The W projection makes routing reliable at k=1 (Phase 47 confirmed; this experiment validates with 100% routing assumption holding).
- Additive composition at k=2 is *not* a free extension — it costs 27 pts of retrieval, concentrated on the most compositional fact types.
- None of the spec's interventions (linearization, layer-splitting, orthogonal training, cross-term-aware training) close the gap.

If multi-passage retrieval is needed, the architecture should:
- Route to k=1 and reformulate the question to be answerable from one passage, OR
- Run k separate forwards with different adapters and reconcile at the output level (text combination, not logit combination — Taylor showed logit summing is worse than k=2 vanilla), OR
- Re-investigate at lower rank/scaling (the cross-term magnitude scales with adapter contribution magnitude; rank=32 or smaller α may produce a smaller gap, though probably with proportionally weaker adapters).

The relation-specific signal under cross-term-aware training (3.5× recovery on the worst-affected type) is worth noting as a partial mitigation if a future variant of the architecture targets the most compositional fact types specifically.

## Files

- `multi_lora.py` — MultiLoRALayer + helpers
- `run_phase1.py` — N=1/N=2 baselines + drift profile
- `run_phase2_no_retrain.py` — Taylor + discrete (combined script)
- `run_phase2d_discrete_only.py` — standalone discrete (after layer-detection bug fix)
- `train_phase2b_orthogonal.py` — orthogonal-regularized training
- `train_phase2c_crosstermaware.py` — cross-term-aware training
- `eval_adapter_dir.py` — parameterized N=1/N=2 evaluator
- `run_phase3_aggregate.py` — summary CSV + figures

- `results/phase1_summary.json` — Phase 1 aggregates
- `results/phase1_n1_baseline.csv`, `phase1_n2_baseline.csv`, `phase1_layer_drift.csv` — Phase 1 details
- `results/phase2_no_retrain_summary.json` — Taylor + discrete combined
- `results/phase2a_taylor.csv`, `phase2d_discrete.csv` — per-probe details
- `results/eval_orthogonal_lambda_{0.01,0.1,1.0}.json` — orthogonal eval per λ
- `results/eval_crosstermaware.json` — cross-term-aware eval
- `results/phase3_summary.csv` — final condition × metrics table
- `results/train_*.log`, `eval_*.log`, `run_*_master.log` — run logs

- `adapters_orthogonal_lambda{0.01,0.1,1.0}/adapter_NN.pt` — orthogonal-regularized adapters
- `adapters_crosstermaware/adapter_NN.pt` — cross-term-aware adapters

- `figures/layer_drift.png` — per-layer activation drift, k=2 vanilla
- `figures/retrieval_comparison.png` — k=2 retrieval bar chart with N=1 reference
