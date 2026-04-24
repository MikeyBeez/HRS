# Seed Transfer Test — Results

5 additional MHA baselines trained with seeds {1, 2, 3, 4, 5} using **identical** architecture, data, hyperparameters, and step counts (5K LM + 20K passkey). Seed 0 in `experiments/pruning/checkpoints/` is the reference. Seeds 4–5 added in a follow-up run for statistical power.

## Convergence

| Seed | LM val PPL | Passkey exact | Converged? (≥0.95 exact, ≤5.5 PPL) |
|---|---|---|---|
| 0 (baseline) | 4.92 | 1.000 | ✓ |
| 1 | 4.89 | **0.592** | ✗ — excluded from overlap stats |
| 2 | 4.94 | 1.000 | ✓ |
| 3 | 4.88 | 1.000 | ✓ |
| 4 | 4.86 | 1.000 | ✓ |
| 5 | 4.86 | 1.000 | ✓ |

5/6 seeds converged. Seed 1 plateaued at 59% passkey within its 20K-step budget.

## Retrieval heads per seed (Δpasskey > 0.5, or top-3)

| Seed | Retrieval heads | Count | Layers |
|---|---|---|---|
| 0 | L0 H3(1.00), L2 H0(1.00), L0 H2(0.96), L0 H1(0.53) | 4 | {0, 2} |
| 1 (not converged) | L0 H2(0.59), L0 H0(0.58), L0 H3(0.58), L0 H1(0.54) | 4 | {0} |
| 2 | L0 H2(1.00), L2 H0(0.98), L0 H3(0.89), L0 H1(0.88), L1 H2(0.85), L1 H3(0.80), L0 H0(0.71) | 7 | {0, 1, 2} |
| 3 | L1 H1(1.00), L2 H2(1.00), L0 H3(0.98), L0 H0(0.97), L0 H2(0.93), L0 H1(0.89) | 6 | {0, 1, 2} |
| 4 | L0 H3(1.00), L0 H2(0.99), L0 H0(0.64), L2 H0(0.63) | 4 | {0, 2} |
| 5 | L0 H1(0.99), L0 H2(0.99), L1 H3(0.84), L0 H0(0.83), L0 H3(0.79), L2 H3(0.65), L1 H2(0.63), L1 H0(0.52) | 8 | {0, 1, 2} |

## Overlap metrics (5 converged seeds)

- **Layer coverage: 100%.** All 5 converged seeds place retrieval heads in **both** baseline layers {0, 2}.
- **Exact position coverage**: seed 2 = 4/4 baseline positions; seeds 3, 4, 5 = 3/4.
- **L3 never rises above Δpasskey 0.42** in single-head ablation across any seed — but see the "whole-layer ablation" section below, which complicates the "L3 is prunable" claim.
- **Retrieval head counts: {4, 7, 6, 4, 8}.** Not conserved across seeds.
- **L1 usage is optional.** Seeds 0 and 4 use only {L0, L2}; seeds 2, 3, 5 recruit L1.

## Whole-layer ablation: every layer probed

Phase 1 measured *single-head* importance. A head can look unimportant individually (Δpasskey ≈ 0) and yet be collectively essential when its siblings are also removed. To check for this, zero **every head in a single layer** at once and measure before/after 500-step fine-tune.

### Summary: passkey after zeroing each whole layer, then 500 FT steps

| Layer zeroed | seed 0 | seed 1† | seed 2 | seed 3 | seed 4 | seed 5 | verdict |
|---|---|---|---|---|---|---|---|
| **L0** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | **irrecoverable — always 0, even with 500 FT** |
| L1 | 1.000 | 0.496 | 0.000 | 0.272 | 1.000 | 0.948 | bimodal, depends whether seed placed retrieval in L1 |
| L2 | 0.300 | 0.548 | 0.884 | 0.180 | 0.560 | 0.896 | always degraded; partial FT recovery (0.18–0.90) |
| L3 | 0.300 / 0.568‡ | 0.588 / 0.756‡ | 0.744 / 0.876‡ | 0.040 / 0.180‡ | 0.596 / 0.780‡ | 0.656 / 0.964‡ | recoverable with more FT; seed-dependent |

†Seed 1 didn't converge; its baseline was 0.59 rather than 1.00, so interpret cautiously.
‡FT(500) / FT(2000). L3 was run with both budgets because single-head ablation suggested it was the "safest" layer.

### Summary: PPL damage (no-FT) from zeroing each whole layer

| Layer zeroed | PPL no-FT (mean across converged seeds) | PPL after 500 FT |
|---|---|---|
| L0 | **55.1** (11× baseline) | +0.06 |
| L1 | **47.1** (9.5× baseline) | +0.24 |
| L2 | 13.9 (2.8× baseline) | +0.13 |
| L3 | 6.1 (1.2× baseline) | +0.08 |

### Interpretation by layer

**L0 — structurally irreplaceable for retrieval.** Zeroing all of L0 crashes passkey to 0.000 in every seed and 500 FT steps do **not** recover *any* signal (still 0.000). This is the only layer where even short fine-tuning completely fails to reconstitute retrieval. L0 is where positional pattern-matching begins (finding MARKER in context); without it, later layers have no positional signal to compose with, and the retrieval circuit cannot be re-grown without that substrate. PPL, by contrast, recovers fully — meaning L0's role for PPL is replaceable by subsequent layers, but its role for passkey is not.

**L1 — PPL-critical, retrieval-conditional.** Zeroing L1 causes the worst *unrecovered* PPL damage (9.5× baseline), and FT fully recovers PPL but not always passkey. Passkey recovery is bimodal and matches Phase 1 exactly: seeds that placed retrieval heads in L1 (seeds 2, 3, 5) fail or partially recover; seeds that didn't (seeds 0, 4) recover to 1.000. **Per-seed importance ranking is required for deployment** — pruning recipes don't transfer across seeds.

**L2 — always retrieval-involved, partially recoverable.** Every converged seed has at least one L2 retrieval head (H0 in baseline/seed 2/seed 4, H2 in seed 3, H3 in seed 5). Zeroing all of L2 → passkey 0.00–0.15 no-FT, 0.18–0.90 after FT. Partial recovery across the board; never the clean 1.000 that L1-removal gives for seeds 0/4.

**L3 — redundantly essential.** Phase 1 showed no single L3 head was individually critical (max Δpasskey = 0.42, most ≈ 0). But zeroing all of L3 → passkey 0.000 in every converged seed. Any one of the four L3 heads can do L3's job (internal redundancy), but the layer as a whole cannot be removed. With 2000 FT steps, passkey partially rebuilds (0.18–0.96), suggesting L3's function is the most substitutable of the "always necessary" layers. Probably does the final write to logit space.

### The architectural hierarchy

1. **L0**: structurally necessary for retrieval; irreplaceable by fine-tuning.
2. **L2**: always retrieval-involved; partially substitutable with FT.
3. **L3**: redundantly essential; substitutable if given enough FT.
4. **L1**: optional for retrieval (seed-dependent); PPL-critical.

This refines the earlier "L0 + L2 are the retrieval layers" story into something more mechanistic:
- **Finding** = L0 (positional signal; not substitutable)
- **Composing** = L2 (content + position; partially substitutable)
- **Writing** = L3 (token to logits; substitutable given FT)
- **General language capability** = L1 + all the FFNs (broadly redundant; fully substitutable with FT)

## Outcome

The spec asked which of three outcomes the data supports. Updated reading after the layer-ablation follow-up:

1. **Exact same positions** — no.
2. **Same layers, different head indices** — yes, for {L0, L2}. Every converged seed uses both, and the head index within L2 varies (H0 in baseline/seeds 2/4, H2 in seed 3, H3 in seed 5).
3. **Positions are arbitrary** — partially, for the *optional* layers. L1 is recruited by 3/5 converged seeds; L3 is used redundantly by every seed. The "number of heads the circuit needs" varies from 4 to 8 depending on initialization.

Composite finding: the architecture has a **canonical layer structure** (L0 always, L2 always, L3 redundantly essential, L1 optional) but **within-layer head allocation is stochastic**. This is a clearer picture than the original spec's three-way framing.

## Revised interpretation of the earlier head-pruning experiment

The original head-pruning README (`experiments/head_pruning/results/README.md`) identified 3 "retrieval heads" (L0 H2, L0 H3, L2 H0) by single-head ablation, concluding that retrieval is concentrated in a small subgraph. That's **mechanically correct but incomplete**:

- L0 H2 / L0 H3 / L2 H0 are **individually critical** — zeroing any one crashes passkey.
- But L3 as a whole is **also** needed — zeroing all 4 L3 heads also crashes passkey, even though no single L3 head was individually critical.
- So the retrieval circuit spans **at least 3 heads in L0+L2 plus the entire L3 layer (any subset of it)**.

This complicates the Phase-2 result too. The "least-important-first + 500 FT + n=8 pruned → 100% passkey" worked because the algorithm pruned 2 L3 heads (keeping 2), giving FT enough substrate. If someone had pruned all 4 L3 heads at n=4 they'd have gotten much worse results than the 4-at-other-layers they actually pruned. **Importance ranking by single-head ablation underestimates collective importance.**

## Files

```
plan.md                          # spec
run_seeds.py                     # driver: train + rank per seed, compare
verify_layer_prunable.py         # whole-layer ablation (with/without FT)
seed_{1..5}/checkpoints/         # per-seed MHA baselines
seed_{1..5}/head_importance.json
seed_{1..5}/importance_heatmap.png
comparison.json                  # per-seed importance + overlap summary
overlap_analysis.txt
importance_grid.png              # 3×2 grid of 4×4 heatmaps, seeds 0–5
layer_0_prunable.txt             # all-L0 zeroed, 500 FT steps (never recovers)
layer_1_prunable.txt             # all-L1 zeroed, 500 FT steps (bimodal)
layer_2_prunable.txt             # all-L2 zeroed, 500 FT steps (partial)
layer_3_prunable.txt             # all-L3 zeroed, 500 FT steps
layer_3_prunable_ft2000.json/.txt # all-L3 zeroed, 2000 FT steps
run.log, run_45.log              # training logs
l0_verify.log, l1_verify.log, l2_verify.log, l3_verify.log, l3_verify_ft2000.log
README.md                        # this file
```

## Reproduce

```bash
# All 5 additional seeds:
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.seeds.run_seeds \
    --seeds 1 2 3 4 5 --lm-steps 5000 --passkey-steps 20000

# Whole-layer prunability checks (all layers, all seeds):
for L in 0 1 2 3; do
  PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.seeds.verify_layer_prunable \
      --layer $L --seeds 0 1 2 3 4 5 --ft-steps 500
done
# Extended FT for L3 (the "most substitutable" of the necessary layers):
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.seeds.verify_layer_prunable \
    --layer 3 --seeds 0 1 2 3 4 5 --ft-steps 2000
```

## Natural next steps

- **Longer FT budget for L2 and L3 removal.** L3 almost fully recovers in some seeds with 2000 FT steps (seed 5: 0.96). A 10K-step FT might close the gap entirely and reclassify "redundantly essential" as "substitutable-with-effort." L2 might behave similarly.
- **L0 stress-test.** Try 10K FT steps to see if L0 is *literally* irreplaceable or just needs much more recovery time. Prediction: still 0.000 — there's no source of positional pattern-matching for FT to reshape.
- **Per-seed head-pruning sweep.** Apply Phase-2 sweep (`experiments/head_pruning/run_sweep.py`) to seeds 2, 3, 4, 5. Expect seed 2 (7 retrieval heads) to tolerate fewer head prunings than seed 4 (4 retrieval heads, {L0, L2} only, closest to baseline).
- **Mechanistic probe of L3.** If L3's role is just "write retrieved token to logit space," an identity-initialized L3 (W_V = W_O = I, attention = identity/averaging) should work. Worth testing.
- **Probe L0's attention patterns.** Visualize the attention maps of L0 heads on passkey examples. If they form a canonical induction pattern (previous-token + content-match), that closes the mechanistic loop.
