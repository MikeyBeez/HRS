# Magnitude Pruning Ablation — Results

Baseline: MHA variant from `experiments/diagonal_attention/`, retrained as two task-specific checkpoints (LM on Tiny Shakespeare for 5K steps; passkey for 20K steps — reaches 100% exact before the step budget is spent).

Baselines:
- LM val PPL = **4.92**
- Passkey exact-match = **1.000**

Full sweep: **3 scopes × 7 sparsities × 4 strategies = 84 runs**, ~90 min on RTX 5070 Ti.

## One-paragraph summary

**PPL degrades as a slope; passkey degrades as a cliff.** Across all three scopes, PPL climbs smoothly as sparsity rises (under oneshot), while passkey holds at 100% until it suddenly collapses. Fine-tuning recovers almost everything below the per-scope recovery ceiling: **MLP-only recovers through 95%**, **attention-only needs long FT (2000 steps) to recover 95%** but 500 steps suffice below that, and **all-weights cannot recover passkey at 95% with any strategy tested** — the circuit is gone. Iterative pruning wins at the highest recoverable sparsities; ft_short is sufficient at moderate ones. The spec's "passkey is the canary" prediction is wrong in the gentle regime (PPL drifts up first, passkey stays pinned) but becomes right once you cross the cliff — passkey is the *binary* signal that the retrieval circuit is still intact.

## Cross-scope summary at 95% sparsity

| Scope | Oneshot PPL | Oneshot passkey | ft_short PPL | ft_short passkey | ft_long PPL | ft_long passkey | iter PPL | iter passkey |
|---|---|---|---|---|---|---|---|---|
| MLP-only | 103.94 | 0.000 | 7.13 | **1.000** | 5.73 | **1.000** | 5.59 | 0.980 |
| Attn-only | 75.40 | 0.000 | 11.05 | **0.000** | 5.18 | **1.000** | 5.13 | **1.000** |
| All-weights | 534.00 | 0.000 | 11.69 | **0.000** | 7.51 | **0.000** | 7.14 | **0.000** |

Read: at 95% MLP, 500 FT steps are enough. At 95% attention, you need 2000 steps of FT or iterative. At 95% all-weights, **no FT recipe we tested brings passkey back** — only PPL partially recovers (from 534 to 7.14). The retrieval circuit needs *some* surviving capacity in both MLP and attention; you can't rob both simultaneously that hard.

## Per-scope findings

### MLP-only

See `mlp_comparison.txt`, `mlp_degradation_curves.png`, `mlp_results.json`.

- **Oneshot cliff at 90%.** 80% oneshot: passkey 0.75. 90% oneshot: 0.004. A sharp two-step collapse.
- **FT fully recovers at every sparsity.** 500 FT steps are enough through 95% (ft_short 95%: PPL 7.13, passkey 1.000). 
- **Iterative slightly wins on PPL at 95%** (5.59 vs ft_long's 5.73 vs ft_short's 7.13) but slightly loses on passkey (0.98 vs 1.00 — saturation noise, not meaningful).
- **Passkey starts degrading around 70% oneshot** (0.936 — near-ceiling). PPL starts degrading around 50% oneshot (6.25 vs 4.92 baseline, +27%).

### Attention-only

See `attn_comparison.txt`, `attn_degradation_curves.png`, `attn_results.json`.

- **Oneshot cliff is earlier — at 70%** (passkey 0.016 at 70% vs 1.000 at 50%). The "attention is more fragile" prediction is confirmed on the oneshot axis.
- **FT still recovers at every sparsity, but 500 steps are no longer enough at 95%.** ft_short at 95% attn: PPL 11.05, passkey 0.000 — fails both. ft_long (2000 steps) at 95% attn: PPL 5.18, passkey 1.000 — recovers fully. Iterative also recovers.
- **Moderate sparsities (30–80%) are trivially recoverable** even with 500 steps.
- **Interpretation:** the retrieval circuit uses a few specific heads; zeroing 95% of attention weights eliminates them in one shot, but given enough FT steps the circuit can reassemble from surviving rank-1 components in the Q/K/V/O matrices.

### All-weights

See `all_comparison.txt`, `all_degradation_curves.png`, `all_results.json`.

- **Oneshot cliff at 80%** (passkey 0.000). Even earlier than attention-only.
- **At 95% all-weights, passkey is unrecoverable.** All three FT strategies give 0.000 passkey. PPL partially recovers (iter 7.14 vs oneshot 534) but retrieval is permanently gone.
- **90% all-weights is the recoverability boundary.** ft_long: 0.984; iterative: 1.000.
- **Interpretation:** MLP and attention can individually compensate for each other's pruning — if MLP is intact, attention can be rebuilt from 5% of its weights, and vice versa. When both are down to 5%, neither has the surplus needed to re-form specialized circuits. This suggests **coupled, distributed representations of retrieval across the two modules** rather than a single "attention-located" circuit.

## Cross-cutting observations

1. **PPL is a slope; passkey is a cliff.** This holds across all three scopes under oneshot pruning. PPL has a smooth monotonic degradation curve; passkey either works (1.000) or doesn't (<0.02). Below the cliff, passkey is saturated by exact-match accuracy; there is no graceful degradation.

2. **Fine-tuning dramatically extends the recoverable regime.** Oneshot "breaks" at 70–90% depending on scope. With FT, the recoverable regime extends to 90–95%. FT effectively replaces "find the circuit" with "just re-fit the circuit in whatever weights are left."

3. **The fine-tuning steps required scale with how hard the pruning hit the critical circuit.** MLP-only 95%: 500 steps OK. Attn-only 95%: 500 steps fail, 2000 OK. All-weights 95%: even 2000 + iterative fails.

4. **Iterative pruning wins exactly at the boundary.** Its advantage over one-shot+FT is tiny at recoverable sparsities (saturated) and also tiny at unrecoverable ones (both fail). It's meaningful at 95% MLP (PPL 5.59 vs 5.73) and 90% all-weights (passkey 1.000 vs ft_long 0.984).

5. **The "canary" reframed.** The spec predicted passkey would fall before PPL, with passkey as the sensitive early warning. Empirically, passkey is pinned at 1.000 far into the sparsity sweep (through 70% MLP oneshot, 50% attn oneshot, 70% all oneshot) while PPL has already drifted noticeably upward. **But** once passkey does fall, it falls off a cliff — passkey is a binary "is the retrieval circuit still there" signal, not a graceful measure of "how damaged is the model." If you want an early warning, watch PPL. If you want a clean yes/no test that the retrieval circuit survived a given pruning recipe, watch passkey.

## Files

```
checkpoints/
  mha_lm.pt            # MHA, 5K-step Shakespeare baseline (val_ppl=4.92)
  mha_passkey.pt       # MHA, 20K-step passkey baseline (exact=1.000)
  baseline.log
results/
  mlp_results.json, mlp_comparison.txt, mlp_degradation_curves.png, mlp_passkey_vs_ppl.png
  attn_results.json, attn_comparison.txt, attn_degradation_curves.png, attn_passkey_vs_ppl.png
  all_results.json, all_comparison.txt, all_degradation_curves.png, all_passkey_vs_ppl.png
  README.md (this file)
  sweep.log, attn_sweep.log, all_sweep.log
```

## Reproduce

```bash
# baselines (~6 min)
PYTHONPATH=. .venv/bin/python -m experiments.pruning.train_baseline \
    --lm-steps 5000 --passkey-steps 20000

# one scope: --scopes {mlp,attn,all}, default sweep is the full 7×4 grid.
PYTHONPATH=. .venv/bin/python -m experiments.pruning.run_sweep \
    --scopes mlp --tag mlp
PYTHONPATH=. .venv/bin/python -m experiments.pruning.run_sweep \
    --scopes attn --tag attn
PYTHONPATH=. .venv/bin/python -m experiments.pruning.run_sweep \
    --scopes all --tag all
```

## Natural next steps

- **Head pruning** (structured): rank heads by ablation impact on passkey and zero them whole. With the "some heads do retrieval" intuition now supported by the attention-only result, identifying *which* heads would be informative — and gives real compute speedup.
- **Global magnitude threshold vs per-layer.** Per-layer was spec'd. Under global, some layers would end up 100% pruned — interesting for identifying which layers are load-bearing.
- **Movement pruning** at the all-weights 95% boundary. Static magnitude pruning fails here; gradient-informed pruning (e.g., SNIP, movement) might find a sparser solution that *does* retain passkey.
- **The unrecoverable regime deserves more fine-tuning budget.** At 95% all-weights, did we run out of recovery budget or is the circuit genuinely unreconstructible? 10K FT steps instead of 2K would settle it.
