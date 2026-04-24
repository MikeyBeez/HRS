# Diagonal Attention Ablation

Four attention variants compared in the **same** tiny transformer (4 layers, d_model=256, 4 heads, d_ff=1024, ctx=256). Only the score computation differs; V and O projections are standard multi-head in all variants.

| Variant | Score computation | Score params (all layers) |
|---|---|---|
| **MHA** | `softmax(QK^T / √d_head)` with per-head Q, K | 524,288 |
| **Bilinear** | `softmax(X W X^T / √d)`, shared across heads | 262,144 |
| **Diagonal** | `softmax((X ⊙ w) X^T / √d)`, shared d-vector | 1,024 |
| **Identity** | `softmax(X X^T / √d)`, no learned interaction | 0 |

## Tasks

- **LM**: char-level Tiny Shakespeare, 2,000 steps, batch 32, AdamW 3e-4 with cosine decay.
- **Passkey**: synthetic. Digit noise + `MARKER` + 4-digit passkey + digit noise + `QUERY` → produce the 4 digits. Marker position varies across `[0, ctx-2K-2]`. 10,000 steps. Loss masked to the 4 answer tokens per sample. Eval bucketed by marker position (5 buckets × 50 samples).

The passkey task replaces the existing TTT-based passkey infra in `experiments/identity_ae/phase10_passkey.py`, because that infra targets a pretrained HRSTransformer + test-time adaptation, which doesn't fit a from-scratch 4-layer ablation.

## Results

```
Variant      Val PPL  Passkey Exact  Passkey Digit  Score Params  Step (ms)  Peak Mem MB
----------------------------------------------------------------------------------------
mha             5.19          0.372          0.763       524,288      13.51          754
bilinear        5.05          0.000          0.102       262,144      11.57          623
diagonal       11.91          0.000          0.102         1,024      11.41          620
identity       11.91          0.000          0.102             0      10.87          588
```

(`0.10` on digit-accuracy is the random baseline — 1/10 digits.)

### Accuracy vs. marker position (MHA)

MHA is roughly flat (28–48%) across all five position buckets; no position-dependent collapse. See `results/passkey_accuracy.png`.

## Interpretation

**LM — bilinear beats MHA at half the score params.** Bilinear (5.05 PPL, 262K params) edges out MHA (5.19, 524K). Diagonal and identity both land at 11.91 PPL — identical to 2 decimals, because the learned diagonal `w` moves very little (means 0.80–0.99 per layer, stds ≤ 0.09) and essentially just rescales the identity-based scores. For smooth language statistics, content-similarity attention (X·X) with a learnable diagonal adds nothing on top of raw content similarity.

**Passkey — only MHA works. Bilinear collapses to random.** This is the decisive finding. Bilinear's full-rank d×d `W` encodes *a* bilinear form, but it produces one attention map **shared across all heads**. Exact retrieval ("attend to the token `k` positions after MARKER") seems to need multiple specialized attention patterns across heads — the per-head Q, K pairs in standard MHA. A single shared map can't simultaneously perform "find MARKER" and "offset-k lookup" for each of the four answer positions. Diagonal is even more restricted and also collapses. All three non-MHA variants sit at the random digit floor (10%).

### So is the diagonal hypothesis dead?

For **language modeling**, diagonal alone is not enough — it tracks identity. But **bilinear matches MHA on PPL at half the parameters**, which is a real win for that task class.

For **exact retrieval / induction**, diagonal fails and **so does full bilinear**. The shared-attention-map structure, not the d×d rank, is what breaks it. Natural next test: a per-head bilinear (`H` separate d_head×d_head score matrices, one per head) — that preserves per-head specialization while keeping the bilinear interaction pattern. If that works, it isolates "shared vs per-head" as the load-bearing axis.

## Caveats

- **Below 5–6K steps, MHA hasn't crossed the induction-head phase transition for this task.** An earlier 2,000-step run had MHA at 13% digit-acc — indistinguishable from the other variants. The first positive control only appears around step ~5,500. See `results/run_2k/` for the 2K-step archive that demonstrates this.
- **Diagonal w barely moved.** Means stayed at 0.80–0.99, stds ≤ 0.09. AdamW's weight decay (0.01) pulls it toward 0 while tiny gradients push it around 1 — the equilibrium is near identity. A larger init or a separate LR for `w` could change this.
- **Bilinear had no weight decay separation from other params.** A d×d dense matrix with wd=0.01 may be over-regularized; worth retuning.
- **Tiny model, tiny task.** 3.3M params total; char Shakespeare. Results are about structural inductive bias at this scale, not final capability.

## Files

```
config.py          # ModelConfig, TrainConfig, PasskeyConfig, VARIANTS
attention.py       # StandardMHA, FullBilinear, DiagonalBilinear, Identity
model.py           # TinyTransformer swaps in any variant
data.py            # Tiny Shakespeare + synthetic passkey generator
train.py           # LM training loop
eval_passkey.py    # Passkey training + position-bucketed eval
run_all.py         # Orchestrator + comparison table + plot
results/           # JSON per run, comparison.txt, summary.json, plot
results/run_2k/    # Archived 2K-step run (all variants at random on passkey)
```

## Reproduce

```bash
# full run (default: LM 2K, passkey 10K)
PYTHONPATH=. .venv/bin/python -m experiments.diagonal_attention.run_all \
    --steps 2000 --passkey-steps 10000

# one variant, one task
PYTHONPATH=. .venv/bin/python -m experiments.diagonal_attention.train \
    --variant bilinear --steps 2000
PYTHONPATH=. .venv/bin/python -m experiments.diagonal_attention.eval_passkey \
    --variant mha --steps 10000
```

## Stretch (not run)

- **Per-head bilinear**: `H` separate d_head×d_head score matrices. Isolates "shared vs per-head" from "diagonal vs full." This is the natural follow-up the passkey result points to.
- **Diagonal + low-rank**: `W = diag(w) + U V^T`, rank 4 or 8. Only meaningful for the LM task given diagonal's passkey result.
- Per-head diagonal (each head gets its own d_head vector).
