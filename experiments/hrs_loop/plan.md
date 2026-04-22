# HRS-Loop: Testing the MPAR-Bias Hypothesis for Looped Transformers

## Hypothesis

The loop-to-loop signal in a looped transformer is functionally an MPAR (mean-pooled abstract representation) acting as a low-rank bias, not a full refined hidden state. A rank-128 MPAR + small per-loop LoRA should match a full-rank refinement baseline.

## Variants

All share Prelude (2 blocks) + Recurrent (1 block, reused T times) + Coda (2 blocks) at d=256, 4 heads.

- **A — Refinement baseline (LTI)**
  `h_{t+1} = A · h_t + B · e + RecurrentBlock(h_t)` with `A = diag(-exp(log_A) · exp(log_dt))`, discretized. Learnable `log_A ∈ R^d`, scalar `log_dt`, `B = Linear(d, d)`.

- **B — MPAR-bias (the hypothesis)**
  `h_t = e + MPAR_project(m_t)`; `h_out = RecurrentBlock(h_t) + LoRA_t(h_t)`; `m_{t+1} = mean_pool(h_out)`.
  `MPAR_project: rank_m → d`, `mean_pool: d → rank_m` then mean over sequence. `rank_m = 128`. Per-loop LoRA of rank 16 on the recurrent block's attention output projection.

- **C — MPAR-bias with no per-loop differentiation**
  Variant B minus the `LoRA_t` branch (all loops share the same recurrent forward).

## Tests

1. **Loss parity**: Train each variant on wikitext (see dataset note). Report final val PPL. Success: `PPL(B) − PPL(A) < 0.3`, `PPL(C) − PPL(B) > 0.5`.
2. **Rank floor**: SVD-truncate trained B's `MPAR_project` to ranks {256, 128, 64, 32, 16, 8, 4}. Report val PPL. Success: flat 256→128, degrades below 64.
3. **MPAR cosine**: For held-out batch, pairwise cosine of `m_1..m_4` averaged over batch vs random-batch baseline. Success: cos(m_t, m_{t+1}) ≥ 0.5; random < 0.05.
4. **Depth extrapolation**: Train at T=4, evaluate at T ∈ {2,4,6,8,12}. Success: B's T=8 PPL within ±0.4 of T=4 PPL.
5. **Order invariance (inference-only)**: Mode 1 canonical, Mode 2 permuted MPAR accumulation at final loop, Mode 3 independent-loop MPARs averaged. Report per-token KL, top-1 agreement, PPL, and L2 norm of the biased MPAR per mode. Calibrate against B-vs-A KL on same batch.

## Pragmatic adaptations from spec

- **File layout**: `experiments/hrs_loop/` subdir rather than `models/…`. HRS top-level `model.py` is the Bonsignore-kernel HRSTransformer, unsuitable for clean ablation. Reuse `experiments/diagonal_attention/`'s TinyTransformer as the block/FFN foundation and `experiments/identity_ae/lora_wrapper.py`'s LoRALayer for LoRA.
- **Dataset**: Try wikitext-2 via existing `load_wikitext`; fall back to char-level Tiny Shakespeare if download fails in the sandbox. All three variants use the same dataset.
- **Scale**: d=256, 5 blocks → ~4M transformer params + embeddings. With GPT-2 BPE embeddings ~17M, with char vocab ~1M. Within range of spec's 10M target.
- **Branch**: `hrs-loop`.

## Deliverables

- Three trained checkpoints in `checkpoints/`.
- Analysis scripts under `experiments/hrs_loop/analysis/`.
- Plots in `results/`.
- `results/hrs_loop_report.md` with one section per test, a single summary table at the end.
