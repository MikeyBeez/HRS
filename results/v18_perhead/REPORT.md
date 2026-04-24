# Task 9 — Attention-Pooled Per-Head Engrams

**Hypothesis tested.** V18's single-vector engram broadcast across all 32
buffer slots (every head sees the same content) is the root cause of the
0/138 cleaned-recall floor observed in tasks 6–8. Giving each head its own
engram slot, populated by attention-pooling the source document with that
head's query, should let different heads specialize to different content
and unlock per-position token routing.

**Outcome (initial).** Weak positive. Recall rose to **3/138 (2.2%)** at
step 5000 — the first time any variant has exceeded 1/138 — with **MAUVE
0.986**, the best generation-quality score of any variant so far. But the
intermediate checkpoints oscillated 0–1/138, so the 3/138 result could be
a single-eval artifact. Gates drifted *down* slightly (L1: 0.272 → 0.240,
L3: 0.278 → 0.249) rather than opening, suggesting the model is not
preferring per-head content over V18's baseline behavior.

**Outcome (after diagnostics — see `DIAGNOSTICS.md`).** Null. The 3/138
did not replicate: three seed-varied re-runs of the step-5000 eval all
returned 0/138. Per-head-disabled control also returned 0/138 (identical
to enabled). Attention entropy at all three layers is log(32) = 3.466 —
exactly uniform over all 32 slots. The architectural affordance exists;
the trained model is not using it. The gate drift and the MAUVE
improvement are real but are not attributable to the per-head mechanism.

## Config

- Base: `results/v18_cross_attn/best.pt` (V18 pretrained, val_ppl 23.26;
  *not* RAFT-2000, because the architectural change makes RAFT-trained
  weights for the old behavior non-transferable)
- Architecture: slots 0–15 = shared engram (V18 broadcast); slots 16–31 =
  per-head pooled content, with slot `h` carrying head-h's specialized
  `E_k_h`, `E_v_h` at head-h's head slice and the shared engram's K/V
  at all other heads' slices (so non-matching heads treat it as another
  shared slot).
- Per-head pooling: for each CA block at layers 1/3/5, source doc's
  layer-4 hidden states are projected through that block's current
  `W_k, W_v`; pooling weights are `softmax(q_mean_h @ K_src_h^T / sqrt(64))`,
  where `q_mean_h` is the current sequence's mean query for head h.
  Source hidden states are pre-computed once using V18's projections
  (`engram_hiddens_v18_layer4.pt`, 912 MB); the projections used for
  pooling are the *current* ones and update during training.
- Monkey-patch: `EngramCrossAttention.forward` is patched at module load
  to check `self._perhead_src_h`. If None → V18 behavior. If set → per-head
  pooling + concat path.
- Training recipe (matches task 7 except):
  - Base is V18 (not RAFT-2000)
  - No grounding aux loss — isolate the architectural change
  - Everything else identical: 5000 steps, B=4, seq_len=512, backbone
    frozen, gate LR 100× base (1e-3), cross-attn+cat 1e-5, cosine decay,
    warmup 500, 50/25/25 retrieve/baseline/wrong mix
- Eval: per-head aware (`eval_niah_perhead`). For each of the 25
  expanded-benchmark needles, compute hiddens of fact+20 distractors,
  inject per-head state, generate 100 tokens from query, score substring
  recall on cleaned answer tokens. Shared engram is mean-pool of that
  same context. *Note:* this is simpler than task-4's `v1_baseline`
  (single pass per needle instead of the variant harness); the absolute
  recall numbers may differ slightly from the task-4 eval protocol, but
  the comparison to tasks 6–8's floor is still meaningful because they
  all used task-4's `eval_niah_v2` runner.
- Output: `results/v18_perhead/`
- Total wall clock: 58 min.

## Trajectory

| step | recall   | MAUVE   | gates L1 / L3 / L5   | ce    | cat   |
|:----:|:--------:|:-------:|:--------------------:|:-----:|:-----:|
| 0    | 1/138    | 0.9249  | 0.272 / 0.278 / 0.327 | —     | —     |
| 1000 | 1/138    | 0.9151  | 0.237 / 0.264 / 0.344 | 3.247 | 0.451 |
| 2000 | 0/138    | 0.9533  | 0.234 / 0.254 / 0.340 | 3.072 | 2.281 |
| 3000 | 1/138    | 0.9535  | 0.238 / 0.248 / 0.342 | 3.249 | 0.600 |
| 4000 | 1/138    | 0.9779  | 0.240 / 0.247 / 0.347 | 3.583 | 0.410 |
| 5000 | **3/138**| **0.986** | 0.240 / 0.249 / 0.346 | 3.197 | 1.225 |

The pre-training MAUVE of 0.9249 at step 0 is **without** per-head
injection — it's measuring V18's MAUVE on its normal inference path
(to verify load succeeded). The in-training MAUVE numbers are also
measured without per-head injection, so the MAUVE trajectory is just
tracking V18's fluency under the normal inference path as the cross-attn
weights are updated.

## Comparison to tasks 6, 7, 8

| variant                         | final recall | final MAUVE | gates L1/L3/L5     | notes                                |
|:--------------------------------|:------------:|:-----------:|:------------------:|:-------------------------------------|
| Task 6 (contrastive w=0.1)      | 0/138        | 0.947       | 0.45 / 0.41 / 0.46 | gates opened ~70%, diff +0.33        |
| Task 7 (contrastive w=1.0)      | 0/138        | 0.919       | 2.03 / 1.99 / 1.32 | gates saturated softplus, diff +0.70 |
| Task 8 (REINFORCE w=0.3)        | 1/138        | 0.960       | 0.29 / 0.27 / 0.35 | reward never grew                    |
| **Task 9 (per-head)**           | **3/138**    | **0.986**   | 0.24 / 0.25 / 0.35 | gates drifted down; MAUVE best-ever  |

Task 9 is the only variant that cracked above 1/138. It is also the
only variant where MAUVE *improved* over the course of training rather
than paying a cost. But the improvement in recall is small and noisy:

- 3/138 at step 5000 represents a single needle contributing its 3
  cleaned tokens in a run where other needles contributed 0. The final
  needle's 3 tokens account for the entire signal.
- The trajectory 1→0→1→1→3 across checkpoints is consistent with each
  eval being a single stochastic generation run (temperature 0.9,
  top-k 50). A single eval without seed-averaging is noise-prone at
  this magnitude.

## Reading the gate drift

Gates closed slightly and stabilized: L1 −12%, L3 −10%, L5 +6%. In
tasks 6/7 gates *opened* under grounding-loss pressure; here with only
LM CE + cat loss pushing them, they drift downward as the model
accommodates the new per-head content by reducing cross-attn gain
overall. This is a mild form of the "gates close on per-head half"
failure mode — but because the gates are *per-block*, not per-slot-half,
the model can't selectively close only the per-head slots. It's closing
the whole cross-attn path a little, which implies the per-head content
isn't adding enough signal to justify keeping the gates as open as V18
had them.

The corollary: we don't know whether the 3/138 at step 5000 comes from
the per-head slots doing useful work, or from the cross-attn path
simply being quieter (closer to baseline no-cross-attn behavior) at
eval time. A useful diagnostic would be to re-run the step-5000 eval
with per-head state *disabled* and compare — if recall stays at 3/138,
the per-head slots aren't the cause; if it drops back to 0–1/138, the
per-head slots are real.

## Decision

This is an ambiguous result. Per the task-9 spec:

- "Cleaned recall ≥ 5/138 consistently" → **no** (we got 3/138 once)
- "Cleaned recall unchanged, gates close on per-head half" → **partial**
  (recall not unchanged, gates closing overall not just on per-head half)
- "Cleaned recall unchanged, gates open on per-head half" → **no**
- "MAUVE crashes" → **no** (MAUVE actually improved)

The cleanest match is: *weak positive that doesn't cross the threshold*.
The architectural change didn't break anything (MAUVE best-ever) and
shows a small improvement (3/138 vs the previous 0–1/138 floor), but
the signal is ambiguous enough that the right next step is either:

1. **Replicate**: re-run step-5000 eval 3 times with different seeds
   to measure whether 3/138 is stable or noise. Cheap (~5 min).
2. **Diagnostic**: re-run step-5000 eval with per-head disabled to
   test whether the slots caused the gain. Cheap (~5 min).
3. **Layer grounding on top (task 10 per spec)**: if per-head content
   is being used but doesn't carry enough grounding on its own, the
   grounding aux loss from tasks 6/7/8 may push it past the threshold.
4. **Different architecture**: per-head content may be too redundant
   with shared content. Next move: learned per-head projections (own
   W_k, W_v per head for the engram path, separate from the main
   attention projections) or slot-based content buffer.

My read: run the two cheap diagnostics (1) and (2) before committing
to a 3-hour task 10 or a bigger architectural change. If step-5000
recall replicates at 3/138 and is sensitive to per-head disablement,
task 10 is justified. If not, the per-head architecture isn't carrying
real signal and we move to learned projections.

## Artifacts

- Checkpoints: `checkpoint_{1000,2000,3000,4000,5000}.pt`
- Log: `training_log.json`
- Training stdout: `/tmp/raft_perhead_full.log`
- Run config: `run_config.json`
- Code: `experiments/hrs_loop/raft_perhead_train.py`,
  `experiments/hrs_loop/build_perhead_cache.py`
- Cache: `engram_store_data/engram_hiddens_v18_layer4.pt` (912 MB)

## Notes on implementation

- Monkey-patch approach (modify `EngramCrossAttention.forward` at
  module load in the training script) avoided touching `engram.py`,
  preserving V18 eval paths for other scripts.
- Pre-computed layer-4 hidden states for 870 store entries fit
  comfortably in GPU memory as fp16 (912 MB on a 16 GB card).
- Per-head pooling re-projects source hiddens with the **current**
  `W_k, W_v` at each step, so training can update the projections
  and pooling tracks them — as specified.
- Training rate: ~2 it/s with per-head active, vs task 6/7's 2.5 it/s.
  Per-head pooling adds modest overhead (one W_k, W_v projection of a
  ~500-token source doc per block per step, then einsum pooling).
- No grounding aux loss in this run — per spec, the architectural
  change is isolated. Task 10 would layer grounding on top.
