# Task 10 — Learned Per-Head Projections for Engram Path

**Hypothesis tested.** Task 9 failed because all heads saw per-head slots
through the same self-attention `W_k`/`W_v` and had no inductive bias
to prefer slot 16+h over any other slot. Adding private
`W_k_engram_ph[h]`, `W_v_engram_ph[h]` (each `(H, head_dim, head_dim)`)
zero-initialized should give head h a dedicated pathway into slot 16+h
that other heads' queries can't align with — breaking the uniform-
attention symmetry.

**Outcome.** Null. The architectural affordance produced gradient and
the projection norms grew monotonically (`L5 v-norm: 0 → 0.22` across
5000 steps). But the gradient was too weak to reshape the attention
distribution: **entropy stayed at log(32) = 3.466 at every checkpoint**,
and L5's diagonal attention actually *decreased* slightly over training
(0.0328 → 0.0318) — effectively no specialization.

## Trajectory

| step | recall | MAUVE  | gates L1/L3/L5      | ph_v-norm L1/L3/L5  | L5 diag | L5 entropy |
|:----:|:------:|:------:|:-------------------:|:-------------------:|:-------:|:----------:|
| 0    | 1/138  | 0.9249 | 0.272/0.278/0.327   | 0.000/0.000/0.000   | 0.0328  | 3.466      |
| 1000 | 0/138  | 0.9645 | 0.242/0.266/0.341   | 0.065/0.064/0.081   | 0.0328  | 3.466      |
| 2000 | 0/138  | 0.9491 | 0.238/0.253/0.335   | 0.100/0.099/0.148   | 0.0324  | 3.466      |
| 3000 | 1/138  | 0.9615 | 0.240/0.246/0.336   | 0.117/0.115/0.193   | 0.0321  | 3.466      |
| 4000 | 0/138  | 0.9482 | 0.243/0.247/0.340   | 0.125/0.121/0.215   | 0.0319  | 3.466      |
| 5000 | 0/138  | 0.9650 | 0.243/0.248/0.338   | 0.127/0.123/0.223   | 0.0318  | 3.466      |

Uniform baseline across 32 slots: `1/32 = 0.03125`, entropy `log(32) = 3.466`.

## Reading the result against the spec's decision tree

The spec's four-way split:

- **"Cleaned recall ≥ 5/138 with per-head ON and < 3/138 with per-head OFF"** — no.
- **"Projections grow, entropy drops, recall still at floor"** — half-true:
  projections grew, but entropy did not drop. Recall stayed at floor.
- **"Projections don't grow"** — technically no; norms reached 0.22 at L5.
- **"MAUVE crashes"** — no, MAUVE stayed 0.94–0.97.

The cleanest match is **"projections grow but LM CE alone isn't strong
enough to shift attention."** Architecture is well-formed and receiving
gradient, but the training objective doesn't care about which slot head
h attends to — LM loss on WT-103 continuations doesn't require the
model to route through any particular engram slot. So the projections
drift to small but non-negligible norms on residual gradient, while
attention entropy stays pinned at uniform.

## Comparison across tasks 6–10

| variant             | final recall | final MAUVE | gates L1/L3/L5       | specialization diag (L5) | notes                               |
|:--------------------|:------------:|:-----------:|:--------------------:|:------------------------:|:------------------------------------|
| Task 6 (contrast 0.1) | 0/138     | 0.947       | 0.45/0.41/0.46       | —                        | gates opened, diff +0.33            |
| Task 7 (contrast 1.0) | 0/138     | 0.919       | 2.03/1.99/1.32       | —                        | gates saturated, diff +0.70         |
| Task 8 (REINFORCE)    | 1/138     | 0.960       | 0.29/0.27/0.35       | —                        | reward never grew                   |
| Task 9 (perhead, pool only) | 3/138→0 (diag) | 0.986   | 0.24/0.25/0.35       | 0.035 (uniform 0.031)    | single-eval noise; attn uniform     |
| **Task 10 (learned per-head proj)** | **0/138** | **0.965** | 0.24/0.25/0.34 | 0.032 (uniform 0.031) | projections grew; attn still uniform |

Five consecutive experiments, zero consistent grounding signal. The
shared assumption — that the cross-attn pathway is the bottleneck and
training pressure can open it — is showing strain.

## Decision

The user pivoted to **task 11**: a read-only diagnostic testing whether
OOD content injected directly into the residual stream survives through
the MLPs to the output logits. If the answer is "no, MLPs wash it out,"
then every task 6–10 experiment was attacking the wrong layer of the
pipeline and we need to stop iterating on cross-attention fixes. Task 11
is running next.

## Artifacts

- Checkpoints: `checkpoint_{1000,2000,3000,4000,5000}.pt`
- Log: `training_log.json`
- Training stdout: `/tmp/raft_perhead_proj_full.log`
- Code: `experiments/hrs_loop/raft_perhead_proj_train.py`

Total wall clock: 67 min.

## Note on implementation

Monkey-patch (`_perhead_proj_forward`) extends task-9's patched forward
with per-head `W_k_ph`, `W_v_ph` applied via einsum to the pooled content
before diagonal placement. Projections are attached as `nn.Parameter`
on each cross_attn module, zero-initialized. They go into the
`ca_other_params` optimizer group at LR 1e-5. Gradients verified at
step 100 (grad L2 non-zero for both `W_k_ph` and `W_v_ph`).
