# Task 8 — Sampled-Token REINFORCE Grounding

**Hypothesis tested.** Replace task 6/7's mean-logp contrastive grounding
with an on-policy sampled-token REINFORCE objective: at retrieve-active
items, run a K=8-step rollout from a random prefix position, reward each
sampled token that lands in the source engram's distinctive-content set,
and push the policy with REINFORCE. The argmax-flip threshold that the
mean-logp loss couldn't cross should be crossable by a direct per-token
objective.

**Outcome.** Failed on the earliest-stage branch of the task spec:
**reward never rose above chance.** Across 5000 steps, mean reward
stayed at 0.038–0.049 — roughly the chance rate of a top-50 sample
landing in a ~100-token distinctive set. No learning signal reached the
gates or cross-attn. Cleaned NIAH recall stayed at the 0–1/138 floor;
MAUVE held above 0.88 at all evals.

## Config

- Base: `results/v18_raft/checkpoint_2000.pt`
- Change vs task 7: contrastive `compute_grounding_loss` replaced with
  `rollout_reinforce_backward`. K=8 sampled tokens per active item, top-k
  50, temperature 1.0, prefix position P sampled uniformly in
  `[32, seq_len-16-K]`. Binary reward on distinctive-D+ hit
  (len ≥ 4 OR uppercase OR digit; minus prefix-appearing tokens).
  Running baseline (init 0.05, window 100). `total = LM_CE + 0.1*cat_loss + 0.3*grounding_loss`.
- Per-step backward on the rollout loss contribution (weight scaled by
  1/(N_active*K)) to cap peak VRAM at a single rollout forward pass —
  the batched K-pass graph OOM'd on the 16 GB RTX 5070 Ti.
- Everything else matches task 7.
- Output: `results/v18_raft_sampled/`
- Total wall clock: 88 min.

## Trajectory

| step | recall | MAUVE  | gates L1/L3/L5      | reward_ma100 | baseline |
|:----:|:------:|:------:|:-------------------:|:------------:|:--------:|
| 0    | 0/138  | —      | 0.272 / 0.277 / 0.327 | —          | 0.050    |
| 1000 | 1/138  | 0.9722 | 0.287 / 0.290 / 0.338 | 0.0439     | 0.0439   |
| 2000 | 0/138  | 0.9352 | 0.293 / 0.283 / 0.345 | 0.0388     | 0.0388   |
| 3000 | 1/138  | 0.8793 | 0.291 / 0.278 / 0.350 | 0.0493     | 0.0493   |
| 4000 | 1/138  | 0.9766 | 0.289 / 0.273 / 0.355 | 0.0472     | 0.0472   |
| 5000 | 1/138  | 0.9599 | 0.290 / 0.272 / 0.354 | 0.0428     | 0.0428   |

NIAH retrieval acc@1 stayed at 0.60 and mean rank at 2.20–2.28 throughout.
The one-hit flickering (0/138 ↔ 1/138) is noise: one specific needle
gets one token correct at some temperatures, not at others.

## Comparison with tasks 6, 7, 8 (gates, diff/reward, recall at step 5000)

| variant              | gates L1/L3/L5      | signal metric          | recall (step 5000) | MAUVE (step 5000) |
|:---------------------|:-------------------:|:----------------------:|:------------------:|:-----------------:|
| Task 6 (contrastive, w=0.1)  | 0.45 / 0.41 / 0.46 | diff = +0.33 | 0/138 | 0.9465 |
| Task 7 (contrastive, w=1.0)  | 2.03 / 1.99 / 1.32 | diff = +0.70 | 0/138 | 0.9185 |
| Task 8 (REINFORCE, w=0.3)    | 0.29 / 0.27 / 0.35 | reward_ma = 0.043 | 1/138 | 0.9599 |

Task 8's gates are indistinguishable from the pre-training checkpoint
(0.272/0.277/0.327). The REINFORCE loss, despite being weighted 0.3,
produced almost no gate movement — in some layers the gate drifted
*downward* (L3: 0.290 → 0.272).

## Reading the failure

The spec's first diagnostic branch applies cleanly:

> **Reward stays at baseline.** Model never samples D+ tokens at a rate
> above chance. This either means the engram isn't carrying enough
> signal about specific tokens (architecture problem) or the rollout
> positions are too far from where source content would naturally
> appear (setup problem).

Reward moving average was 0.044 at step 1000 and 0.043 at step 5000.
Across 5000 steps, with the REINFORCE gradient actively pushing for
more D+ tokens, the rate didn't budge. This is a statistically-strong
null.

The setup-problem explanation is weak: a top-50 sample from a fluent
LM distribution lands on a distinctive token from the active source
roughly 4% of the time even without any steering, which matches what
we see. The engram is in the buffer, the gates are open at
softplus(≈0.3) ≈ 0.85, cross-attn is operating — but the effect on
which specific tokens win the argmax at specific positions is
vanishingly small.

The architecture-problem explanation is strong. In V18, a single
engram vector gets broadcast into 32 identical slots (`.expand(32,-1)`)
and cross-attended into every token position. That's a per-sequence
bias, not a per-position content channel. The mean-logp experiments
in tasks 6/7 could show the bias raising D+ logits in aggregate —
because that is what a bias does — but no bias can cause one specific
content token to win at one specific position without some other
mechanism routing tokens to positions.

REINFORCE needs that per-position mechanism to exist in order to
discover it. It doesn't, so there's nothing to discover, and reward
stays at the unconditional sampling prior.

## Decision

Per the task-8 spec:

> **Reward doesn't grow:** Architecture can't carry the grounding
> signal. Move to per-head engrams or attention-pooled injection from
> the backlog.

This run gives a definitive "no" to the question "can more clever
training get grounding out of V18 cross-attention?" Tasks 6, 7, and 8
tried three different objectives (soft contrastive, hard contrastive,
on-policy policy gradient) and all three hit the same floor. The next
move is architectural: replace the single-vector engram with a
structure that has per-position routing capacity — per-head engrams,
attention-pooled injection over the engram store, or a slot-based
buffer with content in the slots rather than 32 copies of one vector.

## Artifacts

- Checkpoints: `checkpoint_{1000,2000,3000,4000,5000}.pt`
- Log: `training_log.json`
- Training stdout: `/tmp/raft_sampled_full.log`
- Run config: `run_config.json`
- Code: `experiments/hrs_loop/raft_sampled_train.py`

## Notes on implementation

- The batched (N, K, ...) REINFORCE graph OOM'd at 16 GB because the
  K=8 rollout forwards all held PEER-expert activations for backward.
  Fix: per-step, per-item backward with the loss contribution scaled
  to preserve the `-mean(A * logp)` expectation. Peak memory then
  equals one rollout forward plus accumulated parameter-gradient
  tensors. No further optimization needed.
- Rollout added ~40% compute overhead: main run at 1.07 it/s vs task
  7's 1.96 it/s at step 5000. Step-5000 wall-clock was 5293 s vs task
  7's 3561 s.
- Negative `grounding_loss` values in the log are expected: the
  REINFORCE loss is `-(A * logp).mean()`. When a batch's average
  advantage is negative (sampled tokens had sub-baseline reward), the
  sign flips. Not an error.
