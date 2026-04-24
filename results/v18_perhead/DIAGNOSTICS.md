# Task 9 Diagnostics — Step-5000 Checkpoint

The 3/138 recall at step 5000 (reported in `REPORT.md`) could have been
either the first real grounding signal or a single-eval lucky roll. Three
cheap diagnostics to disambiguate, all run on `checkpoint_5000.pt`:

1. **Seed variance** — re-run the per-head-enabled eval at seeds 42, 43, 44
2. **Per-head disabled at inference** — same trained checkpoint, per-head
   state never set; monkey-patched forward falls through to V18 behavior
3. **Attention entropy** — during per-head-enabled eval, capture attention
   weights at each cross-attn block; measure mean attention head h places
   on its "own" slot 16+h vs other per-head slots vs shared slots

Code: `experiments/hrs_loop/diagnostics_perhead.py`. Full output:
`results/v18_perhead/diagnostics.json`. Wall clock: ~16 min (7 evals).

## Results

### 1. Seed variance (per-head ENABLED)

| seed | hits | recall |
|:----:|:----:|:------:|
| 42   | 0/138 | 0.000 |
| 43   | 0/138 | 0.000 |
| 44   | 0/138 | 0.000 |

The 3/138 from the training run's step-5000 eval was a lucky roll. Three
different seeds all come back at the floor.

### 2. Per-head DISABLED at inference

| seed | hits | recall |
|:----:|:----:|:------:|
| 42   | 0/138 | 0.000 |
| 43   | 0/138 | 0.000 |
| 44   | 0/138 | 0.000 |

Identical to per-head enabled. The architecture is not contributing at
inference — disabling per-head state makes no difference.

### 3. Attention entropy (per-head ENABLED, averaged over 25 needle contexts)

| layer | mean attn on diag (slot 16+h) | mean attn off-diag per-head | mean attn shared | entropy |
|:-----:|:-----------------------------:|:---------------------------:|:----------------:|:-------:|
| L1    | 0.0317                        | 0.0312                      | 0.0312           | 3.466   |
| L3    | 0.0328                        | 0.0312                      | 0.0312           | 3.466   |
| L5    | 0.0350                        | 0.0311                      | 0.0311           | 3.466   |

Uniform-distribution baseline over 32 slots: 1/32 = 0.03125; entropy = log(32) = 3.466.

All three layers sit essentially at the uniform baseline. L5 shows the
largest deviation — head h attends to its own slot 16+h about 12% more
often than to other per-head slots (0.0350 vs 0.0311). L3 shows ~5%
deviation; L1 essentially zero. The entropy of each head's attention
distribution is indistinguishable from log(32) at all three layers.

Interpretation: the heads are **not** specializing to their designated
slots. The architectural affordance exists; the trained model is not
using it.

## Conclusion

Task 9 is a null result. Both the behavioral diagnostic (recall matches
at 0/138 with per-head ON vs OFF) and the mechanistic diagnostic
(attention uniform over all 32 slots) agree: the per-head slots are
inert. The reported 3/138 at training-eval step 5000 was single-eval
noise.

The one interesting quantitative finding is that MAUVE improved during
task 9 training (0.9249 → 0.9860). This improvement is independent of
the per-head mechanism — diagnostic 2 shows that turning per-head OFF
at inference changes nothing. The MAUVE gain is presumably from general
cross-attn weight refinement under the backbone-frozen training recipe
acting on a cleaner starting point (V18 rather than RAFT-2000), not
from grounding-related specialization.

## What this means for task 10

The task-9 spec anticipated this exact outcome:

> **Nothing changes. Per-head content is semantically redundant with
> the shared engram. Response: test with per-head content only (no
> shared engram) to force the model to use per-head routing. If that
> also fails, the broadcast-vs-specialized distinction isn't where
> the action is and we move to different architectural ideas (learned
> per-head projections for the engram path specifically, rather than
> reusing the attention projections).**

We got "nothing changes." The decision-tree-prescribed next move is
either:

- **Variant A**: Ablate shared engram entirely — force the model to
  route through per-head slots by removing the 16 shared slots. If
  recall stays at 0/138, the architecture can't carry grounding.
  If it crashes MAUVE, the shared path was doing all the work.
- **Variant B**: Learned per-head projections — separate `W_k_ph`,
  `W_v_ph` matrices for the engram path, not shared with the
  self-attention projections. Gives the model a dedicated pathway
  that isn't also trying to do normal attention. More invasive,
  more parameters.

Of these, A is cheap (small training-script edit, ~1 hour run). B is
a real architectural change with its own risks. A first, then decide.

Independent of A/B: **task 10 should NOT layer grounding aux loss on
top of task 9's architecture.** Diagnostic 2 proves the per-head slots
are inert; no amount of training pressure on an inert pathway will
produce grounding. Tasks 6/7/8 already explored what training pressure
can do on the V18 shared-engram architecture — it can't produce
per-position content routing.
