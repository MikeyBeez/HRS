# Stage A pre-flight — V22-as-shipped engram pathway: FAIL

**Verdict: stop, do not proceed to Stage B.** The cross-attention engram
pathway in `results/v22_learned_kernel/best.pt` (the canonical 17.07 V22
checkpoint, step 61000 of 63000, val_ppl in checkpoint = 17.073) contributes
only **+0.114 PPL** on the WT-103 validation set, well below the spec's
≥1.0 PPL pass criterion. Stage B is not warranted in this configuration —
the pathway isn't carrying enough load for dropout regularization to be
detectable.

This is the spec's anticipated *more-interesting-than-passing* outcome:
"if V23's engram pathway turns out to have a small ablation gap, that's
actually a more interesting finding than anything Stage B could produce."

## Numbers

| metric | value | note |
|--|--|--|
| ppl_on | 19.142 | 50 val batches; canonical 17.07 used different eval protocol (run_all_metrics, more batches, possibly fp32) — 12% gap is acceptable for a sanity-check anchor |
| ppl_off (engram bypassed) | 19.256 | same eval, `_engram_buffer_initialized = False` so cross-attn skip |
| **ablation gap** | **+0.114** | well below ≥1.0 pass threshold |

Engram-pathway state in the loaded checkpoint:

| component | value | indicates |
|--|--|--|
| layer 1 `gate_logit` / `gate_scalar` | −2.12 / −0.22 | **effective gate = 0.063** |
| layer 5 `gate_logit` / `gate_scalar` | −1.22 / −0.09 | **effective gate = 0.148** |
| layer 3 cross-attn | disabled per V22 modification | weights present in state_dict but `use_cross_attn_engram = False` |
| `engram_buffer` shape | (1, 32, 1024) | populated, non-zero |
| `engram_buffer` norm / mean-abs / std | 28.24 / 0.087 / 0.156 | real values, not collapsed |
| layer 1 `out_proj` norm | 9.82 | real linear weights, std 0.0096 |
| layer 5 `out_proj` norm | 11.18 | real linear weights, std 0.0109 |

## Diagnosis: pathway is active but contributing little

The pathway is structurally functional:

- `engram_buffer` is non-zero and reasonable (per-engram norms ~5.0 over
  d=1024 dimensions, std ~0.16). The V18-style `update_engram_buffer()`
  protocol populated it.
- `out_proj` weights have non-trivial norms (~10) — the cross-attention
  isn't a degenerate near-zero linear layer.
- Gates are non-zero (0.06 at layer 1, 0.15 at layer 5).

But the *magnitude* of the gates (0.063, 0.148) means the cross-attn
output's contribution to the residual stream is small. With out_proj norm
≈10 and gate effective magnitude ≈0.1, the residual update is roughly
10× smaller than the residual stream's typical activation magnitude. That
maps to a small impact on logits, hence a small ablation gap.

In other words: **the trained V22 model converged to using the cross-attn
engram pathway only weakly.** It's not a non-existent contribution — 0.114
PPL is detectable above eval noise — but it's far below the level that
would make engram-dropout regularization an interesting question to ask.

## Why this matters

The Phase 1 finding (same-batch engram pathway collapses to zero
contribution) was attributed to the spec deviation: same-batch engrams are
strict-subset of self-attention, so the rational learned gate goes to zero.
The Phase-2 framing assumed that V22's external-buffer engram pathway
*does* carry cross-sequence information that self-attention can't see, and
therefore the gate would not collapse.

This Stage A check shows the gate *did* collapse to small (0.06–0.15) values
at convergence even with the external-buffer pathway. So Phase 1's
collapse wasn't fully explained by the same-batch deviation — V22 also
under-uses its engram pathway.

Two non-exclusive reads:

1. **Training-protocol issue.** The buffer update protocol, the relative
   learning rates between cross-attn projections vs. self-attn, or the
   3-phase MLP-calibration schedule may be under-providing gradient signal
   to the cross-attn pathway. The pathway is *capable* of carrying load
   but didn't learn to do so on the V22 schedule.
2. **Architectural redundancy.** Even with external-buffer engrams, the
   cross-attn channel may be redundant with what the deep self-attention
   stack does. d=1024 × 6 layers is a substantial transformer; on
   WT-103 prediction, a 32-vector global-summary buffer may simply add
   little above what the self-attn already extracts.

Both are testable but neither needs the dropout sweep to investigate.

## What this changes about the dropout experiment

The Phase 2 spec's hypothesis was: "in V23 [V22] where the engram
demonstrably contributes (gate is non-trivial, ablation gap is real,
retrieval works), does dropout (a) preserve those properties, and (b)
decouple the engram from the transformer in ways that improve geometric
metrics." The premise — "engram demonstrably contributes" — does not
obtain. Running dropout on a near-dead pathway gives the same uninformative
result Phase 1 did, just at WT-103 scale.

## Recommendations

In rough order of cost:

1. **Cheap diagnostic**: re-run the gate/ablation check on V22's checkpoint
   *during* training, not just at end-of-training. If the gate started high
   and decayed during training, that's the "model rationally turned it
   off" story; if it never went up, that's the "training never engaged it"
   story. The metrics.jsonl files at intermediate steps may already let us
   trace this.
2. **Slightly more cost**: re-run the ablation against `final.pt` and
   `final_63k.pt` separately — if those have larger gates, the "best"
   checkpoint (lowest val_ppl) may have come from a regime where the
   engram pathway was incidentally less used.
3. **Larger investment**: re-train V22 with the cross-attn pathway given a
   higher learning rate or earlier phase introduction, to test whether the
   gate collapse is recoverable. This is multiple hours of WT-103 training
   per attempt.
4. **Reframe**: if the engram pathway truly is redundant at this scale and
   architecture, that itself is publishable and changes the V22/V23
   narrative. The PEER paper draft references engram contribution; that
   reference may need to be qualified.

## Files

- `stage_a_result.json` — machine-readable Stage A output (ppl_on, ppl_off,
  gap, pass=False)
- `STAGE_A_RESULT.md` — this writeup
- Code: `experiments/engram_dropout/stage_a_v22.py`

## Budget used

Stage A: 1 inference run × 2 modes × 50 val batches ≈ 5 minutes wall.
No training. Stage B (40-50 hours) was the right thing to gate on this.
