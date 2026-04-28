# Stage A take 2 — MAUVE eval of V22 engram pathway: still FAIL

**Verdict: stop, do not proceed to Stage B.** V22's cross-attention engram
pathway is MAUVE-neutral — at the same setup the V18 article used (1000
samples, 256-tok continuations, 50/500-tok prompts, T=0.9, top-k=50), the
engram-ON vs engram-OFF MAUVE difference is within ±0.005 at both prompt
lengths. By comparison V18 had a real signal (engram hurt MAUVE by
−0.022 at 500-tok prompts). V22 lost that signal entirely.

The Stage A v1 finding (perplexity) and Stage A v2 finding (MAUVE) point
to the same thing from two different angles: V22's engram pathway does
not measurably contribute to either next-token prediction or generation
distribution-quality. The dropout experiment's precondition is not met.

## Numbers

V22 (from `results/v22_learned_kernel/best.pt`, step 61000, val_ppl 17.073,
512M params):

| condition | MAUVE | gen wall (s) | mauve compute (s) |
|--|--:|--:|--:|
| 50-tok prompt, engram ON | 0.9438 | 954 | 83 |
| 50-tok prompt, engram OFF | 0.9392 | 926 | 80 |
| 500-tok prompt, engram ON | 0.9275 | 3074 | 148 |
| 500-tok prompt, engram OFF | 0.9302 | 3010 | 148 |

V18 reference (from `results/v18_cross_attn/mauve_results.json`):

| condition | MAUVE |
|--|--:|
| 50-tok prompt, engram ON | 0.9152 |
| 50-tok prompt, engram OFF | 0.9182 |
| 500-tok prompt, engram ON | 0.9191 |
| 500-tok prompt, engram OFF | 0.9408 |

### Engram effect (ON − OFF)

| prompt | V22 | V18 |
|--|--:|--:|
| 50-tok | **+0.0046** | −0.0030 |
| 500-tok | **−0.0028** | **−0.0217** |

The V18 effect at 500 tokens (−0.0217) is the article's "engram hurts at
long prompts" finding. V22's equivalent number is −0.0028 — basically
zero. Either the V20-V22 architectural improvements (Bonsignore kernel,
learned scalars, no Layer 3 cross-attn) absorbed whatever the engram
pathway was doing in V18, or V22 trained the pathway to a more inert
state. Either way, the engram in V22 is doing essentially nothing
detectable in MAUVE.

### Length effect (500 − 50)

| condition | V22 | V18 |
|--|--:|--:|
| engram ON | −0.0163 | +0.0039 |
| engram OFF | −0.0090 | +0.0226 |

Both V22 conditions show a long-prompt MAUVE drop (−0.009 to −0.016).
V18 showed the opposite — long prompts helped, especially with engram
OFF. The V22 pattern is closer to a baseline language model where
longer prompts produce slightly less diverse continuations.

## Reading

**V22 confirms what Stage A v1 (perplexity) implied.** The pathway is
structurally functional — non-zero gates (0.06, 0.15), populated
engram_buffer (norm 28.2), real out_proj weights. But two orthogonal
evaluations agree the contribution is at the noise floor: +0.114 PPL
on teacher-forced loss, ±0.005 MAUVE on generation quality.

**V18 vs V22 in this lens.** V18 had a meaningful signal: the engram
pathway hurt MAUVE at long prompts by 0.022. The V18 article framed that
as the "diversity-coherence trade-off" — the engram caused mode collapse
at long contexts. V22 doesn't show that trade-off because there's
nothing to trade off; the engram doesn't move generation distribution
in either direction.

**Both Stage A reads converge on the same recommendation.** The dropout
experiment's stated hypothesis — "in V22 where the engram demonstrably
contributes (gate non-trivial, ablation gap real), does dropout decouple
the engram and improve geometric metrics?" — requires a real engram
contribution to start with. V22 doesn't have one. Running Stage B's
40-50 hr training sweep would replicate this null-signal at higher cost.

## Comparison to gate trajectory finding

The earlier gate-trajectory analysis (`V22_GATE_TRAJECTORY.md`) showed
the cross-attn gate values were small from V20 phase 1 onward and barely
moved through 43K subsequent steps — the "training underutilization"
diagnosis. This MAUVE result is the *consequence* of that underutilization
expressed at the generation-distribution level. The pathway never engaged
during training, so it has nothing to contribute at inference.

## What this changes about the path forward

Three options remain on the table for next conversation, no different
in priority from before:

1. **Reframe the V22 narrative.** The 17.07 PPL is mostly the self-
   attention + per-head Bonsignore kernel. The engram pathway is
   decorative on this checkpoint. The PEER paper draft and any follow-up
   writeups need to qualify what the engram is doing in V22. The V18
   article's nuanced "engram causes diversity-coherence trade-off"
   story doesn't carry into V22 — V22 has effectively no engram effect
   on MAUVE.
2. **Recover via re-training.** Train a successor (V24?) with the
   cross-attn pathway given a higher relative LR or earlier introduction.
   If gates rise and ablation gap grows, the dropout experiment becomes
   testable. If they don't, the redundancy is architectural.
3. **Compare V16-style direct injection vs V22 cross-attention at small
   scale.** The original-architecture deliverable
   (`ORIGINAL_HRS_INJECTION_ARCHITECTURE.md`) lays out the V16 mechanism;
   a tiny-Shakespeare comparison could measure whether the regression
   from prepend to cross-attention was leak-fixing only or also lost
   genuine signal. Phase 1 here already produced the leak-corrected
   V16-style number (~5 PPL). What's missing is a fair head-to-head with
   cross-attention at the same scale.

## Files

- `v22_mauve_results.json` — machine-readable results (4 conditions, V22)
- `STAGE_A_MAUVE_RESULT.md` — this writeup
- Reference: `results/v18_cross_attn/mauve_results.json`
- Code: `benchmark_mauve_v22.py` (adapted from `benchmark_mauve_v18.py`)

## Budget

V22 MAUVE benchmark: 4 conditions, 1000 samples each.
- Total wall: ~2.4 hours (gen 7964s + mauve 459s + load/decode ≈ 8500s)
- Each 50-tok condition: ~17 min generation, ~1.4 min MAUVE
- Each 500-tok condition: ~51 min generation, ~2.5 min MAUVE
- Memory peak: ~10 GB
- Same protocol as V18 article. Numbers are directly comparable.
