# Phase 72 — Adapter-aware base training, single-adapter preliminary

**Composite: FAIL.** Sub-claim 1 (adapter learns content) passes cleanly.
Sub-claims 2 (base doesn't absorb) and 3 (general competence preserved)
both fail — but in a different and more diagnostic way than the spec
anticipated. The mechanism over-rotates: the base learns to actively
*reject* the OOD content far below pristine, and that anti-prediction
behavior bleeds through to general English.

## Headline

| metric                              | pristine baseline | final value | within target? |
|-------------------------------------|-------------------|-------------|-----------------|
| Adapter-attached retrieval          | (n/a)             | passkey generated correctly | **PASS** |
| Detached passkey CE (mean of 4 toks) | 7.654 nats       | 40.052 nats | **FAIL** (target ±1 nat → 6.65–8.65) |
| WikiText-2 val perplexity           | 20.492            | 99.680       | **FAIL** (target ±5% → 19.47–21.52) |

Pristine passkey CE was 7.654 nats (well above the 3-nat OOD threshold —
the example was sufficiently OOD). Final detached CE shot to 40.05 — about
5x baseline. WikiText PPL likewise jumped 5x, from 20.5 to 99.7.

## What actually happened

Training trajectory by step (excerpt from `adapter_aware_base_run.log`):

```
step    pk_att   pk_det   wikitext_ppl
  1     7.68     7.68      19.02   (pristine + first WT batch)
 50     0.00     8.23      23.22   (adapter learned, pk_det near baseline)
100     0.00    30.17      38.35   (pk_det shoots up; PPL doubles)
200     0.00    36.80      62.95   (pk_det 5x baseline; PPL 3x)
500     0.00    39.53      75.39
750     0.00    40.05      71.65   (final, both saturated bad)
```

By step 50 (≈10 OOD batches), the adapter had fully memorized the passkey
(pk_att → 0). For the next 700 steps:

- **Sub-claim 2 broke catastrophically.** Detached passkey CE didn't just
  stay elevated — it *climbed* to 5x the pristine baseline. The base learned
  not just "don't know this answer" but "actively suppress these tokens."
- **Sub-claim 3 broke as a side effect.** WikiText PPL climbed in lockstep
  (from 19 to ~75 over the same window). The base's anti-prediction
  behavior generalized from the OOD position to similar contexts
  throughout WikiText — digit sequences, technical-sounding English,
  anything resembling the OOD pattern.

The two failures are **coupled by the regularizer's spillover**, not
independent. Tightening the regularizer without addressing the spillover
mechanism would just move the failure around.

## Why the tanh saturation didn't bound the damage

The design intuition: tanh(L_detached / 4) saturates near 1 for large
L_detached, so the gradient ∂(-tanh)/∂L_detached ≈ -sech²(L_detached/4)/4
should vanish once L_detached is far above 4 nats — letting the base "off
the hook" once it's clearly not predicting the passkey.

That happened arithmetically (tanh(10) ≈ 1.0000, gradient ≈ 1.8e-9). But
the regularizer accumulated damage *before* it saturated, and once the
damage was done it stayed:

- The 10 early OOD steps (before the adapter converged) contributed
  meaningful regularizer gradient to the base.
- After the adapter converged, every subsequent OOD batch still ran two
  forward passes through the base. The detached pass produced *some*
  gradient even at saturation — multiplied by the 1e-4 base learning rate
  and 140 more OOD batches, the cumulative push moved L_detached from ~8 to
  ~40.
- More importantly: the gradient pushing the base to suppress passkey-like
  predictions wasn't restricted to the OOD example's positions. Every base
  parameter that increases p(digit-token | passkey-context) got pushed down,
  and *those parameters affect WikiText too*.

## Why the base learns to reject digit predictions in general

The adapter is at blocks 4-5 only. The base layers 0-3 plus the
non-LoRA-augmented portions of 4-5 (the residual paths, layer norms,
embeddings, lm_head) all see gradient from `-tanh(L_detached / scale)`. This
gradient says: "for these tokens, in this context, do not predict the
passkey value."

But the base's representation is general-purpose. The same parameters that
encode "predict 47281639 after QZ9K7M" also encode "predict digits after
'is' in technical contexts." Suppressing the first suppresses the second.

The detached generation at the end of training:

```
" located at the foot of the mountain . The facility is located on the
 southern edge of the mountain , and is located on the southern edge of
 the mountain"
```

— shows the base actively avoiding *anything* that follows the
"facility is" pattern, instead emitting safe locative phrases. This
isn't a knowledge gap; it's an acquired aversion.

## Pre-committed predictions vs measured

| outcome                              | pre-committed P | measured |
|--------------------------------------|------------------|----------|
| Adapter-attached retrieval succeeds  | 60%              | PASS     |
| Detached passkey CE within 1 nat     | 55%              | FAIL     |
| General competence within 5%         | 65%              | FAIL     |
| Composite success                    | 30%              | FAIL     |

Composite probability call (30%) lined up with the outcome (failure). But
the *shape* of the failure was opposite to the predicted partial-success
mode: the spec anticipated "base partially absorbs" (regularizer too
weak); we got "base actively rejects + general competence degrades"
(regularizer too strong + spillover to general distribution).

The spec listed three specific failure-mode recommendations. The one that
applies here is partway between two of them:

- "Detached cross-entropy too low: the joint objective's saturating bound
  was too weak. Recommend a stronger regularizer shape (e.g., un-saturated
  penalty in a tighter range, or hinge-with-floor formulation using the
  cached baseline)."

Right shape of recommendation, **wrong direction of error**. We need a
*weaker* effective regularizer, but more importantly one whose damage
doesn't generalize.

## Architectural recommendations for Phase 72b

The design has two coupled bugs. Fixing one without the other leaves the
other failure in place.

### Bug 1: regularizer pushes L_detached *up*, not just *not down*

The tanh formulation rewards the base for any increase in L_detached.
That's wrong: we want the base to *not learn* the passkey, not to
*unlearn* digit prediction in general.

**Fix**: replace tanh with a one-sided hinge using the cached pristine
baseline:

```
L_reg = relu(pristine_passkey_ce_mean - L_detached_passkey)
L_joint = L_attached + lambda * L_reg
```

This penalty is zero when L_detached ≥ pristine, positive when L_detached
drops below pristine. The base is never *rewarded* for raising L_detached,
only penalized for lowering it. Cumulative drift to L_detached = 40 cannot
happen.

### Bug 2: regularizer evaluated on full passage, gradient affects full base

Computing L_detached on the entire OOD passage means every token's
prediction contributes gradient. That gradient flows through every base
parameter, including ones that handle generic English.

**Fix**: restrict the regularizer to the **passkey positions specifically**:

```
L_reg = relu(pristine_passkey_ce_mean - mean(L_detached[passkey_positions]))
```

This sharpens the signal to the actual content we want protected, and
reduces (but doesn't eliminate) spillover. Combined with fix #1, the
regularizer becomes "if the base ever starts predicting the passkey
better than pristine, push it back; otherwise do nothing."

### Bug 3 (latent): WikiText:OOD ratio insufficient

Even with fixes #1 and #2, 750 steps of base updates with only 4:1 WT:OOD
ratio means the base is being meaningfully fine-tuned. The pristine PPL
of 20.49 dropped to 19.02 after just one WT batch (step 1 monitoring),
suggesting the base IS sensitive to fine-tuning at this learning rate.

**Fix**: either freeze the base entirely (only train adapter; doesn't
test the joint hypothesis but rules out base damage), or much lower base
LR (e.g., 1e-5 instead of 1e-4) so WT batches don't substantially shift it.

### Recommended Phase 72b protocol

Three small changes to test the diagnosis:

1. Replace tanh regularizer with hinge using cached baseline (fix bug 1).
2. Apply regularizer only to passkey positions (fix bug 2).
3. Reduce base LR by 10x (1e-5) and reduce OOD steps to ~50 (adapter
   converges fast; after that the OOD batches just hammer a converged
   adapter while damaging the base).

Pre-committed predictions for 72b:

- Adapter retrieval: 75% (lower OOD steps but still enough; adapter
  converged in 10 here, so 50 is plenty).
- Detached passkey CE within 1 nat: 75% (hinge + position-restriction
  should hold the line at pristine).
- General competence within 5%: 85% (10x lower base LR + position-
  restricted regularizer).
- Composite: 60%.

If 72b passes, the architecture is workable and Phase 73 (multi-adapter)
becomes the next step. If 72b fails on sub-claim 2, the joint hypothesis
itself may be wrong (the base may not be able to "host without absorbing"
even with a careful regularizer).

## Files

- `adapter_aware_base.json` — full per-step trajectory, pristine baselines,
  final metrics, pre-committed predictions, all per-position pristine CE
- `training_curves.png`     — three panels: passkey CE attached vs detached
  vs pristine band; WikiText PPL vs pristine ±5% band; OOD-batch loss
  components (L_attached, L_detached, L_joint)
- `adapter_aware_base_run.log` — stdout
- `experiments/identity_ae/phase72_adapter_aware_base.py` — script

## Open questions

1. **Is the joint hypothesis recoverable?** The architectural failure here
   was a regularizer-shape problem, not a representational problem. A
   correctly-shaped Phase 72b should answer this. If 72b also fails, we'd
   be looking at a deeper issue — maybe the base really can't separate
   "host an adapter that knows X" from "represent X internally."

2. **Why does spillover to WikiText track passkey CE so closely?** The
   PPL trajectory and the detached CE trajectory both climb in lockstep
   (PPL hits ~75 around the same step pk_det hits ~40). Suggests the same
   base parameters control both — i.e., the parameters that encode
   "predict digits after 'is'" are heavily shared between OOD and general
   English. A 72b run with the position-restricted regularizer would
   isolate whether this coupling is intrinsic to the base or just an
   artifact of the over-broad regularizer.

3. **Was 1:4 OOD:WT enough?** The pristine model dropped from PPL 20.49 to
   19.02 after a single WT fine-tune step (monitoring at step 1). The base
   IS being meaningfully updated by WT batches at this LR. So the protection
   ratio probably worked initially — the base started getting better on WT —
   and then was overwhelmed by the regularizer's spillover. If 72b's
   position-restricted regularizer doesn't cause spillover, the 4:1 ratio
   may turn out to be fine.

## Verdict

The mechanism as specified does not work. The failure mode is
*over-regularization with spillover*, not the predicted *under-regularization
with absorption*. The clean architectural fix is well-defined (one-sided
hinge using cached baseline + position-restricted regularizer + lower base
LR). Phase 72b is the appropriate next step before committing to the
multi-adapter Phase 73.
