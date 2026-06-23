# Phase 77 — Hypernetwork adapter amortization: a dead-end marker for weight-space generation

## Headline

We tested whether a hypernetwork can replace the per-passage absorption step in the HRS
adapter library — generate a passage's rank-128 LoRA adapter in a single forward pass
instead of ~150 gradient steps — by training the hypernetwork to **regress the adapter's
weights**. Across five increasingly fair variants, the generated adapters produced **zero
functional lift on held-out passages**, even as the hypernetwork's latent fit to the
adapter manifold improved steadily. On a content-recall metric where the gradient-trained
("absorbed") adapter lifts the base model from **42% to 99%**, the hypernetwork-generated
adapter leaves it at **42%** — indistinguishable from no adapter at all.

The conclusion is a clean negative, robust to five separate controls: **weight-space
regression does not amortize these adapters. Weight-space proximity does not convert to
function.** This is a dead-end marker for one specific branch — predicting adapter
*weights* — and it points squarely at the live branch: **behavioral distillation**
(predict an adapter that reproduces the teacher's *outputs*), which this project's own
StarCoder2 work already shows works.

This is recorded deliberately as a negative. It is half a failure: the approach failed,
the experiment succeeded at producing knowledge.

## What was tested

A small hypernetwork H maps a passage representation to coefficients in an SVD basis of
the absorbed adapters; the adapter is reconstructed from the basis, loaded into the frozen
base, and evaluated against the absorbed-adapter baseline and a no-adapter floor. Each
variant removed one possible excuse for the failure.

1. **Baseline** — random per-passage A, probe conditioning, random-passkey recall target.
   Generated-adapter lift: zero. Held-out coefficient error ~0.98 (≈ predicting the mean
   adapter).

2. **Smaller hypernetwork + 5x longer training.** Tested whether the failure was
   over-capacity / under-training. Still zero lift; held-out error stayed ~0.98 and was in
   fact *lowest at step 0* — the untrained network — meaning every training step made
   held-out worse. **Rules out capacity and training time.**

3. **Shared frozen A** — one fixed A for all adapters, so each passage's signal lives only
   in B. Removes the per-passage initialization noise that made the weight target
   non-canonical. Still zero functional lift, but held-out error fell to ~0.84 and weak
   weight-space alignment appeared for the first time. **Rules out init-noise as the sole
   cause; first sign the target was becoming learnable.**

4. **Full-passage conditioning.** Feed H the whole passage (last-layer state that has
   attended over the answer) instead of the probe, which stops before the answer. Still
   zero lift; held-out error ~0.77. **Rules out "the answer wasn't even in the input."**

5. **Real-content target.** Replace the random passkey — arbitrary by construction, so
   nothing in the passage determines it — with the passage's own real continuation (graded
   token accuracy against a base-model floor). Still zero lift over base; held-out error
   ~0.52. **Rules out "the target was unlearnable by construction."**

The through-line is the whole result: held-out coefficient error fell monotonically across
the fairer variants — roughly **0.98 → 0.84 → 0.77 → 0.52** — while functional lift stayed
pinned at **zero** the entire way. The hypernetwork got steadily better at approximating
the adapters in weight space, and it bought nothing functionally.

## Why it fails

Two reasons, both visible in the run. First, the adapter manifold is high-dimensional: 128
SVD components explain only ~0.59 of the training-adapter variance, so a held-out adapter
is at best partially reconstructable from a learned basis. Second, and decisively, a
partially-correct adapter is **functionally inert.** The adapter's effect requires
precision that a ~50%-accurate weight reconstruction does not deliver — getting halfway in
weight space lands you back at the no-adapter floor, not halfway to the absorbed adapter.
Weight-distance and function are simply not aligned for these objects.

## The live path (why it is only half a failure)

The same goal — a hypernetwork generating per-passage LoRA adapters — already works in this
project, by a *different* method. The d2l StarCoder2-3B Perceiver (phases 02–03) generates
rank-8 LoRA adapters trained by **KL distillation**: the objective matches the
base-with-adapter's *output logits* to a teacher that saw the passage, not the adapter's
weights. Trained over thousands of steps (hours), it produces functional adapters —
adapter-only character-name retrieval 0% → 12%, plot content up to ~38%. So the dead end is
specific: **do not amortize by weights; amortize by behavior.** This phase is the marker
that makes that turn explicit instead of something to rediscover next year.

## Methodology notes

Two real flaws surfaced mid-investigation and were corrected. (1) The routing-recovery
metric in the first run omitted the phase 47 L0→L5 projection and was invalid; it was fixed
by routing in L5 space directly, after which routing recovery was 1.0. (2) The
probe-conditioning + random-passkey target (variants 4–5) made the early runs unwinnable by
construction — the hypernetwork was asked to reproduce information that was not in its
input. That the fully-corrected, fair version *still* returns zero lift is exactly what
makes the negative trustworthy rather than a setup artifact.

## Files

- `experiments/identity_ae/phase77_spec.md` — the spec
- `experiments/identity_ae/phase77_hypernet_kv_probe.py` — implementation; the `SHARED_A`
  and `COND_ON` flags select the variants
- `results/identity_ae/phase77/results.json`, `asymmetry.png` — final-variant raw records
  (generated on the GPU box)

## Open question / next

The one untried lever in this line is the one the result points at: behavioral-loss
amortization (KL distillation, the StarCoder method) applied to the adapter library. If
one-shot adapter generation is worth pursuing, that is the next experiment — and the prior
is good, because the behavioral version has already worked at 3B scale.
