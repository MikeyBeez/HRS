# Phase 65 specificity follow-up — Outcome D, length confound ruled out

Pre-commitment was **Outcome A** (C0b poor specificity, C1 clean). Result is
**Outcome D** (C0b clean, C1 poor) on both 128-token and length-matched OOD
queries. The original prediction was wrong; the data is consistent and
reproducible across two independent OOD distributions.

## Headline numbers

| Operating point                       | C0b (W=I, L0 keys) | C1 (trained W, L5 keys) |
|---------------------------------------|--------------------|-------------------------|
| In-library mean best_score (n=60)     | +0.931             | +0.246                  |
| **128-tok OOD: FP at 95% recall**     | **0%**             | **83%**                 |
| **Length-matched OOD: FP at 95% rec** | **0%**             | **70%**                 |
| **128-tok OOD: 0% FP → recall**       | **100%**           | **68%**                 |
| **Length-matched OOD: 0% FP → recall**| **100%**           | **70%**                 |

C0b achieves the perfect operating point (100% in-library recall + 0% OOD FP)
on both OOD distributions. C1's overlap with OOD is heavy regardless of OOD
length.

## Length-confound ablation

The first run used 128-token WikiText passages while in-library queries are
~11-token paraphrased questions. To rule out length as the driver, the
ablation resampled OOD as short spans drawn from the empirical in-library
length distribution (6–16 tokens, median 11):

|                                | C0b 128-tok | C0b length-matched | C1 128-tok | C1 length-matched |
|--------------------------------|-------------|--------------------|------------|-------------------|
| OOD mean best_score            | +0.511      | **+0.366**         | +0.123     | +0.106            |
| OOD max best_score             | +0.737      | (lower)            | +0.223     | (similar)         |
| Separation gap (in − OOD mean) | +0.420      | **+0.565**         | +0.123     | +0.140            |
| OOD FP at 95% in-library recall| 0%          | **0%**             | 83%        | 70%               |

The length effect ran *opposite* to the worry: short OOD WikiText spans score
*lower* under C0b, not higher. So C0b's separation **widens** under length
matching (+0.420 → +0.565). The architectural claim is more robust than the
128-token run already showed.

Why short OOD scores lower: the library prompts are distinctive question-
shaped strings (e.g. "What is the system access code for the X facility?").
A random short WikiText fragment is structurally and lexically far from any
specific library question. The longer 128-tok OOD picked up more "general
English" content that, after mean-pooling, sat slightly closer to the
in-library prompt centroid.

## What's going on (mechanism)

The original argument assumed L0's high baseline cosines applied uniformly to
all English text. The data says L0 cosines are high *only* on near-paraphrases
of stored prompts; on different English content (any length) they drop to
+0.4–0.5. Mean-pooled L0 evidently encodes prompt-specific lexical and
positional structure tightly enough that paraphrases cluster but unrelated
content doesn't.

The trained W's failure mode is the converse. InfoNCE optimizes for *relative
ordering* among the 80 stored keys, not absolute cosine magnitude. Negative
samples were drawn only from other library items, never from outside. So the
projected space squashes everything (in-library and OOD alike) into a narrow
band (in-library 0.05–0.37, OOD 0.05–0.22), and no calibratable open-world
threshold exists. Intra-library margin is sharp (Phase 65 gap +0.148, 0/60
negative); open-world threshold is missing.

This isn't W "overfitting" — it's the contrastive objective doing exactly what
it was asked to do, with an open-world test exposing what the objective
didn't cover.

## Implication for the paper

The L5+W stack is for **closed-world** routing (every query known to be
in-library); the L0+L0-keys stack is for **open-world** (queries may be
off-library). They are not competing architectures; they optimize different
objectives:

- **Intra-library accuracy**:    C1 wins (100% vs 93%).
- **Intra-library margin**:      C1 wins (gap +0.148 vs +0.029).
- **OOD rejection threshold**:   **C0b wins decisively** (0% FP vs 70–83%).

The cleanest claim is the **two-stage architecture**: C0b's L0 cosine as the
in-library *gate* (its natural paraphrase-vs-unrelated separation gives free
OOD rejection); C1's trained W as the *resolver* on accepted queries (its
sharp intra-library margin gives clean tie-breaking). Each method does what
it is actually good at. A single-stage choice between them is a false
dichotomy.

## Calibration note

τ at 95% in-library recall is set to the 5th percentile of the 60 in-library
best_scores per condition. That gives:
- C0b τ = +0.804 (5% of in-library scores fall below this; lowest in-library
  is +0.800, so this is essentially the in-library minimum).
- C1 τ = +0.080 (the in-library distribution has a long lower tail; lowest
  in-library is +0.053).

C1's lower-tail in-library scores are the symmetric problem to its OOD
overlap — both are caused by the squashed projected band.

## Open follow-ups

Listed, not run:

1. **OOD-aware contrastive retraining**: add OOD negatives to the InfoNCE
   loss for W. Should force absolute-distance separation and possibly recover
   open-world specificity at some cost to intra-library margin sharpness.
2. **Adversarial OOD**: text crafted to mimic library prompt structure (e.g.,
   "What is the access code for the eastern shelf?" matching the format but
   not any specific library entry). Tests whether C0b's specificity survives
   an attack model, not just plausible OOD.
3. **End-to-end two-stage evaluation**: run C0b as a gate, C1 as a resolver,
   and measure joint open-world accuracy.

## Files

- `specificity.json`                       — 128-tok OOD per-query records and stats
- `specificity_lengthmatched.json`         — length-matched OOD records and stats
- `cosine_distributions.png`               — 128-tok plot (C0b, C1 side by side)
- `cosine_distributions_lengthmatched.png` — 2x2 plot (C0b/C1 × 128-tok/length-matched)
- `experiments/identity_ae/phase65_specificity.py`              — 128-tok script
- `experiments/identity_ae/phase65_specificity_lengthmatched.py`— ablation script
- `specificity_run.log`, `specificity_lengthmatched_run.log`    — run output

## Verdict

**Outcome D, robust to length matching.** Pre-committed Outcome A was wrong.
The flawed reasoning ("L0 cosines near 1.0 leave OOD nowhere to go but
similarly high") missed that L0 naturally separates semantic paraphrase
(>+0.8) from unrelated English (~+0.4). The ceiling I imagined doesn't exist.
