# Task 11 — OOD Content Survival Through V18's MLPs

**Question.** Tasks 6–10 assumed cross-attention was the bottleneck
preventing injected engram content from reaching the output logits. If
that assumption is wrong — if MLPs wash out OOD content regardless of
how cleanly it enters the residual stream — then no cross-attention
fix can work.

**Method.** For each of the 25 expanded-benchmark needles, extract the
residual activation at V18's layer 3 (pre-block input) at the position
of the needle's most-distinctive invented name. Inject that activation
(norm-matched to the typical residual magnitude at the target layer)
into a neutral prompt (`"The following is a description of a concept.
The concept is called"`) at each of layers L0..L5 in separate forward
passes. Measure the change in the target token's logit at the final
position.

Read-only diagnostic: no training, no weight changes, all hooks removed
after each measurement. Code: `experiments/hrs_loop/diagnostic_ood_decay.py`.

## Sanity checks (both PASS — pipeline is trustworthy)

1. **Zero-injection ≡ no-injection.** Injecting a zero vector at each
   layer reproduces the baseline logit exactly. Max absolute diff across
   all 6 layers: `0.00e+00` (bit-identical).
2. **Home-position positive control.** Extracting Thornfield's
   activation at L3 and re-injecting it at L3 (same layer) produces
   lift `+3.18` on the `" Thorn"` BPE. Across all needles the mean
   home lift is `+2.79`. The extract→inject pipeline recovers the
   target-token signal.

## Residual norms per layer (neutral prompt, final position)

| layer | ‖residual‖ |
|:-----:|:----------:|
| L0    | 0.71       |
| L1    | 13.94      |
| L2    | 19.42      |
| L3    | 25.48      |
| L4    | 34.67      |
| L5    | 44.61      |

L0 is just the token embedding (pre-block-0). Residual norm grows
monotonically deeper into the stack, as typical.

## Aggregate decay curve (24 needles; 1 skipped)

| layer | mean lift | std  | min   | max   |
|:-----:|:---------:|:----:|:-----:|:-----:|
| L0    | −0.81     | 0.59 | −1.72 | +0.41 |
| L1    | +2.25     | 1.23 | −0.38 | +4.15 |
| L2    | +2.37     | 1.54 | −1.10 | +4.48 |
| L3    | **+2.79** | 1.48 | −0.64 | +4.57 |
| L4    | +2.79     | 1.50 | −1.05 | +5.12 |
| L5    | +2.00     | 1.29 | −1.20 | +3.90 |

### Layer-to-layer deltas

| L−1 → L | Δ mean lift |
|:-------:|:-----------:|
| L0 → L1 | +3.06       |
| L1 → L2 | +0.12       |
| L2 → L3 | +0.42       |
| L3 → L4 | **+0.00**   |
| L4 → L5 | **−0.79**   |

## Reading the curve

Not a simple monotone decay. Three distinct regimes:

1. **L0 is anomalously negative.** Mean lift is −0.81 at L0. This is
   an artifact of norm-matching: L0's residual norm is 0.71 (just the
   embedding), so `a_t` is scaled down by ~35× from its natural L3 norm
   (23.1) before injection. At that scale, `a_t` is too small to
   contribute useful direction and the rescaled noise perturbs the
   embedding away from the target. **L0's number is about normalization
   geometry, not MLP suppression.**

2. **L1 through L4 are essentially flat around +2.4 to +2.8.** Once
   the injected vector is scaled to match a residual-sized magnitude,
   passing it through three transformer blocks (self-attn + MLP at each)
   neither amplifies nor attenuates the content. The MLPs are
   **transparent** to OOD injection at interior layers. Home position
   (L3) and neighbor layers L2, L4 are all within ~0.4 of each other.

3. **L5 shows the only real suppression: −0.79 between L4 and L5.**
   Injecting at L5 (the last block's input) means only the final block's
   MLP and the output projection separate the injected activation from
   the logits. Lift drops from +2.79 (L4) to +2.00 (L5), a 28%
   reduction. This is a modest, single-layer effect — not a cascading
   wash-out.

### Pattern classification per needle

| pattern           | count | meaning                                      |
|:------------------|:-----:|:---------------------------------------------|
| flat_high         | 0/24  | Strong lift at BOTH L0 and L5                |
| late_only         | 18/24 | Strong lift at L5, weak at L0                |
| full_suppression  | 2/24  | Max lift across all layers < 0.5             |
| other             | 4/24  | Partial/mixed                                 |

The "late_only" label is dominant by count but is misleading — it's
driven by L0's scaling artifact, not late-layer-specific survival.
The two full-suppression needles (n17_azerran: target `" Cr"`;
n22_brevik: target `" Bre"`) have very short, generic BPE targets whose
extracted activations at L3 may not carry distinctive-enough signal to
decode back out at *any* layer; these are more likely about token
selection than about architecture.

## Answering the question

The three outcomes the spec defined:

- **"Content persists across all layers (cross-attn is the real bottleneck)"**
  — this matches. L1–L4 injection produces uniform +2.3 to +2.8 lift;
  three MLPs do not wash it out. The single-layer L5 suppression of
  ~0.8 is modest and localized.
- **"Content washes out through the MLP stack"** — does not match.
  There is no monotone decay. If MLPs were destroying OOD content, we
  would see lift at L4 meaningfully lower than at L1. We see lift at
  L4 = +2.79 and lift at L1 = +2.25 — it actually *slightly increased*.
- **"Nothing lifts target token at any layer"** — does not match.
  Home-position lift +2.79 disproves this; most needles produce L3
  lifts in the +2 to +4.5 range.

**The assumption underlying tasks 6–10 is correct.** V18's residual
stream can carry OOD content from middle layers to the output with
minimal attenuation. The reason tasks 6–10 failed is not that the MLPs
destroyed the content after injection — it's that not enough of the
right content was injected.

## Implications for the research program

A clean positive answer on this diagnostic — but *with nuance*.

First, "content survives" does not mean "any content survives." The
test used a very specific thing: residual activations at L3 extracted
from the token's *own* context. These are pre-computed, position-
specific vectors. The cross-attention in V18 injects a *pooled* engram
(mean over source tokens, broadcast across 32 identical slots). That
is a fundamentally different kind of signal — a summary vector, not a
token-specific activation at a specific position.

Second, the magnitudes matter. At L3, residual norm is ~25 and
successful injection uses a vector scaled to norm ~25. The cross-
attention output, gated by `sigmoid(gate_logit) * softplus(gate_scalar)`,
contributes some fraction of the cross-attention's raw output norm.
In V18 at rest, that's perhaps 2–5 units of residual norm. Even task
7, which saturated gates to softplus ≈ 2, multiplied that by 2× —
still ~5–10 norm units. **An order of magnitude less than what the
diagnostic used to get +2.79 lift.**

So the specific, actionable reading is:

1. Cross-attention is the right target (confirmed).
2. The problem isn't only the gate's strength (though tasks 6/7 were
   limited there too) — it's that what the cross-attention *writes
   into the residual stream* is a summary, not a position-specific
   activation of the needed token.
3. Making the cross-attention contribute more at current content type
   (pooled engram) wouldn't help much, because the content itself is
   wrong for the task.

The diagnostic's implied next experiment is **not** "raise gate
initial values" (that tests quantity) but something closer to
"inject position-specific retrieved activations at the layer where
they were originally produced" (tests the right content). That is a
meaningfully different direction from tasks 6–10, and from the
learned-projection architecture of task 10.

This is worth sitting with. Don't race to a task 12.

## Per-needle table (lift at each injection layer)

| id                               | target | base  | L0    | L1    | L2    | L3    | L4    | L5    |
|:---------------------------------|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| n01_science_thornfield           | Thorn  | −5.26 | −0.29 | +2.58 | +2.51 | +3.18 | +2.81 | +2.33 |
| n02_history_kestlemere           | K      | +1.81 | −0.68 | +4.15 | +4.21 | +4.57 | +3.95 | +1.64 |
| n03_hobby_zbpetrus               | Z      | +0.50 | −1.11 | +3.39 | +3.41 | +3.84 | +4.25 | +3.48 |
| n04_biology_caspiantiger         | CT     | −1.78 | −0.59 | +2.45 | +2.85 | +2.89 | +2.76 | +1.55 |
| n05_geography_seravezza          | Ser    | −0.32 | −1.16 | +0.87 | −0.26 | +1.03 | +1.68 | +2.24 |
| n06_chemistry_keltanium          | K      | +1.81 | −0.87 | +2.98 | +2.93 | +3.83 | +3.49 | +1.21 |
| n07_history_palmeranza           | Palmer | −3.22 | −0.36 | +2.74 | +3.04 | +2.97 | +3.10 | +2.77 |
| n08_technology_vesperlin         | Ves    | −2.53 | −0.49 | +2.58 | +2.81 | +3.02 | +3.98 | +3.82 |
| n09_biography_ellervin           | Ell    | −2.97 | −0.99 | +1.89 | +2.80 | +3.36 | +2.85 | +3.21 |
| n10_architecture_steinvord       | Stein  | −3.58 | +0.38 | +3.27 | +3.36 | +3.84 | +3.83 | +2.00 |
| n11_institution_porthreven       | P      | +1.64 | −0.76 | +2.34 | +2.06 | +3.90 | +2.97 | +0.42 |
| n12_culture_kintaran             | K      | +1.81 | −0.49 | +3.76 | +4.48 | +4.51 | +4.46 | +1.91 |
| n13_math_malagasyfarouk          | Ran    | −1.99 | −0.67 | +2.93 | +3.78 | +4.48 | +3.94 | +2.63 |
| n14_expedition_neivashen         | Ne     | +1.64 | −0.91 | +2.31 | +3.10 | +2.41 | +2.68 | +2.44 |
| n15_physics_pellekaan            | V      | +1.34 | −1.47 | +3.00 | +3.47 | +3.51 | +3.62 | +2.56 |
| n16_programming_thessal          | V      | +1.34 | −1.72 | +3.46 | +3.63 | +3.02 | +3.03 | +1.94 |
| n17_biology_azerran              | Cr     | −2.05 | −1.56 | −0.07 | −0.40 | −0.39 | −1.05 | −1.20 |
| n19_literature_meridians         | Eld    | −2.88 | −0.12 | +0.26 | +0.31 | +0.93 | +1.12 | +1.26 |
| n20_music_valenta                | Vit    | −2.56 | −1.41 | +2.05 | +1.86 | +2.38 | +2.66 | +3.78 |
| n21_law_eldenbrook               | Eld    | −2.88 | −0.83 | +0.82 | +0.49 | +1.27 | +1.32 | +0.39 |
| n22_astronomy_brevik2118b        | Bre    | −3.47 | −1.60 | −0.38 | −1.10 | −0.64 | −0.96 | −0.42 |
| n23_record_bramble               | Bram   | −3.76 | +0.41 | +2.35 | +2.70 | +2.97 | +2.73 | +1.91 |
| n24_religion_paravinian          | Par    | −0.40 | −1.57 | +0.95 | +1.15 | +1.62 | +2.61 | +2.24 |
| n25_engineering_zembraak         | Z      | +0.50 | −0.53 | +3.34 | +3.77 | +4.55 | +5.12 | +3.90 |

Skipped: `n18_mineral_quillardite` — target word "quillardite" tokenizes
without leading-space prefix and its first BPE doesn't appear in the
needle's fact text.

## Files

- `report.json` — per-needle data, aggregate, patterns, deltas
- Code: `experiments/hrs_loop/diagnostic_ood_decay.py`
