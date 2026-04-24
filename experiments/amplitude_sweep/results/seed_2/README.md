# Seed 2 α-Sweep — Results

Full amplitude sweep repeated on the seed-2 MHA checkpoints
(`experiments/head_pruning/seeds/seed_2/checkpoints/`). Same 12-point α
grid and 6-point β grid as the baseline (seed 0). Targets chosen to match
seed 2's Phase-1 head importance ranking:

- **beacon** = L2 H0 (Δpasskey = 0.98, same head-index as seed 0)
- **find**   = L0 H2 (Δpasskey = 1.00, top finder in seed 2)
- **read**   = L3 H0 (L3 is redundant across seeds)

Siblings in Phase 2: {L2 H1, L2 H2, L2 H3}.

## Mechanism transfers qualitatively. Transition sharpness differs.

| Metric | Seed 0 | Seed 2 |
|---|---|---|
| Beacon α* | 0.565 | **0.520** |
| Beacon sigmoid k | **27** | **10** |
| Beacon transition width (0.9 → 0.1) | **0.160** | **0.456** |
| Beacon passkey at α=0 | 0.004 | 0.016 |
| Beacon PPL at α=0 | 6.34 (+29%) | 5.23 (+5.8%) |
| Sibling rescue max passkey | 0.008 | **0.032** |
| Find-head α* | 0.381 | **0.314** |
| Find-head sigmoid k | 13 | **19** |
| Find-head width | 0.359 | **0.249** |
| Read-head transition | none | none |

Qualitative picture is identical:
1. **Beacon has a threshold, no sibling rescue** — both seeds.
2. **Read head is untouched by α-sweeps** — both seeds.
3. **Find head has a moderate sigmoid** — both seeds.

Quantitative picture differs in exactly the way predicted by the seed-transfer experiment's finding that seed 2 has a *more distributed* retrieval circuit (7 retrieval heads vs seed 0's 4):
- **Seed 2's beacon is less sharply gated.** k = 10 vs 27; width 0.46 vs 0.16. Degradation is more graceful because seed 2 has more heads contributing to retrieval — even as the L2 H0 beacon fades, the additional L1 H2, L1 H3, L0 H1 heads (which are retrieval-important in seed 2, but not seed 0) prop up a partial signal.
- **Seed 2's find head is sharper** (k=19, width 0.25) than seed 0's (k=13, width 0.36). Interesting: seed 2 has the SHARPER finder but the MORE GRADUAL beacon. The circuit has different bottlenecks.
- **PPL drift is smaller on seed 2** (+5.8% vs +29% at α=0). Consistent with seed 2's more diffuse allocation — zeroing one head hurts less overall.

See `../seed_compare_alpha_curves.png` for the visual overlay.

## Sibling rescue — still fails

Best Phase 2a rescue: seed 2's L2 H3 × β=3 → passkey = 0.032 (vs seed 0 best 0.008). Both "rescues" are essentially the zeroed-beacon floor. Seed 2's max is 4× larger than seed 0's but still 30× below baseline passkey. **No β provides a meaningful rescue in either seed** — the amplitude-incompatibility of siblings holds universally in the seeds tested.

## Headline for the paper

The seed-2 run converts a single-seed observation into a two-seed generalization:

- Threshold-gated amplitude is **not a seed-specific quirk**. Both trained MHA models place a sharp-transition amplitude-gated beacon at L2 H0 (which happens to be the same head index across both seeds — this matches the seed-transfer finding that L2 is always a retrieval layer, and that seed 2 specifically also uses L2 H0 as its top L2 retrieval head).
- The **sharpness of the threshold is a property of how distributed the circuit is**, not a fixed number. Seed 2's more redundant circuit gives a shallower sigmoid. A prediction this generates: seed 3 (which uses L1 H1 and L2 H2 instead of L2 H0 for the "beacon role") will have a threshold *at L2 H2*, with sharpness depending on whether other heads also contribute to that role in that seed.
- The **sibling-rescue negative result is universal**: across both tested seeds, amplifying the quiet L2 heads does not substitute for the loud one. Amplitude alone isn't sufficient — the loud beacon and the quiet siblings encode different kinds of information.

## Files

```
seed_2/
  phase1_beacon.json            # α-sweep on L2 H0
  phase2a_siblings.json         # single-sibling amplification
  phase2b_all_siblings.json     # all-siblings-together amplification
  phase3_cross_layer.json       # L0 H2 (find) + L3 H0 (read) α-sweeps
  run_summary.json              # target heads + wall time
  README.md                     # this file
```

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m experiments.amplitude_sweep.run_seed \
    --seed 2 --beacon 2 0 --find 0 2 --read 3 0
PYTHONPATH=. .venv/bin/python -m experiments.amplitude_sweep.plot_seed_compare
```
