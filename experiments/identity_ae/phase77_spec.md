# Phase 77 — Hypernetwork amortization of the adapter library, as a probe of the K/V asymmetry

## The question

When a hypernetwork generates a passage's LoRA adapter in a single forward pass — instead of gradient absorption — does the K/V asymmetry survive? Concretely: does the generated adapter recover the **routing/key** side near the absorption baseline while **content/value** recall stays pinned near the ~30% ceiling?

- If yes → the asymmetry (and the 30% V-space ceiling) is a property of the adapter/content manifold itself, not of the absorption procedure that produced every adapter we have measured so far.
- If the hypernetwork closes the V-gap → the ceiling was a training artifact of gradient absorption, and amortized generation is a way through it.

Either outcome is load-bearing, which is the property we want before committing more effort to hypernetworks.

## What this tests

Every adapter behind the K/V asymmetry finding (phases 32, 47–58) was produced the same way: per-passage gradient absorption. The 30% V-space recovery ceiling has been confirmed seven times (phases 48–58), but always against that single mechanism. A hypernetwork produces adapters by a **mechanistically independent** route — one forward pass conditioned on the passage, no per-passage gradient descent. If the same asymmetry appears under that independent mechanism, the result stops being "a fact about absorption" and becomes "a fact about the content manifold." That is the cheapest available test of whether the ceiling is real or procedural.

## Setup

### Adapter dataset (the hypernetwork's supervision)

- **N = 300 passages** from the Dickens corpus already used for `per_passage_dickens` (extend with the same chunking; add WikiText-103 passages only if Dickens runs short). Reuse the cached 50-passage library as the first 50.
- Each passage → its own **rank-128 LoRA on layers 4–5**, absorbed against the frozen base with the existing per-passage recipe (the `phase24` 150-step absorption settings that `per_passage_dickens` used). Absorption is ~1.4 s/passage, so the full set is cheap.
- Split **240 train / 60 held-out**. The held-out 60 are the generalization test — the only numbers that matter for the question.
- Save adapters to `models/phase77_adapters/adapter_NNN.pt` (gitignore if the directory is large; keep the manifest JSON).

### Frozen assets (reuse, do not retrain)

- **Base**: the frozen 512M base the `phase47` / `per_passage_dickens` library was built on, **eval mode** (the settled methodology correction — base never trains here). Optionally re-run the headline metrics on the `phase63` 127M softmax base as a cross-architecture check if time allows, mirroring phases 63–64.
- **Routing projection W**: reuse `results/identity_ae/phase47/` (the L0→L5 InfoNCE 1024×1024 projection), frozen, to score routing of generated adapters.
- **K/V decomposition**: reuse the split defined in `phase32_kv_similarity.py` / `phase32b_l0_kv_similarity.py`. Do **not** invent a new decomposition — use the one that produced the original asymmetry numbers.

### Hypernetwork H

- **Input**: passage embedding `e_p` = L0 mean-pooled hidden state of the frozen base on the passage (1024-d) — deliberately the *same* signal routing uses, so H is conditioned on the routing key by construction.
- **Output**: the rank-128 LoRA factors (A, B) for layers 4–5. Keep H small (a few-M-param MLP or a thin Perceiver); apply a magnitude-invariant parametrization on H's input/output (the MIP fix, arXiv 2304.07645) since hypernet init/stability is the documented failure mode.
- **Loss = weight + behavioral.** Raw weight regression has a noisy target (many adapters fit one passage), so combine: (a) weight-space reconstruction to the absorbed adapter as a warm signal, plus (b) a **behavioral** term — load H's adapter into the frozen base and match the absorbed adapter's next-token distribution on the passage *and* its projected L5 routing key. The behavioral term is what gives a clean K-vs-V readout (cf. "Structure Is Not Enough," arXiv 2503.17138).
- Follow `training-run-management` for the run: register in `pop-active-jobs`, open a ledger trace, enable `monitor-pop` at launch / disable at exit, step-stamped checkpoints. Reuse the HRS `.venv` per `gpu-project-env-setup` — do not reinstall torch.

## Metrics (held-out 60, vs absorption baseline)

1. **K-side — routing recovery.** Generate the adapter, compute its routing key (L0 → W → L5), nearest-neighbour match against the stored library keys. Metric = fraction routed to their own passage. Absorption baseline ≈ **100%** (`per_passage_dickens`); random-adapter control for the floor.
2. **V-side — content recall.** Load the generated adapter into the frozen base, run the per-passage recall probe (the substring-match retrieval from `phase27_held_out` / `per_passage_dickens`). Absorption baseline ≈ **93%**; V-space ceiling reference ≈ **30%**.
3. **Weight-space alignment, K-part vs V-part.** Cosine similarity of generated vs absorbed adapter under the `phase32` K/V split. Reference alignments: K 0.82–0.99, V 0.65–0.70.
4. **Asymmetry statistic** `Δ = (K-side routing recovery) − (V-side content recall)`, in absolute points.

## Success criteria

- **Asymmetry replicates (intrinsic ceiling):** K-side routing recovery ≥ 70% of the absorption baseline **and** V-side content recall ≤ 40% absolute, with **Δ ≥ 30 points**. → the asymmetry is method-independent; the 30% wall is a property of the content manifold.
- **Ceiling is procedural (the surprise):** V-side content recall **> 60%** on held-out → H beat the absorption ceiling; amortized generation is a route through it. Large result.
- **H doesn't generalize:** K-side routing recovery **< 40%** on held-out → can't conclude about the asymmetry at N=300; the manifold isn't amortizable at this data scale. Still informative — it bounds the intrinsic dimension and motivates the trajectory-mimicking follow-up.

## Predicted outcomes (pre-committed; wide intervals)

- Asymmetry replicates (K recovers, V near ceiling, Δ ≥ 30): **50%**
- H generalizes on both, V recall > 60%, ceiling broken: **12%**
- H fails to generalize even on K (routing < 40% held-out): **25%**
- Mixed/ambiguous (Δ 10–30, no bar cleanly hit): **8%**
- Experiment breaks: **5%**

Composite "asymmetry is method-independent (intrinsic)": **~55%**.

Reasoning per signal: K-side should be the *easy* half — routing keys align 0.82–0.99 across passages, which says the key manifold is low-dimensional and smooth, so an H conditioned on the L0 embedding should map it even from 240 examples. V-side is the bet: if content genuinely carries only ~30% recoverable structure, H should hit the same wall. The main failure risk is H underfitting *everything* at N=300 (the 25% branch), which is why the dataset is 300 and not 50.

## Files

- `experiments/identity_ae/phase77_hypernet_kv_probe.py`
- `experiments/identity_ae/phase77_spec.md` (this file)
- `data/phase77_passages.json` (300 passages + train/held-out split + provenance)
- `models/phase77_adapters/adapter_NNN.pt` (300 absorbed adapters; gitignore if large, keep manifest)
- `models/phase77_hypernet.pt` (trained H + config)
- `results/identity_ae/phase77/kv_probe_README.md`
- `results/identity_ae/phase77/results.json` (per-passage K-side, V-side, weight-alignment records)
- `results/identity_ae/phase77/asymmetry.png` (K-recovery vs V-recovery, per passage)

## Directive for Code

1. Build the 300-passage adapter dataset: reuse the cached 50-passage `per_passage_dickens` adapters, absorb 250 more with the same recipe against the frozen 512M base (eval mode). Write `data/phase77_passages.json` with the split and provenance.
2. Train H (MIP-stabilized small MLP/Perceiver, weight + behavioral loss) on the 240 train adapters; hold out 60. Follow `training-run-management` (register/trace/monitor/stamped checkpoints).
3. Evaluate the four metrics on the 60 held-out passages, plus the random-adapter floor. Save per-passage records to `results.json` and the scatter to `asymmetry.png`.
4. After the run completes, write `results/identity_ae/phase77/kv_probe_README.md` following the Phase 65 specificity README format: headline paragraph, headline table, per-condition results, pre-committed predictions vs measured outcomes, architectural interpretation, file manifest, open questions. Then append a one-line entry to the experiments index in `~/Code/HRS/README.md` (table row, format `| experiments/identity_ae/phase77_hypernet_kv_probe.py | <one-line description with key result> |`).
5. `git add` all new/modified files (script, adapters manifest, results JSON, plot, phase README, project README), commit as `Phase 77: hypernetwork K/V asymmetry probe — <one-line result>`, and push to origin.

## Runtime budget

- Adapter dataset build: ~250 × 1.4 s ≈ 6–10 min (50 cached).
- H training: small model, ~30–90 min on the 5070 Ti.
- Evaluation + plots: minutes.

Total: **~1.5–2.5 hours**. Treat as a sanity ceiling, not a hang threshold.

## Follow-ups (do not run now; for the next spec)

- **77b — trajectory-mimicking H (HyperNet Fields, arXiv 2412.17040):** train H to follow the absorption *trajectory* logged during step 1 instead of regressing the endpoint, removing the non-canonical-target problem. Compare V-side recovery to phase 77.
- **77c — set-conditioned H against the K=2→K=4 composition ceiling:** condition H on a set of passages and emit one (or K non-interfering) adapters; test whether amortized generation evades the operator-level interference that caps composition at K=2.
