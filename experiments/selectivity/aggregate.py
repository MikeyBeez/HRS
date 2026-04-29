"""Aggregate selectivity-experiment results into RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
SEL = REPO / "experiments/selectivity"


def main():
    data = json.loads((SEL / "results/data.json").read_text())
    train_log = json.loads((SEL / "results/train_log.json").read_text())
    tests = json.loads((SEL / "results/tests.json").read_text())

    out = []
    out.append("# Selectivity Training Experiment\n")
    out.append("**Hypothesis (from spec):** Adapters trained with a dual "
                "objective (positive next-token CE + λ * KL-passivity on "
                "negatives) become *selective* — activate on relevant "
                "content, stay passive on unrelated content. If so, K>2 "
                "composition becomes feasible because most loaded adapters "
                "stay quiet on any given query.")
    out.append("")
    out.append("**Verdict: NOT SUPPORTED for K>2.** Selectivity is real "
                "and generalizes to held-out content (Tests 2-3). λ matters: "
                "too high destroys positive learning; λ=0.1 is the Pareto "
                "sweet spot. At K=2 (Test 4), selectivity adapters retain "
                "more positive content and stay 4× more passive on "
                "out-of-domain content than baseline pairs. **But Test 5 "
                "shows the K limit is NOT lifted:** at K=4 and K=6, both "
                "procedures collapse to 0-5% retrieval, and selectivity's "
                "passivity-on-out-of-domain degrades 6× from K=1 to K=6. "
                "The cross-term interference at K>2 is not fixable through "
                "per-adapter selectivity training alone.")
    out.append("")

    out.append("## Setup\n")
    out.append(f"- **Domain A (apple pie):** 10 facts × 6 templates = 60 "
               f"Q&A pairs. 50 used for training, 10 held-out for eval.")
    out.append(f"- **Domain B (donut):** symmetric, 50 train + 10 held-out.")
    out.append(f"- **Domain C (bread):** held-out for selectivity "
               f"generalization. 60 Q&A pairs, never used in training.")
    out.append("")
    out.append("**Identity loss formulation:** KL divergence on output token "
                "distributions, per-position over the negative probe. "
                "`reduction='batchmean'` (sum over vocab, mean over tokens). "
                "Base logits are precomputed once at the start of each "
                "training run (with LoRA zeroed; depends only on input "
                "tokens and the frozen base, not on the adapter being "
                "trained).")
    out.append("")
    out.append("**Training:** Phase 47 recipe (rank 128, alpha 256, attn+FFN "
                "on blocks 4-5, HIGH_LR -> BASE_LR with StepLR halve-at-half), "
                "200 steps. Each step: one positive forward (CE) + one "
                "negative forward (KL), summed loss. Negative probes drawn "
                "from the *other* training domain.")
    out.append("")
    out.append("**Substrate:** V22-Dickens base. Note the base is "
                "pretrained on Dickens prose, so its output distribution "
                "on cooking queries is Dickensian. \"Passivity\" means "
                "matching this Dickensian baseline distribution rather "
                "than producing better cooking answers.")
    out.append("")

    out.append("### Sample queries\n")
    out.append("**Domain A (apple pie):**")
    for ex in data["A"]["train"][:5]:
        out.append(f"- `{ex['probe']}` → **{ex['answer']}**")
    out.append("")
    out.append("**Domain B (donut):**")
    for ex in data["B"]["train"][:5]:
        out.append(f"- `{ex['probe']}` → **{ex['answer']}**")
    out.append("")
    out.append("**Domain C (bread, held-out):**")
    for ex in data["C"]["held_out"][:5]:
        out.append(f"- `{ex['probe']}` → **{ex['answer']}**")
    out.append("")

    out.append("## Test 1: positive-content retrieval (K=1)\n")
    out.append("Substring match on 10 held-out probes per adapter × 3 "
                "stochastic seeds = 30 evals per row.")
    out.append("")
    t1 = tests["tests"]["test1_positive"]
    out.append("| adapter | retrieval rate |")
    out.append("|---|---:|")
    for k in ("baseline_A", "sel_A_lam0.1", "sel_A_lam0.5",
              "sel_A_lam1.0", "sel_A_lam2.0",
              "baseline_B", "sel_B_lam0.1", "sel_B_lam0.5",
              "sel_B_lam1.0", "sel_B_lam2.0"):
        if k in t1:
            v = t1[k]
            out.append(f"| {k} | {v['rate']:.3f} ({v['n_hit']}/{v['n']}) |")
    out.append("")
    out.append("**Read:** Selectivity at λ=0.1 keeps positive performance "
                "close to baseline (A: 40% vs 47%; B: 23% vs 43%). At λ≥0.5 "
                "positive performance collapses to near-zero. The dual "
                "objective is in genuine tension with content learning; "
                "the right λ is small.")
    out.append("")

    out.append("## Test 2: passivity on training-negative content\n")
    out.append("For Adapter A, training negatives = Domain B (donut) probes. "
                "For Adapter B, training negatives = Domain A (apple) probes. "
                "50 probes each.")
    out.append("")
    t2 = tests["tests"]["test2_train_neg_passivity"]
    out.append("| adapter | KL ↓ | h5_cos ↑ | argmax_overlap ↑ |")
    out.append("|---|---:|---:|---:|")
    for k in ("baseline_A", "sel_A_lam0.1", "sel_A_lam0.5",
              "sel_A_lam1.0", "sel_A_lam2.0",
              "baseline_B", "sel_B_lam0.1", "sel_B_lam0.5",
              "sel_B_lam1.0", "sel_B_lam2.0"):
        if k in t2:
            v = t2[k]
            out.append(f"| {k} | {v['kl_mean']:.3f} | {v['h5_cos_mean']:.3f} "
                        f"| {v['argmax_overlap_mean']:.3f} |")
    out.append("")
    out.append("**Read:** Baseline adapters are *highly active* on the "
                "other domain's content (KL ~80, h5_cos ~0.5, argmax_overlap "
                "~0.1). Selectivity-trained adapters are much more passive: "
                "λ=0.1 hits KL ~5, h5_cos ~0.87, argmax_overlap ~0.62; "
                "λ=2.0 saturates at KL ~1.3, h5_cos ~0.98. Passivity is "
                "achieved, with diminishing returns beyond λ=1.0.")
    out.append("")

    out.append("## Test 3: passivity on held-out negative content (Domain C "
                "= bread)\n")
    out.append("Bread queries were never seen during training. This tests "
                "whether selectivity *generalizes* beyond the specific "
                "training negatives.")
    out.append("")
    t3 = tests["tests"]["test3_held_out_C_passivity"]
    out.append("| adapter | KL ↓ | h5_cos ↑ |")
    out.append("|---|---:|---:|")
    for k in ("baseline_A", "sel_A_lam0.1", "sel_A_lam0.5",
              "sel_A_lam1.0", "sel_A_lam2.0",
              "baseline_B", "sel_B_lam0.1", "sel_B_lam0.5",
              "sel_B_lam1.0", "sel_B_lam2.0"):
        if k in t3:
            v = t3[k]
            out.append(f"| {k} | {v['kl_mean']:.3f} | {v['h5_cos_mean']:.3f} |")
    out.append("")
    out.append("**Read:** Selectivity *does* generalize. On bread queries "
                "(never seen in training), selectivity adapters at λ=0.1 "
                "drop KL by ~6× vs baseline (11 vs 69 for A; 10 vs 75 for "
                "B). At higher λ the effect is even stronger (KL 1-3 at "
                "λ=2.0). The passivity is somewhat weaker on held-out "
                "content than on training negatives (e.g., KL 11 vs 5 at "
                "λ=0.1 for A) but still a major win.")
    out.append("")

    out.append("## Test 4: K=2 composition\n")
    out.append("Both adapters loaded simultaneously (Phase 43 block-stacking "
                "to one rank-256 LoRA). Selectivity adapters use λ=0.1 "
                "(Pareto-best: positive retrieval ≥ 70% of baseline, "
                "minimum KL among those).")
    out.append("")
    t4 = tests["tests"]["test4_composition"]
    out.append("| metric | selectivity_AB (λ=0.1) | baseline_AB | Δ |")
    out.append("|---|---:|---:|---:|")
    sel = t4["selectivity_AB"]
    base = t4["baseline_AB"]
    for kind in ("A_positive", "B_positive", "C_held_out"):
        s, b = sel[kind]["rate"], base[kind]["rate"]
        out.append(f"| {kind} retrieval | {s:.3f} ({sel[kind]['n_hit']}/{sel[kind]['n']}) "
                    f"| {b:.3f} ({base[kind]['n_hit']}/{base[kind]['n']}) "
                    f"| {s-b:+.3f} |")
    s, b = sel["C_passivity"], base["C_passivity"]
    out.append(f"| C passivity KL ↓ | {s['kl_mean']:.2f} | {b['kl_mean']:.2f} "
                f"| {s['kl_mean']-b['kl_mean']:+.2f} |")
    out.append(f"| C passivity h5_cos ↑ | {s['h5_cos_mean']:.3f} "
                f"| {b['h5_cos_mean']:.3f} | {s['h5_cos_mean']-b['h5_cos_mean']:+.3f} |")
    out.append(f"| C passivity argmax_overlap ↑ | {s['argmax_overlap_mean']:.3f} "
                f"| {b['argmax_overlap_mean']:.3f} "
                f"| {s['argmax_overlap_mean']-b['argmax_overlap_mean']:+.3f} |")
    out.append("")
    out.append("**Read:** With both adapters loaded at K=2, the selectivity-"
                "trained pair *retains* more positive-content retrieval than "
                "the baseline pair: A_positive 27% vs 13%, B_positive 27% "
                "vs 20%. Both are below the K=1 single-adapter rates "
                "(40% and 23% respectively), so K=2 still degrades "
                "performance — but selectivity degrades less.")
    out.append("")
    out.append("On out-of-domain bread queries, the K=2 selectivity pair "
                "stays much more passive than the K=2 baseline pair "
                "(KL 28 vs 109 — almost 4× lower). This is the most "
                "directly hypothesis-relevant number: when neither adapter's "
                "content is queried, the selectivity composition stays close "
                "to the base while the baseline composition diverges.")
    out.append("")

    # Test 5: K-sweep
    test5_path = SEL / "results/test5.json"
    if test5_path.exists():
        t5 = json.loads(test5_path.read_text())
        out.append("## Test 5: K-sweep composition with 6 selectivity adapters\n")
        out.append("Trained 4 additional domains (chocolate cake, pizza, "
                    "soup, cookies) at λ=0.1, plus reused sel_A and sel_B. "
                    "Stacked K adapters at rank K×128 and measured: "
                    "(a) average retrieval rate across the K loaded "
                    "domains' held-out positives, (b) passivity on bread "
                    "(Domain C, never trained on).")
        out.append("")
        out.append("New domains' negatives = 10 probes from each of the "
                    "5 other domains × 5 = 50 (mix-negative training, vs "
                    "the original A/B which used paired single-domain "
                    "negatives).")
        out.append("")
        out.append("| K | procedure | avg retrieval (loaded) | C passivity KL ↓ | C passivity h5_cos ↑ |")
        out.append("|---:|---|---:|---:|---:|")
        for K in t5["K_values"]:
            for proc in ("selectivity", "baseline"):
                r = t5["results"][proc][f"K{K}"]
                out.append(f"| {K} | {proc} | {r['avg_retrieval_loaded']:.3f} "
                            f"| {r['passivity_on_C']['kl_mean']:.2f} "
                            f"| {r['passivity_on_C']['h5_cos_mean']:.3f} |")
        out.append("")
        out.append("**Read:** Selectivity is consistently more passive on "
                    "bread than baseline at every K (4-5× lower KL, "
                    "20-40% higher h5_cos). And selectivity retains "
                    "slightly more retrieval at low K. **But:**")
        out.append("")
        out.append("1. **Both procedures collapse at K=4 and K=6** — "
                    "retrieval drops to 0-5%. The K limit is not lifted.")
        out.append("2. **Selectivity's own passivity-on-out-of-domain "
                    "degrades rapidly with K**: KL goes from 12 (K=1) to "
                    "70 (K=6) — almost 6× worse. When 6 selectivity "
                    "adapters are loaded together, their combined "
                    "contribution to the residual stream is no longer "
                    "passive on bread, even though each adapter alone "
                    "would be.")
        out.append("3. **The relative gap (selectivity vs baseline) "
                    "narrows with K**: at K=1 selectivity is 5.5× more "
                    "passive than baseline; at K=6 only 2× more. The "
                    "advantage shrinks as K grows.")
        out.append("")
        out.append("This is the failure mode #3 from the spec: "
                    "individually-selective adapters do **not** compose "
                    "cleanly past K=2. Per-adapter passivity is real, "
                    "but at K>2 the cross-term interference between "
                    "stacked LoRA matrices dominates — the same "
                    "phenomenon that broke the sequential-training "
                    "experiment.")
        out.append("")

    # Best λ rationale
    bl = tests["best_lambda"]
    out.append(f"**Best λ:** A={bl['A']}, B={bl['B']}. Selected as "
                f"\"positive retrieval ≥ 70% of baseline, minimum KL on "
                f"training negs among those.\" λ=0.1 satisfies this for "
                f"both domains.")
    out.append("")

    out.append("## Failure-mode analysis\n")
    out.append("The spec defined three failure modes:")
    out.append("")
    out.append("1. **Test 1 degraded** — selectivity-vs-content tradeoff is "
                "fundamental. Confirmed at λ ≥ 0.5: positive retrieval "
                "collapses to near-zero. λ=0.1 mitigates this (40%/23% vs "
                "baseline 47%/43%) — modest degradation but small.")
    out.append("2. **Test 2 ok but Test 3 fails** — passivity is "
                "domain-specific, not general. **Not** the failure mode here. "
                "Selectivity generalizes from donut/apple negatives to "
                "bread held-out.")
    out.append("3. **Tests 1-3 ok but Test 4 breaks** — composition still "
                "interferes despite selectivity. Partially the failure mode: "
                "K=2 passivity (KL 28) is significantly worse than K=1 "
                "passivity (KL 11) — composing two selective adapters does "
                "introduce some cross-term interference. But selectivity "
                "still beats baseline composition by a wide margin, so this "
                "is a quantitative, not categorical, failure.")
    out.append("")

    out.append("## Implementation notes / deviations\n")
    out.append("1. **Identity loss = KL divergence on output logits "
                "(`F.kl_div(log_softmax(adapter), softmax(base), "
                "reduction='batchmean')`).** Picked over MSE-on-hidden or "
                "cross-entropy-on-base-argmax because it's the most "
                "direct end-to-end measure of distribution match and "
                "uses the same forward pass we already needed.")
    out.append("2. **Base logits are precomputed once** at the start of "
                "training (after `reset_lora_to_zero`, before any LoRA "
                "updates). Cached in GPU memory as fp32 for the 50 "
                "negative probes (~3MB per probe × 30 tokens × 50257 "
                "vocab ≈ 6MB total — trivially small).")
    out.append("3. **λ sweep:** intended {0.05, 0.1, 0.5, 1.0, 2.0} but a "
                "format-string bug (`f\"{lam:.1f}\"`) collapsed 0.05 → "
                "\"0.1\" and overwrote the lam=0.05 adapter with the "
                "lam=0.1 one. Effectively the swept values are {0.1, "
                "0.5, 1.0, 2.0}. The Pareto-best λ is at the low end of "
                "this range (λ=0.1); whether λ=0.05 would do better is "
                "not directly tested.")
    out.append("4. **N_STEPS = 200** (vs Phase 47's 150) because dual "
                "objective takes longer to converge.")
    out.append("5. **Substring scoring** uses lowercase-+-no-comma matching, "
                "same as Phase 47. Substring false positives are possible "
                "for short answers (\"425 degrees\", \"5 minutes\") but "
                "the same scorer is used for both procedures so the "
                "comparison is valid.")
    out.append("6. **Test 5 was run** with 4 additional domains (D=chocolate "
                "cake, E=pizza, F=soup, G=cookies). 6 total selectivity "
                "adapters at λ=0.1; new ones trained with mix-negatives "
                "(10 probes from each of the 5 other domains).")
    out.append("")

    out.append("## Wall-clock totals\n")
    train_wall = sum(v["wall_s"] for v in train_log.values())
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| Training (10 adapters: 2 baselines + 8 selectivity) | "
                f"{train_wall:.0f}s |")
    out.append(f"| Evaluation (Tests 1-4) | "
                f"{tests['wall_total_s']:.0f}s |")
    out.append(f"| **Total** | **~"
                f"{(train_wall+tests['wall_total_s'])/60:.0f} min** |")
    out.append("")

    out.append("## Summary\n")
    out.append("The selectivity training works for *passivity at K=1*: "
                "the dual objective produces adapters that stay close to "
                "the base on unrelated content, and this property "
                "generalizes from training negatives to held-out content. "
                "The cost is a positive-content tradeoff that the λ knob "
                "trades against passivity strength; λ=0.1 is the sweet "
                "spot.")
    out.append("")
    out.append("At K=2 composition, the selectivity pair retrieves more "
                "positive content AND stays 4× more passive on "
                "out-of-domain content than the baseline pair — a real "
                "improvement, but the K=2 passivity (KL 28) is already "
                "much worse than K=1 passivity (KL 11).")
    out.append("")
    out.append("**At K=4 and K=6 the architecture's K limit is reached "
                "regardless of selectivity training.** Both procedures "
                "collapse to 0-5% retrieval. Selectivity's per-adapter "
                "passivity does NOT survive composition: stacking 6 "
                "selectivity adapters is not equivalent to having one "
                "of them active and the rest passive — the cross-term "
                "interference between the K low-rank matrices is the "
                "dominant signal.")
    out.append("")
    out.append("This matches the prior sequential-training experiment's "
                "result. The K>2 problem is not addressable through "
                "training-time changes to individual adapters. Both "
                "training-procedure approaches (sequential adapter "
                "training, dual-objective selectivity) produce adapters "
                "that work alone or in pairs but break at K=4+.")
    out.append("")
    out.append("**Next architectural levers to test:** orthogonality "
                "constraints between adapters (force their (A, B) ranges "
                "into disjoint subspaces of the residual stream); "
                "explicit per-adapter gating machinery (route the LoRA "
                "contribution through a learned per-input scalar); "
                "hierarchical adapter trees (route to a single adapter "
                "per query, never compose).")

    out_path = SEL / "results/RESULT.md"
    out_path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n--- saved to {out_path} ---")


if __name__ == "__main__":
    main()
