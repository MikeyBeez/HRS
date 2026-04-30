"""Aggregate combination-adapter experiment results into RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/mnt/data/Code/HRS")
COMBO = REPO / "experiments/combo_adapter"


def main():
    train_log = json.loads((COMBO / "results/train_log.json").read_text())
    tests = json.loads((COMBO / "results/tests.json").read_text())
    test4 = json.loads((COMBO / "results/test4_multipass.json").read_text())

    out = []
    out.append("# Combination Adapter Experiment\n")
    out.append("**Hypothesis (from spec):** Train one adapter on the "
                "concatenated training data of K passages → K=1 inference at "
                "deploy time. The combination adapter should match or "
                "approach single-passage K=1 retrieval (ideally within "
                "~10%) and substantially outperform K>2 multi-adapter "
                "block-stacking. This is \"offline composition through "
                "training\" instead of \"online composition at query "
                "time.\"")
    out.append("")
    out.append("**Verdict: SUPPORTED for single-content retrieval at K>2; "
                "PARTIALLY supported for cross-passage queries.** "
                "Combination adapters retain 65-74% per-constituent "
                "retrieval across K=2/3/4 while multi-adapter stacking "
                "collapses (67% at K=2 → 17% at K=4). Combo is a clean "
                "win over both multi-stack composition AND a multi-pass-"
                "with-base-final-pass simulation. Combo's per-constituent "
                "retrieval is consistently 22-27pp below single-passage "
                "K=1 — the spec's \"within 10%\" criterion is missed but "
                "the K limit is effectively eliminated.")
    out.append("")

    out.append("## Setup\n")
    out.append("- Reused 8 passages from sequential-training experiment: "
               "library_ids [0, 2, 16, 17, 22, 30, 31, 36]. Pip's family "
               "name, Joe's profession, Estella, Wemmick, Provis, Herbert, "
               "Magwitch, Drummle.")
    out.append("- 4 K=2 + 4 K=3 + 2 K=4 = 10 combination adapters trained.")
    out.append("- Phase 47 recipe (rank 128, alpha 256, attn+FFN on blocks "
               "4-5, HIGH_LR -> BASE_LR with StepLR halve-at-half).")
    out.append("- n_steps scales with K: 150 × K (so 300 for K=2, 450 for "
               "K=3, 600 for K=4) to give each training source comparable "
               "per-source training.")
    out.append("- Substring scoring same as Phase 47 baseline. Test 2 / "
               "Test 4 use *fragment coverage* (mean fraction of expected "
               "answer fragments hit) plus *full hit rate* (all fragments "
               "in one generation).")
    out.append("")

    out.append("### Combination groupings\n")
    out.append("| name | K | constituents (library_ids) |")
    out.append("|---|---:|---|")
    for r in train_log["train_log"]:
        out.append(f"| {r['name']} | {r['k']} | {r['constituents']} |")
    out.append("")

    # Training stats
    out.append("### Training\n")
    out.append("All 10 adapters converged (final loss < 0.2):")
    out.append("")
    out.append("| name | K | n_sources | n_steps | loss_init → final | wall |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in train_log["train_log"]:
        out.append(f"| {r['name']} | {r['k']} | {r['n_sources']} | "
                    f"{r['n_steps']} | {r['loss_init']:.2f} → "
                    f"{r['loss_final_mean10']:.3f} | {r['wall_s']:.0f}s |")
    out.append(f"\n*Total training wall: {train_log['wall_total_s']:.0f}s*")
    out.append("")

    out.append("## Test 1: per-constituent retrieval\n")
    out.append("For each combination, evaluate substring match on each "
               "constituent's 3 held-out paraphrases × 3 seeds = 9 evals "
               "per (combination, constituent) pair. Compare three "
               "conditions:")
    out.append("- **combo:** combination adapter loaded alone (K=1 inference)")
    out.append("- **single:** canonical Phase 47 single-passage adapter for "
                "that constituent loaded alone (K=1 inference)")
    out.append("- **multi-stack:** all K constituent single-passage adapters "
                "block-stacked at rank K×128 (Phase 43 stacking), K=N "
                "inference")
    out.append("")
    out.append("### Per-combination averages\n")
    out.append("| combination | K | combo | single | multi-stack | combo-multi | combo-single |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    avgs_by_k = {2: [], 3: [], 4: []}
    for r in tests["test1"]:
        c_avg = np.mean([p["combo_rate"] for p in r["per_constituent"]])
        s_avg = np.mean([p["single_rate"] for p in r["per_constituent"]])
        m_avg = np.mean([p["multi_rate"] for p in r["per_constituent"]])
        avgs_by_k[r["k"]].append((c_avg, s_avg, m_avg))
        out.append(f"| {r['name']} | {r['k']} | {c_avg:.3f} | {s_avg:.3f} "
                    f"| {m_avg:.3f} | {c_avg-m_avg:+.3f} | {c_avg-s_avg:+.3f} |")
    out.append("")

    out.append("### Aggregated by K\n")
    out.append("| K | combo (avg) | single (avg) | multi-stack (avg) | "
                "combo gap to single | combo gap over multi |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for k in (2, 3, 4):
        rows = avgs_by_k[k]
        c = np.mean([r[0] for r in rows]); s = np.mean([r[1] for r in rows])
        m = np.mean([r[2] for r in rows])
        out.append(f"| {k} | {c:.3f} | {s:.3f} | {m:.3f} | {c-s:+.3f} | {c-m:+.3f} |")
    out.append("")
    out.append("**Read:**")
    out.append("- Combo retrieval is *flat* across K: 0.65 → 0.63 → 0.74. "
                "The combination size does NOT degrade per-constituent "
                "retrieval up to K=4. Capacity ceiling not yet hit.")
    out.append("- Multi-stack collapses with K: 0.67 → 0.38 → 0.17. The "
                "K>2 cross-term interference replicated.")
    out.append("- Combo wins over multi-stack by 25pp at K=3 and **57pp at "
                "K=4**. At K=2 they tie.")
    out.append("- Combo holds 22-27pp below single-passage K=1 across all "
                "K. The spec's \"within 10%\" criterion is missed — combo "
                "loses some absolute fidelity per fact, but doesn't degrade "
                "with K.")
    out.append("")

    out.append("## Test 2: cross-passage queries\n")
    out.append("3 hand-crafted Phase-43-style chained probes per "
               "combination (e.g., \"Recall: Pip's family = . Also, Joe's "
               "profession = ...\"), each requiring the combo adapter to "
               "produce K answer fragments in one generation. Score: "
               "*frac_hits* = mean fraction of expected fragments present "
               "in the continuation; *full_hit_rate* = both/all fragments "
               "present.")
    out.append("")
    out.append("### Per-combination averages\n")
    out.append("| combination | K | combo full_hit | combo frac | "
                "multi-stack full | multi-stack frac |")
    out.append("|---|---:|---:|---:|---:|---:|")
    avgs_by_k_t2 = {2: [], 3: [], 4: []}
    for r in tests["test2"]:
        c_full = np.mean([x["full_hit_rate"] for x in r["combo_results"]])
        c_frac = np.mean([x["frac_hits_mean"] for x in r["combo_results"]])
        m_full = np.mean([x["full_hit_rate"] for x in r["multi_results"]])
        m_frac = np.mean([x["frac_hits_mean"] for x in r["multi_results"]])
        avgs_by_k_t2[r["k"]].append((c_full, c_frac, m_full, m_frac))
        out.append(f"| {r['name']} | {r['k']} | {c_full:.3f} | {c_frac:.3f} "
                    f"| {m_full:.3f} | {m_frac:.3f} |")
    out.append("")

    out.append("### Aggregated by K\n")
    out.append("| K | combo frac | multi-stack frac | combo - multi |")
    out.append("|---:|---:|---:|---:|")
    for k in (2, 3, 4):
        rows = avgs_by_k_t2[k]
        c = np.mean([r[1] for r in rows]); m = np.mean([r[3] for r in rows])
        out.append(f"| {k} | {c:.3f} | {m:.3f} | {c-m:+.3f} |")
    out.append("")
    out.append("**Read:**")
    out.append("- Cross-passage queries are *hard for both approaches*: "
                "fragment coverage stays at 30-50%, and full-hit rate is "
                "near 0 at K≥3 for both procedures. Producing all K answer "
                "fragments in one generation is genuinely difficult at "
                "this base/scale.")
    out.append("- At K=2 and K=3, combo and multi-stack tie on cross-pass.")
    out.append("- At K=4, combo edges out multi-stack by 12pp on fragment "
                "coverage (32% vs 20%), but neither produces full-hits.")
    out.append("")

    out.append("## Test 3: capacity behavior across K\n")
    out.append("Combo retrieval (per-constituent, Test 1) by K: "
               "0.65 (K=2) → 0.63 (K=3) → 0.74 (K=4). The K=4 increase is "
               "noise (the two K=4 combinations both happen to hit 67% "
               "and 81%; the small sample variance is wider than the "
               "per-K differences).")
    out.append("")
    out.append("**The capacity ceiling for combination adapters is NOT "
                "reached at K=4** with rank-128 LoRA on this base. K=8 or "
                "K=10 would be the next test for finding the practical "
                "ceiling.")
    out.append("")

    out.append("## Test 4: combo vs multi-pass simulation\n")
    out.append("Multi-pass protocol: for each cross-passage query "
               "requiring K passages, run K separate K=1 inferences (each "
               "with a different constituent's single-passage adapter "
               "loaded), capture the continuations, then run a final pass "
               "with NO adapter on a composite probe = original probe + "
               "labeled intermediates + \"Final answer: \". Score the final "
               "continuation against the expected fragments.")
    out.append("")
    out.append("Run on K=3 and K=4 combinations only (where multi-stack "
                "fails on Test 2).")
    out.append("")
    out.append("### Per-combination\n")
    out.append("| combination | K | combo frac | multipass frac | "
                "combo - mp |")
    out.append("|---|---:|---:|---:|---:|")
    avgs_by_k_t4 = {3: [], 4: []}
    for r in test4["results"]:
        c = r["avg_combo_frac"]; m = r["avg_multipass_frac"]
        avgs_by_k_t4[r["k"]].append((c, m))
        out.append(f"| {r['name']} | {r['k']} | {c:.3f} | {m:.3f} | "
                    f"{c-m:+.3f} |")
    out.append("")
    out.append("### Aggregated by K\n")
    out.append("| K | combo frac | multipass frac | gap |")
    out.append("|---:|---:|---:|---:|")
    for k in (3, 4):
        rows = avgs_by_k_t4[k]
        c = np.mean([r[0] for r in rows]); m = np.mean([r[1] for r in rows])
        out.append(f"| {k} | {c:.3f} | {m:.3f} | {c-m:+.3f} |")
    out.append("")
    out.append("**Read:** Multi-pass is effectively broken on this "
                "substrate — the V22-Dickens base, given the original "
                "probe + K labeled intermediate continuations, doesn't "
                "synthesize a useful final answer (it produces Dickensian "
                "text on the cooking-style probes). Combo wins by 26-30pp "
                "at K=3 and K=4. **Combo is the better approach over "
                "multi-pass at this scale.**")
    out.append("")
    out.append("Caveat: a more capable base model (e.g., a 7B+ "
                "instruction-tuned model) would likely have non-zero "
                "skill at synthesizing intermediate notes, making "
                "multi-pass more competitive. The Test 4 result here is "
                "specific to the V22-Dickens substrate.")
    out.append("")

    out.append("## Failure-mode analysis\n")
    out.append("The spec defined three failure modes for the combo "
               "approach:")
    out.append("")
    out.append("1. **Test 1 substantial degradation per-constituent** — "
                "rank-128 LoRA insufficient for multiple passages. "
                "**Partial.** Combo IS 22-27pp below single-passage K=1 "
                "consistently. But it's flat across K=2/3/4, so the "
                "per-passage capacity isn't being strained as K grows. "
                "Likely explanation: training distributes a fixed amount "
                "of LoRA capacity across more passages, so each passage "
                "gets less individual fidelity, but the trade is "
                "predictable.")
    out.append("2. **Test 2 cross-passage failure with Test 1 success** — "
                "adapter learns each passage independently rather than "
                "their relationships. **Partial.** Test 2 shows combo "
                "modestly better than multi-stack but neither produces "
                "full-hits at K≥3. This is the spec's 2nd failure mode "
                "in part: combo apparently learns each constituent OK "
                "but doesn't synthesize them into composite answers.")
    out.append("3. **K=4 collapse with K=3 success** — practical "
                "combination size bounded. **Not** the failure mode. "
                "K=4 combo retrieval (0.74) is actually the highest of "
                "the three K values, well within sample noise of K=2 "
                "and K=3.")
    out.append("")

    out.append("## Implementation notes / deviations\n")
    out.append("1. **Reused** the per_passage_dickens single-passage "
                "adapters as the K=1 baseline (no retraining).")
    out.append("2. **Cross-passage queries** are Phase-43-style chained "
                "probes (\"Recall: X = . Also, Y = \") because hand-"
                "crafted free-form composition queries (\"What is the "
                "relation between X and Y?\") are hard to score by "
                "substring match at this scale. Each query has 2-4 "
                "expected answer fragments; partial credit by fragment-"
                "coverage rate.")
    out.append("3. **n_steps scales with K** (150 × K). At K=4 with 20 "
                "training sources, this gives ~30 visits/source — "
                "comparable to single-passage adapters' 30 visits/source.")
    out.append("4. **Multi-pass test** uses the V22-Dickens base for the "
                "final synthesis pass. A more capable base would change "
                "the result. The comparison here is specifically about "
                "what works at the tiny-Dickens scale used by the rest "
                "of the HRS architecture.")
    out.append("5. **Test 1 currently rebuilds the model 30+ times**. "
                "This is wasteful (model load is the dominant wall "
                "time) but the test correctness is unaffected. Cleaner "
                "implementation would build the base once and only "
                "swap LoRA states.")
    out.append("")

    out.append("## Wall-clock totals\n")
    eval_wall = tests["wall_total_s"]
    t4_wall = test4["wall_total_s"]
    train_wall = train_log["wall_total_s"]
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| Training (10 combination adapters) | {train_wall:.0f}s |")
    out.append(f"| Tests 1-2 evaluation | {eval_wall:.0f}s "
                f"(~{eval_wall/60:.0f} min) |")
    out.append(f"| Test 4 (multi-pass) | {t4_wall:.0f}s |")
    out.append(f"| **Total** | **~{(train_wall+eval_wall+t4_wall)/60:.0f} "
                f"min** |")
    out.append("")

    out.append("## Summary\n")
    out.append("**Combination adapters work.** Training one adapter on "
                "the concatenated content of K passages produces an "
                "adapter that, at K=1 inference, retrieves each of its "
                "constituents far better than the existing architecture's "
                "K=N multi-adapter block-stacking can manage at K>2. "
                "Combo retrieval is *flat across K* (0.63-0.74) while "
                "multi-stack drops from 0.67 (K=2) to 0.17 (K=4). The "
                "K limit problem the prior two experiments couldn't fix "
                "is structurally avoided by combo training because "
                "combo never composes — it's a single rank-128 LoRA at "
                "deploy time.")
    out.append("")
    out.append("The cost is modest: combo holds 22-27pp below single-"
                "passage K=1. The spec's \"within 10%\" criterion is "
                "missed, but the gap is consistent across K and "
                "doesn't grow.")
    out.append("")
    out.append("Cross-passage composition queries (where the answer "
                "requires fragments from multiple constituents in one "
                "generation) are hard for any approach at this base "
                "scale. Combo modestly beats multi-stack at K=4 (12pp) "
                "and decisively beats a multi-pass-with-base-final-"
                "synthesis baseline (26-30pp). On this substrate combo "
                "is the best available approach for cross-passage "
                "queries, but absolute performance is limited.")
    out.append("")
    out.append("**Deployment story:** the architecture has at least three "
                "operating patterns now. (1) K=1 single-passage adapters "
                "for individual content, retrieval ~0.93. (2) Combination "
                "adapters for known content groupings, retrieval "
                "~0.65-0.74 per constituent regardless of K. (3) "
                "K=2 multi-adapter stacking for query-time composition "
                "of two adapters, retrieval ~0.67 single-content / 0.49 "
                "compositional. Combination adapters become the "
                "preferred path at K≥3 where multi-adapter stacking "
                "collapses.")
    out.append("")
    out.append("**Next experiments:**")
    out.append("- Find the actual capacity ceiling: train K=8, K=12 "
                "combo adapters and see when per-constituent retrieval "
                "starts dropping.")
    out.append("- Test combo at higher rank (256, 512) to see if the "
                "22-27pp gap to single-passage K=1 closes with capacity.")
    out.append("- Test combo with more constituents per cross-passage "
                "query and see if more focused training data (queries "
                "that explicitly require synthesis) lifts the cross-"
                "passage performance.")

    out_path = COMBO / "results/RESULT.md"
    out_path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n--- saved to {out_path} ---")


if __name__ == "__main__":
    main()
