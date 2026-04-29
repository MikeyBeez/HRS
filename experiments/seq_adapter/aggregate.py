"""Aggregate sequential-adapter experiment results into a single RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
RES = REPO / "experiments/seq_adapter/results"


def fmt(x):
    return f"{x:.3f}" if x is not None else "—"


def main():
    case1 = json.loads((RES / "case1.json").read_text())
    case2 = json.loads((RES / "case2.json").read_text())
    proc_b_log = json.loads((RES / "proc_b_train_log.json").read_text())

    out = []
    out.append("# Sequential Adapter Training Experiment")
    out.append("")
    out.append("**Hypothesis (from spec): Procedure B (sequential training, "
                "with prior adapters frozen and active) produces adapters that "
                "compose more cleanly past K=2 than Procedure A (independent "
                "training).**")
    out.append("")
    out.append("**Verdict: Hypothesis NOT supported.** Procedure B's adapters "
                "are *context-dependent* — they only retrieve when their "
                "training-time context (prior adapters baked in) is approximately "
                "reproduced. At K=2 (the regime where Procedure A works), "
                "Procedure B collapses below A's K=2; at K=4 and K=8, Procedure "
                "B partially recovers but never surpasses A's K=2 baseline.")
    out.append("")

    out.append("## Setup")
    out.append("")
    out.append(f"- 8 distinct Dickens passages chosen from per_passage_dickens "
               f"library: indices {proc_b_log['chosen_ids']}.")
    out.append(f"- Procedure A: reuse the canonical Phase 47 adapters for "
               f"these 8 indices (rank 128, L45 targets).")
    out.append(f"- Procedure B: train 8 adapters sequentially. Adapter k is "
               f"trained with adapters 0..k-1 baked into the V22-Dickens base "
               f"weights, so the new adapter learns corrections on top of the "
               f"prior modified forward pass. Total wall: "
               f"{proc_b_log['wall_total_s']:.0f}s.")
    out.append(f"- Composition: Phase 43-style block-stacking. K rank-128 "
               f"adapters → one rank-(K×128) state dict via "
               f"`stack_k_state_dicts`. Same scoring (substring match, 3 "
               f"stochastic seeds, T=0.8, top_k=50, 30 generated tokens).")
    out.append("")

    # Procedure B training sanity
    out.append("### Procedure B training sanity")
    out.append("")
    out.append("All 8 sequential adapters retrieved their target answer on "
                "greedy generation *during their training context* (= prior "
                "adapters baked in):")
    out.append("")
    out.append("| local k | library_id | answer | greedy hit | wall |")
    out.append("|---:|---:|---|---|---:|")
    for r in proc_b_log["log"]:
        out.append(f"| {r['local_k']} | {r['library_id']} | "
                    f"{r['answer']} | {r['greedy_hit']} | {r['wall_s']:.0f}s |")
    out.append("")

    # Case 1
    out.append("## Case 1: single-adapter retrieval at K ∈ {1, 2, 4, 8}")
    out.append("")
    out.append("24 questions = 8 adapters × 3 held-out paraphrases × 3 stochastic "
                "seeds = 72 evals per cell.")
    out.append("")
    out.append("Adapter subset: relevant adapter i + (i+1)%8 + ... + "
                "(i+K-1)%8 (deterministic offset).")
    out.append("")
    out.append("| K | Procedure A | Procedure B | Δ (B - A) |")
    out.append("|---:|---:|---:|---:|")
    for ka, kb in zip(case1["results"]["A"], case1["results"]["B"]):
        delta = kb["retrieval"] - ka["retrieval"]
        sign = "+" if delta >= 0 else ""
        out.append(f"| {ka['K']} | {ka['retrieval']:.3f} ({ka['n_correct']}/{ka['n_total']}) | "
                    f"{kb['retrieval']:.3f} ({kb['n_correct']}/{kb['n_total']}) | "
                    f"{sign}{delta:+.3f} |")
    out.append("")

    # Case 2
    out.append("## Case 2: 12 hand-crafted composition queries at K ∈ {2, 4, 8}")
    out.append("")
    out.append("Each query has two target answer fragments (from passages a "
                "and b). Score: hit_a, hit_b, both. **`rate_both`** is the "
                "headline metric — both answers retrieved in the same "
                "generation.")
    out.append("")
    out.append("Adapter subset for K=4: [a, b] + 2 distractors (round-robin "
                "from remaining indices). K=8: all 8 adapters.")
    out.append("")
    out.append("### Procedure A (independent training)")
    out.append("")
    out.append("| K | rate_a | rate_b | rate_both |")
    out.append("|---:|---:|---:|---:|")
    for r in case2["results"]["A"]:
        out.append(f"| {r['K']} | {r['rate_a']:.3f} | {r['rate_b']:.3f} | "
                    f"**{r['rate_both']:.3f}** |")
    out.append("")
    out.append("### Procedure B (sequential training)")
    out.append("")
    out.append("| K | rate_a | rate_b | rate_both |")
    out.append("|---:|---:|---:|---:|")
    for r in case2["results"]["B"]:
        out.append(f"| {r['K']} | {r['rate_a']:.3f} | {r['rate_b']:.3f} | "
                    f"**{r['rate_both']:.3f}** |")
    out.append("")

    # Side-by-side rate_both comparison
    out.append("### Side-by-side rate_both")
    out.append("")
    out.append("| K | Procedure A | Procedure B | Δ (B - A) |")
    out.append("|---:|---:|---:|---:|")
    for ka, kb in zip(case2["results"]["A"], case2["results"]["B"]):
        d = kb["rate_both"] - ka["rate_both"]
        sign = "+" if d >= 0 else ""
        out.append(f"| {ka['K']} | {ka['rate_both']:.3f} | "
                    f"{kb['rate_both']:.3f} | {sign}{d:+.3f} |")
    out.append("")

    out.append("## Summary")
    out.append("")
    out.append("Procedure A reproduces the documented pattern: clean K=2 "
               "composition (47% both-hit on hand-crafted compositional "
               "queries; 68% single-adapter retrieval at K=2), collapsing at "
               "K=4 (0% both-hit, 14% single-adapter retrieval) and K=8 (0% "
               "everywhere).")
    out.append("")
    out.append("Procedure B fails to fix the K=2 ceiling. Worse, it breaks "
               "the K=1 baseline: when an adapter is loaded alone, retrieval "
               "drops to 42% (vs Procedure A's 88%). The sequential adapters "
               "are specific to the composition context they were trained in. "
               "When prior adapters aren't loaded (K=1), the adapter's "
               "contributions misapply to a base it wasn't designed for.")
    out.append("")
    out.append("Procedure B's K=4 and K=8 retrieval rates are slightly "
               "*higher* than its K=1 — consistent with the interpretation "
               "that adding more adapters partially reconstructs the training "
               "context.")
    out.append("")
    out.append("This is informative-negative for the K=2 problem. The "
               "interference between independently-trained LoRA matrices at "
               "K>2 is **not** addressable by training-time freezing alone. "
               "Alternative approaches — orthogonality constraints, gating, "
               "hierarchical adapters — should be tested next.")
    out.append("")

    out.append("## Implementation notes / deviations")
    out.append("")
    out.append("1. **Procedure B training trick:** rather than wrap the model "
                "with a rank-(k×128) LoRA holding k-1 frozen and 1 trainable, "
                "I bake adapters 0..k-1 into the wrapped Linear weights "
                "(`W += scaling * (A @ B).T`), reset LoRA to fresh state, and "
                "train the new adapter. This gives the same forward-pass "
                "context the spec describes (\"adapter 2 sees the adapter-1-"
                "modified forward pass\") with simpler bookkeeping. The "
                "canonical base is restored before evaluation.")
    out.append("2. **Hyperparameters:** Phase 47's recipe (rank 128, alpha "
                "256, n_steps 150, HIGH_LR→BASE_LR StepLR halving). All 8 "
                "Procedure B adapters converged to greedy retrieval *in their "
                "training context*; no divergence behavior observed.")
    out.append("3. **Block-stacking for evaluation:** Phase 43's "
                "`stack_k_state_dicts` block-diagonal-concatenates the K "
                "rank-128 (A, B) pairs into a single rank-(K×128) (A, B). "
                "This makes the wrapped Linear's forward equivalent to "
                "summing the K LoRA contributions: "
                "`x @ A_stacked @ B_stacked = Σₖ x @ Aₖ @ Bₖ`.")
    out.append("4. **Composition queries (Case 2):** 12 hand-constructed "
                "Phase 43-style chained probes, each requiring content from "
                "two specific adapters. Listed verbatim in `queries.py`. The "
                "answers are unambiguous proper nouns or single common words "
                "(\"Pirrip\", \"Estella\", \"blacksmith\", \"Drummle\", "
                "\"Provis\", \"Magwitch\", \"Wemmick\", \"Herbert\").")
    out.append("5. **No hyperparameter search** for Procedure B was attempted. "
                "The spec explicitly allowed re-framing as \"establish what "
                "hyperparameters sequential training needs\" if convergence "
                "failed. Convergence didn't fail; the failure is at "
                "evaluation time, not training time. A different LR schedule "
                "wouldn't change the context-dependence.")
    out.append("")

    out.append("## Wall-clock totals")
    out.append("")
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| Procedure B training (8 sequential adapters) | "
                f"{proc_b_log['wall_total_s']:.0f}s |")
    out.append(f"| Case 1 evaluation (4 K-values × 2 procedures × "
                f"72 evals) | {case1['wall_total_s']:.0f}s |")
    out.append(f"| Case 2 evaluation (3 K-values × 2 procedures × "
                f"36 evals) | {case2['wall_total_s']:.0f}s |")
    out.append(f"| **Total** | **~"
                f"{proc_b_log['wall_total_s']+case1['wall_total_s']+case2['wall_total_s']:.0f}s "
                f"(~{(proc_b_log['wall_total_s']+case1['wall_total_s']+case2['wall_total_s'])/60:.0f} min)** |")
    out.append("")

    out_path = RES / "RESULT.md"
    out_path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n--- saved to {out_path} ---")


if __name__ == "__main__":
    main()
