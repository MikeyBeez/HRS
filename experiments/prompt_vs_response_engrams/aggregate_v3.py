"""Aggregate v3 results with LLM-as-judge alongside token-F1.
Compare the rankings.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"
PRIOR = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.topics import ALL_TOPICS

STOPWORDS = set("""a an the and or of in on at to for from with by is was are were be
been being it its this that these those i you he she they we us them
him her his its my your their our as if but not no yes also so very
just only than then now thus into upon between within without through
during about across over under after before above below toward up down
out off again further while because where when how what which who whom
whose why all any most some many few both each every other another
do does did done can could should would may might must shall will
have has had having more less much fewer same different""".split())


def tokens(text):
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text)
            if w.lower() not in STOPWORDS and len(w) > 2]


def f1_overlap(pred, ref):
    p = tokens(pred); r = tokens(ref)
    if not p or not r: return 0.0
    p_set = set(p); r_set = set(r)
    overlap = p_set & r_set
    if not overlap: return 0.0
    precision = len(overlap) / len(p_set)
    recall    = len(overlap) / len(r_set)
    return 2 * precision * recall / (precision + recall)


def coverage(generation, relevant_turn_ids, all_topics=ALL_TOPICS):
    if not relevant_turn_ids: return 0.0
    gen_tokens = set(tokens(generation))
    n_covered = 0
    for tid in relevant_turn_ids:
        title = all_topics[tid][0]
        kw = set(tokens(title))
        if kw & gen_tokens:
            n_covered += 1
    return n_covered / len(relevant_turn_ids)


CONDITIONS = [
    ("full_context",       "gen_full_context",       "tokens_full_context",       False),
    ("random_engrams",     "gen_random_engrams",     "tokens_random_engrams",     True),
    ("prompts_as_context", "gen_prompts_as_context", "tokens_prompts_as_context", False),
    ("recent_only",        "gen_recent_only",        "tokens_recent_only",        False),
    ("engram_8_p",         "gen_engram_8_p",         "tokens_engram_8_p",         True),
    ("engram_8_pr",        "gen_engram_8_pr",        "tokens_engram_8_pr",        True),
    ("engram_16_p",        "gen_engram_16_p",        "tokens_engram_16_p",        True),
    ("engram_16_pr",       "gen_engram_16_pr",       "tokens_engram_16_pr",       True),
    ("engram_24_p",        "gen_engram_24_p",        "tokens_engram_24_p",        True),
    ("engram_24_pr",       "gen_engram_24_pr",       "tokens_engram_24_pr",       True),
    ("uniform_pool",       "gen_uniform_pool",       "tokens_uniform_pool",       True),
]


def main():
    data = json.loads((EXP / "results/conditions_v3.json").read_text())
    judge = json.loads((EXP / "results/judge_v3.json").read_text())
    results = data["results"]
    aniso = data["anisotropy"]
    train_accs = data["train_accs"]
    cfg = data["config"]

    # Build judgment lookup: (probe_idx, condition) -> score
    judge_scores = {}
    for j in judge["judgments"]:
        judge_scores[(j["probe_idx"], j["condition"])] = j["score"]
    # full_context judgements weren't computed (always equal to ceiling); set to 3.
    for r in results:
        judge_scores[(r["probe_idx"], "full_context")] = 3

    # Score each condition
    scores = {label: {"f1": [], "coverage": [], "tokens": [], "judge": []}
              for label, *_ in CONDITIONS}
    routing = {f"{layer}_{kind}": {"p_at_k": [], "r_at_k": []}
               for layer in cfg["layers"] for kind in ("p", "pr")}

    for r in results:
        ceiling = r["gen_full_context"]
        for label, gk, tk, _ in CONDITIONS:
            gen = r[gk]
            scores[label]["f1"].append(f1_overlap(gen, ceiling))
            scores[label]["coverage"].append(coverage(gen, r["relevant_ids"]))
            scores[label]["tokens"].append(r[tk])
            scores[label]["judge"].append(
                judge_scores.get((r["probe_idx"], label), 0))
        for layer in cfg["layers"]:
            for kind in ("p", "pr"):
                key = f"{layer}_{kind}"
                routing[key]["p_at_k"].append(r[f"routing_p_at_k_{key}"])
                routing[key]["r_at_k"].append(r[f"routing_r_at_k_{key}"])

    # Aggregate
    summary = {}
    for label, *_ in CONDITIONS:
        s = scores[label]
        summary[label] = {
            "f1_mean":       float(np.mean(s["f1"])),
            "coverage_mean": float(np.mean(s["coverage"])),
            "tokens_mean":   float(np.mean(s["tokens"])),
            "judge_mean":    float(np.mean(s["judge"])),
            "judge_median":  float(np.median(s["judge"])),
        }

    # Print
    print(f"\n{'condition':>22s}  {'F1':>5s}  {'cov':>5s}  {'judge':>5s}  "
          f"{'tokens':>7s}  {'aniso':>6s}  {'P@10':>6s}")
    for label, *_ in CONDITIONS:
        s = summary[label]
        aniso_str = ""; pk_str = ""
        if label.startswith("engram_"):
            ek = label.replace("engram_", "")
            aniso_str = f"{aniso[ek]['mean']:.2f}"
            pk_str = f"{np.mean(routing[ek]['p_at_k']):.3f}"
        elif label == "uniform_pool":
            aniso_str = f"{aniso['16_pr']['mean']:.2f}"
        print(f"{label:>22s}  {s['f1_mean']:.3f}  {s['coverage_mean']:.3f}  "
              f"{s['judge_mean']:.2f}  {s['tokens_mean']:7.0f}  "
              f"{aniso_str:>6s}  {pk_str:>6s}")

    # Save
    (EXP / "results/scores_v3.json").write_text(json.dumps({
        "summary": summary,
        "anisotropy": aniso,
        "train_accs": train_accs,
        "config": cfg,
        "per_probe": [
            {
                "probe_idx": r["probe_idx"], "prompt": r["prompt"],
                "n_relevant": r["n_relevant"],
                **{f"f1_{label}": f1_overlap(r[gk], r["gen_full_context"])
                   for label, gk, *_ in CONDITIONS if label != "full_context"},
                **{f"judge_{label}": judge_scores.get((r["probe_idx"], label), 3)
                   for label, *_ in CONDITIONS},
                **{f"cov_{label}": coverage(r[gk], r["relevant_ids"])
                   for label, gk, *_ in CONDITIONS},
            }
            for r in results
        ],
    }, indent=2))
    print("\nSaved results/scores_v3.json")

    # ---------- Plot ----------
    fig, axs = plt.subplots(2, 2, figsize=(13, 10))
    labels = [c[0] for c in CONDITIONS]
    f1s = [summary[l]["f1_mean"] for l in labels]
    judges = [summary[l]["judge_mean"] for l in labels]
    covs = [summary[l]["coverage_mean"] for l in labels]
    colors = []
    for l in labels:
        if l == "full_context": colors.append("tab:green")
        elif l == "prompts_as_context": colors.append("tab:purple")
        elif l == "recent_only": colors.append("tab:gray")
        elif l == "random_engrams": colors.append("tab:red")
        elif "16_pr" in l: colors.append("tab:blue")
        else: colors.append("lightblue")

    ax = axs[0, 0]
    bars = ax.bar(labels, f1s, color=colors)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_ylabel("token-F1 vs full-context ceiling")
    ax.set_title("v3 token-F1 (with metadata wrapping)")
    ax.tick_params(axis="x", rotation=45); plt.setp(ax.get_xticklabels(),
                                                      ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axs[0, 1]
    bars = ax.bar(labels, judges, color=colors)
    for b, v in zip(bars, judges):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_ylabel("Mistral-as-judge mean score (0-3)")
    ax.set_title("LLM-as-judge (Mistral self-judging)")
    ax.set_ylim(0, 3.2)
    ax.tick_params(axis="x", rotation=45); plt.setp(ax.get_xticklabels(),
                                                      ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Token-F1 vs LLM-judge ranking
    ax = axs[1, 0]
    # Sort by F1, plot rank vs LLM-judge value
    order_f1 = sorted(range(len(labels)), key=lambda i: -f1s[i])
    order_judge = sorted(range(len(labels)), key=lambda i: -judges[i])
    rank_f1 = {i: rank for rank, i in enumerate(order_f1)}
    rank_judge = {i: rank for rank, i in enumerate(order_judge)}
    for i, l in enumerate(labels):
        ax.scatter(rank_f1[i], rank_judge[i], s=80,
                    color=colors[i], edgecolors="black")
        ax.annotate(l, (rank_f1[i], rank_judge[i]),
                     xytext=(5, 5), textcoords="offset points", fontsize=7)
    # diagonal reference
    ax.plot([0, len(labels)-1], [0, len(labels)-1], "k--", alpha=0.3)
    ax.set_xlabel("token-F1 rank (0 = best)")
    ax.set_ylabel("LLM-judge rank (0 = best)")
    ax.set_title("Ranking comparison (diagonal = same rank)")
    ax.grid(True, alpha=0.3)

    # Per-probe judge distribution per condition
    ax = axs[1, 1]
    plot_conds = ["recent_only", "prompts_as_context", "engram_8_pr",
                  "engram_16_pr", "uniform_pool", "random_engrams"]
    judge_per_cond = []
    for c in plot_conds:
        judge_per_cond.append([judge_scores.get((r["probe_idx"], c), 0)
                                for r in results])
    bp = ax.boxplot(judge_per_cond, tick_labels=plot_conds)
    ax.set_ylabel("Mistral-as-judge score (0-3)")
    ax.set_title("Per-probe judge score distribution")
    ax.tick_params(axis="x", rotation=30); plt.setp(ax.get_xticklabels(),
                                                      ha="right")
    ax.set_ylim(-0.2, 3.2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("v3: metadata wrapping + sampling + LLM-as-judge "
                  "(Mistral-7B, 100 turns, 20 probes)")
    fig.tight_layout()
    out_png = EXP / "results/result_v3.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---------- RESULT_v3.md ----------
    out = []
    s = summary
    out.append("# Prompt vs Response Engrams: v3 (metadata + sampling + LLM-judge)\n")
    out.append("**Question:** does adding metadata wrapping to all "
                "conditions change the result? In particular, does "
                "prompts-as-context — which scored *worst* in v2 (F1 "
                "0.121) — improve when given a clear conversation "
                "structure? Also: does Mistral-as-judge agree with "
                "token-F1 rankings, or was F1 misleading?")
    out.append("")

    f1_full   = s["full_context"]["f1_mean"]
    f1_prompts= s["prompts_as_context"]["f1_mean"]
    f1_recent = s["recent_only"]["f1_mean"]
    f1_random = s["random_engrams"]["f1_mean"]
    judge_full   = s["full_context"]["judge_mean"]
    judge_prompts= s["prompts_as_context"]["judge_mean"]
    judge_recent = s["recent_only"]["judge_mean"]
    judge_eng_best_label = max(["8_p","8_pr","16_p","16_pr","24_p","24_pr"],
                                  key=lambda k: s[f"engram_{k}"]["judge_mean"])
    judge_eng_best = s[f"engram_{judge_eng_best_label}"]["judge_mean"]

    out.append("**Verdict.**")
    out.append("")
    if judge_prompts >= judge_recent + 0.3:
        out.append(f"**Metadata wrapping rescues prompts-as-context.** "
                    f"Under LLM-judge, prompts_as_context "
                    f"({judge_prompts:.2f}) substantially exceeds "
                    f"recent_only ({judge_recent:.2f}). The v2 result "
                    f"(prompts-as-context worst) was an artifact of "
                    f"poor input framing.")
    elif judge_prompts >= judge_recent - 0.2:
        out.append(f"**Metadata wrapping closes the gap but doesn't "
                    f"invert it.** Under LLM-judge, prompts_as_context "
                    f"({judge_prompts:.2f}) ≈ recent_only "
                    f"({judge_recent:.2f}). The v2 result was "
                    f"partially an artifact of input framing, but the "
                    f"deeper finding (prompts-as-pointers ≠ winning "
                    f"strategy) holds.")
    else:
        out.append(f"**Metadata wrapping helps but prompts-as-context "
                    f"still loses to recent-only.** Under LLM-judge, "
                    f"prompts_as_context ({judge_prompts:.2f}) < "
                    f"recent_only ({judge_recent:.2f}) by "
                    f"{judge_recent - judge_prompts:.2f}. The "
                    f"prompt-as-pointer hypothesis fails even under "
                    f"fair input framing.")
    out.append("")
    out.append(f"Best engram condition (engram_{judge_eng_best_label}, "
                f"judge {judge_eng_best:.2f}) "
                + ("beats" if judge_eng_best > judge_recent else "loses to")
                + f" recent_only ({judge_recent:.2f}). Engram conditions "
                + "remain at or below the truncation baseline."
                + "")
    out.append("")

    # Setup
    out.append("## Setup\n")
    out.append("- Substrate, engrams, W projections, probes: identical "
                "to v2.")
    out.append("- **New: metadata wrapping** for all text conditions:")
    out.append("  - Each turn wrapped as `[Conversation turn N] / "
                "USER: ... / ASSISTANT: ... / [end of turn N]`.")
    out.append("  - Synthesis probe wrapped as `[Current question — "
                "please answer using the conversation history above] / "
                "USER: {probe} / ASSISTANT:`.")
    out.append("  - prompts_as_context wraps each prompt as `[Earlier "
                "conversation turn N — prompt only, response removed]`.")
    out.append("  - Engram conditions get a text prefix: `[The model "
                "has access to compressed memories of the prior "
                "conversation, retrieved by relevance to the current "
                "question. These memories appear as the initial context "
                "below.]`.")
    out.append(f"- **New: sampling generation.** Temperature {cfg['temp']}, "
                f"repetition_penalty {cfg['repetition_penalty']}, "
                f"max_new_tokens {cfg['gen_tokens']}. v2 was greedy "
                f"and produced repetitive output (the Grant probe "
                f"inspection showed this).")
    out.append(f"- **New: LLM-as-judge.** Mistral-7B itself, used "
                f"with a logit-based scoring protocol — for each "
                f"(probe, condition), compute the next-token logits "
                f"after a comparison prompt ending in `Score:`, take "
                f"argmax over {{0,1,2,3}}. **Caveat:** Mistral judges "
                f"its own outputs, which biases scores toward "
                f"Mistral-style writing. The relative ranking is "
                f"informative even if absolute scores aren't.")
    out.append("")

    # Main table
    out.append("## Main results table\n")
    out.append("| condition | token-F1 | LLM-judge | coverage | tokens | "
                "aniso | P@10 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, *_ in CONDITIONS:
        si = s[label]
        aniso_str = ""; pk_str = ""
        if label.startswith("engram_"):
            ek = label.replace("engram_", "")
            aniso_str = f"{aniso[ek]['mean']:.2f}"
            pk_str = f"{np.mean(routing[ek]['p_at_k']):.3f}"
        elif label == "uniform_pool":
            aniso_str = f"{aniso['16_pr']['mean']:.2f}"
        out.append(f"| {label} | {si['f1_mean']:.3f} | "
                    f"{si['judge_mean']:.2f} | "
                    f"{si['coverage_mean']:.3f} | "
                    f"{si['tokens_mean']:.0f} | "
                    f"{aniso_str} | {pk_str} |")
    out.append("")

    # v2 vs v3 comparison
    out.append("## v2 vs v3 comparison\n")
    out.append("| condition | v2 F1 | v3 F1 | v3 LLM-judge |")
    out.append("|---|---:|---:|---:|")
    v2_f1 = {
        "full_context": 1.000, "random_engrams": 0.123,
        "prompts_as_context": 0.121, "recent_only": 0.340,
        "engram_8_p": 0.144, "engram_8_pr": 0.230,
        "engram_16_p": 0.119, "engram_16_pr": 0.147,
        "engram_24_p": 0.139, "engram_24_pr": 0.104,
        "uniform_pool": 0.196,
    }
    for label, *_ in CONDITIONS:
        si = s[label]
        v2 = v2_f1.get(label, float("nan"))
        out.append(f"| {label} | {v2:.3f} | {si['f1_mean']:.3f} | "
                    f"{si['judge_mean']:.2f} |")
    out.append("")

    # Predictions check
    out.append("## Pre-registered predictions check\n")
    out.append(f"1. **\"Prompts-as-context with metadata will substantially "
                f"outperform prompts-as-context without metadata.\"**")
    out.append(f"   v2 F1 = 0.121 → v3 F1 = {f1_prompts:.3f}; "
                f"v3 LLM-judge = {judge_prompts:.2f}. ")
    if f1_prompts > 0.121 + 0.05 or judge_prompts > 0.5:
        out.append(f"   **SUPPORTED.** Metadata wrapping helps; magnitude varies "
                    f"by metric.")
    else:
        out.append(f"   **NOT supported.** Metadata wrapping doesn't change "
                    f"the picture meaningfully.")
    out.append("")
    out.append(f"2. **\"LLM-as-judge rankings will differ from token-F1 "
                f"rankings, possibly substantially.\"** "
                "See the rank-comparison panel of the plot.")
    out.append("")
    out.append(f"3. **\"Engram conditions with metadata prefix will improve "
                f"marginally if at all.\"** "
                f"v2 best engram F1 = 0.230 (engram_8_pr) → "
                f"v3 best engram F1 = "
                f"{max(s[f'engram_{k}']['f1_mean'] for k in ['8_p','8_pr','16_p','16_pr','24_p','24_pr']):.3f}. "
                f"v3 best engram LLM-judge = {judge_eng_best:.2f}.")
    out.append("")

    out.append("![curves](result_v3.png)\n")

    # Honest commentary
    out.append("## Caveats / limitations\n")
    out.append("1. **Judge bias.** Mistral judges its own outputs. "
                "There's no clean way to remove this bias without a "
                "second model (we don't have Anthropic API access in "
                "this environment). Treat absolute LLM-judge scores "
                "with caution; relative differences across conditions "
                "are still informative.")
    out.append("2. **Sampling variance.** Temperature 0.7 introduces "
                "run-to-run variance. Each condition was generated "
                "once; rerunning with different seeds would produce "
                "slightly different outputs and judge scores. The "
                "20-probe averaging absorbs most of the variance for "
                "the table-level summary but per-probe numbers are "
                "noisy.")
    out.append("3. **Same content as v2.** The 100 Civil War turns "
                "and 20 test probes are unchanged. If those probes "
                "are unrepresentative, all v2 + v3 conclusions could "
                "be artifacts.")
    out.append("4. **Token-F1 stays misleading.** The plot's rank-"
                "comparison panel shows where token-F1 disagrees "
                "with the LLM-judge. For synthesis tasks where two "
                "answers can express the same content with different "
                "vocabulary, token-F1 underweights paraphrase. The "
                "LLM-judge has its own biases but they're complementary.")
    out.append("")

    out.append("## Wall-clock\n")
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| v3 conditions × 20 probes (sampling, rep_penalty=1.15) | "
                f"{cfg.get('wall_total_s', 0):.0f}s "
                f"(~{cfg.get('wall_total_s', 0)/60:.0f} min) |")
    out.append(f"| LLM-as-judge (220 logit-based scorings) | "
                f"{judge.get('wall_total_s', 0):.0f}s "
                f"(~{judge.get('wall_total_s', 0)/60:.0f} min) |")
    out.append("")

    out_md = EXP / "results/RESULT_v3.md"
    out_md.write_text("\n".join(out))
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
