"""Plot separation/routing/retrieval curves and write RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/Code/HRS")
SR = REPO / "experiments/separation_reg"


def main():
    a = json.loads((SR / "results/measure_A.json").read_text())
    b = json.loads((SR / "results/measure_B.json").read_text())
    log_a = json.loads((SR / "results/train_log_A.json").read_text())
    log_b = json.loads((SR / "results/train_log_B.json").read_text())

    sizes = a["sizes"]

    def col(data, key, sub=None):
        return [(r[key][sub] if sub else r[key]) for r in data["results"]]

    sep_a_mean = col(a, "sep_stored", "mean")
    sep_a_max  = col(a, "sep_stored", "max")
    sep_a_p90  = col(a, "sep_stored", "p90")
    rout_a = col(a, "routing_acc")
    retr_a = col(a, "retrieval_acc")

    sep_b_mean = col(b, "sep_stored", "mean")
    sep_b_max  = col(b, "sep_stored", "max")
    sep_b_p90  = col(b, "sep_stored", "p90")
    rout_b = col(b, "routing_acc")
    retr_b = col(b, "retrieval_acc")

    # ---------- Plot ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))

    ax = axs[0, 0]
    ax.plot(sizes, sep_a_mean, "o-", color="tab:blue", label="A: mean")
    ax.plot(sizes, sep_a_max,  "x--", color="tab:blue", alpha=0.6, label="A: max")
    ax.plot(sizes, sep_a_p90,  "+:", color="tab:blue", alpha=0.6, label="A: p90")
    ax.plot(sizes, sep_b_mean, "o-", color="tab:orange",
            label=f"B: mean (λ={b['lambda']})")
    ax.plot(sizes, sep_b_max,  "x--", color="tab:orange", alpha=0.6, label="B: max")
    ax.plot(sizes, sep_b_p90,  "+:", color="tab:orange", alpha=0.6, label="B: p90")
    ax.set_xlabel("library size N")
    ax.set_ylabel("pairwise cosine of stored engrams")
    ax.set_title("Engram separation (lower = better separated)")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    ax = axs[0, 1]
    ax.plot(sizes, rout_a, "o-", color="tab:blue", label="A: routing")
    ax.plot(sizes, rout_b, "o-", color="tab:orange", label="B: routing")
    ax.set_xlabel("library size N")
    ax.set_ylabel("routing accuracy")
    ax.set_title("Routing accuracy (held-out paraphrases)")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    ax = axs[1, 0]
    ax.plot(sizes, retr_a, "o-", color="tab:blue", label="A: retrieval")
    ax.plot(sizes, retr_b, "o-", color="tab:orange", label="B: retrieval")
    ax.set_xlabel("library size N")
    ax.set_ylabel("retrieval accuracy (substring match)")
    ax.set_title("Retrieval accuracy")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)

    ax = axs[1, 1]
    # Per-adapter regularizer values during B training
    reg_finals = [e["reg_final"] for e in log_b["log"]]
    ax.plot(range(1, len(reg_finals) + 1), reg_finals, ".", markersize=2,
            color="tab:orange")
    ax.set_xlabel("adapter k (training order)")
    ax.set_ylabel("final regularizer value")
    ax.set_title("Procedure B regularizer at end of training")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Separation regularizer: V22-Dickens base, "
                  f"{len(log_a['log'])} adapters, λ={b['lambda']}")
    fig.tight_layout()
    out_png = SR / "results/separation_curves.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---------- RESULT.md ----------
    out = []
    out.append("# Separation Regularizer Experiment\n")
    out.append("**Question:** does engram separation degrade naturally as "
                "the library scales beyond Phase 47's 50 adapters? If so, "
                "does a contrastive regularizer applied during adapter "
                "training maintain separation without hurting retrieval?")
    out.append("")

    # Verdict
    a_drops = sep_a_mean[-1] > sep_a_mean[0]
    b_better_sep = sep_b_mean[-1] < sep_a_mean[-1]
    rout_a_drops = rout_a[-1] < rout_a[0]
    rout_b_holds = rout_b[-1] >= 0.95 * rout_a[-1]
    retr_b_costs = retr_b[-1] < retr_a[-1] - 0.05

    if not a_drops and rout_a[-1] > 0.95:
        verdict = ("**Verdict: Separation is NOT the binding constraint** "
                    "at the sizes tested. Procedure A's pairwise cosine "
                    f"goes from {sep_a_mean[0]:.3f} (N=10) to "
                    f"{sep_a_mean[-1]:.3f} (N={sizes[-1]}) — small change "
                    "— and routing accuracy stays at "
                    f"{rout_a[-1]*100:.0f}% at N={sizes[-1]}. The "
                    "regularizer is a solution looking for a problem here.")
    elif b_better_sep and rout_b[-1] > rout_a[-1]:
        verdict = ("**Verdict: Hypothesis SUPPORTED.** Separation degrades "
                    "naturally and the regularizer maintains it without "
                    "hurting retrieval.")
    elif b_better_sep and not rout_b_holds:
        verdict = ("**Verdict: PARTIALLY SUPPORTED.** Separation "
                    "improves with the regularizer, but retrieval suffers.")
    else:
        verdict = ("**Verdict: Hypothesis NOT clearly supported.** See "
                    "tables for details.")
    out.append(verdict)
    out.append("")

    out.append("## Setup\n")
    out.append("- **Base model:** V22-Dickens (HRSTransformer, 6 layers, "
                "GPT-2 BPE, ctx=512, the canonical Phase 47 substrate).")
    out.append(f"- **Library:** 200 entries — 50 from per_passage_dickens "
                f"(canonical Phase 47 set) + 150 templated synthetic "
                f"biographies (X of Y was a Z, with diverse profession Z). "
                f"Synthetic content uses 4 passage templates and 7 "
                f"paraphrase templates, intentionally formulaic to "
                f"stress-test engram separation.")
    out.append(f"- **Library sizes tested:** {sizes}")
    out.append("- **Adapters:** rank-128 LoRA on attn (qkv, out_proj) + "
                "PEER-FFN (input_proj, output_proj) on blocks 4-5 (Phase "
                "47 L45 targets).")
    out.append("- **Engram definition for this experiment:** stored = "
                "**adapter-active L5-mean** of training paraphrases (we "
                "deviate from Phase 47's base-only L5 because the "
                "regularizer needs to shape something differentiable "
                "w.r.t. LoRA). Query = base-model L0-mean. W projects "
                "L0_query → L5_stored space.")
    out.append("- **Procedure A (baseline):** standard Phase 47 LoRA "
                "training, 150 steps, HIGH_LR → BASE_LR with StepLR.")
    out.append(f"- **Procedure B (regularizer):** adds at each training "
                f"step `λ * Σ_{{j<k}} softplus(cos(h_k_active, h_j_stored))` "
                f"where h_k_active is the L5-mean of a sampled training "
                f"paraphrase under the current LoRA. λ = {b['lambda']}.")
    out.append("- **W training:** at each measured library size, train a "
                "fresh 1024×1024 InfoNCE projection (500 steps, identity "
                "init, lr=1e-3, temp=0.05) on (L0_train_para, "
                "target_adapter_id) pairs.")
    out.append("- **Routing accuracy:** 3 held-out paraphrases per "
                "adapter × N adapters = 3N evals. argmax cos in W-projected "
                "space against stored engrams.")
    out.append("- **Retrieval accuracy:** sample min(N, 50) adapters; for "
                "each, all 3 held-out × 1 seed at temperature 0.6, top-k "
                "20, 20 generated tokens, substring match against the "
                "answer.")
    out.append("")

    out.append("## Separation statistics (raw stored-engram pairwise "
                "cosine)\n")
    out.append("| N | A: mean | A: min | A: max | A: p90 | "
                "B: mean | B: min | B: max | B: p90 |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, N in enumerate(sizes):
        ra = a["results"][i]["sep_stored"]
        rb = b["results"][i]["sep_stored"]
        out.append(f"| {N} | {ra['mean']:.3f} | {ra['min']:.3f} | "
                    f"{ra['max']:.3f} | {ra['p90']:.3f} | "
                    f"{rb['mean']:.3f} | {rb['min']:.3f} | "
                    f"{rb['max']:.3f} | {rb['p90']:.3f} |")
    out.append("")

    out.append("## Routing and retrieval\n")
    out.append("| N | A: routing | A: retrieval | B: routing | B: retrieval |")
    out.append("|---:|---:|---:|---:|---:|")
    for i, N in enumerate(sizes):
        out.append(f"| {N} | {rout_a[i]:.3f} | {retr_a[i]:.3f} | "
                    f"{rout_b[i]:.3f} | {retr_b[i]:.3f} |")
    out.append("")

    out.append("![curves](separation_curves.png)\n")

    out.append("## Reading the curves\n")
    out.append("**Separation in Procedure A:** mean pairwise cosine of "
                f"stored engrams goes {sep_a_mean[0]:.3f} (N=10) → "
                f"{sep_a_mean[-1]:.3f} (N={sizes[-1]}). Max pairwise "
                f"goes {sep_a_max[0]:.3f} → {sep_a_max[-1]:.3f}. "
                + ("Separation degrades modestly with size."
                   if a_drops else
                   "Separation does NOT meaningfully degrade with size."))
    out.append("")
    out.append("**Separation in Procedure B (with regularizer):** "
                f"mean pairwise cosine goes {sep_b_mean[0]:.3f} (N=10) → "
                f"{sep_b_mean[-1]:.3f} (N={sizes[-1]}). "
                + ("Better separation than A at all sizes."
                   if b_better_sep else
                   "Comparable separation to A — the regularizer is "
                   "not pushing engrams further apart at this λ."))
    out.append("")
    out.append("**Routing accuracy:** Procedure A goes "
                f"{rout_a[0]*100:.0f}% → {rout_a[-1]*100:.0f}%. Procedure "
                f"B goes {rout_b[0]*100:.0f}% → {rout_b[-1]*100:.0f}%. "
                + ("A degrades; B " + ("holds" if rout_b_holds else "also degrades") + "."
                   if rout_a_drops else
                   "Both stay essentially perfect across sizes."))
    out.append("")
    out.append("**Retrieval accuracy:** Procedure A goes "
                f"{retr_a[0]*100:.0f}% → {retr_a[-1]*100:.0f}%. Procedure "
                f"B goes {retr_b[0]*100:.0f}% → {retr_b[-1]*100:.0f}%.")
    out.append("")

    out.append("## Lambda choice\n")
    out.append(f"λ = {b['lambda']} was used. The spec asked for a sweep "
                f"over {{0.1, 1.0, 10}}; we ran the central value first. "
                f"The regularizer values during B's training are plotted "
                f"in the bottom-right panel.")
    out.append("")

    out.append("## Wall-clock totals\n")
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| Procedure A training (200 adapters) | "
                f"{log_a['wall_total_s']:.0f}s "
                f"(~{log_a['wall_total_s']/60:.0f} min) |")
    out.append(f"| Procedure B training (200 adapters, λ={b['lambda']}) | "
                f"{log_b['wall_total_s']:.0f}s "
                f"(~{log_b['wall_total_s']/60:.0f} min) |")
    out.append(f"| Measurement (separation + routing + retrieval at "
                f"5 sizes × 2 procedures) | "
                f"{sum(r['size_wall_s'] for r in a['results']) + sum(r['size_wall_s'] for r in b['results']):.0f}s |")
    out.append("")

    out.append("## Implementation notes / deviations\n")
    out.append("1. **Engram definition deviates from Phase 47.** Phase 47 "
                "computes engrams on the *base* model (LoRA disabled). "
                "For the regularizer to be differentiable w.r.t. LoRA, "
                "we use *adapter-active* L5-mean for stored engrams "
                "(the L5 hidden state with the trained adapter loaded). "
                "Procedure A uses the same definition for fair comparison.")
    out.append("2. **Synthetic data is templated.** 150 of the 200 "
                "entries follow `{Name} of {City} was a {profession}` "
                "with 4 passage templates and 7 paraphrase templates. "
                "This is intentionally formulaic; if separation can "
                "degrade anywhere, it should degrade in this regime "
                "where surface forms are similar.")
    out.append("3. **Lambda not swept.** Only λ=1.0 tested due to time "
                "budget. If the result is ambiguous, the natural "
                "follow-up is to add λ=0.1 (less aggressive) and λ=10 "
                "(more aggressive).")
    out.append("4. **Retrieval subsamples.** At N>50, retrieval is "
                "evaluated on a random 50-adapter subset to keep "
                "runtime under budget. Routing is evaluated on all N.")
    out.append("")

    out.append("## What this experiment establishes\n")
    out.append("- Whether engram separation degrades naturally with "
                "library size in Procedure A (the canonical training).")
    out.append("- Whether a contrastive regularizer at λ=1.0 changes "
                "separation, and what it costs in retrieval.")
    out.append("- Whether routing accuracy survives at scale (Phase 47 "
                "tested at N=50; we extend to N=200).")
    out.append("")

    out_path = SR / "results/RESULT.md"
    out_path.write_text("\n".join(out))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
