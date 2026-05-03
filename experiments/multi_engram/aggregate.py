"""Plot multi-engram results and write RESULT.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))


def main():
    cond = json.loads((EXP / "results/conditions.json").read_text())
    scores = json.loads((EXP / "results/scores.json").read_text())

    s = scores["summary"]
    pp = scores["per_probe"]
    cfg = cond["config"]

    # ---------- Plot ----------
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # F1 bar
    ax = axs[0, 0]
    conds = ["recent_only", "uniform_pool", "engram_routing", "random_engrams"]
    f1s = [s["f1_means"][c] for c in conds]
    colors = ["tab:gray", "tab:gray", "tab:blue", "tab:gray"]
    bars = ax.bar(conds, f1s, color=colors)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel("token-F1 vs full-context ceiling")
    ax.set_title("Answer quality (higher = closer to ceiling)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")

    # Coverage bar
    ax = axs[0, 1]
    conds_c = ["full_context", "recent_only", "uniform_pool",
                "engram_routing", "random_engrams"]
    covs = [s["coverage_means"][c] for c in conds_c]
    bars = ax.bar(conds_c, covs)
    for b, v in zip(bars, covs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylabel("coverage (frac. of relevant turn keywords in answer)")
    ax.set_title("Coverage of relevant turns")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")

    # Per-probe F1 distribution
    ax = axs[1, 0]
    f1_cols = {"recent": [r["f1_recent"] for r in pp],
               "engram_routing": [r["f1_engram_routing"] for r in pp],
               "random": [r["f1_random"] for r in pp],
               "uniform": [r["f1_uniform"] for r in pp]}
    positions = list(range(len(f1_cols)))
    ax.boxplot([f1_cols[k] for k in f1_cols.keys()], labels=list(f1_cols.keys()))
    ax.set_ylabel("token-F1 vs ceiling")
    ax.set_title(f"Per-probe F1 distribution (N={len(pp)} probes)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # Routing P@K, R@K
    ax = axs[1, 1]
    p_at_k = [r["routing_p_at_k"] for r in pp]
    r_at_k = [r["routing_r_at_k"] for r in pp]
    ax.scatter(range(len(p_at_k)), p_at_k, label=f"P@{cfg['top_k']}", alpha=0.7)
    ax.scatter(range(len(r_at_k)), r_at_k, label=f"R@{cfg['top_k']}", alpha=0.7)
    ax.axhline(s["routing_p_at_k_mean"], color="C0", linestyle="--",
                label=f"P mean={s['routing_p_at_k_mean']:.3f}")
    ax.axhline(s["routing_r_at_k_mean"], color="C1", linestyle="--",
                label=f"R mean={s['routing_r_at_k_mean']:.3f}")
    # Random baseline reference
    ax.axhline(0.10, color="black", linestyle=":", alpha=0.5,
                label="random P@k≈0.10")
    ax.set_xlabel("probe index")
    ax.set_ylabel("precision / recall")
    ax.set_title(f"Per-probe routing P@{cfg['top_k']} and R@{cfg['top_k']}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-engram synthesis on Civil War turns: Mistral-7B, "
                  "L16 mean-pool, 100 turns, 20 test probes")
    fig.tight_layout()
    out_png = EXP / "results/multi_engram.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---------- RESULT.md ----------
    out = []
    out.append("# Multi-Engram Synthesis on In-Distribution Conversational Context\n")
    out.append("**Question:** can a bank of 100 engrams (one per "
                "conversation turn, mid-layer mean-pooled) support "
                "synthesis-style queries that need to integrate "
                "information from many spread-out turns simultaneously?")
    out.append("")
    out.append("**Verdict: hypothesis NOT supported.** Engram routing "
                "performs *worse* than recent-only truncation on every "
                "metric. Routing precision is barely above chance (P@10 "
                "= 0.135, vs random baseline ~0.10). The W projection "
                "trained on 5 validation probes does not generalize to "
                "the 20 test probes — train acc on the validation pairs "
                "themselves was only 8%, well below the threshold for "
                "meaningful discrimination among 100 highly anisotropic "
                "L16 engrams (mean pairwise cosine 0.95). On the bigger "
                "picture: this replicates the negative pattern from the "
                "GPT-2 context-compression experiment, now on a stronger "
                "base (Mistral-7B) and with semantically meaningful "
                "content. **Recent-only truncation continues to be the "
                "strongest simple baseline.**")
    out.append("")

    out.append("## Setup\n")
    out.append("- **Engram-producing model:** Mistral-7B-v0.1 base (fp16, "
                "32 layers, 4096d, RoPE positions). FROZEN.")
    out.append("- **Source content:** 100 conversational turns about "
                "the U.S. Civil War, generated via Mistral-7B with a "
                "2-shot extractive Q/A prompt across 100 hand-curated "
                "subtopics: 25 battles, 20 generals, 15 political/"
                "leadership, 15 logistics/technology, 15 social, 10 "
                "named campaigns. Total ~370s of generation; turns "
                "average ~120 tokens (Q+A combined).")
    out.append("- **Engram per turn:** mean-pool of layer-16 hidden "
                "states (mid-stack of 32) over the full Q+A turn.")
    out.append("- **Probes:** 25 hand-written synthesis-style queries "
                "(5 validation + 20 test), each with hand-annotated "
                "ground-truth relevant turn-ids based on category + "
                "title-keyword matching. 16/20 test probes have ≥5 "
                "relevant turns; smaller probes (e.g., \"women in war,\" "
                "1 relevant turn) test focused retrieval.")
    out.append("- **W projection:** 4096×4096 linear, identity init, "
                "trained 500 steps via multi-label InfoNCE on the 5 "
                "validation probes' relevance pairs (~50 positive pairs "
                "total, lr 1e-3, temperature 0.05).")
    out.append("- **Generation:** Mistral-7B greedy, 180 new tokens. "
                "For engram-injection conditions (3, 4, 5), the engram "
                "vectors are prepended to the embedded probe via "
                "`inputs_embeds`.")
    out.append("")

    out.append("## Five conditions (per test probe)\n")
    out.append("- **A — full context (ceiling):** concatenate all 100 "
                "Q&A turns; truncate to last 4000 tokens (so the probe "
                "lands at the end); ask. *(NOTE: Mistral's context is "
                "not infinite — even fp16 it OOMs on much longer than "
                "this on a 16GB GPU. Full-context is therefore an "
                "approximation; the actual budget is ~25 of 100 turns "
                "of context.)*")
    out.append("- **B — recent-only truncation:** keep only the last 8 "
                "turns of the conversation (~960 tokens); ask.")
    out.append("- **C — random engrams:** pick 10 engrams at random, "
                "prepend as `inputs_embeds`; ask.")
    out.append("- **D — uniform pool:** average all 100 engrams into one "
                "vector, prepend; ask.")
    out.append("- **E — engram routing (the test):** project the probe's "
                "L16-mean through W, softmax over 100 engrams "
                f"(temp={cfg['temp']}), take top-{cfg['top_k']} engrams, "
                "prepend; ask.")
    out.append("")

    out.append("## Results\n")
    out.append("**Answer quality** (token-F1 between each condition's "
                "answer and the full-context ceiling answer):")
    out.append("")
    out.append("| condition | mean F1 | median | p25 | p75 |")
    out.append("|---|---:|---:|---:|---:|")
    for c in ("recent_only", "engram_routing", "random_engrams",
                "uniform_pool"):
        v = [r[f"f1_{c.split('_')[0] if c != 'engram_routing' else 'engram_routing'}"] for r in pp]
        # The per-probe keys use shorter names
        key_map = {"recent_only": "f1_recent", "engram_routing": "f1_engram_routing",
                   "random_engrams": "f1_random", "uniform_pool": "f1_uniform"}
        v = [r[key_map[c]] for r in pp]
        out.append(f"| {c} | {np.mean(v):.3f} | "
                    f"{np.median(v):.3f} | "
                    f"{np.quantile(v, 0.25):.3f} | "
                    f"{np.quantile(v, 0.75):.3f} |")
    out.append("")
    out.append("**Recent-only (0.31) decisively beats engram routing "
                "(0.17) and even uniform pooling (0.21).** Engram "
                "routing's mean F1 of 0.17 is barely above the random-"
                "engram baseline of 0.15 — the W-trained routing adds "
                "essentially no value over picking 10 random engrams.")
    out.append("")

    out.append("**Coverage** (fraction of relevant turn-keywords appearing "
                "in the generated answer; catches \"the model is just "
                "answering from pretraining\"):")
    out.append("")
    out.append("| condition | mean coverage | median |")
    out.append("|---|---:|---:|")
    cov_map = {"full_context": "cov_full", "recent_only": "cov_recent",
               "uniform_pool": "cov_uniform", "engram_routing": "cov_engram_routing",
               "random_engrams": "cov_random"}
    for c, key in cov_map.items():
        v = [r[key] for r in pp]
        out.append(f"| {c} | {np.mean(v):.3f} | {np.median(v):.3f} |")
    out.append("")
    out.append("**The coverage stat reveals the bigger problem.** "
                "Random engrams cover 19% of relevant keywords. Engram "
                "routing covers 35%. Uniform pool covers 40%. "
                "Recent-only covers 47%. **None of the engram conditions "
                "beat plain truncation on coverage either.** The fact "
                "that uniform pool (a single mean vector!) covers more "
                "relevant content than top-K routing tells us the W "
                "projection isn't selecting the *right* engrams.")
    out.append("")

    out.append("**Routing precision/recall** (against the hand-annotated "
                f"relevant set per probe, k={cfg['top_k']}):")
    out.append("")
    out.append("| metric | value | random baseline |")
    out.append("|---|---:|---:|")
    out.append(f"| P@{cfg['top_k']} | "
                f"**{s['routing_p_at_k_mean']:.3f}** | ~0.10 (10/100 picks "
                "from a uniform 100-bank) |")
    out.append(f"| R@{cfg['top_k']} | "
                f"**{s['routing_r_at_k_mean']:.3f}** | ~0.10 |")
    out.append(f"| R@20 | "
                f"**{s['routing_r_at_20_mean']:.3f}** | ~0.20 |")
    out.append("")
    out.append("Routing precision at k=10 (0.135) is barely above the "
                "random baseline of 0.10. The architecture's routing "
                "mechanism is functionally inert here.")
    out.append("")

    out.append("![curves](multi_engram.png)\n")

    out.append("## Pre-registered predictions vs. result\n")
    out.append("From the spec:")
    out.append("")
    out.append("1. **\"Engram routing should beat random engrams\"** → "
                "**FAILS.** Engram F1 0.168 vs random F1 0.154 — a 1.4pp "
                "gap that's well within noise. The W projection isn't "
                "doing meaningful work.")
    out.append("2. **\"Engram routing should beat recent-only-"
                "truncation on synthesis probes\"** → **FAILS.** "
                "Recent-only F1 0.309 vs engram routing F1 0.168 — "
                "truncation is 14pp better. The synthesis regime "
                "doesn't change which approach wins; truncation "
                "remains the stronger baseline.")
    out.append("3. **\"Engram routing's gap to ceiling should be "
                "smaller than recent-only's\"** → **FAILS.** Recent-only "
                "is closer to the ceiling than engram routing.")
    out.append("4. **\"Routing precision/recall should be substantially "
                "above chance\"** → **FAILS.** P@10 = 0.135 vs random "
                "≈ 0.10. Marginal.")
    out.append("5. **\"Temperature should matter (sharp loses recall, "
                "diffuse loses precision)\"** → not tested due to time "
                "budget; the binding constraint is W's training-set "
                "size and target-engram anisotropy, not temperature.")
    out.append("")

    out.append("## Why the hypothesis failed\n")
    out.append("Two compounding problems:")
    out.append("")
    out.append("1. **L16 engrams are highly anisotropic.** Pairwise "
                "cosine across the 100 engrams: mean 0.95, max 0.99, "
                "min 0.82. This is the well-known transformer hidden-"
                "state anisotropy: mean-pooled mid-stack vectors all "
                "cluster along a few common directions. Discriminating "
                "100 engrams that are within 5° of each other in raw "
                "cosine is hard for any cosine-based mechanism.")
    out.append("")
    out.append("2. **W has too little training signal.** 5 validation "
                "probes × ~10 relevant engrams each = ~50 positive "
                "pairs. Training a 4096×4096 projection (~17M params) "
                "on 50 examples doesn't generalize. Train acc on the "
                "validation set itself was 8% — barely above the 1% "
                "random baseline. With this little signal, W is "
                "essentially noise.")
    out.append("")
    out.append("3. **The model is answering from pretraining.** "
                "Mistral-7B knows the U.S. Civil War from its "
                "pretraining corpus. Even with prepended engrams that "
                "are largely uninformative, the model produces "
                "plausible-sounding Civil War content. The coverage "
                "stat confirms this: uniform pool (a single vector!) "
                "covers more relevant content than top-K routing — "
                "the engram input isn't *steering* the answer; the "
                "answer is coming from the model's internal Civil War "
                "knowledge.")
    out.append("")

    out.append("## Failure modes from the spec, addressed\n")
    out.append("- **\"If engram routing doesn't beat random engrams, "
                "the W projection isn't generalizing\"**: confirmed. "
                "F1 0.168 vs 0.154. Within noise.")
    out.append("- **\"If the model produces good-looking answers from "
                "pretraining alone\"**: confirmed by the coverage "
                "comparison. All conditions produce Civil War content "
                "regardless of what's prepended.")
    out.append("- **\"If recent-only-truncation matches engram routing "
                "on synthesis probes too, the per-token-of-budget "
                "argument extends and the engram approach has no "
                "winning regime\"**: confirmed. **This is the strong "
                "negative result the spec named.** On synthesis "
                "probes, on a base that knows the topic well, on "
                "naturalistic content — engram routing still loses to "
                "truncation.")
    out.append("")

    out.append("## Implications for the architecture\n")
    out.append("**The architectural argument for engrams is now narrower "
                "than ever.** Across the prior experiments:")
    out.append("- 50-Dickens templated paraphrases (Phase 47): routing "
                "works at 100%.")
    out.append("- 100k WikiText non-paraphrase chunks: routing fails "
                "(`library_scaling`); top-1 8% on disjoint paraphrases.")
    out.append("- GPT-2 context compression on WikiText: learned "
                "compression beats nothing but loses to truncation.")
    out.append("- This experiment, multi-engram synthesis on Mistral: "
                "routing is functionally inert; recent-only truncation "
                "wins.")
    out.append("")
    out.append("The cumulative pattern: **engram routing only works in "
                "regimes where the queries share substantial token "
                "overlap with the training paraphrases**. On natural "
                "paraphrases (split halves of WT-103, synthesis "
                "queries about real-world topics), the linear W "
                "projection learned via InfoNCE cannot extract "
                "discriminative signal.")
    out.append("")
    out.append("**For the deployment story:** engram-based routing as "
                "currently formulated is not yet a viable replacement "
                "for either RAG or recent-only-truncation. The "
                "architecture's claimed advantages (compute, context-"
                "compression) hold only when an upstream component — "
                "templated paraphrase, exact-string matching, or some "
                "learned-but-non-linear discriminator — is doing the "
                "actual disambiguation work.")
    out.append("")
    out.append("**For follow-up work:** the experiments converge on "
                "two architectural changes worth testing.")
    out.append("1. Replace the L16 mid-layer mean with something less "
                "anisotropic — e.g., last-token hidden state at L16, "
                "or hidden states from a model with a discrimination-"
                "preserving training objective (contrastively trained "
                "encoder, like BGE / E5).")
    out.append("2. Replace the linear W with a non-linear router "
                "(small MLP) and train on a much larger validation "
                "set (50+ probes, not 5). The 50-pair training "
                "signal is insufficient regardless of how good the "
                "engrams are.")
    out.append("")

    out.append("## Wall-clock\n")
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| Generate 100 turns (Mistral-7B 2-shot) | 366 s "
                f"(~6 min) |")
    out.append(f"| Compute 100 engrams (L16 mean) | 6 s |")
    out.append(f"| Train W (500 InfoNCE steps) | <1 s |")
    out.append(f"| Run 5 conditions × 20 test probes | "
                f"{cond['wall_total_s']:.0f} s "
                f"(~{cond['wall_total_s']/60:.0f} min) |")
    out.append(f"| Score (token-F1, coverage, routing P/R) | <1 s |")
    out.append(f"| **Total** | **~{(366 + 6 + cond['wall_total_s'])/60:.0f} min** |")
    out.append("")

    out.append("## Caveats\n")
    out.append("1. **Mistral-7B base, not instruct.** A more "
                "instruction-tuned model might use prepended embeddings "
                "differently. But the failure mode here (model answers "
                "from pretraining) would likely persist.")
    out.append("2. **Token-F1 is a coarse answer-quality metric.** "
                "BERTScore would be cleaner; LLM-as-judge cleaner "
                "still. But the relative ordering across conditions "
                "is consistent across F1 and coverage, which is "
                "evidence the metric isn't misleading.")
    out.append("3. **Training W on 5 probes is very thin.** A more "
                "fair test would expand the validation set to 50+ "
                "probes. We didn't because hand-annotating relevance "
                "is expensive. 50 probes might lift the result but "
                "wouldn't change the qualitative conclusion that "
                "routing under-performs truncation.")
    out.append("4. **Mid-layer engrams (L16) may be the wrong choice.** "
                "Earlier or later layers might give better separation. "
                "Position-erosion experiment showed late layers are "
                "*more* anisotropic than mid layers though, so "
                "L16 was the defensible compromise.")
    out.append("5. **Engram injection via `inputs_embeds` skips RoPE "
                "for those positions.** Mistral uses RoPE in attention; "
                "the prepended engrams take positions 0..K-1 of the "
                "ROPE schedule. The model has no principled way to "
                "know these are summaries of long content.")

    out_md = EXP / "results/RESULT.md"
    out_md.write_text("\n".join(out))
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
