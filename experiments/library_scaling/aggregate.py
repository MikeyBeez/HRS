"""Plot library-scaling curves and write RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/Code/HRS")
LS = REPO / "experiments/library_scaling"


def main():
    data = json.loads((LS / "results/measure_v2.json").read_text())
    schemes = data["schemes"]
    sizes = data["sizes"]

    # ---------- Plot ----------
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    colors = {"A_same_chunk_L0first→L5first": "tab:green",
              "B_overlapping_L0[50-150]→L5first": "tab:orange",
              "C_split_halves_L0second→L5first": "tab:red"}
    labels_short = {"A_same_chunk_L0first→L5first": "A: same-chunk (upper bound)",
                    "B_overlapping_L0[50-150]→L5first": "B: overlapping (50-token shift)",
                    "C_split_halves_L0second→L5first": "C: split halves (disjoint)"}

    ax = axs[0, 0]
    # NOTE: at N=1000 the eval queries (1000..1099) are outside the library;
    # top-1 is meaningless. Skip that column for the plot.
    valid_sizes = sizes[1:]
    for s in schemes:
        top1 = [r["top1"] for r in s["rows"]][1:]
        top5 = [r["top5"] for r in s["rows"]][1:]
        c = colors[s["label"]]
        l = labels_short[s["label"]]
        ax.plot(valid_sizes, top1, "o-", color=c, label=f"{l} (top-1)")
        ax.plot(valid_sizes, top5, "x--", color=c, alpha=0.5,
                label=f"{l} (top-5)")
    ax.set_xscale("log")
    ax.set_xlabel("library size N")
    ax.set_ylabel("routing accuracy")
    ax.set_title("Routing accuracy vs library size")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)

    ax = axs[0, 1]
    for s in schemes:
        sep = [r["sep"]["mean"] for r in s["rows"]]
        ax.plot(sizes, sep, "o-", color=colors[s["label"]],
                label=f"sep mean")
    # All three schemes use the same stored engrams, so curves overlap;
    # plot once.
    sep = [r["sep"]["mean"] for r in schemes[0]["rows"]]
    sep_max = [r["sep"]["max"] for r in schemes[0]["rows"]]
    sep_p90 = [r["sep"]["p90"] for r in schemes[0]["rows"]]
    ax.cla()
    ax.plot(sizes, sep, "o-", label="mean")
    ax.plot(sizes, sep_max, "x--", label="max", alpha=0.5)
    ax.plot(sizes, sep_p90, "+:", label="p90", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("library size N")
    ax.set_ylabel("pairwise cosine of stored engrams")
    ax.set_title("Engram separation (sampled 10k pairs at each N)")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axs[1, 0]
    cost = [r["cost_us"] for r in schemes[0]["rows"]]
    ax.plot(sizes, cost, "o-", color="tab:purple")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("library size N")
    ax.set_ylabel("μs per query (cosine vs N + argmax)")
    ax.set_title("Routing computation cost")
    ax.grid(True, alpha=0.3)

    # Top-K rank distribution at largest N for each scheme
    ax = axs[1, 1]
    for s in schemes:
        ranks = [r for r in s["rows"][-1]["ranks"] if r >= 0]
        if not ranks: continue
        # Cumulative top-K
        ks = np.arange(1, 101)
        cum = []
        for k in ks:
            cum.append(sum(1 for r in ranks if r < k) / len(ranks))
        ax.plot(ks, cum, color=colors[s["label"]],
                label=labels_short[s["label"]])
    ax.set_xscale("log")
    ax.set_xlabel("k (top-k threshold)")
    ax.set_ylabel("fraction of queries with correct in top-k")
    ax.set_title(f"Top-K cumulative at N={sizes[-1]}")
    ax.set_ylim(-0.05, 1.05); ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle("Library scaling: V22-Dickens base, WikiText-103 100k chunks, "
                  "W trained on 1000 pairs")
    fig.tight_layout()
    out_png = LS / "results/library_scaling.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---------- RESULT.md ----------
    out = []
    out.append("# Library Scaling Experiment\n")
    out.append("**Question:** does the engram-based routing architecture "
                "scale to 1k-100k libraries? Which failure mode (separation "
                "degradation, W projection capacity, or compute cost) "
                "binds first?")
    out.append("")
    out.append("**Verdict: NONE OF THE THREE EXPECTED FAILURE MODES "
                "BIND AT SCALE.** Engram separation is flat across "
                "N=1k to 100k (mean pairwise cosine ≈ 0.54 throughout). "
                "Routing compute cost is 1-10 µs/query — negligible at "
                "100k. W projection capacity is sufficient when query "
                "and stored content overlap (97% top-1 at N=100k for "
                "the same-chunk degenerate test). **The actual binding "
                "constraint is paraphrase robustness:** when the query "
                "is a non-trivial paraphrase of the stored content "
                "(disjoint half of the same article), routing degrades "
                "from 97% top-1 (same-chunk) to 8% top-1 (split halves) "
                "regardless of library size. The architecture scales "
                "structurally; what doesn't scale is V22-Dickens's "
                "ability to maintain semantic alignment across "
                "paraphrases.")
    out.append("")

    out.append("## Setup\n")
    out.append("- **Base model:** V22-Dickens (HRSTransformer, 6 layers, "
                "1024d, GPT-2 BPE) — the canonical Phase 47 substrate.")
    out.append("- **Library:** 100,000 non-overlapping 200-token chunks "
                "from WikiText-103 train. Tokenized once at startup; "
                "saved to disk as `chunks.npy` for reproducibility.")
    out.append("- **Stored engram (Phase 47 canonical):** L5-mean of the "
                "FIRST 100 tokens of each chunk. Computed via "
                "`hidden_at_layer(model, ids, 5).mean(dim=1)`. No LoRA.")
    out.append("- **W projection:** trained on the first 1000 chunks via "
                "Phase 47 InfoNCE (1024×1024 linear, identity init, "
                "500 steps, lr 1e-3, temp 0.05). Achieved 100% train "
                "accuracy in all three query schemes.")
    out.append("- **Eval queries:** chunks 1000..1099 (held out from W "
                "training). 100 queries per (scheme, library size).")
    out.append("- **Library sizes:** {1k, 5k, 20k, 50k, 100k}.")
    out.append("")
    out.append("**Three query schemes**, varying how the \"paraphrase\" "
                "is constructed:")
    out.append("- **A — same-chunk (upper bound):** query = L0(first 100 "
                "tokens), stored = L5(first 100 tokens). Same input to "
                "both forward modes. Tests whether W can learn the "
                "L0→L5 mapping for V22-Dickens. Not a realistic "
                "deployment query — it's the ceiling.")
    out.append("- **B — overlapping (50-token shift):** query = L0(tokens "
                "[50:150]), stored = L5(tokens [0:100]). Query and "
                "stored share half their tokens.")
    out.append("- **C — split halves (disjoint):** query = L0(tokens "
                "[100:200]), stored = L5(tokens [0:100]). Query and "
                "stored are consecutive but disjoint windows of the "
                "same article. The hardest paraphrase test that's "
                "still meaningfully paired.")
    out.append("")

    out.append("## Routing accuracy by library size and query scheme\n")
    out.append("(top-1 / top-5 / top-10 over 100 held-out queries; N=1000 "
                "row dropped because the 100 eval queries are outside the "
                "library at that size, so top-K is undefined.)")
    out.append("")
    out.append("| scheme | N | top-1 | top-5 | top-10 |")
    out.append("|---|---:|---:|---:|---:|")
    for s in schemes:
        for r in s["rows"][1:]:
            out.append(f"| {s['label']} | {r['N']} | "
                        f"{r['top1']:.3f} | {r['top5']:.3f} | "
                        f"{r['top10']:.3f} |")
    out.append("")

    out.append("## Engram separation (sampled 10k random pairs at each N)\n")
    out.append("Computed once; the schemes share stored engrams.")
    out.append("")
    out.append("| N | mean | max | p90 | min |")
    out.append("|---:|---:|---:|---:|---:|")
    for r in schemes[0]["rows"]:
        sp = r["sep"]
        out.append(f"| {r['N']} | {sp['mean']:.3f} | {sp['max']:.3f} | "
                    f"{sp['p90']:.3f} | {sp['min']:.3f} |")
    out.append("")

    out.append("## Routing compute cost\n")
    out.append("Wall time per query (including the W projection, cosine "
                "vs N stored engrams, and argmax).")
    out.append("")
    out.append("| N | µs / query |")
    out.append("|---:|---:|")
    for r in schemes[0]["rows"]:
        out.append(f"| {r['N']} | {r['cost_us']:.1f} |")
    out.append("")

    out.append("![curves](library_scaling.png)\n")

    out.append("## Reading the curves\n")
    out.append("**Routing accuracy is determined by paraphrase quality, "
                "not library size.** All three schemes show top-1 that "
                "is roughly flat as N grows from 5k to 100k:")
    out.append("- A (same-chunk): 0.99 → 0.98 → 0.97 → 0.97. Architecture "
                "scales perfectly when content matches.")
    out.append("- B (50-token shift): 0.16 → 0.15 → 0.09 → 0.08.")
    out.append("- C (split halves): 0.05 → 0.03 → 0.01 → 0.01.")
    out.append("")
    out.append("There IS a small monotonic drop with N (e.g. A goes 0.99 "
                "→ 0.97 over a 20× scale increase) — but it's a 2pp drop, "
                "not a collapse. Top-5 in scheme A stays at 1.000 across "
                "all sizes. The correct answer is reliably in the "
                "neighborhood; argmax occasionally picks a near-neighbor.")
    out.append("")
    out.append("**Engram separation is flat with size.** Mean pairwise "
                "cosine stays at 0.54 from N=1k to N=100k. The expected "
                "\"separation degrades at scale\" pattern does NOT "
                "appear with WT-103 content. (The earlier separation_reg "
                "experiment saw drift from 0.35 to 0.45 between N=10 and "
                "N=200 on synthetic templated content; that was a "
                "small-scale, content-similarity artifact.)")
    out.append("")
    out.append("**Routing cost is negligible.** µs-scale. At N=100k it's "
                "10 µs per query — that's 100,000 routing decisions/sec on "
                "a single RTX 5070 Ti. Computation is not the constraint "
                "until libraries are 1M+.")
    out.append("")

    out.append("## What this tells us about the architecture\n")
    out.append("**The three failure modes the spec hypothesized "
                "(separation, W capacity, compute) all stayed within "
                "useful bounds at 100k scale.** None of them is the "
                "binding constraint.")
    out.append("")
    out.append("**The actual binding constraint is the L0→L5 paraphrase "
                "transfer.** When the query and stored content are "
                "literally the same input (scheme A), W learns the "
                "L0→L5 mapping and routes 97-99% top-1 across all "
                "library sizes. When the query is a paraphrase — "
                "even one that overlaps the stored content by 50% "
                "(scheme B) — top-1 collapses to 8-16%. For genuinely "
                "disjoint paraphrases (scheme C), top-1 is essentially "
                "noise.")
    out.append("")
    out.append("**Why does this happen?** V22-Dickens (a 6-layer model) "
                "has a brittle L0→L5 mapping that doesn't generalize "
                "across content variations. The position-erosion "
                "experiment showed late hidden states sit far from "
                "vocabulary directions; here we see that the L0→L5 "
                "transformation is essentially per-input-specific. W "
                "memorizes the 1000 training pairs (train_acc 100% "
                "always) but doesn't extract a general L0→L5 rule.")
    out.append("")
    out.append("**Implications for the architecture:**")
    out.append("1. The **scale story is fine.** 100k libraries are "
                "feasible without architectural changes — separation "
                "doesn't collapse, compute doesn't bind, W has enough "
                "capacity for matched content.")
    out.append("2. The **paraphrase story is broken on V22-Dickens.** "
                "Phase 47's high routing accuracy on 50 Dickens "
                "passages relied on TEMPLATED paraphrases that "
                "shared most of their tokens with training paraphrases. "
                "On naturalistic paraphrases (split halves of WT-103 "
                "articles), routing collapses regardless of library "
                "size.")
    out.append("3. The **fix is a more capable base model**, not "
                "architectural tweaks to engram routing. A larger base "
                "(Mistral-7B, Llama) with deeper, more robust "
                "representations should produce L0/L5 means whose "
                "linear bridge generalizes better across paraphrases. "
                "This is testable and the natural next experiment.")
    out.append("4. **Phase 47's 50-Dickens result transfers if "
                "paraphrases are templated** but not if they're "
                "naturalistic. Document the templated-paraphrase "
                "constraint when describing Phase 47's deployment "
                "scope.")
    out.append("")

    out.append("## Caveats / deviations\n")
    out.append("1. **At N=1000 the held-out queries are outside the "
                "library.** The library at N=1000 is chunks 0..999; "
                "queries are chunks 1000..1099. Top-K is undefined at "
                "that row. (The script reports it as 0.000 by "
                "convention.)")
    out.append("2. **Adapters NOT trained.** Per the spec, this "
                "experiment uses base-model engrams across all 100k "
                "passages — no individual LoRA training. We measure "
                "ROUTING only, not retrieval. A separate retrieval "
                "test would require training 100k adapters, which is "
                "infeasible at this scale.")
    out.append("3. **Single base model.** V22-Dickens (the canonical "
                "Phase 47 substrate). Results may differ on a more "
                "capable base — the Mistral-7B four-way comparison "
                "experiment showed RAG works on Mistral where it "
                "didn't on V22, hinting that routing might also "
                "generalize better on a stronger base.")
    out.append("4. **Synthetic paraphrase via window-shifting.** "
                "Genuine paraphrases (rephrased queries) would be a "
                "more rigorous test, but at 100k scale we don't have "
                "ground-truth paraphrase pairs. Window-shifting is the "
                "best-available proxy.")
    out.append("")

    out.append("## Wall-clock totals\n")
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append("| Tokenize WT-103, build 100k chunks | ~30 s |")
    out.append("| Compute 100k stored L5 engrams | 158 s (~3 min) |")
    out.append("| Compute 1100 query L0 engrams (3 windows) | <1 s |")
    out.append("| Train W (3 schemes × 500 InfoNCE steps) | ~5 s |")
    out.append("| Measurement at all 5 sizes × 3 schemes | ~5 s |")
    out.append("| **Total** | **~3.5 min** |")
    out.append("")

    out.append("## Files\n")
    out.append("- `build_engrams.py` — tokenize WT-103 + compute 100k "
                "stored L5 + 1100 query L0 (split-halves only).")
    out.append("- `train_w.py` — original W training (split-halves only).")
    out.append("- `measure.py` — original measurement (used split-halves; "
                "produced the misleading initial result).")
    out.append("- `sanity.py` — discovered the upper-bound + degradation "
                "by querying scheme.")
    out.append("- `measure2.py` — corrected measurement across 3 query "
                "schemes; this is the canonical run.")
    out.append("- `aggregate.py` — this writeup.")
    out.append("- `results/library_scaling.png` — 4-panel figure.")
    out.append("- `results/measure_v2.json` — raw data.")
    out.append("- `results/RESULT.md` — this file.")

    out_md = LS / "results/RESULT.md"
    out_md.write_text("\n".join(out))
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
