"""Aggregate the 3-seed sweep, fit Spearman correlation, generate figures + table."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_corpus_dependency"

CORPORA = ["ts", "code", "wt103"]
VARIANTS = ["baseline", "compressed"]
SEEDS = [0, 1, 2]


def main():
    # Load MI ratios
    mi = {}
    for c in CORPORA:
        d = json.loads((EXP / f"results/mi_{c}.json").read_text())
        mi[c] = d["mi_ratio_16_over_1"]

    # Load all training results
    rows = {}
    for c in CORPORA:
        for v in VARIANTS:
            ppls = []; anchors = []
            for s in SEEDS:
                p = EXP / f"results/train_{c}_{v}_seed{s}.json"
                d = json.loads(p.read_text())
                ppls.append(d["final_val_ppl"])
                anchors.append(d["anchor_acc"])
            rows[(c, v)] = {"ppl": ppls, "anchor": anchors,
                              "ppl_mean": mean(ppls), "ppl_std": stdev(ppls),
                              "anchor_mean": mean(anchors),
                              "anchor_std": stdev(anchors)}

    print(f"\n{'='*100}")
    print("3-SEED SWEEP RESULTS")
    print(f"{'='*100}\n")
    print(f"{'corpus':>8} {'MI@16/1':>9} | {'baseline PPL':>20} | {'compressed PPL':>20} | "
          f"{'gap':>14} | {'gap %':>10}")
    print("-" * 100)
    summary = []
    for c in CORPORA:
        b = rows[(c, "baseline")]
        cp = rows[(c, "compressed")]
        # Gap is per-seed (compressed - baseline) at the same seed; report mean & std
        gaps = [cp["ppl"][i] - b["ppl"][i] for i in range(len(SEEDS))]
        gap_pcts = [100 * (cp["ppl"][i] - b["ppl"][i]) / b["ppl"][i]
                    for i in range(len(SEEDS))]
        gap_mean = mean(gaps); gap_std = stdev(gaps)
        gap_pct_mean = mean(gap_pcts); gap_pct_std = stdev(gap_pcts)
        print(f"{c:>8} {mi[c]:9.4f} | "
              f"{b['ppl_mean']:8.2f} ± {b['ppl_std']:6.2f} | "
              f"{cp['ppl_mean']:8.2f} ± {cp['ppl_std']:6.2f} | "
              f"{gap_mean:+7.2f} ± {gap_std:5.2f} | "
              f"{gap_pct_mean:+5.2f}% ± {gap_pct_std:.2f}%")
        summary.append({
            "corpus": c, "mi_ratio": mi[c],
            "baseline_ppl_mean": b["ppl_mean"], "baseline_ppl_std": b["ppl_std"],
            "compressed_ppl_mean": cp["ppl_mean"],
            "compressed_ppl_std": cp["ppl_std"],
            "gap_mean": gap_mean, "gap_std": gap_std,
            "gap_pct_mean": gap_pct_mean, "gap_pct_std": gap_pct_std,
            "baseline_anchor_mean": b["anchor_mean"],
            "baseline_anchor_std": b["anchor_std"],
            "compressed_anchor_mean": cp["anchor_mean"],
            "compressed_anchor_std": cp["anchor_std"],
            "gap_signal_to_noise": abs(gap_mean) / gap_std if gap_std > 0 else float("inf"),
        })
    print()

    # Anchor accuracy
    print(f"{'corpus':>8} | {'baseline anchor':>20} | {'compressed anchor':>20}")
    print("-" * 60)
    for c in CORPORA:
        b = rows[(c, "baseline")]
        cp = rows[(c, "compressed")]
        print(f"{c:>8} | "
              f"{b['anchor_mean']:.3f} ± {b['anchor_std']:.3f}    | "
              f"{cp['anchor_mean']:.3f} ± {cp['anchor_std']:.3f}")
    print()

    # Spearman correlation: gap_pct vs MI ratio
    mi_vals = [s["mi_ratio"] for s in summary]
    gap_vals = [s["gap_pct_mean"] for s in summary]
    # Rank
    mi_rank = sorted(range(len(mi_vals)), key=lambda i: mi_vals[i])
    gap_rank = sorted(range(len(gap_vals)), key=lambda i: gap_vals[i])
    ranks_mi = [mi_rank.index(i) + 1 for i in range(len(mi_vals))]
    ranks_gap = [gap_rank.index(i) + 1 for i in range(len(gap_vals))]
    n = len(mi_vals)
    d2 = sum((ranks_mi[i] - ranks_gap[i])**2 for i in range(n))
    spearman = 1 - 6 * d2 / (n * (n*n - 1))

    # Pearson for completeness
    mi_arr = np.array(mi_vals); gap_arr = np.array(gap_vals)
    pearson = float(np.corrcoef(mi_arr, gap_arr)[0, 1])

    print(f"\n{'='*100}\nCORRELATION ANALYSIS\n{'='*100}")
    print(f"Corpora ranked by MI ratio:")
    for i, (c, m, g) in enumerate(zip(CORPORA, mi_vals, gap_vals)):
        print(f"  rank-MI={ranks_mi[i]}  rank-gap={ranks_gap[i]}  "
              f"{c}: MI={m:.3f}  gap%={g:+.2f}")
    print(f"\nSpearman rank correlation (MI ratio vs gap %): rho = {spearman:.3f}")
    print(f"Pearson correlation:                              r   = {pearson:.3f}")
    print(f"Predicted sign: NEGATIVE (higher MI -> smaller gap)")
    print(f"Observed sign:  {'NEGATIVE' if spearman < 0 else 'POSITIVE'}")
    print(f"With n=3 the test is severely underpowered; |rho|=1 would be the only "
          f"unambiguous direction.")

    # Save
    out = {
        "summary": summary,
        "spearman_rho": spearman, "pearson_r": pearson,
        "n_corpora": len(CORPORA), "n_seeds": len(SEEDS),
        "predicted_sign_of_correlation": "negative",
        "observed_sign": "negative" if spearman < 0 else "positive",
    }
    (EXP / "results/aggregate.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {EXP / 'results/aggregate.json'}")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = mi_vals; ys = gap_vals
        ystds = [s["gap_pct_std"] for s in summary]
        ax.errorbar(xs, ys, yerr=ystds, fmt='o', markersize=10, capsize=5,
                     color='steelblue')
        for x, y, c in zip(xs, ys, CORPORA):
            ax.annotate(c, (x, y), textcoords="offset points",
                         xytext=(8, 0), fontsize=11, va='center')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel("MI ratio MI(X_t; X_{t+16}) / MI(X_t; X_{t+1})")
        ax.set_ylabel("PPL gap (%): (compressed - baseline) / baseline × 100")
        ax.set_title(f"Compression PPL gap vs corpus MI ratio "
                      f"(3 seeds; rho={spearman:.2f}, r={pearson:.2f})")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        (EXP / "figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(EXP / "figures/gap_vs_mi.png", dpi=120)
        plt.close(fig)
        print(f"Saved {EXP / 'figures/gap_vs_mi.png'}")
    except Exception as e:
        print(f"WARN: figure failed: {e}")


if __name__ == "__main__":
    main()
