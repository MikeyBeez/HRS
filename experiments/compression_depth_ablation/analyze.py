"""Aggregate depth ablation: pull existing depth_4/baseline + new depth_6/8."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_depth_ablation"
PRIOR = REPO / "experiments/compression_corpus_dependency/results"

CORPORA = ["ts", "wt103"]
SEEDS = [0, 1, 2]


def load_metric(path):
    d = json.loads(path.read_text())
    return {"ppl": d["final_val_ppl"], "anchor": d["anchor_acc"]}


def collect(corpus):
    """Return dict: depth -> list of {ppl, anchor} per seed."""
    out = {}
    # Baseline (no compression) — from corpus_dependency
    out["baseline"] = []
    for s in SEEDS:
        p = PRIOR / f"train_{corpus}_baseline_seed{s}.json"
        out["baseline"].append(load_metric(p))
    # depth_4 — reuse from corpus_dependency (file uses "compressed")
    out["depth_4"] = []
    for s in SEEDS:
        p = PRIOR / f"train_{corpus}_compressed_seed{s}.json"
        out["depth_4"].append(load_metric(p))
    # depth_6 + depth_8 — from this experiment
    for d in ["depth_6", "depth_8"]:
        out[d] = []
        for s in SEEDS:
            p = EXP / f"results/train_{corpus}_{d}_seed{s}.json"
            out[d].append(load_metric(p))
    return out


def main():
    rows = {}
    for c in CORPORA:
        rows[c] = collect(c)

    print(f"\n{'='*100}\nDEPTH ABLATION SUMMARY (3 seeds each)\n{'='*100}\n")
    summary = {}
    for c in CORPORA:
        print(f"--- corpus: {c} ---")
        print(f"{'depth':>10} {'val PPL (mean ± std)':>26} {'anchor (mean ± std)':>22} "
              f"{'gap vs base PPL':>20}")
        base_ppls = [r["ppl"] for r in rows[c]["baseline"]]
        base_mean = mean(base_ppls)
        for d in ["baseline", "depth_4", "depth_6", "depth_8"]:
            ppls = [r["ppl"] for r in rows[c][d]]
            ancs = [r["anchor"] for r in rows[c][d]]
            ppl_mean = mean(ppls); ppl_std = stdev(ppls)
            anc_mean = mean(ancs); anc_std = stdev(ancs)
            if d == "baseline":
                gap_str = "—"
            else:
                gap_pct = (ppl_mean - base_mean) / base_mean * 100
                gap_str = f"{ppl_mean - base_mean:+8.2f} ({gap_pct:+5.2f}%)"
            print(f"{d:>10} {ppl_mean:8.2f} ± {ppl_std:6.2f}{'':>5} "
                  f"{anc_mean:.3f} ± {anc_std:.3f}{'':>5} {gap_str}")
            summary.setdefault(c, {})[d] = {
                "ppl_mean": ppl_mean, "ppl_std": ppl_std,
                "anchor_mean": anc_mean, "anchor_std": anc_std,
                "ppls_per_seed": ppls,
            }
        print()

    # Cross-corpus comparison
    print(f"\n{'='*100}\nGAP-VS-BASELINE BY DEPTH\n{'='*100}\n")
    print(f"{'corpus':>8} {'depth':>10} {'gap %':>10} {'gap-S/N':>10}")
    print("-" * 45)
    for c in CORPORA:
        base_ppls = summary[c]["baseline"]["ppls_per_seed"]
        base_mean = mean(base_ppls)
        for d in ["depth_4", "depth_6", "depth_8"]:
            comp_ppls = summary[c][d]["ppls_per_seed"]
            gaps_pct = [(comp_ppls[i] - base_ppls[i]) / base_ppls[i] * 100
                        for i in range(len(base_ppls))]
            gap_mean = mean(gaps_pct); gap_std = stdev(gaps_pct)
            sn = abs(gap_mean) / gap_std if gap_std > 0 else float("inf")
            print(f"{c:>8} {d:>10} {gap_mean:+7.2f}% ± {gap_std:4.2f}% {sn:7.1f}")
    print()

    # Save
    (EXP / "results/aggregate.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {EXP / 'results/aggregate.json'}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, c in zip(axes, CORPORA):
            depths = [4, 6, 8]
            base_mean = summary[c]["baseline"]["ppl_mean"]
            base_std = summary[c]["baseline"]["ppl_std"]
            comp_means = [summary[c][f"depth_{d}"]["ppl_mean"] for d in depths]
            comp_stds = [summary[c][f"depth_{d}"]["ppl_std"] for d in depths]
            ax.axhline(base_mean, color='gray', linestyle='--',
                        label=f"baseline ({base_mean:.1f})")
            ax.fill_between([3.5, 8.5], base_mean - base_std, base_mean + base_std,
                              alpha=0.15, color='gray')
            ax.errorbar(depths, comp_means, yerr=comp_stds, fmt='o-',
                          markersize=10, capsize=5, color='steelblue',
                          label="compressed")
            for d, m in zip(depths, comp_means):
                ax.annotate(f"{m:.1f}", (d, m), textcoords="offset points",
                             xytext=(0, 8), ha='center', fontsize=9)
            ax.set_xlabel("compression stack depth (layers)")
            ax.set_ylabel("validation PPL")
            ax.set_title(f"corpus: {c}")
            ax.set_xticks([4, 6, 8])
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(alpha=0.3)
        plt.tight_layout()
        (EXP / "figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(EXP / "figures/depth_ablation.png", dpi=120)
        plt.close(fig)
        print(f"Saved {EXP / 'figures/depth_ablation.png'}")
    except Exception as e:
        print(f"WARN: figure failed: {e}")


if __name__ == "__main__":
    main()
