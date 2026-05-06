"""Generate figures for trainable-decoder sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/trainable_decoder"


def main():
    d = json.loads((EXP / "results/sweep_summary.json").read_text())
    snaps = d["snapshots"]
    Ns = [s["N"] for s in snaps]
    mean = [s["mean_retrieval"] for s in snaps]
    std = [s["std_retrieval"] for s in snaps]
    minr = [s["min_retrieval"] for s in snaps]
    maxr = [s["max_retrieval"] for s in snaps]
    ad1 = [s["adapter_1_retrieval"] for s in snaps]
    adN = [s["adapter_N_retrieval"] for s in snaps]

    # Figure 1: forgetting curve
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(Ns, ad1, marker='o', linestyle='-', color='crimson',
             linewidth=2, label='adapter 1 (oldest)')
    ax.plot(Ns, adN, marker='s', linestyle='-', color='steelblue',
             linewidth=2, label='adapter N (most recent)')
    ax.plot(Ns, mean, marker='^', linestyle='--', color='black',
             linewidth=1.5, alpha=0.7, label='mean across all adapters')
    ax.axhline(0.93, color='gray', linestyle=':', alpha=0.6,
                label='frozen-decoder baseline (~0.93)')
    ax.axhline(0.85, color='red', linestyle=':', alpha=0.4,
                label='spec acceptable threshold (0.85)')
    ax.set_xscale('log')
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('sweep point N (adapters trained)')
    ax.set_ylabel('retrieval accuracy')
    ax.set_title('Forgetting curve — shared trainable decoder, Dickens-50')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='center left', fontsize=9)
    for n, a, b in zip(Ns, ad1, adN):
        ax.annotate(f"{a:.2f}", (n, a), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=8, color='crimson')
        ax.annotate(f"{b:.2f}", (n, b), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8, color='steelblue')
    plt.tight_layout()
    (EXP / "figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(EXP / "figures/forgetting_curve.png", dpi=120)
    plt.close(fig)
    print(f"Saved {EXP/'figures/forgetting_curve.png'}")

    # Figure 2: per-position retrieval at each snapshot (heatmap-ish)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.fill_between(Ns, minr, maxr, alpha=0.2, color='steelblue',
                     label='min/max range')
    ax.plot(Ns, mean, marker='o', color='steelblue', linewidth=2,
             label='mean')
    ax.errorbar(Ns, mean, yerr=std, fmt='none', color='steelblue', alpha=0.5)
    ax.axhline(0.93, color='gray', linestyle=':', alpha=0.6,
                label='frozen-decoder baseline')
    ax.axhline(0.85, color='red', linestyle=':', alpha=0.4,
                label='0.85 threshold')
    ax.set_xscale('log')
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('sweep point N')
    ax.set_ylabel('retrieval')
    ax.set_title('Capacity curve — mean ± std + min/max')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)

    # Per-fact-type
    ax = axes[1]
    types = ["entity", "numeric", "place", "relation"]
    colors = {"entity": "steelblue", "numeric": "darkorange",
              "place": "seagreen", "relation": "crimson"}
    for t in types:
        xs = []; ys = []
        for s in snaps:
            if t in s["per_fact_type"]:
                xs.append(s["N"]); ys.append(s["per_fact_type"][t])
        if xs:
            ax.plot(xs, ys, marker='o', linestyle='-', color=colors[t],
                     linewidth=2, label=t)
    ax.set_xscale('log')
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel('sweep point N')
    ax.set_ylabel('retrieval')
    ax.set_title('Per-fact-type retrieval across the sweep')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(EXP / "figures/capacity_curve.png", dpi=120)
    plt.close(fig)
    print(f"Saved {EXP/'figures/capacity_curve.png'}")


if __name__ == "__main__":
    main()
