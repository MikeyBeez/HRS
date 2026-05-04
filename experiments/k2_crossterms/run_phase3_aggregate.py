"""Phase 3 — aggregate all conditions into one summary table + two figures.

Reads:
  results/phase1_summary.json
  results/phase2_no_retrain_summary.json   (Taylor + discrete)
  results/eval_orthogonal_lambda_0.01.json
  results/eval_orthogonal_lambda_0.1.json
  results/eval_orthogonal_lambda_1.0.json
  results/eval_crosstermaware.json

Writes:
  results/phase3_summary.csv        condition x metrics
  figures/layer_drift.png           per-layer drift (Phase 1 vanilla)
  figures/retrieval_comparison.png  retrieval bar chart with N=1 reference

Decision:
  Pick interventions where N=2 is within 3 pts of single-adapter (target
  >= 0.894 given 0.924 baseline) AND single-adapter retrieval doesn't
  drop more than 2 pts (>= 0.904).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/k2_crossterms"


CONDITIONS_TO_REPORT = [
    ("vanilla_addition",    "phase1_summary.json",     None),
    ("taylor",              "phase2_no_retrain_summary.json", "taylor"),
    ("discrete",            "phase2_no_retrain_summary.json", "discrete"),
    ("orthogonal_l_0.01",   "eval_orthogonal_lambda_0.01.json", None),
    ("orthogonal_l_0.1",    "eval_orthogonal_lambda_0.1.json",  None),
    ("orthogonal_l_1.0",    "eval_orthogonal_lambda_1.0.json",  None),
    ("crosstermaware",      "eval_crosstermaware.json", None),
]


def get_n1_n2(filename, key=None):
    """Load (n1, n2) from a results JSON with various shapes."""
    p = EXP / "results" / filename
    if not p.exists():
        return None, None
    data = json.loads(p.read_text())
    if filename == "phase1_summary.json":
        return data.get("n1_overall_retrieval"), data.get("n2_overall_retrieval")
    if filename == "phase2_no_retrain_summary.json":
        ctx = data.get("context", {})
        n1 = ctx.get("phase1_n1")
        if key == "taylor":
            return n1, data["taylor"]["overall"]
        if key == "discrete":
            return n1, data["discrete"]["overall"]
        return n1, None
    # generic eval_*.json
    return data.get("n1_overall"), data.get("n2_overall")


def main():
    rows = []
    for cond, filename, key in CONDITIONS_TO_REPORT:
        n1, n2 = get_n1_n2(filename, key)
        rows.append({
            "condition": cond,
            "n1_retrieval": n1,
            "n2_retrieval": n2,
            "gap_n1_n2": (None if (n1 is None or n2 is None) else n1 - n2),
            "available": n1 is not None and n2 is not None,
        })

    # CSV
    csv_path = EXP / "results/phase3_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n1_retrieval", "n2_retrieval", "gap_n1_n2"])
        for r in rows:
            w.writerow([r["condition"],
                         f"{r['n1_retrieval']:.3f}" if r["n1_retrieval"] is not None else "",
                         f"{r['n2_retrieval']:.3f}" if r["n2_retrieval"] is not None else "",
                         f"{r['gap_n1_n2']:.3f}" if r["gap_n1_n2"] is not None else ""])

    print(f"\n{'='*78}\nPHASE 3 AGGREGATE\n{'='*78}")
    print(f"  {'condition':<20s} {'N=1':>8s}  {'N=2':>8s}  {'gap':>8s}  {'within 3pt?':>12s}")
    print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
    for r in rows:
        n1 = f"{r['n1_retrieval']:.3f}" if r["n1_retrieval"] is not None else "—"
        n2 = f"{r['n2_retrieval']:.3f}" if r["n2_retrieval"] is not None else "—"
        gap = f"{r['gap_n1_n2']:.3f}" if r["gap_n1_n2"] is not None else "—"
        ok = ""
        if r["gap_n1_n2"] is not None:
            ok = "yes" if r["gap_n1_n2"] <= 0.03 else ""
        print(f"  {r['condition']:<20s} {n1:>8s}  {n2:>8s}  {gap:>8s}  {ok:>12s}")
    print(f"\nSaved {csv_path}")

    # Decision
    decision = "no intervention closes the gap"
    for r in rows:
        if (r["gap_n1_n2"] is not None and r["gap_n1_n2"] <= 0.03
                and r["n1_retrieval"] >= 0.904
                and r["condition"] != "vanilla_addition"):
            decision = f"adopt {r['condition']} (gap {r['gap_n1_n2']:.3f}, N=1 {r['n1_retrieval']:.3f})"
            break
    print(f"\nDECISION: {decision}")

    # Figures (matplotlib optional — try, skip on failure)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Figure 1: per-layer drift profile from Phase 1
        p1 = json.loads((EXP / "results/phase1_summary.json").read_text())
        prof = p1["layer_drift_profile"]
        layers = [p["layer"] for p in prof]
        means = [p["mean"] for p in prof]
        stds = [p["std"] for p in prof]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(layers, means, yerr=stds, marker='o', linestyle='-',
                     color='steelblue', capsize=4)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"Drift $\|h_{i,j}^L - h_i^L\| / \|h_i^L\|$")
        ax.set_title("Per-layer activation drift, k=2 vanilla composition")
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.grid(alpha=0.3)
        ax.set_xticks(layers)
        for L, m in zip(layers, means):
            ax.annotate(f"{m:.3f}", (L, m), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8)
        plt.tight_layout()
        (EXP / "figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(EXP / "figures/layer_drift.png", dpi=120)
        plt.close(fig)
        print(f"Saved {EXP/'figures/layer_drift.png'}")

        # Figure 2: retrieval bar chart
        labels = [r["condition"] for r in rows if r["available"]]
        n2_vals = [r["n2_retrieval"] for r in rows if r["available"]]
        n1_ref = rows[0]["n1_retrieval"]  # vanilla N=1 baseline
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ['#888888' if v is None else
                  ('#cc4444' if r["gap_n1_n2"] > 0.05 else
                   ('#dd9933' if r["gap_n1_n2"] > 0.03 else '#44aa66'))
                  for r, v in zip([r for r in rows if r["available"]], n2_vals)]
        bars = ax.bar(labels, n2_vals, color=colors)
        ax.axhline(n1_ref, color='black', linestyle='--',
                    label=f"N=1 baseline ({n1_ref:.3f})")
        ax.axhline(n1_ref - 0.03, color='gray', linestyle=':',
                    label=f"3-pt threshold ({n1_ref - 0.03:.3f})")
        ax.set_ylabel("k=2 retrieval")
        ax.set_title("k=2 retrieval across interventions")
        ax.set_ylim(0, max(0.95, n1_ref + 0.05))
        plt.xticks(rotation=20, ha='right')
        for b, v in zip(bars, n2_vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width()/2, v),
                         textcoords="offset points", xytext=(0, 3),
                         ha='center', fontsize=8)
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(EXP / "figures/retrieval_comparison.png", dpi=120)
        plt.close(fig)
        print(f"Saved {EXP/'figures/retrieval_comparison.png'}")
    except Exception as e:
        print(f"WARN: figure generation failed: {e}")


if __name__ == "__main__":
    main()
