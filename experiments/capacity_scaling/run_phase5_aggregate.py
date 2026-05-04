"""Phase 5 — aggregate scaling results into table + figures."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/capacity_scaling"


SIZE_LABELS = [
    ("size_01",  1),
    ("size_05",  5),
    ("size_25", 25),
    ("size_50", 50),
]


def main():
    rows = []
    for label, size in SIZE_LABELS:
        p = EXP / f"results/eval_{label}.json"
        if not p.exists():
            print(f"WARN: {p} missing — skipping")
            continue
        d = json.loads(p.read_text())
        rows.append({
            "size": size,
            "label": label,
            "training_time_s": d["training_time_s"],
            "n_steps": d["n_steps"],
            "final_loss": d["final_loss_mean50"],
            "overall_retrieval": d["overall_retrieval"],
            "per_passage_mean": d["per_passage_mean"],
            "per_passage_min": d["per_passage_min"],
            "per_passage_max": d["per_passage_max"],
            "per_fact_type": d["per_fact_type"],
        })

    # CSV
    csv_path = EXP / "results/phase5_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "adapter_size", "training_time_s", "n_steps", "final_loss",
            "overall_retrieval", "per_passage_mean", "per_passage_min",
            "per_passage_max",
            "ret_entity", "ret_numeric", "ret_place", "ret_relation",
        ])
        for r in rows:
            ft = r["per_fact_type"]
            w.writerow([
                r["size"],
                f"{r['training_time_s']:.1f}",
                r["n_steps"],
                f"{r['final_loss']:.4f}",
                f"{r['overall_retrieval']:.3f}",
                f"{r['per_passage_mean']:.3f}",
                f"{r['per_passage_min']:.3f}",
                f"{r['per_passage_max']:.3f}",
                f"{ft.get('entity', float('nan')):.3f}",
                f"{ft.get('numeric', float('nan')):.3f}",
                f"{ft.get('place', float('nan')):.3f}",
                f"{ft.get('relation', float('nan')):.3f}",
            ])

    print(f"\n{'='*82}\nPHASE 5 — capacity scaling summary\n{'='*82}")
    print(f"  {'size':>4s}  {'train_s':>8s}  {'n_steps':>8s}  {'overall':>8s}  "
          f"{'pp_mean':>8s}  {'pp_min':>7s}  {'pp_max':>7s}  "
          f"{'entity':>7s} {'numeric':>7s} {'place':>7s} {'relation':>7s}")
    for r in rows:
        ft = r["per_fact_type"]
        print(f"  {r['size']:4d}  {r['training_time_s']:8.0f}  "
              f"{r['n_steps']:8d}  {r['overall_retrieval']:8.3f}  "
              f"{r['per_passage_mean']:8.3f}  {r['per_passage_min']:7.3f}  "
              f"{r['per_passage_max']:7.3f}  "
              f"{ft.get('entity', float('nan')):7.3f} "
              f"{ft.get('numeric', float('nan')):7.3f} "
              f"{ft.get('place', float('nan')):7.3f} "
              f"{ft.get('relation', float('nan')):7.3f}")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = [r["size"] for r in rows]
        rets = [r["overall_retrieval"] for r in rows]
        rets_min = [r["per_passage_min"] for r in rows]
        rets_max = [r["per_passage_max"] for r in rows]
        train_s = [r["training_time_s"] for r in rows]

        # Figure 1: retrieval vs size
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sizes, rets, marker='o', linestyle='-', color='steelblue',
                 label='overall', linewidth=2)
        ax.fill_between(sizes, rets_min, rets_max, alpha=0.2, color='steelblue',
                         label='per-passage min/max range')
        ax.axhline(0.93, color='gray', linestyle='--', linewidth=0.8,
                    label='Dickens-50 single-adapter baseline (0.93)')
        ax.axhline(0.70, color='red', linestyle=':', linewidth=0.8,
                    label='spec acceptable threshold (0.70)')
        ax.set_xlabel('adapter size (passages combined)')
        ax.set_ylabel('retrieval accuracy')
        ax.set_title('rank-128 LoRA capacity scaling on Dickens-50')
        ax.set_xscale('log')
        ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        for s, r in zip(sizes, rets):
            ax.annotate(f"{r:.3f}", (s, r), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8)
        plt.tight_layout()
        (EXP / "figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(EXP / "figures/retrieval_vs_size.png", dpi=120)
        plt.close(fig)
        print(f"\nSaved {EXP/'figures/retrieval_vs_size.png'}")

        # Figure 2: training time vs size
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sizes, train_s, marker='s', linestyle='-', color='darkorange',
                 linewidth=2)
        ax.set_xlabel('adapter size (passages combined)')
        ax.set_ylabel('training wall time (s)')
        ax.set_title('Training time scaling')
        ax.set_xscale('log')
        ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes])
        ax.grid(alpha=0.3)
        for s, t in zip(sizes, train_s):
            ax.annotate(f"{t:.0f}s", (s, t), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(EXP / "figures/training_time_vs_size.png", dpi=120)
        plt.close(fig)
        print(f"Saved {EXP/'figures/training_time_vs_size.png'}")
    except Exception as e:
        print(f"WARN: figure generation failed: {e}")

    # Decision
    decision = ""
    if all(r["overall_retrieval"] >= 0.90 for r in rows):
        decision = "FLAT — capacity not binding even at size 50"
    elif rows and rows[-1]["overall_retrieval"] >= 0.70:
        decision = "SMOOTH DEGRADATION — capacity bounded but stays above 0.70 at size 50"
    else:
        # Look for cliff
        prev = rows[0]["overall_retrieval"] if rows else 1.0
        cliff_at = None
        for r in rows[1:]:
            if prev - r["overall_retrieval"] > 0.20:
                cliff_at = r["size"]
                break
            prev = r["overall_retrieval"]
        if cliff_at:
            decision = f"CLIFF — sharp drop between sizes near {cliff_at}"
        else:
            decision = "BELOW THRESHOLD — retrieval falls below 0.70 by size 50"
    print(f"\n  Scaling regime: {decision}")

    summary = {
        "phase": 5,
        "scaling_regime": decision,
        "by_size": rows,
    }
    (EXP / "results/phase5_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {EXP/'results/phase5_summary.json'}")


if __name__ == "__main__":
    main()
