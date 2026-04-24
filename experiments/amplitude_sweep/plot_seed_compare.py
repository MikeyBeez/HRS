"""Overlay seed-0 and seed-2 α-curves (beacon + find + read) side by side."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"


def _records_xy(records):
    xs = np.array([r["alpha"] for r in records])
    exact = np.array([r["passkey_exact"] for r in records])
    order = np.argsort(xs)
    return xs[order], exact[order]


def load_seed(seed):
    if seed == 0:
        # Baseline uses the original phase-*.json filenames in RESULTS_ROOT.
        p1 = json.loads((RESULTS_ROOT / "phase1_l2h0.json").read_text())
        p3 = json.loads((RESULTS_ROOT / "phase3_cross_layer.json").read_text())
        return {
            "beacon": p1["records"],
            "beacon_fit": p1["fit_sigmoid"],
            "beacon_tw": p1["transition"],
            "find": p3["L0H3"]["records"],
            "find_fit": p3["L0H3"]["fit_sigmoid"],
            "find_tw": p3["L0H3"]["transition"],
            "read": p3["L3H0"]["records"],
        }
    seed_dir = RESULTS_ROOT / f"seed_{seed}"
    p1 = json.loads((seed_dir / "phase1_beacon.json").read_text())
    p3 = json.loads((seed_dir / "phase3_cross_layer.json").read_text())
    find_key = [k for k in p3 if k != "wall_seconds" and k.startswith("L0")][0]
    read_key = [k for k in p3 if k != "wall_seconds" and k.startswith("L3")][0]
    return {
        "beacon": p1["records"],
        "beacon_fit": p1["fit_sigmoid"],
        "beacon_tw": p1["transition"],
        "find": p3[find_key]["records"],
        "find_fit": p3[find_key]["fit_sigmoid"],
        "find_tw": p3[find_key]["transition"],
        "read": p3[read_key]["records"],
    }


def main():
    seed0 = load_seed(0)
    seed2 = load_seed(2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    roles = [("beacon", "L2 H0 (beacon)"),
              ("find",   "find head"),
              ("read",   "read head")]

    for ax, (key, title) in zip(axes, roles):
        x0, y0 = _records_xy(seed0[key])
        x2, y2 = _records_xy(seed2[key])
        ax.plot(x0, y0, marker="o", color="#c03030", label="seed 0", lw=1.8)
        ax.plot(x2, y2, marker="s", color="#3060c0", label="seed 2", lw=1.8)
        ax.set_xlabel("α")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.axhline(0.5, color="gray", lw=0.5, alpha=0.4)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("passkey exact accuracy")
    fig.suptitle("α-sweep comparison: seed 0 vs seed 2", fontsize=13)
    fig.tight_layout()
    out = RESULTS_ROOT / "seed_compare_alpha_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
