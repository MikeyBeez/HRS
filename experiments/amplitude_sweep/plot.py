"""Plot the two main figures: α-curves and sibling rescue."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def plot_alpha_curves():
    p1 = json.loads((RESULTS_DIR / "phase1_l2h0.json").read_text())
    p3 = json.loads((RESULTS_DIR / "phase3_cross_layer.json").read_text())

    series = [
        ("L2 H0 (beacon)", p1["records"], "#c03030"),
        ("L0 H3 (find)", p3["L0H3"]["records"], "#3060c0"),
        ("L3 H0 (read)", p3["L3H0"]["records"], "#30a030"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, records, color in series:
        xs = [r["alpha"] for r in records]
        exact = [r["passkey_exact"] for r in records]
        digit = [r["passkey_digit"] for r in records]
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        exact = np.array(exact)[order]
        digit = np.array(digit)[order]
        axes[0].plot(xs, exact, marker="o", color=color, label=label, lw=1.8)
        axes[1].plot(xs, digit, marker="o", color=color, label=label, lw=1.8)

    # Annotate α* for L2 H0.
    astar = p1["fit_sigmoid"]["alpha_star"]
    axes[0].axvline(astar, color="#c03030", alpha=0.35, linestyle=":")
    axes[0].text(astar + 0.01, 0.55, f"α*={astar:.2f}\n(L2 H0)",
                  color="#c03030", fontsize=9)

    for ax in axes:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("scaling α on head output")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
        ax.axhline(0.5, color="gray", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("passkey exact accuracy")
    axes[0].set_title("Passkey exact vs α (head output scaling)")
    axes[1].set_ylabel("passkey digit accuracy")
    axes[1].set_title("Passkey digit vs α")

    fig.tight_layout()
    p = RESULTS_DIR / "alpha_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def plot_sibling_rescue():
    a = json.loads((RESULTS_DIR / "phase2a_siblings.json").read_text())
    b = json.loads((RESULTS_DIR / "phase2b_all_siblings.json").read_text())
    p1 = json.loads((RESULTS_DIR / "phase1_l2h0.json").read_text())

    # Reference: passkey at α=0 for L2 H0 (no rescue).
    ref = [r for r in p1["records"] if r["alpha"] == 0.0][0]["passkey_exact"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 2a: per-sibling curves.
    colors = {1: "#c06050", 2: "#50a050", 3: "#5060c0"}
    for sib in [1, 2, 3]:
        rs = sorted([r for r in a["records"] if r["sibling"] == sib],
                     key=lambda r: r["beta"])
        xs = [r["beta"] for r in rs]
        ys = [r["passkey_exact"] for r in rs]
        axes[0].plot(xs, ys, marker="o", color=colors[sib],
                      label=f"only L2 H{sib} amplified")
    axes[0].axhline(ref, color="black", alpha=0.5, lw=0.8, linestyle="--",
                     label=f"no rescue (ref={ref:.3f})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("amplification β (log scale)")
    axes[0].set_ylabel("passkey exact accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Phase 2a — zero L2 H0, amplify one sibling at a time")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    # 2b: all siblings together.
    rs = sorted(b["records"], key=lambda r: r["beta"])
    xs = [r["beta"] for r in rs]
    ys = [r["passkey_exact"] for r in rs]
    axes[1].plot(xs, ys, marker="o", color="#8030b0",
                  label="L2 H1/H2/H3 all × β")
    axes[1].axhline(ref, color="black", alpha=0.5, lw=0.8, linestyle="--",
                     label=f"no rescue (ref={ref:.3f})")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("amplification β (log scale)")
    axes[1].set_ylabel("passkey exact accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Phase 2b — zero L2 H0, amplify all three siblings")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    p = RESULTS_DIR / "sibling_rescue.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    plot_alpha_curves()
    plot_sibling_rescue()


if __name__ == "__main__":
    main()
