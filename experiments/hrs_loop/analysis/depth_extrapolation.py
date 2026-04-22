"""Test 4: Eval at T ∈ {2, 4, 6, 8, 12} for each trained variant.

Variants A/B/C were trained at T=4. See whether B extrapolates past T=4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.hrs_loop.analysis._shared import (
    RESULTS_DIR, eval_ppl_with_T, load_checkpoint, load_val_loader,
)


T_VALUES = [2, 4, 6, 8, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--n-batches", type=int, default=30)
    ap.add_argument("--out", default=str(RESULTS_DIR / "test4_depth_extrap.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, _tok = load_val_loader(batch_size=16)

    all_results = {}
    for v in args.variants:
        try:
            model, cfg, _ = load_checkpoint(v, device)
        except FileNotFoundError:
            print(f"  variant {v} checkpoint missing, skipping")
            continue
        per_T = []
        for T in T_VALUES:
            ppl = eval_ppl_with_T(model, val_loader, device, T=T,
                                    n_batches=args.n_batches)
            print(f"  variant {v}  T={T}: val_ppl={ppl:.3f}")
            per_T.append({"T": T, "val_ppl": ppl})
        all_results[v] = per_T

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"wrote {Path(args.out)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"A": "#c03030", "B": "#3060c0", "C": "#30a030", "D": "#b030c0"}
    for v in args.variants:
        xs = [r["T"] for r in all_results[v]]
        ys = [r["val_ppl"] for r in all_results[v]]
        ax.plot(xs, ys, marker="o", lw=1.8, color=colors.get(v),
                 label=f"variant {v}")
    ax.set_xlabel("T (loops at inference)")
    ax.set_ylabel("val PPL")
    ax.set_title("Depth extrapolation (trained at T=4)")
    ax.axvline(4, color="gray", ls=":", alpha=0.6, label="train T")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = RESULTS_DIR / "test4_depth_extrap.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
