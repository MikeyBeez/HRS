"""Phase 1: α-sweep on L2 H0."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.amplitude_sweep._common import eval_point, load_baselines
from experiments.amplitude_sweep.alpha_hooks import fit_sigmoid, transition_width


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
TARGET = (2, 0)  # (layer, head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=TARGET[0])
    ap.add_argument("--head", type=int, default=TARGET[1])
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--out", default=str(RESULTS_DIR / "phase1_l2h0.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = load_baselines(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for alpha in args.alphas:
        scales = {(args.layer, args.head): alpha}
        rec = eval_point(bl, scales, measure_lm=True)
        rec["alpha"] = alpha
        rec["target"] = [args.layer, args.head]
        records.append(rec)
        print(f"  α={alpha:>4.2f}  passkey={rec['passkey_exact']:.3f} "
              f"digit={rec['passkey_digit']:.3f} ppl={rec['val_ppl']:.3f}")
        Path(args.out).write_text(json.dumps(records, indent=2))

    # Fit sigmoid + transition width.
    alphas = [r["alpha"] for r in records]
    passkeys = [r["passkey_exact"] for r in records]
    astar, k, r2 = fit_sigmoid(alphas, passkeys)
    tw = transition_width(alphas, passkeys)

    summary = {
        "target_layer": args.layer,
        "target_head": args.head,
        "records": records,
        "fit_sigmoid": {"alpha_star": astar, "k": k, "r2": r2},
        "transition": {
            "alpha_at_hi": tw[0], "alpha_at_lo": tw[1], "width": tw[2]
        },
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nfit: α*={astar:.3f}  k={k:.2f}  r²={r2:.3f}")
    if tw[2] is not None:
        print(f"transition width (α@0.9 → α@0.1): {tw[2]:.3f}  "
              f"[{tw[0]:.3f} → {tw[1]:.3f}]")
    else:
        print(f"transition width: undefined (α_hi={tw[0]}, α_lo={tw[1]})")


if __name__ == "__main__":
    main()
