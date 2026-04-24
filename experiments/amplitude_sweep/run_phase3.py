"""Phase 3: α-sweep on L0 H3 (find) and L3 H0 (read). Passkey only."""
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
TARGETS = [(0, 3), (3, 0)]


def sweep(bl, layer, head, alphas):
    records = []
    for alpha in alphas:
        rec = eval_point(bl, {(layer, head): alpha}, measure_lm=False)
        rec["alpha"] = alpha
        rec["target"] = [layer, head]
        records.append(rec)
        print(f"  L{layer}H{head} α={alpha:>4.2f}  "
              f"passkey={rec['passkey_exact']:.3f} "
              f"digit={rec['passkey_digit']:.3f}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--out", default=str(RESULTS_DIR / "phase3_cross_layer.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = load_baselines(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_records = {}
    for (l, h) in TARGETS:
        key = f"L{l}H{h}"
        print(f"\n=== {key} ===")
        records = sweep(bl, l, h, args.alphas)
        alphas = [r["alpha"] for r in records]
        passkeys = [r["passkey_exact"] for r in records]
        try:
            astar, k, r2 = fit_sigmoid(alphas, passkeys)
        except Exception as e:
            astar, k, r2 = None, None, None
            print(f"  sigmoid fit failed: {e}")
        tw = transition_width(alphas, passkeys)
        all_records[key] = {
            "records": records,
            "fit_sigmoid": {"alpha_star": astar, "k": k, "r2": r2},
            "transition": {
                "alpha_at_hi": tw[0], "alpha_at_lo": tw[1], "width": tw[2]
            },
        }
        if astar is not None:
            print(f"  fit: α*={astar:.3f} k={k:.2f} r²={r2:.3f}")
        print(f"  width={tw[2]}  "
              f"[α@0.9={tw[0]}, α@0.1={tw[1]}]")
        Path(args.out).write_text(json.dumps(all_records, indent=2))

    all_records["wall_seconds"] = time.time() - t0
    Path(args.out).write_text(json.dumps(all_records, indent=2))


if __name__ == "__main__":
    main()
