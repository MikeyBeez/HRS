"""Phase 2b: zero L2 H0, amplify all three siblings together."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.amplitude_sweep._common import eval_point, load_baselines


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
BETAS = [1, 2, 3, 5, 10, 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="+", type=float, default=BETAS)
    ap.add_argument("--out", default=str(RESULTS_DIR / "phase2b_all_siblings.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = load_baselines(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for beta in args.betas:
        scales = {(2, 0): 0.0, (2, 1): beta, (2, 2): beta, (2, 3): beta}
        rec = eval_point(bl, scales, measure_lm=False)
        rec["beta"] = beta
        records.append(rec)
        print(f"  L2 H0=0, L2 H1/H2/H3 × {beta}: "
              f"passkey={rec['passkey_exact']:.3f} "
              f"digit={rec['passkey_digit']:.3f}")
        Path(args.out).write_text(json.dumps(records, indent=2))

    summary = {
        "records": records,
        "best": max(records, key=lambda r: r["passkey_exact"]),
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nbest: β={summary['best']['beta']} "
          f"-> passkey={summary['best']['passkey_exact']:.3f}")


if __name__ == "__main__":
    main()
