"""Phase 2a: zero L2 H0, amplify one sibling at a time."""
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
SIBLINGS = [1, 2, 3]  # head indices in L2 other than H0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="+", type=float, default=BETAS)
    ap.add_argument("--siblings", nargs="+", type=int, default=SIBLINGS)
    ap.add_argument("--out", default=str(RESULTS_DIR / "phase2a_siblings.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = load_baselines(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for sibling in args.siblings:
        for beta in args.betas:
            scales = {(2, 0): 0.0, (2, sibling): beta}
            rec = eval_point(bl, scales, measure_lm=False)
            rec["sibling"] = sibling
            rec["beta"] = beta
            records.append(rec)
            print(f"  L2 H0=0, L2 H{sibling}×{beta}: "
                  f"passkey={rec['passkey_exact']:.3f} "
                  f"digit={rec['passkey_digit']:.3f}")
            Path(args.out).write_text(json.dumps(records, indent=2))

    summary = {
        "records": records,
        "best": max(records, key=lambda r: r["passkey_exact"]),
        "wall_seconds": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    best = summary["best"]
    print(f"\nbest: sibling H{best['sibling']} × β={best['beta']} "
          f"-> passkey={best['passkey_exact']:.3f}")


if __name__ == "__main__":
    main()
