"""End-to-end runner: given trained checkpoints, execute Tests 1-5 on the
four variants and produce a results table.

Test 1 is already covered by the final_val_ppl in each checkpoint's
train log. Tests 2-5 run their respective analysis scripts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    t = time.time()
    rc = subprocess.call(cmd, cwd="/mnt/data/Code/HRS",
                           env={"PYTHONPATH": "/mnt/data/Code/HRS",
                                 "PATH": "/mnt/data/Code/HRS/.venv/bin:"
                                          "/usr/bin:/bin"})
    print(f"<<< exit {rc}  ({time.time() - t:.1f}s)")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+",
                    default=["test2", "test3", "test4", "test5", "variantD"],
                    help="Subset of analyses to run.")
    ap.add_argument("--use-best", action="store_true",
                    help="For analyses that accept --variant, prefer *_best "
                         "checkpoints by creating a symlink-like copy.")
    args = ap.parse_args()

    py = "/mnt/data/Code/HRS/.venv/bin/python"

    if args.use_best:
        # Copy variant_{X}_best.pt over variant_{X}.pt so the shared loader
        # reads the best checkpoint. This is reversible — the originals are
        # overwritten only for the duration of analysis.
        import shutil
        for v in ("A", "B", "C", "D"):
            best = CKPT_DIR / f"variant_{v}_best.pt"
            final = CKPT_DIR / f"variant_{v}.pt"
            if best.exists():
                shutil.copy(best, final)
                print(f"using BEST checkpoint for variant {v}")

    if "test2" in args.which:
        run([py, "-m", "experiments.hrs_loop.analysis.rank_floor_sweep",
             "--variant", "B"])
    if "test3" in args.which:
        run([py, "-m", "experiments.hrs_loop.analysis.mpar_cosine",
             "--variant", "B"])
    if "test4" in args.which:
        run([py, "-m", "experiments.hrs_loop.analysis.depth_extrapolation",
             "--variants", "A", "B", "C", "D"])
    if "test5" in args.which:
        run([py, "-m", "experiments.hrs_loop.analysis.order_invariance"])
    if "variantD" in args.which:
        run([py, "-m", "experiments.hrs_loop.analysis.variant_d_tests"])

    print("\n=== All analyses complete ===")
    print("Summary of results:")
    for f in sorted(RESULTS_DIR.glob("*.json")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
