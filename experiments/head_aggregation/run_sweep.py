"""Run all variants × seeds sequentially and emit a summary.

7 variants × 4 seeds each + 2 extra seeds for D (high init variance) = 30 runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.head_aggregation.config import TrainConfig, VARIANTS
from experiments.head_aggregation.train import train_one


# Default seeds per variant. D gets 6 seeds (extra variance control); rest get 4.
DEFAULT_SEEDS = {v: [0, 1, 2, 3] for v in VARIANTS}
DEFAULT_SEEDS["D"] = [0, 1, 2, 3, 4, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/head_aggregation/results")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    help="subset of variants to run (default: all 7)")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="if >0, stop after this many runs (for smoke tests)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, int]] = []
    for v in args.variants:
        for s in DEFAULT_SEEDS[v]:
            existing = out / f"{v}_seed{s}.json"
            if existing.exists():
                print(f"  [skip] {v}/seed{s} — existing result at {existing}")
                continue
            todo.append((v, s))

    if args.max_runs > 0:
        todo = todo[:args.max_runs]

    print(f"Planned runs: {len(todo)}")
    for i, (v, s) in enumerate(todo):
        print(f"  {i+1:2d}. variant={v} seed={s}")

    t0 = time.time()
    completed = []
    for i, (v, s) in enumerate(todo):
        print(f"\n=== run {i+1}/{len(todo)}: variant={v} seed={s} ===")
        tcfg = TrainConfig(steps=args.steps, seed=s)
        rec = train_one(v, s, out, tcfg)
        completed.append({"variant": v, "seed": s, "final_val_ppl": rec["final_val_ppl"],
                          "wall_seconds": rec["wall_seconds"]})
        elapsed = time.time() - t0
        remaining = len(todo) - (i + 1)
        est = (elapsed / (i + 1)) * remaining if (i + 1) > 0 else 0
        print(f"  -- sweep progress: {i+1}/{len(todo)}  "
              f"elapsed={elapsed:.0f}s  est_remaining={est:.0f}s")

    # Summary
    summary_path = out / "sweep_summary.json"
    summary = {
        "total_runs": len(todo),
        "elapsed_s": time.time() - t0,
        "runs": completed,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSweep complete in {time.time()-t0:.0f}s. Summary: {summary_path}")


if __name__ == "__main__":
    main()
