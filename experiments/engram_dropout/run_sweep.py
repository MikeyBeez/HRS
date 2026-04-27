"""Sweep: engram_dropout p ∈ {0.0, 0.1, 0.25, 0.5} × seeds {0, 1, 2}, 5000 steps."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.engram_dropout.train import train_one


SWEEP_PS = [0.0, 0.1, 0.25, 0.5]
SWEEP_SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="experiments/engram_dropout/results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo = []
    for p in SWEEP_PS:
        for s in SWEEP_SEEDS:
            existing = out / f"p{p}_seed{s}.json"
            if existing.exists():
                print(f"  [skip] p={p}/seed{s}")
                continue
            todo.append((p, s))

    print(f"Planned runs: {len(todo)}  (steps={args.steps})")
    for i, (p, s) in enumerate(todo):
        print(f"  {i+1:2d}. p={p}  seed={s}")

    t0 = time.time()
    completed = []
    p_skip_seeds = set()  # p values where seed1/2 should be skipped (per spec)
    for i, (p, s) in enumerate(todo):
        if p in p_skip_seeds and s in [1, 2]:
            print(f"\n=== [SKIP] p={p}/seed{s} (p={p}/seed0 was >2x worse than p=0/seed0) ===")
            continue
        print(f"\n=== run {i+1}/{len(todo)}: p={p} seed={s} ===")
        rec = train_one(p, s, out, steps=args.steps, save_checkpoint=True)
        completed.append({"p": p, "seed": s,
                          "ppl_on": rec["final_val_ppl_on"],
                          "ppl_off": rec["final_val_ppl_off"],
                          "diverged": rec["diverged"],
                          "wall_s": rec["wall_seconds"]})

        # Per spec stopping condition: if p=X seed=0 is >2x p=0 seed=0 ppl_on, skip seeds 1, 2.
        if s == 0 and p > 0.0:
            # Get p=0 seed=0 result
            try:
                with open(out / "p0.0_seed0.json") as f:
                    p0_ref = json.load(f)["final_val_ppl_on"]
                if rec["final_val_ppl_on"] > 2 * p0_ref:
                    print(f"  ** p={p}/seed0 ppl_on={rec['final_val_ppl_on']:.2f} > "
                          f"2× p=0/seed0 ppl_on={p0_ref:.2f}, skipping seeds 1,2")
                    p_skip_seeds.add(p)
            except FileNotFoundError:
                pass

        elapsed = time.time() - t0
        remaining = len(todo) - (i + 1)
        est = (elapsed / (i + 1)) * remaining
        print(f"  -- progress {i+1}/{len(todo)}  elapsed={elapsed:.0f}s  est_remaining={est:.0f}s")

    summary = {"total_runs": len(todo), "elapsed_s": time.time() - t0,
               "completed": completed, "skipped_seeds_for_p": list(p_skip_seeds)}
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSweep complete in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
