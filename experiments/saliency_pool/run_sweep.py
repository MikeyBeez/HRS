"""Sweep all variants × 4 seeds = 20 runs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.saliency_pool.config import TrainConfig, VARIANTS
from experiments.saliency_pool.train import train_one


DEFAULT_SEEDS = {v: [0, 1, 2, 3] for v in VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/saliency_pool/results")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, int]] = []
    for v in args.variants:
        for s in DEFAULT_SEEDS[v]:
            existing = out / f"{v}_seed{s}.json"
            if existing.exists():
                print(f"  [skip] {v}/seed{s} (existing result)")
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
        completed.append({
            "variant": v, "seed": s,
            "final_val_ppl": rec["final_val_ppl"],
            "long_ctx_val_ppl": rec["long_ctx_val_ppl"],
            "wall_seconds": rec["wall_seconds"],
        })
        elapsed = time.time() - t0
        remaining = len(todo) - (i + 1)
        est = (elapsed / (i + 1)) * remaining
        print(f"  -- progress {i+1}/{len(todo)}  "
              f"elapsed={elapsed:.0f}s  est_remaining={est:.0f}s")

    summary = {
        "total_runs": len(todo),
        "elapsed_s": time.time() - t0,
        "runs": completed,
    }
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSweep complete in {time.time()-t0:.0f}s. "
          f"Summary: {out / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
