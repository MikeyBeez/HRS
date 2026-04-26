"""Sweep: saliency-compression-then-attention (V4-style).

3 variants × 4 seeds × 2000 steps on Tiny Shakespeare.

Reuses baseline / variant_D / cumulative_mean / dual_projection results from
prior sweeps. Saves checkpoints for compress_4 (per spec) for post-hoc
analysis of the learned saliency vector.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.saliency_pool.config import TrainConfig
from experiments.saliency_pool.train import train_one


SWEEP_VARIANTS = ["compress_4", "compress_8", "compress_16"]
DEFAULT_SEEDS = {v: [0, 1, 2, 3] for v in SWEEP_VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/saliency_pool/results_compression_causal")
    ap.add_argument("--variants", nargs="+", default=SWEEP_VARIANTS)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[str, int]] = []
    for v in args.variants:
        for s in DEFAULT_SEEDS[v]:
            existing = out / f"{v}_seed{s}.json"
            if existing.exists():
                print(f"  [skip] {v}/seed{s} (existing)")
                continue
            todo.append((v, s))

    print(f"Planned runs: {len(todo)}")
    for i, (v, s) in enumerate(todo):
        print(f"  {i+1:2d}. variant={v} seed={s}")

    t0 = time.time()
    completed = []
    for i, (v, s) in enumerate(todo):
        print(f"\n=== run {i+1}/{len(todo)}: variant={v} seed={s} ===")
        tcfg = TrainConfig(steps=args.steps, seed=s)
        # Per spec: save checkpoints for compress_4 to allow post-hoc analysis
        # of the learned saliency vector.
        save_ckpt = (v == "compress_4")
        rec = train_one(v, s, out, tcfg, save_checkpoint=save_ckpt)
        completed.append({"variant": v, "seed": s,
                          "final_val_ppl": rec["final_val_ppl"],
                          "wall_seconds": rec["wall_seconds"],
                          "step_time_ms_median": rec["step_time_ms_median"]})
        elapsed = time.time() - t0
        remaining = len(todo) - (i + 1)
        est = (elapsed / (i + 1)) * remaining
        print(f"  -- progress {i+1}/{len(todo)}  elapsed={elapsed:.0f}s  "
              f"est_remaining={est:.0f}s")

    summary = {"total_runs": len(todo), "elapsed_s": time.time() - t0,
               "runs": completed}
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSweep complete in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
