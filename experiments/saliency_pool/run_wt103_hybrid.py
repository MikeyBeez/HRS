"""Driver for the V4-style hybrid experiment at ctx=1024 on WT-103.

Runs two single-seed jobs back-to-back:
  1. baseline (V22 Bonsignore, full attention) at ctx=1024
  2. compress_hybrid (W=128, M=384, r2=4, r3=16) at ctx=1024

Batch size set to 8 (fits in 16 GB at ctx=1024). At batch=8, ctx=1024 the
per-step token count matches the prior batch=32, ctx=256 baselines, so
20000 steps trains on the same number of tokens as the existing 20K
ctx=256 baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.saliency_pool.config import TrainConfig
from experiments.saliency_pool.train_wt103 import train_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/saliency_pool/results_wt103_hybrid")
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "compress_hybrid"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    todo = []
    for v in args.variants:
        existing = out / f"{v}_seed{args.seed}.json"
        if existing.exists():
            print(f"  [skip] {v}/seed{args.seed} (existing)")
            continue
        todo.append(v)

    print(f"Planned runs: {len(todo)}  (steps={args.steps}, ctx={args.ctx_len}, "
          f"batch={args.batch_size}, seed={args.seed})")
    for i, v in enumerate(todo):
        print(f"  {i+1}. variant={v}")

    t0 = time.time()
    for i, v in enumerate(todo):
        print(f"\n=== run {i+1}/{len(todo)}: variant={v} seed={args.seed} ===")
        # eval_every=1000 keeps the convergence trajectory tight
        tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                           seed=args.seed, eval_every=1000)
        train_one(v, args.seed, out, tcfg, ctx_len=args.ctx_len,
                  save_checkpoint=True)
        elapsed = time.time() - t0
        print(f"  -- progress {i+1}/{len(todo)}  elapsed={elapsed:.0f}s")

    print(f"\nDone in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
