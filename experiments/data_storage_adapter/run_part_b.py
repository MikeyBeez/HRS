"""Part B: rank × size × group sweep.

For each thematic group, train adapters at the cross-product of:
  size: 1, 2, 5, 10 passages
  rank: 32, 64, 128, 256
Each adapter trained from scratch on the complete training set.

Step count scales with size to keep visits/source roughly constant.

Output: results/part_b.json with full grid of mean_rate + per_entry rates.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--harness", choices=["1", "2"], default="2")
parser.add_argument("--steps_per_source", type=int, default=80,
                    help="Visits per training source. n_steps = steps_per_source * len(sources).")
parser.add_argument("--min_steps", type=int, default=400)
parser.add_argument("--high_lr", type=float, default=5e-3)
parser.add_argument("--base_lr", type=float, default=1e-4)
parser.add_argument("--ranks", default="32,64,128,256")
parser.add_argument("--sizes", default="1,2,5,10")
parser.add_argument("--groups", default="G1_pip_childhood,G2_domestic_joe,G3_midbook,G4_latebook")
parser.add_argument("--out", default="results/part_b.json")
args = parser.parse_args()

if args.harness == "1":
    from experiments.data_storage_adapter.harness import (
        load_base, reset_lora, train_adapter, eval_retrieval,
        make_training_sources,
    )
else:
    from experiments.data_storage_adapter.harness2 import (
        load_base, reset_lora, train_adapter, eval_retrieval,
        make_training_sources,
    )


def main():
    device = torch.device("cuda")
    out_path = REPO / "experiments/data_storage_adapter" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())

    ranks = [int(x) for x in args.ranks.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    group_names = args.groups.split(",")

    print(f"Part B: ranks={ranks} × sizes={sizes} × groups={group_names}")
    print(f"  steps_per_source={args.steps_per_source}, min_steps={args.min_steps}, "
          f"harness{args.harness}, lr={args.high_lr}->{args.base_lr}")

    EVAL_KW = {"temperature": 0.6, "top_k": 20, "seeds": (0, 1, 2)}
    all_records = []
    t0 = time.time()
    n_total = len(ranks) * len(sizes) * len(group_names)
    n_done = 0
    for gname in group_names:
        entries_full = groups["groups"][gname]
        for size in sizes:
            entries = entries_full[:size]
            for rank in ranks:
                model, cfg, tok = load_base(rank, device)
                reset_lora(model)
                sources = make_training_sources(entries, tok, cfg.ctx_len, device)
                n_steps = max(args.min_steps,
                              args.steps_per_source * len(sources))
                t1 = time.time()
                info = train_adapter(model, sources, n_steps=n_steps,
                                      high_lr=args.high_lr,
                                      base_lr=args.base_lr, seed=0)
                ev = eval_retrieval(model, entries, tok, device, **EVAL_KW)
                rec = {
                    "group": gname, "size": size, "rank": rank,
                    "n_steps": n_steps, "n_sources": len(sources),
                    "loss_init": info["loss_init"],
                    "loss_final": info["loss_mean_last10"],
                    "mean_rate": ev["mean_rate"],
                    "per_entry": [
                        {"id": pe["id"], "answer": pe["answer"], "rate": pe["rate"]}
                        for pe in ev["per_entry"]
                    ],
                    "wall_s": time.time() - t1,
                }
                all_records.append(rec)
                n_done += 1
                print(f"  [{n_done:2d}/{n_total}] {gname} size={size} rank={rank} "
                      f"n_steps={n_steps} loss {info['loss_init']:.2f}->"
                      f"{info['loss_mean_last10']:.2f}  recall={ev['mean_rate']:.3f}  "
                      f"({rec['wall_s']:.1f}s)")
                # Save incrementally so we don't lose work on crash
                out_path.write_text(json.dumps({
                    "config": vars(args), "records": all_records,
                }, indent=2))

    print(f"\nPart B DONE  wall={time.time()-t0:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
