"""Stage 3 eval: after training Variant B on compositional lookup at T=4,
evaluate at each T ∈ {2, 4, 6, 8, 12} and each k ∈ {1, 2, 3, 4, 6, 8}.

Does T=8 help on k=6 / k=8 tasks (deeper reasoning), or does the fixed-point
pattern from wikitext generalize?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig
from experiments.hrs_loop.tasks.compositional_lookup import (
    CompositionalConfig, make_loaders, VOCAB_SIZE,
)
from experiments.hrs_loop.train_composite import eval_accuracy


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T-list", nargs="+", type=int,
                    default=[2, 4, 6, 8, 12])
    ap.add_argument("--k-list", nargs="+", type=int,
                    default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--use-best", action="store_true", default=True)
    ap.add_argument("--out", default=str(RESULTS_DIR / "stage3_compositional.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    tag = "_best" if args.use_best else ""
    path = CKPT_DIR / f"composite_{args.variant}{suffix}{tag}.pt"
    if not path.exists():
        path = CKPT_DIR / f"composite_{args.variant}{suffix}.pt"
    print(f"loading {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    task_cfg = CompositionalConfig(**ckpt["task_cfg"])
    data = make_loaders(task_cfg, n_train=1, n_val_per_k=500,
                          eval_ks=tuple(args.k_list))

    # Grid: T × k.
    grid = {}
    for T in args.T_list:
        grid[T] = {}
        for k in args.k_list:
            acc = eval_accuracy(model, data["val"][k], device, amp_dtype, T=T)
            grid[T][k] = acc
            print(f"  T={T:>2}  k={k:>2}: acc={acc:.3f}")

    # Print table
    print(f"\n{'T\\k':>4} |  " + "  ".join(f"{k:>6}" for k in args.k_list))
    print("-" * (8 + 8 * len(args.k_list)))
    for T in args.T_list:
        row = "  ".join(f"{grid[T][k]:>6.3f}" for k in args.k_list)
        print(f"{T:>4} |  {row}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint_step": ckpt.get("step"),
        "best_val_acc_at_train": ckpt.get("best_val_acc"),
        "grid_T_k": {str(T): {str(k): v for k, v in row.items()}
                       for T, row in grid.items()},
    }, indent=2))
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
