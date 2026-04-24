"""Stage 3 ablation: evaluate the trained Variant-B composite model with the
recurrent stage bypassed. If Prelude→Coda alone reaches the same accuracy on
k ∈ {1..4} and k ∈ {6, 8}, the loops are not doing the chain traversal on
this task and the T-flat result is explained by a feedforward-only solution,
not by fixed-point convergence.

Ablation mechanism: pass skip_recurrent_with_mpar = zeros(B, rank_m) to
HRSLoop.forward. Because MPARUnprojector is bias-free, this reduces the
Coda input to `e + project_up(0) = e` — i.e., Coda sees the Prelude output
directly, no contribution from the recurrent stage or its LoRAs.

Compared to:
  - canonical: T ∈ {2, 4, 6, 8, 12} (already in stage3_compositional.json)
  - ablated:   recurrent bypassed (this script)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig
from experiments.hrs_loop.tasks.compositional_lookup import (
    CompositionalConfig, make_loaders,
)


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


@torch.no_grad()
def eval_bypass(model, loader, device, amp_dtype):
    """Accuracy with recurrent stage skipped (skip_recurrent_with_mpar=0)."""
    model.eval()
    correct = total = 0
    rank_m = model.cfg.rank_m
    for x, y, ans_pos, terminal, k in loader:
        x = x.to(device); ans_pos = ans_pos.to(device); terminal = terminal.to(device)
        B = x.shape[0]
        zero_mpar = torch.zeros(B, rank_m, device=device, dtype=amp_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x, skip_recurrent_with_mpar=zero_mpar)
            if isinstance(out, tuple):
                out = out[0]
        idx = torch.arange(out.shape[0], device=out.device)
        pred = out[idx, ans_pos].argmax(dim=-1)
        correct += (pred == terminal).sum().item()
        total += terminal.shape[0]
    return correct / max(1, total)


@torch.no_grad()
def eval_canonical(model, loader, device, amp_dtype, T):
    model.eval()
    correct = total = 0
    for x, y, ans_pos, terminal, k in loader:
        x = x.to(device); ans_pos = ans_pos.to(device); terminal = terminal.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x, T=T)
            if isinstance(out, tuple):
                out = out[0]
        idx = torch.arange(out.shape[0], device=out.device)
        pred = out[idx, ans_pos].argmax(dim=-1)
        correct += (pred == terminal).sum().item()
        total += terminal.shape[0]
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-list", nargs="+", type=int,
                    default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--T-canonical", type=int, default=4,
                    help="T to use for the canonical (non-ablated) comparison.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "stage3_ablation.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    path = CKPT_DIR / f"composite_{args.variant}{suffix}_best.pt"
    if not path.exists():
        path = CKPT_DIR / f"composite_{args.variant}{suffix}.pt"
    print(f"loading {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B", \
        f"Ablation only implemented for Variant B (has skip_recurrent_with_mpar). Got {cfg.variant}."
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    task_cfg = CompositionalConfig(**ckpt["task_cfg"])
    data = make_loaders(task_cfg, n_train=1, n_val_per_k=500,
                          eval_ks=tuple(args.k_list))

    canonical = {}
    ablated = {}
    for k in args.k_list:
        vl = data["val"][k]
        canonical[k] = eval_canonical(model, vl, device, amp_dtype,
                                        T=args.T_canonical)
        ablated[k] = eval_bypass(model, vl, device, amp_dtype)
        print(f"  k={k:>2}: canonical(T={args.T_canonical})={canonical[k]:.3f}   "
              f"ablated(recurrent bypassed)={ablated[k]:.3f}   "
              f"Δ={canonical[k] - ablated[k]:+.3f}")

    out = {
        "variant": args.variant,
        "seed": args.seed,
        "T_canonical": args.T_canonical,
        "checkpoint_step": ckpt.get("step"),
        "canonical_acc_per_k": {str(k): v for k, v in canonical.items()},
        "ablated_acc_per_k": {str(k): v for k, v in ablated.items()},
        "notes": "ablated = skip_recurrent_with_mpar=zeros, i.e. Coda input = Prelude output. "
                 "Tests whether recurrent stage contributes anything on this task.",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n{'k':>3} | {'canonical':>10}  {'ablated':>10}  {'Δ (canon-abl)':>14}")
    print("-" * 45)
    for k in args.k_list:
        print(f"{k:>3} | {canonical[k]:>10.3f}  {ablated[k]:>10.3f}  "
              f"{canonical[k] - ablated[k]:>+14.3f}")
    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
