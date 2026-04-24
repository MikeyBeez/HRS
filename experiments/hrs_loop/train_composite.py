"""Stage 3: Train Variant B (and optionally others) on the compositional
multi-hop lookup task.

Single-seed training. Saves a best-val checkpoint and a final checkpoint.
Loss is computed only at the ANS position (the one token right after the
ANS marker, which is the terminal value).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig
from experiments.hrs_loop.tasks.compositional_lookup import (
    CompositionalConfig, make_loaders, VOCAB_SIZE,
)


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def _lr_at(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_accuracy(model, loader, device, amp_dtype, T=None):
    """Accuracy = fraction of examples where argmax at ANS predicts terminal."""
    model.eval()
    correct = total = 0
    for x, y, ans_pos, terminal, k in loader:
        x = x.to(device); ans_pos = ans_pos.to(device); terminal = terminal.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x, T=T) if T is not None else model(x)
            if isinstance(out, tuple):
                out = out[0]
        # out: (B, seq, V). predict at ans_pos (model emits token at ans_pos+1 given context up to ans_pos). Actually model outputs next-token at each position, so prediction at ans_pos+1 is the token we want (should be terminal). Equivalently, logits at index ans_pos PREDICT token at ans_pos+1.
        idx = torch.arange(out.shape[0], device=out.device)
        pred = out[idx, ans_pos].argmax(dim=-1)
        correct += (pred == terminal).sum().item()
        total += terminal.shape[0]
    model.train()
    return correct / max(1, total)


def train(variant: str, steps: int, seed: int, batch_size: int, lr: float,
           T: int = 4) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_cfg = CompositionalConfig(seq_len=256, seed=seed)
    data = make_loaders(task_cfg, n_train=20000, n_val_per_k=500,
                          eval_ks=(1, 2, 3, 4, 6, 8))

    cfg = HRSLoopConfig(variant=variant, vocab_size=VOCAB_SIZE,
                          ctx_len=task_cfg.seq_len, T_default=T)
    model = HRSLoop(cfg).to(device)
    print(f"[{variant}] params: {model.num_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                              betas=(0.9, 0.95))
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    warmup = 500

    train_iter = iter(data["train"])
    best_val = -1.0
    best_step = -1
    log = {"train_loss": [], "val": []}
    t0 = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in range(steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(data["train"])
            batch = next(train_iter)
        x, y, ans_pos, terminal, k = batch
        x = x.to(device); ans_pos = ans_pos.to(device); terminal = terminal.to(device)

        for g in opt.param_groups:
            g["lr"] = _lr_at(step, warmup, steps, lr)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x, T=T)
            if isinstance(out, tuple):
                out = out[0]
            idx = torch.arange(out.shape[0], device=out.device)
            logits = out[idx, ans_pos]               # (B, V)
            loss = F.cross_entropy(logits, terminal)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN at step {step}")

        log["train_loss"].append(float(loss.item()))
        if (step + 1) % 500 == 0 or step == 0:
            # Aggregate val accuracy across all k's as training-time signal.
            acc_per_k = {}
            for kk, vl in data["val"].items():
                acc_per_k[kk] = eval_accuracy(model, vl, device, amp_dtype, T=T)
            mean_acc = float(np.mean(list(acc_per_k.values())))
            log["val"].append({"step": step + 1, "acc_per_k": acc_per_k,
                                "mean_acc": mean_acc})
            elapsed = time.time() - t0
            print(f"[{variant}] step {step+1:5d}/{steps} loss={loss.item():.3f} "
                  f"val_acc_mean={mean_acc:.3f} "
                  f"(k=1:{acc_per_k.get(1, 0):.2f} "
                  f"k=2:{acc_per_k.get(2, 0):.2f} "
                  f"k=3:{acc_per_k.get(3, 0):.2f} "
                  f"k=4:{acc_per_k.get(4, 0):.2f} "
                  f"k=6:{acc_per_k.get(6, 0):.2f} "
                  f"k=8:{acc_per_k.get(8, 0):.2f}) "
                  f"({elapsed:.0f}s)")
            if mean_acc > best_val:
                best_val = mean_acc
                best_step = step + 1
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                suffix = "" if seed == 0 else f"_seed{seed}"
                torch.save({
                    "state_dict": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "task_cfg": task_cfg.__dict__,
                    "best_val_acc": mean_acc,
                    "acc_per_k": acc_per_k,
                    "step": best_step,
                    "tag": "best_composite",
                }, CKPT_DIR / f"composite_{variant}{suffix}_best.pt")

    peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                 if device.type == "cuda" else 0)

    # Save final (overfit) checkpoint too.
    suffix = "" if seed == 0 else f"_seed{seed}"
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "task_cfg": task_cfg.__dict__,
        "step": steps,
        "tag": "final_composite",
    }, CKPT_DIR / f"composite_{variant}{suffix}.pt")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"composite_train_{variant}{suffix}.json"
    log_path.write_text(json.dumps({
        "variant": variant,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "n_params": model.num_params(),
        "best_val_acc": best_val,
        "best_step": best_step,
        "peak_mem_mb": peak_mem,
        "wall_seconds": time.time() - t0,
        "val_trajectory": log["val"],
    }, indent=2))
    print(f"[{variant}] done: best_val_acc={best_val:.3f} @ step {best_step}  "
          f"peak_mem={peak_mem:.0f}MB  wall={time.time()-t0:.0f}s")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B",
                    choices=["A", "B", "C", "D",
                              "Bmax", "Battn", "Bfull", "Bcomb"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--T", type=int, default=4)
    args = ap.parse_args()
    train(args.variant, args.steps, args.seed, args.batch_size, args.lr, args.T)


if __name__ == "__main__":
    main()
