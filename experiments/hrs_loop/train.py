"""Train a single Variant (A/B/C) on WikiText-2 (or fallback).

Outputs:
  checkpoints/variant_{A|B|C}.pt
  results/train_log_{A|B|C}.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Force unbuffered stdout so training progress is visible under redirect.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def _load_wikitext():
    """Try wikitext-2 first; fall back to Tiny Shakespeare on failure."""
    try:
        from data import load_wikitext
        splits, tok = load_wikitext("wikitext-2", seq_len=512)
        return {
            "kind": "wikitext2",
            "train": splits["train"],
            "val": splits["validation"],
            "tokenizer": tok,
            "vocab_size": tok.vocab_size,
            "ctx_len": 512,
        }
    except Exception as e:
        print(f"  wikitext-2 failed ({e}); falling back to Tiny Shakespeare")
        raise


def _build_loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                       drop_last=True, num_workers=0)


def _lr_at(step, warmup, total, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def _eval_loss(model, val_loader, device, amp_dtype, n_batches=40):
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            x = batch.to(device)
            y = torch.cat([x[:, 1:], x[:, :1]], dim=1)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x[:, :-1])
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    x[:, 1:].reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_variant(variant: str, steps: int, seed: int,
                    batch_size: int, lr: float) -> dict:
    from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = _load_wikitext()
    cfg = HRSLoopConfig(variant=variant, vocab_size=data["vocab_size"],
                         ctx_len=data["ctx_len"])
    model = HRSLoop(cfg).to(device)
    n_params = model.num_params()
    print(f"[{variant}] params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                              betas=(0.9, 0.95))

    train_loader = _build_loader(data["train"], batch_size, shuffle=True)
    val_loader = _build_loader(data["val"], batch_size, shuffle=False)

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    warmup = 500

    step = 0
    train_iter = iter(train_loader)
    log = {"train_loss": [], "val": []}
    t0 = time.time()
    best_val = float("inf")
    best_step = -1

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if isinstance(batch, (tuple, list)):
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            x = batch.to(device)
            y = torch.cat([x[:, 1:], x[:, :1]], dim=1)

        for g in opt.param_groups:
            g["lr"] = _lr_at(step, warmup, steps, lr)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            out = model(x[:, :-1])
            if isinstance(out, tuple):
                out = out[0]
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]),
                                    x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN at step {step}")

        log["train_loss"].append(float(loss.item()))
        if (step + 1) % 500 == 0 or step == 0:
            val_loss = _eval_loss(model, val_loader, device, amp_dtype,
                                    n_batches=20)
            val_ppl = float(math.exp(val_loss))
            log["val"].append({"step": step + 1, "val_loss": val_loss,
                                "val_ppl": val_ppl})
            elapsed = time.time() - t0
            print(f"[{variant}] step {step+1:5d}/{steps}  "
                  f"train={loss.item():.3f}  val_ppl={val_ppl:.2f}  "
                  f"({elapsed:.0f}s)")
            # Save "best" checkpoint at lowest val.
            if val_ppl < best_val:
                best_val = val_ppl
                best_step = step + 1
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                suffix = "" if seed == 0 else f"_seed{seed}"
                torch.save({
                    "state_dict": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "final_val_ppl": val_ppl,
                    "final_val_loss": val_loss,
                    "steps": best_step,
                    "n_params": n_params,
                    "tag": "best",
                }, CKPT_DIR / f"variant_{variant}{suffix}_best.pt")
        step += 1

    # Final eval.
    final_val_loss = _eval_loss(model, val_loader, device, amp_dtype,
                                  n_batches=80)
    final_val_ppl = float(math.exp(final_val_loss))
    peak_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                 if device.type == "cuda" else 0)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Seed 0 keeps the unadorned filename for backward compatibility with
    # existing analysis scripts; seeds 1+ use a seed-tagged name.
    suffix = "" if seed == 0 else f"_seed{seed}"
    ckpt_path = CKPT_DIR / f"variant_{variant}{suffix}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "final_val_ppl": final_val_ppl,
        "final_val_loss": final_val_loss,
        "steps": steps,
        "n_params": n_params,
    }, ckpt_path)

    log_path = RESULTS_DIR / f"train_log_{variant}{suffix}.json"
    log_path.write_text(json.dumps({
        "variant": variant,
        "steps": steps,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "n_params": n_params,
        "final_val_loss": final_val_loss,
        "final_val_ppl": final_val_ppl,
        "best_val_ppl": best_val,
        "best_step": best_step,
        "peak_mem_mb": peak_mem,
        "wall_seconds": time.time() - t0,
        "val_trajectory": log["val"],
    }, indent=2))
    print(f"[{variant}] done: val_ppl={final_val_ppl:.2f} "
          f"(best {best_val:.2f} @ step {best_step})  "
          f"peak_mem={peak_mem:.0f}MB  wall={time.time()-t0:.0f}s")
    return {"variant": variant, "final_val_ppl": final_val_ppl,
             "ckpt": str(ckpt_path), "log": str(log_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    train_variant(args.variant, args.steps, args.seed,
                    args.batch_size, args.lr)


if __name__ == "__main__":
    main()
