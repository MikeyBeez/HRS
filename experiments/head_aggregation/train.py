"""Train one variant × one seed on Tiny Shakespeare."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.diagonal_attention.data import load_shakespeare, sample_lm_batch
from experiments.head_aggregation.config import ModelConfig, TrainConfig, VARIANTS
from experiments.head_aggregation.model import TinyBonsignoreTransformer


def _lr_at(step: int, tcfg: TrainConfig) -> float:
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    progress = (step - tcfg.warmup_steps) / max(1, tcfg.steps - tcfg.warmup_steps)
    return tcfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_ppl(model, val_data, tcfg, mcfg, device, n_batches: int = 40) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_lm_batch(val_data, tcfg.batch_size, mcfg.ctx_len, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(math.exp(np.mean(losses)))


def train_one(variant: str, seed: int, out_dir: Path,
              tcfg: TrainConfig | None = None) -> dict:
    tcfg = tcfg or TrainConfig(seed=seed)
    tcfg.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, info = load_shakespeare()

    mcfg = ModelConfig(variant=variant, vocab_size=info["vocab_size"])
    model = TinyBonsignoreTransformer(mcfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr, weight_decay=tcfg.weight_decay, betas=tcfg.betas,
    )

    rng = np.random.default_rng(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times = []
    train_losses = []
    eval_points = []

    model.train()
    t_wall_start = time.time()
    for step in range(tcfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tcfg)

        x, y = sample_lm_batch(train_data, tcfg.batch_size, mcfg.ctx_len, device, rng)
        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)

        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/inf at step {step} in variant={variant} seed={seed}")

        if (step + 1) % tcfg.eval_every == 0 or step == 0:
            ppl = eval_ppl(model, val_data, tcfg, mcfg, device)
            eval_points.append({
                "step": step + 1,
                "val_ppl": ppl,
                "train_loss": float(loss.item()),
                "diagnostics": model.per_layer_diagnostics(),
            })
            print(f"  [{variant}/seed{seed}] step {step+1:4d}/{tcfg.steps} "
                  f"loss={loss.item():.3f} val_ppl={ppl:.2f}")
        train_losses.append(float(loss.item()))

    t_wall = time.time() - t_wall_start
    final_ppl = eval_ppl(model, val_data, tcfg, mcfg, device, n_batches=80)
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task": "head_aggregation_shakespeare",
        "variant": variant,
        "seed": seed,
        "final_val_ppl": final_ppl,
        "total_params": model.total_params(),
        "output_path_params_per_layer": model.output_path_params_per_layer(),
        "step_time_ms_median": float(np.median(step_times) * 1000),
        "step_time_ms_mean": float(np.mean(step_times) * 1000),
        "wall_seconds": t_wall,
        "peak_mem_mb": peak_mem,
        "eval_points": eval_points,
        "final_diagnostics": model.per_layer_diagnostics(),
        "train_loss_last_100_mean": float(np.mean(train_losses[-100:])),
        "steps": tcfg.steps,
        "batch_size": tcfg.batch_size,
        "ctx_len": mcfg.ctx_len,
        "vocab_size": mcfg.vocab_size,
        "per_head_hidden": mcfg.per_head_hidden,
    }
    (out_dir / f"{variant}_seed{seed}.json").write_text(json.dumps(record, indent=2))
    print(f"  [{variant}/seed{seed}] DONE val_ppl={final_ppl:.3f} "
          f"total_params={record['total_params']:,} "
          f"output_path/layer={record['output_path_params_per_layer'][0]:,} "
          f"wall={t_wall:.0f}s peak_mem={peak_mem:.0f}MB")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/head_aggregation/results")
    args = ap.parse_args()

    tcfg = TrainConfig(steps=args.steps, seed=args.seed)
    train_one(args.variant, args.seed, Path(args.out), tcfg)


if __name__ == "__main__":
    main()
