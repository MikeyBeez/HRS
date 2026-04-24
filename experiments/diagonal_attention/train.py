"""Train one attention variant on Tiny Shakespeare."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.diagonal_attention.config import ModelConfig, TrainConfig
from experiments.diagonal_attention.data import load_shakespeare, sample_lm_batch
from experiments.diagonal_attention.model import TinyTransformer


def _lr_at(step: int, tcfg: TrainConfig) -> float:
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    # Cosine decay to 10% of peak.
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


def train_lm(variant: str, out_dir: Path, tcfg: TrainConfig | None = None) -> dict:
    tcfg = tcfg or TrainConfig()
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, info = load_shakespeare()

    mcfg = ModelConfig(variant=variant, vocab_size=info["vocab_size"])
    model = TinyTransformer(mcfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
        betas=tcfg.betas,
    )

    rng = np.random.default_rng(tcfg.seed)

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
            raise RuntimeError(f"NaN/inf loss at step {step} in variant {variant}")

        if (step + 1) % tcfg.eval_every == 0 or step == 0:
            ppl = eval_ppl(model, val_data, tcfg, mcfg, device)
            eval_points.append({"step": step + 1, "val_ppl": ppl,
                                 "train_loss": float(loss.item())})
            print(f"  [{variant}] step {step+1:4d}/{tcfg.steps} "
                  f"loss={loss.item():.3f} val_ppl={ppl:.2f}")
        train_losses.append(float(loss.item()))

    t_wall = time.time() - t_wall_start
    final_ppl = eval_ppl(model, val_data, tcfg, mcfg, device, n_batches=80)
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    diagonal_w_stats = None
    if variant == "diagonal":
        ws = [blk.attn.w.detach().cpu().float() for blk in model.blocks]
        diagonal_w_stats = {
            "per_layer_mean": [float(w.mean()) for w in ws],
            "per_layer_std": [float(w.std()) for w in ws],
            "per_layer_min": [float(w.min()) for w in ws],
            "per_layer_max": [float(w.max()) for w in ws],
            "per_layer_l2": [float(w.norm()) for w in ws],
        }

    record = {
        "task": "lm_shakespeare",
        "variant": variant,
        "final_val_ppl": final_ppl,
        "score_params": model.score_params(),
        "total_params": model.total_params(),
        "step_time_ms_median": float(np.median(step_times) * 1000),
        "step_time_ms_mean": float(np.mean(step_times) * 1000),
        "wall_seconds": t_wall,
        "peak_mem_mb": peak_mem,
        "eval_points": eval_points,
        "train_loss_last_100_mean": float(np.mean(train_losses[-100:])),
        "steps": tcfg.steps,
        "batch_size": tcfg.batch_size,
        "ctx_len": mcfg.ctx_len,
        "vocab_size": mcfg.vocab_size,
        "diagonal_w_stats": diagonal_w_stats,
    }
    (out_dir / f"lm_{variant}.json").write_text(json.dumps(record, indent=2))
    print(f"  [{variant}] done: val_ppl={final_ppl:.2f} "
          f"score_params={record['score_params']:,} "
          f"step={record['step_time_ms_median']:.2f}ms "
          f"peak_mem={peak_mem:.0f}MB")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["mha", "bilinear", "diagonal", "identity"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/diagonal_attention/results")
    args = ap.parse_args()

    tcfg = TrainConfig(steps=args.steps)
    train_lm(args.variant, Path(args.out), tcfg)


if __name__ == "__main__":
    main()
