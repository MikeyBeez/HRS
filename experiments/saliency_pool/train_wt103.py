"""Scaled-up saliency-pool experiment: same architecture, WikiText-103.

Loads the existing WT-103 GPT-2-tokenized cache at
`experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt`, re-slices into
seq_len=256 chunks at random offsets for training (fair across all
variants), evaluates on validation chunks.

Keeps the saliency-pool architecture (d=384, n_heads=6, n_layers=6,
d_ff=1536, ctx=256, Bonsignore baseline + saliency variants). Only the
vocab size changes: 50257 (GPT-2 BPE) vs Shakespeare's ~65 chars.

Usage:
    python -m experiments.saliency_pool.train_wt103 --variant baseline --seed 0 --steps 5000
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.saliency_pool.config import ModelConfig, TrainConfig, VARIANTS
from experiments.saliency_pool.model import TinyTransformer

WT103_CACHE = REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt"


def load_wt103_tokens():
    """Return (train_tokens, val_tokens) as 1D LongTensors."""
    c = torch.load(WT103_CACHE, weights_only=False)
    splits = c["splits"]
    return splits["train"].tokens, splits["validation"].tokens


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int,
                 device: torch.device, rng: np.random.Generator):
    """Sample a batch of (x, y) pairs from a 1D token tensor."""
    n = tokens.shape[0]
    max_start = n - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    x = torch.stack([tokens[s:s + seq_len] for s in starts]).to(device)
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts]).to(device)
    return x, y


def _lr_at(step: int, tcfg: TrainConfig) -> float:
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    progress = (step - tcfg.warmup_steps) / max(1, tcfg.steps - tcfg.warmup_steps)
    return tcfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_ppl(model, val_tokens, tcfg, mcfg, device, n_batches: int = 40) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(val_tokens, tcfg.batch_size, mcfg.ctx_len, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(math.exp(np.mean(losses)))


def train_one(variant: str, seed: int, out_dir: Path,
              tcfg: TrainConfig | None = None,
              ctx_len: int = 256) -> dict:
    tcfg = tcfg or TrainConfig(seed=seed)
    tcfg.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading WT-103 tokens from {WT103_CACHE.name}...")
    train_tokens, val_tokens = load_wt103_tokens()
    print(f"  train={train_tokens.shape[0]:,} tokens  val={val_tokens.shape[0]:,} tokens")

    mcfg = ModelConfig(variant=variant, vocab_size=50257, ctx_len=ctx_len)
    model = TinyTransformer(mcfg).to(device)
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
        x, y = sample_batch(train_tokens, tcfg.batch_size, mcfg.ctx_len, device, rng)
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
            raise RuntimeError(f"NaN/inf at step {step} variant={variant} seed={seed}")

        if (step + 1) % tcfg.eval_every == 0 or step == 0:
            ppl = eval_ppl(model, val_tokens, tcfg, mcfg, device)
            eval_points.append({
                "step": step + 1, "val_ppl": ppl,
                "train_loss": float(loss.item()),
                "diagnostics": model.per_layer_diagnostics(),
            })
            el = time.time() - t_wall_start
            print(f"  [{variant}/seed{seed}] step {step+1:5d}/{tcfg.steps} "
                  f"loss={loss.item():.3f} val_ppl={ppl:.2f} elapsed={el:.0f}s")
        train_losses.append(float(loss.item()))

    t_wall = time.time() - t_wall_start
    final_ppl = eval_ppl(model, val_tokens, tcfg, mcfg, device, n_batches=80)
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "task": "saliency_pool_wikitext103",
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
    }
    (out_dir / f"{variant}_seed{seed}.json").write_text(json.dumps(record, indent=2))
    print(f"  [{variant}/seed{seed}] DONE val_ppl={final_ppl:.3f} "
          f"params={record['total_params']:,} "
          f"wall={t_wall:.0f}s peak_mem={peak_mem:.0f}MB")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="experiments/saliency_pool/results_wt103")
    args = ap.parse_args()
    tcfg = TrainConfig(steps=args.steps, seed=args.seed)
    train_one(args.variant, args.seed, Path(args.out), tcfg)


if __name__ == "__main__":
    main()
