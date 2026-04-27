"""Train one (p, seed) of engram_dropout on Tiny Shakespeare."""
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
from experiments.engram_dropout.model import (
    EngramDropoutConfig,
    EngramTinyTransformer,
    engram_reconstruction_loss,
)


def _lr_at(step: int, warmup: int, total: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_ppl(model, val_data, batch_size, ctx_len, device,
             n_batches: int = 40, drop_engram: bool = False) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_lm_batch(val_data, batch_size, ctx_len, device, rng)
        out = model(x, drop_engram=drop_engram)
        loss = F.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]),
                                y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(math.exp(np.mean(losses)))


def train_one(p: float, seed: int, out_dir: Path,
               steps: int, batch_size: int = 32,
               lr: float = 3e-4, recon_weight: float = 0.1,
               eval_every: int = 500,
               save_checkpoint: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, info = load_shakespeare()

    cfg = EngramDropoutConfig(
        d_model=256, n_heads=4, n_layers=6, d_ff=1024, ctx_len=256,
        vocab_size=info["vocab_size"], engram_dropout_p=p,
        recon_loss_weight=recon_weight,
    )
    model = EngramTinyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                             betas=(0.9, 0.95))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rng = np.random.default_rng(seed)
    step_times = []
    history = []
    diverged = False

    model.train()
    t_start = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, warmup=100, total=steps, peak=lr)

        x, y = sample_lm_batch(train_data, batch_size, cfg.ctx_len, device, rng)

        # Per-batch engram dropout: with prob p, zero the engram tensor going
        # into cross-attention. Encoder still runs (recon loss still applies).
        drop_this_batch = (np.random.random() < p)

        t0 = time.time()
        out = model(x, drop_engram=drop_this_batch, return_recon=True)
        ce_loss = F.cross_entropy(
            out["logits"].reshape(-1, out["logits"].shape[-1]), y.reshape(-1)
        )
        recon_loss = engram_reconstruction_loss(
            out["h_at_extract"], out["engram"], cfg.engram_window
        )
        loss = ce_loss + recon_weight * recon_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)

        if not torch.isfinite(loss):
            diverged = True
            print(f"  [p={p}/seed{seed}] DIVERGED at step {step}")
            break

        if (step + 1) % 100 == 0 or step == 0:
            history.append({
                "step": step + 1,
                "train_loss": float(ce_loss.item()),
                "recon_loss": float(recon_loss.item()),
                "grad_norm": float(grad_norm.item()),
                "drop_this_batch": bool(drop_this_batch),
            })

        if (step + 1) % eval_every == 0 or step == 0:
            val_ppl_on = eval_ppl(model, val_data, batch_size, cfg.ctx_len, device,
                                   drop_engram=False)
            val_ppl_off = eval_ppl(model, val_data, batch_size, cfg.ctx_len, device,
                                    drop_engram=True)
            history[-1].update({
                "val_ppl_on": val_ppl_on,
                "val_ppl_off": val_ppl_off,
            })
            elapsed = time.time() - t_start
            gate_v = model.blocks[cfg.engram_layer].cross_attn.gate_value()
            print(f"  [p={p}/seed{seed}] step {step+1:5d}/{steps} "
                  f"loss={ce_loss.item():.3f} recon={recon_loss.item():.3f} "
                  f"ppl_on={val_ppl_on:.2f} ppl_off={val_ppl_off:.2f} "
                  f"gate={gate_v:.4f} elapsed={elapsed:.0f}s")

    # Final eval (80 batches, both engram on and off)
    final_ppl_on = eval_ppl(model, val_data, batch_size, cfg.ctx_len, device,
                              n_batches=80, drop_engram=False)
    final_ppl_off = eval_ppl(model, val_data, batch_size, cfg.ctx_len, device,
                               n_batches=80, drop_engram=True)
    diags = model.get_diagnostics()
    t_wall = time.time() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                if device.type == "cuda" else 0.0)

    record = {
        "task": "engram_dropout_shakespeare",
        "engram_dropout_p": p,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "recon_weight": recon_weight,
        "diverged": diverged,
        "final_val_ppl_on": final_ppl_on,
        "final_val_ppl_off": final_ppl_off,
        "ablation_gap": final_ppl_off - final_ppl_on,
        "total_params": model.total_params(),
        "step_time_ms_median": float(np.median(step_times) * 1000),
        "wall_seconds": t_wall,
        "peak_mem_mb": peak_mb,
        "history": history,
        "final_diagnostics": diags,
        "ctx_len": cfg.ctx_len,
        "vocab_size": cfg.vocab_size,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / f"p{p}_seed{seed}.json"
    rec_path.write_text(json.dumps(record, indent=2))

    if save_checkpoint and not diverged:
        ckpt_path = out_dir / f"p{p}_seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": vars(cfg),
            "p": p, "seed": seed, "steps": steps,
            "final_val_ppl_on": final_ppl_on,
            "final_val_ppl_off": final_ppl_off,
        }, ckpt_path)

    print(f"  [p={p}/seed{seed}] DONE  ppl_on={final_ppl_on:.3f}  "
          f"ppl_off={final_ppl_off:.3f}  gap={final_ppl_off-final_ppl_on:+.3f}  "
          f"wall={t_wall:.0f}s")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--out", default="experiments/engram_dropout/results")
    args = ap.parse_args()
    train_one(args.p, args.seed, Path(args.out), steps=args.steps)


if __name__ == "__main__":
    main()
