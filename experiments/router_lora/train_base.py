"""Train the base TinyTransformer on Tiny Shakespeare. Save checkpoint."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.diagonal_attention.data import load_shakespeare, sample_lm_batch
from experiments.router_lora.model import TinyTransformer, TinyConfig


def lr_at(step, total, peak=3e-4, warmup=100):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_ppl(model, val, batch_size, ctx, device, n_batches=40):
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_lm_batch(val, batch_size, ctx, device, rng)
        logits, _ = model(x)
        l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(l.item())
    model.train()
    return float(math.exp(np.mean(losses)))


def main(steps=3000, batch_size=32, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, info = load_shakespeare()
    cfg = TinyConfig(vocab_size=info["vocab_size"], ctx_len=512)
    model = TinyTransformer(cfg).to(device)
    print(f"params: {model.total_params():,}  base: "
          f"{sum(p.numel() for p in model.base_params()):,}")

    # Train ONLY base parameters (LoRA stays at init zeros — that's the "Shakespeare prior" state).
    opt = torch.optim.AdamW(list(model.base_params()), lr=3e-4, weight_decay=0.01,
                             betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps)
        x, y = sample_lm_batch(train, batch_size, cfg.ctx_len, device, rng)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            ppl = eval_ppl(model, val, batch_size, cfg.ctx_len, device)
            history.append({"step": step + 1, "loss": float(loss.item()), "val_ppl": ppl})
            print(f"  step {step+1:5d}/{steps}  loss={loss.item():.3f}  val_ppl={ppl:.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    final_ppl = eval_ppl(model, val, batch_size, cfg.ctx_len, device, n_batches=80)
    out_dir = REPO / "experiments/router_lora/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "base_shakespeare.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": vars(cfg),
        "final_val_ppl": final_ppl,
        "history": history,
    }, ckpt_path)
    print(f"\nDONE  final_val_ppl={final_ppl:.3f}  wall={time.time()-t0:.0f}s")
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
