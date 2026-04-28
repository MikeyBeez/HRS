"""Three-phase training driver: Phase 1 (base on Shakespeare),
Phase 2 (LoRA on Dickens first half), Phase 3 (router on mixed batches).
"""
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

from experiments.router_lora_phased.model import (
    TinyTransformer, TinyConfig, Router,
)


DATA_DIR = REPO / "experiments/router_lora_phased/data"
RESULT_DIR = REPO / "experiments/router_lora_phased/results"


def lr_at(step, total, peak, warmup=100):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int,
                  device: torch.device, rng: np.random.Generator):
    n = tokens.shape[0]
    starts = rng.integers(0, n - seq_len - 1, size=batch_size)
    x = torch.stack([tokens[s:s + seq_len] for s in starts]).to(device)
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def eval_lm(model, val_tokens, batch_size, seq_len, device, n_batches=20,
             lora_scale=None) -> float:
    model.eval()
    rng = np.random.default_rng(1234)
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(val_tokens, batch_size, seq_len, device, rng)
        logits, _ = model(x, lora_scale=lora_scale)
        l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(l.item())
    model.train()
    return float(math.exp(np.mean(losses)))


def phase1_train_base(steps=1500, batch_size=16, seq_len=512, lr=3e-4, seed=0):
    """Train the base model on Shakespeare BPE."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda")
    info = json.loads((DATA_DIR / "info.json").read_text())
    sh_train = torch.load(DATA_DIR / "shakespeare_train.pt", weights_only=False)
    sh_val = torch.load(DATA_DIR / "shakespeare_val.pt", weights_only=False)

    cfg = TinyConfig(vocab_size=info["vocab_size"], dropout=0.1)
    model = TinyTransformer(cfg).to(device)
    print(f"[Phase 1] params: {model.total_params():,}  base: "
          f"{sum(p.numel() for p in model.base_params()):,}")

    opt = torch.optim.AdamW(list(model.base_params()), lr=lr,
                              weight_decay=0.01, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y = sample_batch(sh_train, batch_size, seq_len, device, rng)
        logits, _ = model(x, lora_scale=0.0)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.base_params()), 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            ppl = eval_lm(model, sh_val, batch_size, seq_len, device,
                           lora_scale=0.0)
            history.append({"step": step + 1, "train_loss": float(loss.item()),
                            "val_ppl": ppl})
            print(f"  step {step+1:5d}/{steps}  train_loss={loss.item():.3f}  "
                  f"val_ppl={ppl:.2f}  elapsed={time.time()-t0:.0f}s")

    final_ppl = eval_lm(model, sh_val, batch_size, seq_len, device, n_batches=80,
                          lora_scale=0.0)
    print(f"[Phase 1] DONE  final_val_ppl={final_ppl:.3f}  wall={time.time()-t0:.0f}s")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": vars(cfg),
        "final_val_ppl": final_ppl,
        "history": history,
    }, RESULT_DIR / "phase1_base.pt")
    return final_ppl, history


def phase2_train_lora(steps=2000, batch_size=16, seq_len=512, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda")
    info = json.loads((DATA_DIR / "info.json").read_text())
    di_lora = torch.load(DATA_DIR / "dickens_lora.pt", weights_only=False)
    # Held-out Dickens for tracking LoRA val_ppl
    di_router = torch.load(DATA_DIR / "dickens_router.pt", weights_only=False)
    sh_val = torch.load(DATA_DIR / "shakespeare_val.pt", weights_only=False)

    cfg = TinyConfig(vocab_size=info["vocab_size"])
    model = TinyTransformer(cfg).to(device)
    ck = torch.load(RESULT_DIR / "phase1_base.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    # Freeze base; only LoRA trainable.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.lora_params():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(list(model.lora_params()), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y = sample_batch(di_lora, batch_size, seq_len, device, rng)
        logits, _ = model(x, lora_scale=1.0)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.lora_params()), 1.0)
        opt.step()

        if (step + 1) % 200 == 0 or step == 0:
            di_ppl_on = eval_lm(model, di_router, batch_size, seq_len, device,
                                  lora_scale=1.0)
            di_ppl_off = eval_lm(model, di_router, batch_size, seq_len, device,
                                   lora_scale=0.0)
            sh_ppl_on = eval_lm(model, sh_val, batch_size, seq_len, device,
                                  lora_scale=1.0)
            sh_ppl_off = eval_lm(model, sh_val, batch_size, seq_len, device,
                                   lora_scale=0.0)
            history.append({
                "step": step + 1, "train_loss": float(loss.item()),
                "dickens_ppl_lora_on": di_ppl_on,
                "dickens_ppl_lora_off": di_ppl_off,
                "shakespeare_ppl_lora_on": sh_ppl_on,
                "shakespeare_ppl_lora_off": sh_ppl_off,
            })
            print(f"  step {step+1:5d}/{steps}  loss={loss.item():.3f}  "
                  f"DickensPPL on/off={di_ppl_on:.2f}/{di_ppl_off:.2f}  "
                  f"ShakePPL on/off={sh_ppl_on:.2f}/{sh_ppl_off:.2f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    print(f"[Phase 2] DONE  wall={time.time()-t0:.0f}s")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": vars(cfg),
        "history": history,
    }, RESULT_DIR / "phase2_lora.pt")
    return history


def phase3_train_router(steps=1500, batch_size=16, seq_len=512, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda")
    info = json.loads((DATA_DIR / "info.json").read_text())
    di_router = torch.load(DATA_DIR / "dickens_router.pt", weights_only=False)
    sh_train = torch.load(DATA_DIR / "shakespeare_train.pt", weights_only=False)
    sh_val = torch.load(DATA_DIR / "shakespeare_val.pt", weights_only=False)

    cfg = TinyConfig(vocab_size=info["vocab_size"])
    model = TinyTransformer(cfg).to(device)
    ck = torch.load(RESULT_DIR / "phase2_lora.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])

    # Freeze base + LoRA. Only router trains.
    for p in model.parameters():
        p.requires_grad_(False)

    router = Router(cfg).to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))

    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()
    router.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        # Alternate Dickens / Shakespeare batches each step (50/50 mix).
        is_dickens = (step % 2 == 0)
        if is_dickens:
            x, y = sample_batch(di_router, batch_size, seq_len, device, rng)
        else:
            x, y = sample_batch(sh_train, batch_size, seq_len, device, rng)

        # Two-pass: 1st with lora_scale=0 to capture post-attn hidden states for
        # the router; 2nd with lora_scale=router_output for the actual loss.
        with torch.no_grad():
            _, hidden = model(x, lora_scale=0.0, capture_router_layer=True)
        weights = router(hidden)                               # (B,)
        logits, _ = model(x, lora_scale=weights)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        opt.step()

        if (step + 1) % 100 == 0 or step == 0:
            history.append({
                "step": step + 1, "is_dickens": bool(is_dickens),
                "train_loss": float(loss.item()),
                "mean_router_weight": float(weights.mean().item()),
            })
            print(f"  step {step+1:5d}/{steps}  is_dickens={is_dickens}  "
                  f"loss={loss.item():.3f}  router_w_mean={weights.mean().item():.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # End-of-training summary: router activation distribution on Dickens vs Shakespeare
    print("\n[Phase 3] router-activation summary:")
    router.eval()
    n_eval_batches = 20
    with torch.no_grad():
        d_acts, s_acts = [], []
        for _ in range(n_eval_batches):
            x, _ = sample_batch(di_router, batch_size, seq_len, device, rng)
            _, hidden = model(x, lora_scale=0.0, capture_router_layer=True)
            d_acts.extend(router(hidden).tolist())
            x, _ = sample_batch(sh_train, batch_size, seq_len, device, rng)
            _, hidden = model(x, lora_scale=0.0, capture_router_layer=True)
            s_acts.extend(router(hidden).tolist())
    d_mean, d_std = float(np.mean(d_acts)), float(np.std(d_acts))
    s_mean, s_std = float(np.mean(s_acts)), float(np.std(s_acts))
    print(f"  Dickens activations:     mean={d_mean:.3f}  std={d_std:.3f}  "
          f"min={min(d_acts):.3f}  max={max(d_acts):.3f}")
    print(f"  Shakespeare activations: mean={s_mean:.3f}  std={s_std:.3f}  "
          f"min={min(s_acts):.3f}  max={max(s_acts):.3f}")
    sep = d_mean - s_mean
    print(f"  Separation (Dickens - Shakespeare mean): {sep:+.3f}")

    print(f"\n[Phase 3] DONE  wall={time.time()-t0:.0f}s")
    torch.save({
        "router_state_dict": router.state_dict(),
        "history": history,
        "activation_stats": {
            "dickens": {"mean": d_mean, "std": d_std,
                          "min": min(d_acts), "max": max(d_acts),
                          "samples": d_acts},
            "shakespeare": {"mean": s_mean, "std": s_std,
                              "min": min(s_acts), "max": max(s_acts),
                              "samples": s_acts},
            "separation": sep,
        },
    }, RESULT_DIR / "phase3_router.pt")
    return history


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["1", "2", "3", "all"])
    args = ap.parse_args()
    if args.phase in ("1", "all"):
        print("=" * 70); print("PHASE 1"); print("=" * 70)
        phase1_train_base()
    if args.phase in ("2", "all"):
        print("\n" + "=" * 70); print("PHASE 2"); print("=" * 70)
        phase2_train_lora()
    if args.phase in ("3", "all"):
        print("\n" + "=" * 70); print("PHASE 3"); print("=" * 70)
        phase3_train_router()
