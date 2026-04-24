"""Train the two task-specific MHA baselines and save checkpoints.

See plan.md "Implementation note" for why we train two baselines instead of one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.diagonal_attention.config import (
    ModelConfig,
    PasskeyConfig,
    TrainConfig,
)
from experiments.diagonal_attention.data import (
    load_shakespeare,
    loss_mask_for_answer,
    make_passkey_eval_set,
    sample_lm_batch,
    sample_passkey_batch,
)
from experiments.diagonal_attention.eval_passkey import evaluate as eval_passkey
from experiments.diagonal_attention.model import TinyTransformer
from experiments.diagonal_attention.train import _lr_at, eval_ppl


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"


def train_lm_baseline(steps: int, seed: int = 0) -> Path:
    tcfg = TrainConfig(steps=steps, seed=seed)
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data, info = load_shakespeare()
    mcfg = ModelConfig(variant="mha", vocab_size=info["vocab_size"])
    model = TinyTransformer(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                             weight_decay=tcfg.weight_decay, betas=tcfg.betas)
    rng = np.random.default_rng(tcfg.seed)

    model.train()
    t0 = time.time()
    for step in range(tcfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tcfg)
        x, y = sample_lm_batch(train_data, tcfg.batch_size, mcfg.ctx_len, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        if (step + 1) % max(1, tcfg.steps // 10) == 0:
            ppl = eval_ppl(model, val_data, tcfg, mcfg, device)
            print(f"  [lm] step {step+1}/{tcfg.steps} loss={loss.item():.3f} val_ppl={ppl:.2f}")

    final_ppl = eval_ppl(model, val_data, tcfg, mcfg, device, n_batches=80)
    wall = time.time() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "mha_lm.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "mcfg": mcfg.__dict__,
        "final_val_ppl": final_ppl,
        "steps": tcfg.steps,
        "vocab_info": {"vocab_size": info["vocab_size"]},
        "task": "lm",
    }, ckpt_path)
    print(f"  saved {ckpt_path}  final_val_ppl={final_ppl:.3f}  wall={wall:.1f}s")
    return ckpt_path


def train_passkey_baseline(steps: int, seed: int = 0) -> Path:
    tcfg = TrainConfig(steps=steps, seed=seed)
    pkcfg = PasskeyConfig()
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = ModelConfig(variant="mha", vocab_size=pkcfg.vocab_size, ctx_len=pkcfg.ctx_len)
    model = TinyTransformer(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                             weight_decay=tcfg.weight_decay, betas=tcfg.betas)
    rng = np.random.default_rng(tcfg.seed + 1)
    mask = loss_mask_for_answer(pkcfg, device)
    eval_set = make_passkey_eval_set(pkcfg, np.random.default_rng(9999))

    model.train()
    t0 = time.time()
    for step in range(tcfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tcfg)
        x, y = sample_passkey_batch(pkcfg, tcfg.batch_size, device, rng)
        logits = model(x)
        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_tgt = y.reshape(-1)
        flat_mask = mask[None].expand(B, T).reshape(-1)
        loss = F.cross_entropy(flat_logits[flat_mask], flat_tgt[flat_mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        if (step + 1) % max(1, tcfg.steps // 20) == 0:
            metrics = eval_passkey(model, pkcfg, eval_set, device)
            print(f"  [pk] step {step+1}/{tcfg.steps} loss={loss.item():.3f} "
                  f"exact={metrics['overall_exact_acc']:.2f} "
                  f"digit={metrics['overall_digit_acc']:.2f}")

    final = eval_passkey(model, pkcfg, eval_set, device)
    wall = time.time() - t0

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / "mha_passkey.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "mcfg": mcfg.__dict__,
        "final_passkey": final,
        "steps": tcfg.steps,
        "pkcfg": pkcfg.__dict__,
        "task": "passkey",
    }, ckpt_path)
    print(f"  saved {ckpt_path}  exact={final['overall_exact_acc']:.3f} "
          f"digit={final['overall_digit_acc']:.3f}  wall={wall:.1f}s")
    return ckpt_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-steps", type=int, default=5000)
    ap.add_argument("--passkey-steps", type=int, default=20000)
    ap.add_argument("--skip-lm", action="store_true")
    ap.add_argument("--skip-passkey", action="store_true")
    args = ap.parse_args()

    if not args.skip_lm:
        print("=== train LM baseline ===")
        train_lm_baseline(args.lm_steps)
    if not args.skip_passkey:
        print("=== train passkey baseline ===")
        train_passkey_baseline(args.passkey_steps)


if __name__ == "__main__":
    main()
