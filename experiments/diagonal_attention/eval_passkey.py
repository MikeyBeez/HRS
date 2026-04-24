"""Train variant from scratch on synthetic passkey retrieval, then eval.

The test: a passkey of K digits is placed at a random position after a
MARKER token. A QUERY token at the end signals the model to reproduce
the passkey. Loss is only on the answer span.

Accuracy is bucketed by marker position (fraction of max placement).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
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
    loss_mask_for_answer,
    make_passkey_eval_set,
    sample_passkey_batch,
)
from experiments.diagonal_attention.model import TinyTransformer


def _lr_at(step: int, tcfg: TrainConfig) -> float:
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    progress = (step - tcfg.warmup_steps) / max(1, tcfg.steps - tcfg.warmup_steps)
    return tcfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, pkcfg: PasskeyConfig, eval_set, device) -> dict:
    model.eval()
    K = pkcfg.passkey_len
    T = pkcfg.ctx_len
    query_idx = T - K - 1

    per_bucket_correct = defaultdict(int)
    per_bucket_total = defaultdict(int)
    per_bucket_digit_correct = defaultdict(int)
    per_bucket_digit_total = defaultdict(int)

    for seq_np, pk_np, frac in eval_set:
        # For eval, we ask the model to predict p1..pK from the prefix that
        # ends at position query_idx + K - 1 (since that predicts pK).
        # Equivalent: feed the full sequence, read logits at positions
        # query_idx .. query_idx + K - 1, argmax -> predicted digits.
        x = torch.from_numpy(seq_np[None]).to(device)
        logits = model(x)  # (1, T, V)
        pred = logits[0, query_idx : query_idx + K].argmax(dim=-1).cpu().numpy()

        per_bucket_total[frac] += 1
        per_bucket_digit_total[frac] += K
        digit_hits = int((pred == pk_np).sum())
        per_bucket_digit_correct[frac] += digit_hits
        if digit_hits == K:
            per_bucket_correct[frac] += 1

    buckets = sorted(per_bucket_total.keys())
    by_bucket = []
    for b in buckets:
        by_bucket.append({
            "position_frac": b,
            "exact_acc": per_bucket_correct[b] / per_bucket_total[b],
            "digit_acc": per_bucket_digit_correct[b] / per_bucket_digit_total[b],
            "n": per_bucket_total[b],
        })
    overall_exact = sum(per_bucket_correct.values()) / sum(per_bucket_total.values())
    overall_digit = (
        sum(per_bucket_digit_correct.values())
        / sum(per_bucket_digit_total.values())
    )
    model.train()
    return {
        "overall_exact_acc": overall_exact,
        "overall_digit_acc": overall_digit,
        "by_bucket": by_bucket,
    }


def train_passkey(variant: str, out_dir: Path,
                   tcfg: TrainConfig | None = None,
                   pkcfg: PasskeyConfig | None = None) -> dict:
    tcfg = tcfg or TrainConfig()
    pkcfg = pkcfg or PasskeyConfig()
    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mcfg = ModelConfig(
        variant=variant,
        vocab_size=pkcfg.vocab_size,
        ctx_len=pkcfg.ctx_len,
    )
    model = TinyTransformer(mcfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
        betas=tcfg.betas,
    )

    rng = np.random.default_rng(tcfg.seed + 1)
    mask = loss_mask_for_answer(pkcfg, device)  # (T,)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    eval_set = make_passkey_eval_set(pkcfg, np.random.default_rng(9999))
    step_times = []
    eval_points = []

    model.train()
    t_wall_start = time.time()
    for step in range(tcfg.steps):
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, tcfg)

        x, y = sample_passkey_batch(pkcfg, tcfg.batch_size, device, rng)
        t0 = time.time()
        logits = model(x)  # (B, T, V)
        # Masked CE: only score answer positions.
        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_tgt = y.reshape(-1)
        flat_mask = mask[None].expand(B, T).reshape(-1)
        loss = F.cross_entropy(
            flat_logits[flat_mask], flat_tgt[flat_mask]
        )
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
            metrics = evaluate(model, pkcfg, eval_set, device)
            eval_points.append({
                "step": step + 1,
                "train_loss": float(loss.item()),
                **metrics,
            })
            print(f"  [{variant}] step {step+1:4d}/{tcfg.steps} "
                  f"loss={loss.item():.3f} "
                  f"exact={metrics['overall_exact_acc']:.2f} "
                  f"digit={metrics['overall_digit_acc']:.2f}")

    t_wall = time.time() - t_wall_start
    final_metrics = evaluate(model, pkcfg, eval_set, device)
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
        "task": "passkey",
        "variant": variant,
        "final": final_metrics,
        "score_params": model.score_params(),
        "total_params": model.total_params(),
        "step_time_ms_median": float(np.median(step_times) * 1000),
        "step_time_ms_mean": float(np.mean(step_times) * 1000),
        "wall_seconds": t_wall,
        "peak_mem_mb": peak_mem,
        "eval_points": eval_points,
        "steps": tcfg.steps,
        "batch_size": tcfg.batch_size,
        "ctx_len": mcfg.ctx_len,
        "passkey_len": pkcfg.passkey_len,
        "diagonal_w_stats": diagonal_w_stats,
    }
    (out_dir / f"passkey_{variant}.json").write_text(json.dumps(record, indent=2))
    print(f"  [{variant}] done: exact={final_metrics['overall_exact_acc']:.2f} "
          f"digit={final_metrics['overall_digit_acc']:.2f} "
          f"score_params={record['score_params']:,}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["mha", "bilinear", "diagonal", "identity"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="experiments/diagonal_attention/results")
    args = ap.parse_args()

    train_passkey(args.variant, Path(args.out), TrainConfig(steps=args.steps))


if __name__ == "__main__":
    main()
