"""Train gated-compression transformer on WT103 with mixed lengths,
or train length-specific baselines.

Mixed-length training cycles through lengths [16, 64, 256, 1024] with
adaptive batch sizes (64, 64, 16, 4) for ~constant 4096 tokens/batch
above length 32. 5000 steps total — each length seen ~1250 times.

For each length k and gate parameter M:
  if k ≤ M: model output is at original positions; standard next-token LM.
  if k > M: model output is at M compressed positions; predict every-(k/M)-th
            token (compressed_targets style).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/gated_compression"
sys.path.insert(0, str(REPO))

from experiments.gated_compression.model import (
    GatedConfig, GatedTransformer, BaselineTransformer,
)
from experiments.compression_corpus_dependency.train_corpus import sample_batch


WT103_CACHE = REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt"

LENGTHS = [16, 64, 256, 1024]
BATCH_BY_LEN = {16: 64, 64: 64, 256: 16, 1024: 4}
M_DEFAULT = 32


def load_wt103():
    c = torch.load(WT103_CACHE, weights_only=False)
    return c["splits"]["train"].tokens, c["splits"]["validation"].tokens


def compressed_targets_at(batch_inputs, ratio):
    """For input length k with compression ratio r = k/M, return targets at
    positions [r, 2r, ..., (M-1)r] = (M-1) targets per sequence.
    Note: r == 1 case (no compression) handled separately as standard LM."""
    B, T = batch_inputs.shape
    n = T // ratio - 1
    target_positions = torch.arange(1, n + 1, device=batch_inputs.device) * ratio
    return batch_inputs[:, target_positions]


def loss_for_gated(model, x, M):
    """Compute loss for gated model on input of length k.

    If k ≤ M: standard next-token LM. Logits at position i predict token i+1.
    If k > M: compressed-targets task. Compressed pos c predicts token (c+1)*r
              where r = k/M.
    """
    B, k = x.shape
    logits = model(x)                                  # (B, M, V)

    if k <= M:
        # Take logits at positions [0..k-2] (length k-1) predicting tokens [1..k-1].
        valid_logits = logits[:, :k-1, :]              # (B, k-1, V)
        targets = x[:, 1:k]                             # (B, k-1)
    else:
        ratio = k // M                                   # k/M, integer division
        # Compressed pos c (0..M-2) predicts token (c+1)*ratio
        n_targets = M - 1
        valid_logits = logits[:, :n_targets, :]        # (B, M-1, V)
        target_positions = torch.arange(1, n_targets + 1, device=x.device) * ratio
        targets = x[:, target_positions]                # (B, M-1)

    return F.cross_entropy(
        valid_logits.reshape(-1, valid_logits.size(-1)),
        targets.reshape(-1),
    )


def loss_for_baseline(model, x, target_at_ratio=None):
    """Compute loss for baseline model.

    If target_at_ratio is None: standard next-token LM loss at every position.
    If target_at_ratio is r: predict only every-r-th-token (matched-task style),
       pulling logits at positions [r-1, 2r-1, ...] predicting tokens [r, 2r, ...].
    """
    B, T = x.shape
    logits = model(x)                                   # (B, T, V)
    if target_at_ratio is None:
        valid_logits = logits[:, :-1, :]
        targets = x[:, 1:]
    else:
        r = target_at_ratio
        n = T // r - 1
        out_positions = torch.arange(1, n + 1, device=x.device) * r - 1
        valid_logits = logits[:, out_positions, :]
        target_positions = torch.arange(1, n + 1, device=x.device) * r
        targets = x[:, target_positions]
    return F.cross_entropy(
        valid_logits.reshape(-1, valid_logits.size(-1)),
        targets.reshape(-1),
    )


@torch.no_grad()
def eval_loss_gated(model, val_tokens, length, M, n_batches, device):
    model.eval()
    bs = BATCH_BY_LEN[length]
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    for _ in range(n_batches):
        x = sample_batch(val_tokens, bs, length, device, generator=g)
        losses.append(loss_for_gated(model, x, M).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


@torch.no_grad()
def eval_loss_baseline(model, val_tokens, length, n_batches, device,
                         target_at_ratio=None):
    model.eval()
    bs = BATCH_BY_LEN[length]
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    for _ in range(n_batches):
        x = sample_batch(val_tokens, bs, length, device, generator=g)
        losses.append(loss_for_baseline(model, x, target_at_ratio=target_at_ratio).item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def train_gated(args, train_tokens, val_tokens, device):
    cfg = GatedConfig(
        vocab_size=50257, d_model=256, n_heads=8, n_layers=6,
        d_ff=1024, M=args.M, dropout=0.1,
    )
    model = GatedTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Gated model params: {n_params:,}  M={cfg.M}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=0.01, betas=(0.9, 0.95))
    def lr_lambda(step):
        if step < args.warmup:
            return step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = args.lr_min / args.lr
        return ratio + (1 - ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    eval_every = max(100, args.steps // 25)
    autocast_dtype = torch.bfloat16

    t0 = time.time()
    model.train()
    for step in range(args.steps):
        # Cycle through lengths
        length = LENGTHS[step % 4]
        bs = BATCH_BY_LEN[length]
        x = sample_batch(train_tokens, bs, length, device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            loss = loss_for_gated(model, x, cfg.M)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step == 0:
            print(f"  step {step+1:5d}/{args.steps}  len={length:4d}  "
                  f"loss={loss.item():.3f}  elapsed={time.time()-t0:.0f}s")
            history.append({"step": step + 1, "length": length,
                              "train_loss": float(loss.item()),
                              "elapsed_s": time.time() - t0})

    # Final eval at all lengths
    final_metrics = {}
    print("\nFinal eval at each length:")
    for length in LENGTHS:
        loss, ppl = eval_loss_gated(model, val_tokens, length, cfg.M,
                                       n_batches=30, device=device)
        final_metrics[f"len{length}"] = {"loss": loss, "ppl": ppl}
        print(f"  len={length:4d}  loss={loss:.3f}  ppl={ppl:.2f}")

    tag = f"gated_M{args.M}_seed{args.seed}"
    ck_path = EXP / f"checkpoints/{tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "args": vars(args),
        "final_metrics": final_metrics,
        "history": history,
        "wall_s": time.time() - t0,
        "n_params": n_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{tag}.json"
    summary_path.write_text(json.dumps({
        "tag": tag, "kind": "gated", "seed": args.seed, "M": cfg.M,
        "steps": args.steps,
        "final_metrics": final_metrics, "n_params": n_params,
        "wall_s": time.time() - t0, "history": history,
    }, indent=2))
    print(f"Saved {ck_path}\nSaved {summary_path}")


def train_baseline(args, train_tokens, val_tokens, device, length):
    """Train length-specific baseline at fixed length."""
    cfg = GatedConfig(
        vocab_size=50257, d_model=256, n_heads=8, n_layers=6,
        d_ff=1024, M=length, dropout=0.1,    # M unused for baseline; just for consistency
    )
    max_seq = max(length, 1024)
    model = BaselineTransformer(cfg, max_seq=max_seq).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline_{length} params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=0.01, betas=(0.9, 0.95))
    def lr_lambda(step):
        if step < args.warmup:
            return step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = args.lr_min / args.lr
        return ratio + (1 - ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    bs = BATCH_BY_LEN[length]
    history = []
    eval_every = max(100, args.steps // 25)
    autocast_dtype = torch.bfloat16

    t0 = time.time()
    model.train()
    for step in range(args.steps):
        x = sample_batch(train_tokens, bs, length, device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            loss = loss_for_baseline(model, x, target_at_ratio=None)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step == 0:
            print(f"  step {step+1:5d}/{args.steps}  loss={loss.item():.3f}  "
                  f"elapsed={time.time()-t0:.0f}s")
            history.append({"step": step + 1,
                              "train_loss": float(loss.item()),
                              "elapsed_s": time.time() - t0})

    # Eval at the trained length (standard LM) and at the gated-equivalent target ratio
    final_metrics = {}
    M = M_DEFAULT
    # Standard LM eval (every-next-token)
    loss, ppl = eval_loss_baseline(model, val_tokens, length, n_batches=30,
                                       device=device, target_at_ratio=None)
    final_metrics[f"len{length}_std"] = {"loss": loss, "ppl": ppl}
    print(f"  baseline_{length} std-LM:  loss={loss:.3f}  ppl={ppl:.2f}")

    # Compressed-targets eval (matches gated model's targets at this length)
    if length > M:
        ratio = length // M
        loss, ppl = eval_loss_baseline(model, val_tokens, length, n_batches=30,
                                           device=device, target_at_ratio=ratio)
        final_metrics[f"len{length}_at_ratio{ratio}"] = {"loss": loss, "ppl": ppl}
        print(f"  baseline_{length} at_ratio={ratio}:  loss={loss:.3f}  ppl={ppl:.2f}")

    tag = f"baseline_len{length}_seed{args.seed}"
    ck_path = EXP / f"checkpoints/{tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "args": vars(args), "length": length,
        "final_metrics": final_metrics, "history": history,
        "wall_s": time.time() - t0, "n_params": n_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{tag}.json"
    summary_path.write_text(json.dumps({
        "tag": tag, "kind": "baseline", "seed": args.seed, "length": length,
        "steps": args.steps,
        "final_metrics": final_metrics, "n_params": n_params,
        "wall_s": time.time() - t0, "history": history,
    }, indent=2))
    print(f"Saved {ck_path}\nSaved {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["gated", "baseline"])
    ap.add_argument("--length", type=int, default=None,
                    help="for kind=baseline, the fixed length to train at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--M", type=int, default=M_DEFAULT, help="gate bottleneck size")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print("Loading WT103 ...")
    train_tokens, val_tokens = load_wt103()
    print(f"  train: {len(train_tokens):,}  val: {len(val_tokens):,}")

    if args.kind == "gated":
        train_gated(args, train_tokens, val_tokens, device)
    else:
        if args.length is None:
            raise ValueError("--length required for baseline")
        train_baseline(args, train_tokens, val_tokens, device, length=args.length)


if __name__ == "__main__":
    main()
