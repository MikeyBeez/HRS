"""Train baseline (no compression) or experimental (16x compression)
on Tiny Shakespeare. Both predict every 16th token (matched task).

Both architectures see the same input tokens[0..T-1] and predict tokens
[16, 32, ..., T-16]. The compressed model produces T/16 outputs natively
at compressed positions; the baseline produces T outputs and we slice
at positions [15, 31, ..., T-17] to match the same target tokens.

CE loss is computed over the same number of (B, n_targets) predictions
in both cases, so perplexity numbers are directly comparable.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python train.py \\
        --variant baseline --steps 2000 --ctx 512
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python train.py \\
        --variant compressed --steps 2000 --ctx 512
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
EXP = REPO / "experiments/compression_conv"
sys.path.insert(0, str(REPO))

from experiments.compression_conv.model import (
    CompressedConfig, CompressedTransformer,
)


COMPRESSION_RATIO = 16   # 2^4


def get_batch(data, batch_size, ctx_len, device, generator=None):
    """Sample a random batch of contiguous sequences of length ctx_len."""
    n = len(data) - ctx_len - 1  # need ctx_len input + 1 final target
    if generator is None:
        idx = torch.randint(0, n, (batch_size,))
    else:
        idx = torch.randint(0, n, (batch_size,), generator=generator)
    x = torch.stack([data[i:i+ctx_len] for i in idx]).to(device)
    return x


def compressed_targets(batch_inputs, ratio):
    """Given batch_inputs of shape (B, T), return target tokens for each
    compressed-aligned prediction:
      target[k] = batch_inputs[:, ratio * (k+1)]  for k in 0..T/ratio - 2
    so we have T/ratio - 1 targets per sequence, matching the number
    of usable compressed-output positions (drop the last one whose
    target would be out of bounds).
    """
    B, T = batch_inputs.shape
    n = T // ratio - 1
    target_positions = torch.arange(1, n + 1, device=batch_inputs.device) * ratio
    # target_positions: [ratio, 2*ratio, ..., n*ratio]; max = n*ratio = T - ratio
    return batch_inputs[:, target_positions]    # (B, n)


def baseline_logits_at_targets(logits, ratio):
    """For baseline (no compression), logits has shape (B, T, V).
    To predict tokens at positions [ratio, 2*ratio, ...], we use logits
    at positions [ratio-1, 2*ratio-1, ...].
    Return the n-position slice (B, n, V) that matches compressed targets."""
    B, T, V = logits.shape
    n = T // ratio - 1
    output_positions = torch.arange(1, n + 1, device=logits.device) * ratio - 1
    return logits[:, output_positions, :]


def compressed_logits_at_targets(logits, ratio):
    """For compressed model, logits has shape (B, T/ratio, V).
    Output position k corresponds to summary of tokens [k*ratio..(k+1)*ratio-1]
    and predicts the next token, which is target[k] = tokens[(k+1)*ratio].
    We have T/ratio compressed positions; the last one's target is out of
    bounds, so we drop it and keep n = T/ratio - 1 positions."""
    return logits[:, :-1, :]   # drop last; keep n = T/ratio - 1


@torch.no_grad()
def eval_loss(model, data, batch_size, ctx_len, device, n_batches=20):
    """Compute mean CE loss + perplexity over n_batches sampled batches."""
    model.eval()
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    for _ in range(n_batches):
        x = get_batch(data, batch_size, ctx_len, device, generator=g)
        logits = model(x)
        targets = compressed_targets(x, COMPRESSION_RATIO)
        if model.compression_ratio == 1:
            logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
        else:
            logits = compressed_logits_at_targets(logits, model.compression_ratio)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "compressed"], required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--out-tag", default=None,
                    help="Suffix for output filenames (default uses variant)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    # Tiny Shakespeare data
    train_data = torch.load(REPO / "experiments/router_lora_phased/data/shakespeare_train.pt",
                              weights_only=False)
    val_data = torch.load(REPO / "experiments/router_lora_phased/data/shakespeare_val.pt",
                            weights_only=False)
    print(f"Train: {len(train_data):,} tokens, val: {len(val_data):,} tokens")

    n_compress = 4 if args.variant == "compressed" else 0
    cfg = CompressedConfig(
        vocab_size=50257, d_model=args.d_model, n_heads=6,
        n_layers=args.n_layers, d_ff=4 * args.d_model,
        ctx_len=args.ctx, n_compress_layers=n_compress, dropout=0.0,
    )
    print(f"Variant: {args.variant}  ctx={args.ctx}  "
          f"n_compress={n_compress}  ratio={2**n_compress}x")

    model = CompressedTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_compress_params = sum(p.numel() for p in model.compress.parameters()) \
        if n_compress > 0 else 0
    print(f"Total params: {n_params:,}  (compression: {n_compress_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=0.01, betas=(0.9, 0.95))

    # Cosine schedule with warmup
    warmup = 100
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    t0 = time.time()
    eval_every = max(100, args.steps // 20)
    model.train()
    for step in range(args.steps):
        x = get_batch(train_data, args.batch_size, args.ctx, device)
        logits = model(x)
        targets = compressed_targets(x, COMPRESSION_RATIO)
        if model.compression_ratio == 1:
            logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
        else:
            logits = compressed_logits_at_targets(logits, model.compression_ratio)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step == 0:
            val_loss, val_ppl = eval_loss(model, val_data, args.batch_size,
                                            args.ctx, device, n_batches=20)
            history.append({
                "step": step + 1, "train_loss": float(loss.item()),
                "val_loss": val_loss, "val_ppl": val_ppl,
                "elapsed_s": time.time() - t0,
            })
            print(f"  step {step+1:5d}/{args.steps}  train_loss={loss.item():.3f}  "
                  f"val_loss={val_loss:.3f}  val_ppl={val_ppl:.2f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # Final eval (more batches for stability)
    final_loss, final_ppl = eval_loss(model, val_data, args.batch_size,
                                        args.ctx, device, n_batches=50)
    print(f"\nFINAL  val_loss={final_loss:.3f}  val_ppl={final_ppl:.2f}  "
          f"wall={time.time()-t0:.0f}s")

    tag = args.out_tag or args.variant
    ck_path = EXP / f"checkpoints/{tag}_ctx{args.ctx}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "args": vars(args),
        "final_loss": final_loss, "final_ppl": final_ppl,
        "history": history, "wall_s": time.time() - t0,
        "n_params": n_params, "n_compress_params": n_compress_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{tag}_ctx{args.ctx}.json"
    summary_path.write_text(json.dumps({
        "variant": args.variant, "ctx": args.ctx, "steps": args.steps,
        "final_val_loss": final_loss, "final_val_ppl": final_ppl,
        "n_params": n_params, "n_compress_params": n_compress_params,
        "wall_s": time.time() - t0,
        "history": history,
    }, indent=2))
    print(f"Saved {ck_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
