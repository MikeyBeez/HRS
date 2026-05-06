"""Train baseline or compressed transformer on WikiText-103.

Same architecture as train.py (Tiny Shakespeare version), but loads
WikiText-103 from the existing cache:
  experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt

Both architectures predict every COMPRESSION_RATIO-th token (matched task).
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

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_conv"
sys.path.insert(0, str(REPO))

from experiments.compression_conv.model import (
    CompressedConfig, CompressedTransformer,
)
from experiments.compression_conv.train import (
    compressed_targets, baseline_logits_at_targets,
    compressed_logits_at_targets, COMPRESSION_RATIO,
)


WT103_CACHE = REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt"


def load_wt103():
    c = torch.load(WT103_CACHE, weights_only=False)
    return c["splits"]["train"].tokens, c["splits"]["validation"].tokens


def sample_batch(tokens, batch_size, ctx, device, generator=None):
    n = tokens.shape[0]
    max_start = n - ctx - 1
    if generator is None:
        starts = torch.randint(0, max_start, (batch_size,))
    else:
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([tokens[s:s+ctx] for s in starts]).to(device)
    return x


@torch.no_grad()
def eval_loss(model, val_tokens, batch_size, ctx, device, n_batches=20):
    model.eval()
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    for _ in range(n_batches):
        x = sample_batch(val_tokens, batch_size, ctx, device, generator=g)
        logits = model(x)
        targets = compressed_targets(x, COMPRESSION_RATIO)
        if model.compression_ratio == 1:
            logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
        else:
            logits = compressed_logits_at_targets(logits, model.compression_ratio)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "compressed"], required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(f"Loading WT103 cache from {WT103_CACHE} ...")
    train_tokens, val_tokens = load_wt103()
    print(f"Train: {len(train_tokens):,} tokens, val: {len(val_tokens):,}")

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
        x = sample_batch(train_tokens, args.batch_size, args.ctx, device)
        logits = model(x)
        targets = compressed_targets(x, COMPRESSION_RATIO)
        if model.compression_ratio == 1:
            logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
        else:
            logits = compressed_logits_at_targets(logits, model.compression_ratio)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % eval_every == 0 or step == 0:
            val_loss, val_ppl = eval_loss(model, val_tokens, args.batch_size,
                                            args.ctx, device, n_batches=20)
            history.append({
                "step": step + 1, "train_loss": float(loss.item()),
                "val_loss": val_loss, "val_ppl": val_ppl,
                "elapsed_s": time.time() - t0,
            })
            print(f"  step {step+1:5d}/{args.steps}  train_loss={loss.item():.3f}  "
                  f"val_loss={val_loss:.3f}  val_ppl={val_ppl:.2f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    final_loss, final_ppl = eval_loss(model, val_tokens, args.batch_size,
                                        args.ctx, device, n_batches=50)
    print(f"\nFINAL  val_loss={final_loss:.3f}  val_ppl={final_ppl:.2f}  "
          f"wall={time.time()-t0:.0f}s")

    tag = args.out_tag or f"{args.variant}_wt103"
    ck_path = EXP / f"checkpoints/{tag}_ctx{args.ctx}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__, "args": vars(args),
        "final_loss": final_loss, "final_ppl": final_ppl,
        "history": history, "wall_s": time.time() - t0,
        "n_params": n_params, "n_compress_params": n_compress_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{tag}_ctx{args.ctx}.json"
    summary_path.write_text(json.dumps({
        "variant": args.variant, "dataset": "wt103",
        "ctx": args.ctx, "steps": args.steps,
        "final_val_loss": final_loss, "final_val_ppl": final_ppl,
        "n_params": n_params, "n_compress_params": n_compress_params,
        "wall_s": time.time() - t0, "history": history,
    }, indent=2))
    print(f"Saved {ck_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
