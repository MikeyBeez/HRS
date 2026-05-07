"""Train compressed transformer with configurable compression depth.

Stride patterns:
  4-layer (depth_4): [2, 2, 2, 2]
  6-layer (depth_6): [2, 1, 2, 1, 2, 2]
  8-layer (depth_8): [2, 1, 1, 2, 1, 1, 2, 2]

Same training protocol as compression_corpus_dependency: ctx 1024,
5000 steps, batch 8, AdamW lr=3e-4 cosine to 3e-5, warmup 200,
weight decay 0.01, bf16, predict every 16th token.
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
EXP = REPO / "experiments/compression_depth_ablation"
sys.path.insert(0, str(REPO))

from experiments.compression_depth_ablation.model import (
    DepthCompressionConfig, DepthCompressedTransformer,
)
from experiments.compression_conv.train import (
    compressed_targets, compressed_logits_at_targets,
    baseline_logits_at_targets, COMPRESSION_RATIO,
)
from experiments.compression_corpus_dependency.train_corpus import (
    load_corpus, sample_batch, eval_loss as _gen_eval_loss,
    coarsened_anchor_acc as _gen_anchor,
)


STRIDE_PATTERNS = {
    "depth_4": [2, 2, 2, 2],
    "depth_6": [2, 1, 2, 1, 2, 2],
    "depth_8": [2, 1, 1, 2, 1, 1, 2, 2],
}


@torch.no_grad()
def eval_ppl(model, val_tokens, batch_size, ctx, device, n_batches):
    model.eval()
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    for _ in range(n_batches):
        x = sample_batch(val_tokens, batch_size, ctx, device, generator=g)
        logits = model(x)
        targets = compressed_targets(x, COMPRESSION_RATIO)
        logits = compressed_logits_at_targets(logits, model.compression_ratio)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


@torch.no_grad()
def coarsened_anchor_acc(model, val_tokens, ctx, n_prompts, n_cycles, device, seed):
    initial_prefix_len = ctx - n_cycles * COMPRESSION_RATIO
    n = len(val_tokens)
    if n <= ctx + 1:
        return 0.0, 0
    g = torch.Generator(); g.manual_seed(seed)
    starts = sorted(torch.randint(0, n - ctx, (n_prompts,),
                                    generator=g).tolist())
    hits = 0
    for start in starts:
        prefix = val_tokens[start:start + initial_prefix_len].tolist()
        cycle_prefix = torch.tensor(prefix, dtype=torch.long,
                                      device=device).unsqueeze(0)
        for k in range(n_cycles):
            target_pos = start + initial_prefix_len + k * COMPRESSION_RATIO
            gt = int(val_tokens[target_pos].item())
            logits = model(cycle_prefix)
            pred = int(logits[0, -1, :].argmax().item())
            if pred == gt:
                hits += 1
            chunk = val_tokens[target_pos:target_pos + COMPRESSION_RATIO]
            chunk = chunk.to(device).unsqueeze(0)
            cycle_prefix = torch.cat([cycle_prefix, chunk], dim=1)
    return hits / (n_prompts * n_cycles), n_prompts * n_cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["ts", "code", "wt103"])
    ap.add_argument("--depth", required=True, choices=list(STRIDE_PATTERNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1024)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    train_tokens, val_tokens = load_corpus(args.corpus)
    print(f"Corpus: {args.corpus}  train: {len(train_tokens):,}  "
          f"val: {len(val_tokens):,}")

    pattern = STRIDE_PATTERNS[args.depth]
    cfg = DepthCompressionConfig(
        vocab_size=50257, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff,
        ctx_len=args.ctx, stride_pattern=pattern, dropout=0.1,
    )
    print(f"Depth: {args.depth} pattern={pattern}  ratio={cfg.stride_pattern} → "
          f"{2**sum(s == 2 for s in pattern)}x")

    model = DepthCompressedTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_compress_params = sum(p.numel() for p in model.compress.parameters())
    print(f"Total params: {n_params:,}  (compression: {n_compress_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = args.lr_min / args.lr
        return ratio + (1 - ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    t0 = time.time()
    eval_every = max(100, args.steps // 25)
    autocast_dtype = torch.bfloat16

    model.train()
    for step in range(args.steps):
        x = sample_batch(train_tokens, args.batch_size, args.ctx, device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            logits = model(x)
            targets = compressed_targets(x, COMPRESSION_RATIO)
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
            val_loss, val_ppl = eval_ppl(model, val_tokens, args.batch_size,
                                            args.ctx, device, n_batches=20)
            history.append({
                "step": step + 1, "train_loss": float(loss.item()),
                "val_loss": val_loss, "val_ppl": val_ppl,
                "elapsed_s": time.time() - t0,
            })
            print(f"  step {step+1:5d}/{args.steps}  train_loss={loss.item():.3f}  "
                  f"val_ppl={val_ppl:.2f}  elapsed={time.time()-t0:.0f}s")

    final_loss, final_ppl = eval_ppl(model, val_tokens, args.batch_size,
                                        args.ctx, device, n_batches=50)
    anchor_acc, anchor_total = coarsened_anchor_acc(
        model, val_tokens, args.ctx, n_prompts=64, n_cycles=1,
        device=device, seed=args.seed,
    )
    print(f"\nFINAL  val_ppl={final_ppl:.2f}  anchor_acc={anchor_acc:.3f}  "
          f"wall={time.time()-t0:.0f}s")

    tag = f"{args.corpus}_{args.depth}_seed{args.seed}"
    ck_path = EXP / f"checkpoints/{tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"vocab_size": cfg.vocab_size, "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads, "n_layers": cfg.n_layers,
                    "d_ff": cfg.d_ff, "ctx_len": cfg.ctx_len,
                    "stride_pattern": pattern, "dropout": cfg.dropout},
        "args": vars(args),
        "final_loss": final_loss, "final_ppl": final_ppl,
        "anchor_acc": anchor_acc, "anchor_total": anchor_total,
        "history": history, "wall_s": time.time() - t0,
        "n_params": n_params, "n_compress_params": n_compress_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{tag}.json"
    summary_path.write_text(json.dumps({
        "corpus": args.corpus, "depth": args.depth,
        "stride_pattern": pattern, "seed": args.seed,
        "ctx": args.ctx, "steps": args.steps,
        "final_val_loss": final_loss, "final_val_ppl": final_ppl,
        "anchor_acc": anchor_acc, "anchor_total": anchor_total,
        "n_params": n_params, "n_compress_params": n_compress_params,
        "wall_s": time.time() - t0, "history": history,
    }, indent=2))
    print(f"Saved {ck_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
