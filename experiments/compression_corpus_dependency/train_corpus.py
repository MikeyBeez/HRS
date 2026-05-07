"""Train baseline or compressed transformer on a specified corpus.

Reads a tokenized corpus from a .pt file with the schema
{"splits": {"train": SplitObj, "validation": SplitObj}}.

Implements the spec's training protocol:
  ctx 1024, 5000 steps, batch 32, AdamW lr 3e-4 cosine to 3e-5,
  warmup 200, weight decay 0.01, gradient clip 1.0, bfloat16.

Predict every 16th token (matched task across architectures).

Saves checkpoint, training log, and final stats per (corpus, variant, seed).
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
EXP = REPO / "experiments/compression_corpus_dependency"
sys.path.insert(0, str(REPO))

from experiments.compression_conv.model import (
    CompressedConfig, CompressedTransformer,
)
from experiments.compression_conv.train import (
    compressed_targets, baseline_logits_at_targets,
    compressed_logits_at_targets, COMPRESSION_RATIO,
)


CORPUS_PATHS = {
    "ts": REPO / "experiments/router_lora_phased/data/shakespeare_train.pt",
    "ts_val": REPO / "experiments/router_lora_phased/data/shakespeare_val.pt",
    "code": EXP / "data/code_corpus.pt",
    "wt103": REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt",
}


def load_corpus(corpus_name):
    """Returns (train_tokens, val_tokens) as 1D LongTensors."""
    if corpus_name == "ts":
        # TS files are flat tensors, no split structure
        train = torch.load(CORPUS_PATHS["ts"], weights_only=False)
        val = torch.load(CORPUS_PATHS["ts_val"], weights_only=False)
        return train, val
    elif corpus_name == "code":
        c = torch.load(CORPUS_PATHS["code"], weights_only=False)
        return c["train"], c["validation"]
    elif corpus_name == "wt103":
        c = torch.load(CORPUS_PATHS["wt103"], weights_only=False)
        return c["splits"]["train"].tokens, c["splits"]["validation"].tokens
    else:
        raise ValueError(f"unknown corpus {corpus_name}")


def sample_batch(tokens, batch_size, ctx, device, generator=None):
    n = tokens.shape[0]
    max_start = n - ctx - 1
    if max_start <= 0:
        raise ValueError(f"corpus too small ({n}) for ctx {ctx}")
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


@torch.no_grad()
def coarsened_anchor_acc(model, val_tokens, ctx, n_prompts, n_cycles, device, seed=0):
    """Same protocol as compression_conv/generation_compare.py."""
    initial_prefix_len = ctx - n_cycles * COMPRESSION_RATIO
    if initial_prefix_len <= 0:
        return 0.0, 0
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
    total = n_prompts * n_cycles
    return hits / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True,
                    choices=["ts", "code", "wt103"])
    ap.add_argument("--variant", required=True, choices=["baseline", "compressed"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1024)
    ap.add_argument("--bfloat16", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    print(f"Loading corpus {args.corpus} ...")
    train_tokens, val_tokens = load_corpus(args.corpus)
    print(f"  train: {len(train_tokens):,}  val: {len(val_tokens):,}")

    n_compress = 4 if args.variant == "compressed" else 0
    cfg = CompressedConfig(
        vocab_size=50257, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff,
        ctx_len=args.ctx, n_compress_layers=n_compress, dropout=0.1,
    )
    print(f"Variant: {args.variant}  ctx={args.ctx}  "
          f"n_compress={n_compress}  ratio={2**n_compress}x")

    model = CompressedTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_compress_params = sum(p.numel() for p in model.compress.parameters()) \
        if n_compress > 0 else 0
    print(f"Total params: {n_params:,}  (compression: {n_compress_params:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return step / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Decay from lr to lr_min: factor goes from 1 to lr_min/lr
        ratio = args.lr_min / args.lr
        return ratio + (1 - ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    t0 = time.time()
    eval_every = max(100, args.steps // 25)

    use_amp = args.bfloat16
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    model.train()
    for step in range(args.steps):
        x = sample_batch(train_tokens, args.batch_size, args.ctx, device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_amp):
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

    # Coarsened-anchor accuracy
    anchor_acc, anchor_total = coarsened_anchor_acc(
        model, val_tokens, args.ctx, n_prompts=64, n_cycles=1, device=device,
        seed=args.seed,
    )

    print(f"\nFINAL  val_loss={final_loss:.3f}  val_ppl={final_ppl:.2f}  "
          f"anchor_acc={anchor_acc:.3f} ({anchor_total} trials)  "
          f"wall={time.time()-t0:.0f}s")

    tag = f"{args.corpus}_{args.variant}_seed{args.seed}"
    ck_path = EXP / f"checkpoints/{tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__, "args": vars(args),
        "final_loss": final_loss, "final_ppl": final_ppl,
        "anchor_acc": anchor_acc, "anchor_total": anchor_total,
        "history": history, "wall_s": time.time() - t0,
        "n_params": n_params, "n_compress_params": n_compress_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{tag}.json"
    summary_path.write_text(json.dumps({
        "corpus": args.corpus, "variant": args.variant, "seed": args.seed,
        "ctx": args.ctx, "steps": args.steps,
        "final_val_loss": final_loss, "final_val_ppl": final_ppl,
        "anchor_acc": anchor_acc, "anchor_total": anchor_total,
        "n_params": n_params, "n_compress_params": n_compress_params,
        "wall_s": time.time() - t0, "history": history,
    }, indent=2))
    print(f"Saved {ck_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
