"""Train multi-mechanism supernet with floor-then-release gate schedule.

Per launch direction (proposal B): ctx 512, 3000 steps, batch 8, all 12
mechanisms. Floor active for steps 0..1500, released 1500..3000.

Matched task: predict every 16th token (consistent with prior experiments).
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
EXP = REPO / "experiments/multi_mechanism_supernet"
sys.path.insert(0, str(REPO))

from experiments.multi_mechanism_supernet.model import (
    SupernetConfig, SupernetTransformer,
)
from experiments.multi_mechanism_supernet.mechanisms import MECHANISM_NAMES
from experiments.compression_conv.train import (
    compressed_targets, baseline_logits_at_targets, COMPRESSION_RATIO,
)
from experiments.compression_corpus_dependency.train_corpus import (
    load_corpus, sample_batch,
)


GATE_LOG_EVERY = 100      # log gate magnitudes every N steps
EVAL_EVERY = 250          # evaluate val PPL every N steps


@torch.no_grad()
def eval_loss(model, val_tokens, batch_size, ctx, device, n_batches=20):
    model.eval()
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    for _ in range(n_batches):
        x = sample_batch(val_tokens, batch_size, ctx, device, generator=g)
        logits = model(x, floor_active=False)   # eval with current released gates
        targets = compressed_targets(x, COMPRESSION_RATIO)
        logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


@torch.no_grad()
def coarsened_anchor_acc(model, val_tokens, ctx, n_prompts, device, seed=0):
    n = len(val_tokens)
    if n <= ctx + 1:
        return 0.0, 0
    g = torch.Generator(); g.manual_seed(seed)
    initial_prefix_len = ctx - COMPRESSION_RATIO
    starts = sorted(torch.randint(0, n - ctx, (n_prompts,), generator=g).tolist())
    hits = 0
    for start in starts:
        prefix = val_tokens[start:start + initial_prefix_len].tolist()
        ids_t = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        target_pos = start + initial_prefix_len
        gt = int(val_tokens[target_pos].item())
        logits = model(ids_t, floor_active=False)
        pred = int(logits[0, -1, :].argmax().item())
        if pred == gt:
            hits += 1
    return hits / n_prompts, n_prompts


def get_gate_snapshot(model):
    """Return per-(layer, mechanism) gate values as a flat dict."""
    snapshot = {}
    for li, block in enumerate(model.blocks):
        gates = block.attn.gates.detach().cpu().tolist()
        for hi, g in enumerate(gates):
            snapshot[f"L{li}_{MECHANISM_NAMES[hi]}"] = g
    return snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wt103", choices=["ts", "wt103", "code"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--floor-released", action="store_true",
                    help="Use floor-then-release schedule (default = pinned for half).")
    ap.add_argument("--floor-pinned", action="store_true",
                    help="Pin floor for entire training (control).")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    if args.floor_released and args.floor_pinned:
        raise ValueError("pass only one of --floor-released or --floor-pinned")
    if not (args.floor_released or args.floor_pinned):
        args.floor_released = True   # default

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    train_tokens, val_tokens = load_corpus(args.corpus)
    print(f"Corpus: {args.corpus}  train: {len(train_tokens):,}  val: {len(val_tokens):,}")

    cfg = SupernetConfig(
        vocab_size=50257, d_model=288, n_heads=12, n_layers=6,
        d_ff=1152, ctx_len=args.ctx, dropout=0.1,
    )
    print(f"Supernet d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"ctx={cfg.ctx_len}  steps={args.steps}  batch={args.batch_size}")
    print(f"Schedule: {'floor-released' if args.floor_released else 'floor-pinned'}")

    model = SupernetTransformer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

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

    floor_release_step = args.steps // 2
    history = []
    gate_history = []
    t0 = time.time()
    autocast_dtype = torch.bfloat16

    model.train()
    for step in range(args.steps):
        # Determine floor state for this step
        if args.floor_pinned:
            floor_active = True
        else:  # floor_released
            floor_active = step < floor_release_step

        x = sample_batch(train_tokens, args.batch_size, args.ctx, device)
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            logits = model(x, floor_active=floor_active)
            targets = compressed_targets(x, COMPRESSION_RATIO)
            logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % GATE_LOG_EVERY == 0 or step == 0:
            gate_history.append({
                "step": step + 1,
                "floor_active": floor_active,
                "gates": get_gate_snapshot(model),
            })

        if (step + 1) % EVAL_EVERY == 0 or step == 0:
            val_loss, val_ppl = eval_loss(model, val_tokens, args.batch_size,
                                            args.ctx, device, n_batches=10)
            history.append({
                "step": step + 1, "train_loss": float(loss.item()),
                "val_loss": val_loss, "val_ppl": val_ppl,
                "elapsed_s": time.time() - t0,
                "floor_active": floor_active,
            })
            print(f"  step {step+1:5d}/{args.steps}  train_loss={loss.item():.3f}  "
                  f"val_ppl={val_ppl:.2f}  floor={'on' if floor_active else 'off'}  "
                  f"elapsed={time.time()-t0:.0f}s")

    # Final eval (more batches for stability)
    final_loss, final_ppl = eval_loss(model, val_tokens, args.batch_size,
                                        args.ctx, device, n_batches=30)
    anchor_acc, anchor_total = coarsened_anchor_acc(
        model, val_tokens, args.ctx, n_prompts=64, device=device, seed=args.seed,
    )
    final_gates = get_gate_snapshot(model)

    print(f"\nFINAL  val_ppl={final_ppl:.2f}  anchor_acc={anchor_acc:.3f}  "
          f"wall={time.time()-t0:.0f}s")
    print(f"Final gates (released):")
    for li in range(cfg.n_layers):
        for hi, name in enumerate(MECHANISM_NAMES):
            key = f"L{li}_{name}"
            print(f"  {key:>30s}: {final_gates[key]:.3f}")

    ck_path = EXP / f"checkpoints/{args.tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "args": vars(args),
        "final_loss": final_loss, "final_ppl": final_ppl,
        "anchor_acc": anchor_acc, "anchor_total": anchor_total,
        "final_gates": final_gates,
        "history": history, "gate_history": gate_history,
        "wall_s": time.time() - t0, "n_params": n_params,
    }, ck_path)
    summary_path = EXP / f"results/train_{args.tag}.json"
    summary_path.write_text(json.dumps({
        "tag": args.tag, "corpus": args.corpus, "seed": args.seed,
        "ctx": args.ctx, "steps": args.steps,
        "floor_schedule": "released" if args.floor_released else "pinned",
        "floor_release_step": floor_release_step if args.floor_released else None,
        "final_val_loss": final_loss, "final_val_ppl": final_ppl,
        "anchor_acc": anchor_acc, "anchor_total": anchor_total,
        "n_params": n_params, "wall_s": time.time() - t0,
        "history": history, "gate_history": gate_history,
        "final_gates": final_gates,
    }, indent=2))
    print(f"Saved {ck_path}\nSaved {summary_path}")


if __name__ == "__main__":
    main()
