"""Extended evaluation of trained compression vs baseline models:

  1. Validation perplexity at the training context length.
  2. Validation perplexity at extended context lengths (ctx grid).
  3. Inference time per forward pass at multiple context lengths.

Loads checkpoints from the train.py outputs.
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
from experiments.compression_conv.train import (
    get_batch, compressed_targets,
    baseline_logits_at_targets, compressed_logits_at_targets,
    COMPRESSION_RATIO,
)


@torch.no_grad()
def eval_at_ctx(model, data, ctx, batch_size, device, n_batches=30):
    """Eval val PPL at a custom context length (which may differ from
    the model's training ctx). Position embeddings beyond training ctx
    are zero — model wasn't trained to use them — so this is a clean
    extrapolation test, not a rigorous long-context measurement."""
    model.eval()
    losses = []
    g = torch.Generator(); g.manual_seed(456)
    cr = model.compression_ratio
    for _ in range(n_batches):
        x = get_batch(data, batch_size, ctx, device, generator=g)
        logits = model(x)
        targets = compressed_targets(x, COMPRESSION_RATIO)
        if cr == 1:
            logits = baseline_logits_at_targets(logits, COMPRESSION_RATIO)
        else:
            logits = compressed_logits_at_targets(logits, cr)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


@torch.no_grad()
def time_forward(model, ctx, device, n_warmup=3, n_runs=10):
    """Average forward-pass wall time at a given context length."""
    model.eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, ctx), device=device)
    # Resize positional embedding if needed (eval extrapolation only)
    if ctx > model.cfg.ctx_len:
        # Pad pos_emb with zeros for the new positions
        old_pos = model.pos_emb
        if old_pos.size(1) < ctx:
            pad = torch.zeros(1, ctx - old_pos.size(1), old_pos.size(2),
                                device=old_pos.device, dtype=old_pos.dtype)
            new_pos = torch.cat([old_pos, pad], dim=1)
            model.pos_emb = torch.nn.Parameter(new_pos, requires_grad=False)
    for _ in range(n_warmup):
        _ = model(x)
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), min(times)


def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    # Filter to known fields
    known = {f for f in CompressedConfig.__dataclass_fields__}
    cfg = CompressedConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = CompressedTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ck", required=True)
    ap.add_argument("--compressed-ck", required=True)
    ap.add_argument("--out", default=str(EXP / "results/extended_eval.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    val_data = torch.load(REPO / "experiments/router_lora_phased/data/shakespeare_val.pt",
                            weights_only=False)
    print(f"Val tokens: {len(val_data):,}")

    baseline, _ = load_model(args.baseline_ck, device)
    compressed, _ = load_model(args.compressed_ck, device)

    print(f"\nBaseline:    n_params={sum(p.numel() for p in baseline.parameters()):,}  "
          f"ctx_train={baseline.cfg.ctx_len}")
    print(f"Compressed:  n_params={sum(p.numel() for p in compressed.parameters()):,}  "
          f"ratio={compressed.compression_ratio}x  ctx_train={compressed.cfg.ctx_len}")

    # ---- 1. PPL at training context length ----
    print(f"\n{'='*72}\nPPL @ training ctx={baseline.cfg.ctx_len}\n{'='*72}")
    base_loss, base_ppl = eval_at_ctx(baseline, val_data, baseline.cfg.ctx_len,
                                        batch_size=16, device=device, n_batches=50)
    comp_loss, comp_ppl = eval_at_ctx(compressed, val_data, compressed.cfg.ctx_len,
                                        batch_size=16, device=device, n_batches=50)
    print(f"  baseline:    ppl = {base_ppl:7.2f}  (loss {base_loss:.3f})")
    print(f"  compressed:  ppl = {comp_ppl:7.2f}  (loss {comp_loss:.3f})")
    print(f"  delta:       compressed/baseline = {comp_ppl/base_ppl:.3f}x")

    # ---- 2. PPL at varying context lengths ----
    # For TS, max meaningful ctx is bounded by val set size. Use ctx grid
    # within range of training ctx and a bit beyond.
    train_ctx = baseline.cfg.ctx_len
    # PPL with pos_emb learned only up to train_ctx; ctx > train_ctx would
    # extrapolate with zero-padded pos_emb and show pos_emb noise rather
    # than compression behavior. Restrict PPL eval to ctx <= train_ctx.
    ctx_grid = sorted({train_ctx // 2, train_ctx, train_ctx // 4})
    ctx_grid = [c for c in ctx_grid
                if c <= train_ctx
                and c <= len(val_data) - 1
                and c % COMPRESSION_RATIO == 0]
    print(f"\n{'='*72}\nPPL across ctx lengths {ctx_grid}\n{'='*72}")
    ctx_results = []
    for ctx in ctx_grid:
        # Note: extrapolation beyond train ctx uses zero-padded pos_emb (rough)
        b_loss, b_ppl = eval_at_ctx(baseline, val_data, ctx, batch_size=8,
                                      device=device, n_batches=20)
        c_loss, c_ppl = eval_at_ctx(compressed, val_data, ctx, batch_size=8,
                                      device=device, n_batches=20)
        marker = "" if ctx <= train_ctx else "(extrapolation)"
        print(f"  ctx={ctx:5d} {marker:<18s}  baseline ppl={b_ppl:7.2f}   "
              f"compressed ppl={c_ppl:7.2f}   ratio={c_ppl/b_ppl:.3f}")
        ctx_results.append({
            "ctx": ctx, "baseline_ppl": b_ppl, "compressed_ppl": c_ppl,
            "is_extrapolation": ctx > train_ctx,
        })

    # ---- 3. Inference timing ----
    print(f"\n{'='*72}\nInference time at varying ctx (batch 1)\n{'='*72}")
    timing_grid = [256, 512, 1024, 2048, 4096]
    timing_results = []
    for ctx in timing_grid:
        if ctx % COMPRESSION_RATIO != 0:
            continue
        try:
            b_mean, b_min = time_forward(baseline, ctx, device)
            c_mean, c_min = time_forward(compressed, ctx, device)
            speedup = b_mean / c_mean
            print(f"  ctx={ctx:5d}  baseline {b_mean*1000:7.2f}ms  "
                  f"compressed {c_mean*1000:7.2f}ms  speedup={speedup:.2f}x")
            timing_results.append({
                "ctx": ctx,
                "baseline_ms_mean": b_mean * 1000,
                "compressed_ms_mean": c_mean * 1000,
                "baseline_ms_min": b_min * 1000,
                "compressed_ms_min": c_min * 1000,
                "speedup": speedup,
            })
        except Exception as e:
            print(f"  ctx={ctx}: FAIL ({e})")

    # ---- Save ----
    summary = {
        "training_ppl": {
            "baseline": base_ppl, "compressed": comp_ppl,
            "ratio": comp_ppl / base_ppl,
        },
        "ctx_scaling": ctx_results,
        "inference_timing": timing_results,
        "baseline_ckpt": args.baseline_ck,
        "compressed_ckpt": args.compressed_ck,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
