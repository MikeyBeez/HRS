"""Variant-D-specific tests.

Test 3 (adapted): post-hoc mean-pool the per-loop outputs in D's cache,
    then compute pairwise cross-loop cosine vs cross-batch baseline. Does
    D's uncompressed loop structure spontaneously converge to MPAR-like
    geometry?

Test 5 (adapted): permute cache entries at the final cross-attention step.
    Does the order of prior loops matter when they're reached via cross
    attention rather than averaged?
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.hrs_loop.analysis._shared import (
    RESULTS_DIR, load_checkpoint, load_val_loader,
)
from experiments.hrs_loop.analysis.mpar_cosine import (
    pairwise_cos_same_input, pairwise_cos_cross_batch,
)


@torch.no_grad()
def collect_pool_of_cache(model, val_loader, device, n_batches: int):
    """Run Variant D with capture_outputs=True; mean-pool each cache entry
    over sequence axis to get a rank-d vector per loop per example.

    Returns (N, T, d_model).
    """
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    all_m = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            logits, cache = model(x[:, :-1], capture_outputs=True)
        # cache is list of (B, seq, d). Mean-pool over seq -> (B, d).
        pooled = [c.mean(dim=1).to(torch.float32).cpu() for c in cache]
        stacked = torch.stack(pooled, dim=1)          # (B, T, d)
        all_m.append(stacked)
    return torch.cat(all_m, dim=0)


@torch.no_grad()
def run_d_with_permuted_cache(model, x, perm, device, amp_dtype):
    """Replay the final combine step with cache permuted. Need to re-run
    loops up to T-1 to collect cache, then combine with permuted cache."""
    # Run all T loops, collect cache.
    rec = model.recurrent
    cfg = model.cfg
    T = cfg.T_default

    # Manually replicate HRSLoop.forward up to cache collection.
    B, L = x.shape
    pos = torch.arange(L, device=x.device)
    e = model.tok_emb(x) + model.pos_emb(pos)[None]
    for blk in model.prelude:
        e = blk(e)

    cache = []
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                         enabled=(device.type == "cuda")):
        for t in range(T):
            h_t = rec._combine(e, cache)
            block_out = rec.block(h_t)
            lora_idx = t if t < len(rec.loras) else (t % len(rec.loras))
            lora_out = rec.loras[lora_idx](h_t)
            cache.append(block_out + lora_out)

        permuted_cache = [cache[i] for i in perm]
        final = rec._combine(e, permuted_cache)
        h = final
        for blk in model.coda:
            h = blk(h)
        h = model.ln_f(h)
        logits = model.head(h)
    return logits.to(torch.float32)


def kl_per_token(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return float((p.exp() * (p - q)).sum(-1).mean())


def top1_agreement(p_logits, q_logits):
    return float((p_logits.argmax(-1) == q_logits.argmax(-1)).float().mean())


def token_loss(logits, tgt):
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                    tgt.reshape(-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=30)
    ap.add_argument("--out-prefix",
                     default=str(RESULTS_DIR / "variantD"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint("D", device)
    val_loader, _tok = load_val_loader(batch_size=16)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Test 3 adapted: mean-pooled per-loop outputs.
    pooled = collect_pool_of_cache(model, val_loader, device, args.n_batches)
    print(f"\nTest 3 (adapted): pooled-cache shape {tuple(pooled.shape)}")
    same = pairwise_cos_same_input(pooled)
    cross = pairwise_cos_cross_batch(pooled)
    print("\nSame-input cross-loop cosine (D, post-hoc mean-pooled):")
    for r in same:
        print(f"  pool_{r['loop_pair'][0]} ↔ pool_{r['loop_pair'][1]}: "
              f"mean={r['mean_cos']:.3f} std={r['std_cos']:.3f}")
    print("\nCross-batch baseline:")
    for r in cross:
        print(f"  loop {r['loop']}: mean={r['mean_cos']:+.3f} "
              f"std={r['std_cos']:.3f}")

    test3_out = Path(args.out_prefix + "_test3_cosine.json")
    test3_out.write_text(json.dumps({
        "variant": "D",
        "same_input_cross_loop": same,
        "cross_batch_same_loop": cross,
        "n_sequences": pooled.shape[0],
    }, indent=2))
    print(f"\nwrote {test3_out}")

    # Test 5 adapted: permute cache ordering at final combine.
    T = cfg.T_default
    perms = [tuple(range(T)), tuple(reversed(range(T)))]
    rng = random.Random(42)
    while len(perms) < 6:
        p = list(range(T))
        rng.shuffle(p)
        if tuple(p) not in perms:
            perms.append(tuple(p))
    perm_labels = ["identity", "reverse"] + [f"random_{i}" for i in range(4)]

    totals = {}
    def _ensure(k):
        if k not in totals:
            totals[k] = {"kl_sum": 0.0, "top1_sum": 0.0, "loss_sum": 0.0,
                           "n_batches": 0}
    _ensure("canonical")
    for lbl in perm_labels:
        _ensure(f"perm_{lbl}")

    for bi, batch in enumerate(val_loader):
        if bi >= args.n_batches:
            break
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        inp = x[:, :-1]
        tgt = x[:, 1:]

        logits_canon = run_d_with_permuted_cache(
            model, inp, list(range(T)), device, amp_dtype)
        t = totals["canonical"]
        t["loss_sum"] += token_loss(logits_canon, tgt)
        t["top1_sum"] += 1.0
        t["n_batches"] += 1

        for pi, lbl in zip(perms, perm_labels):
            logits = run_d_with_permuted_cache(model, inp, list(pi),
                                                  device, amp_dtype)
            tk = totals[f"perm_{lbl}"]
            tk["kl_sum"] += kl_per_token(logits_canon, logits)
            tk["top1_sum"] += top1_agreement(logits_canon, logits)
            tk["loss_sum"] += token_loss(logits, tgt)
            tk["n_batches"] += 1

        if (bi + 1) % 10 == 0:
            print(f"  Test 5 batch {bi + 1}/{args.n_batches}")

    summary = []
    for mode, t in totals.items():
        n = max(1, t["n_batches"])
        summary.append({
            "mode": mode,
            "mean_kl_nats_per_token": t["kl_sum"] / n,
            "mean_top1_agreement": t["top1_sum"] / n,
            "mean_loss": t["loss_sum"] / n,
            "val_ppl": float(math.exp(t["loss_sum"] / n)),
        })
    test5_out = Path(args.out_prefix + "_test5_permute.json")
    test5_out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {test5_out}")
    hdr = f"{'mode':<25} {'KL':>10} {'top1':>8} {'PPL':>10}"
    print(hdr); print("-" * len(hdr))
    for r in summary:
        print(f"{r['mode']:<25} {r['mean_kl_nats_per_token']:>10.4f} "
              f"{r['mean_top1_agreement']:>8.3f} {r['val_ppl']:>10.3f}")


if __name__ == "__main__":
    main()
