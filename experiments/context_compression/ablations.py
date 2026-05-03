"""Ablations comparing alternative compression schemes against the
tiered baseline.

Schemes tested:
  recent_255_only: Drop older 745 tokens entirely. Use only the last 255
                   of the original 1000. No learned projection. Tells us
                   whether the projections add value over discarding the
                   older context.
  recent_200_only: Drop older 800 tokens entirely. Use only the last 200.
                   Same length-budget as our preserved-recent slice
                   without any compressed-older tokens. Pure baseline:
                   "what if we just truncate?"
  full_uniform_255: Compress the FULL 1000 tokens via a single 255×1000
                    learned linear projection (no asymmetry). Tests
                    whether the temporal asymmetry matters.
  recent_400_only:  Last 400 tokens. Mid-length truncation as a sanity
                    point on the truncation curve.

For each scheme: train its projection (if applicable) on next-token CE,
eval on the 100 held-out passages.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/context_compression"
sys.path.insert(0, str(REPO))

from experiments.context_compression.run import (
    MODEL_NAME, CONTEXT, RECENT, MIDDLE, OLDER,
    RECENT_KEEP, MIDDLE_KEEP, OLDER_KEEP,
    N_TRAIN, N_EVAL, TOTAL_TOKENS,
    avg_pool_init, build_passages,
)


def get_logits_from_embeds(model, inputs_embeds):
    out = model(inputs_embeds=inputs_embeds, return_dict=True)
    return out.logits[:, -1, :]


def get_logits_from_ids(model, ids):
    out = model(ids, return_dict=True)
    return out.logits[:, -1, :]


@torch.no_grad()
def eval_scheme(model, label, get_logits_fn, eval_passages, base_logits_cache):
    """get_logits_fn(ids) -> (1, V) logits for the next token."""
    records = []
    for i in range(eval_passages.shape[0]):
        ids = eval_passages[i:i+1]
        ctx = ids[:, :CONTEXT]
        target = ids[:, CONTEXT]
        comp_logits = get_logits_fn(ctx)
        base_logits = base_logits_cache[i:i+1]
        base_logp = F.log_softmax(base_logits, dim=-1)
        comp_logp = F.log_softmax(comp_logits, dim=-1)
        base_p = base_logp.exp()
        ce_base = -base_logp.gather(-1, target.unsqueeze(-1)).squeeze().item()
        ce_comp = -comp_logp.gather(-1, target.unsqueeze(-1)).squeeze().item()
        kl = (base_p * (base_logp - comp_logp)).sum(-1).item()
        top1 = (base_logits.argmax(-1) == comp_logits.argmax(-1)).item()
        records.append({"i": i, "ce_base": ce_base, "ce_comp": ce_comp,
                        "kl": kl, "top1_match": top1})
    return {
        "label": label,
        "n_eval": len(records),
        "mean_ce_base": float(np.mean([r["ce_base"] for r in records])),
        "mean_ce_comp": float(np.mean([r["ce_comp"] for r in records])),
        "mean_ce_gap": float(np.mean([r["ce_comp"] - r["ce_base"] for r in records])),
        "mean_kl": float(np.mean([r["kl"] for r in records])),
        "top1_match_rate": float(np.mean([r["top1_match"] for r in records])),
    }


def train_uniform_projection(model, train_passages, in_len, out_len,
                              steps=2000, batch=4, lr=1e-3, seed=0):
    """Train a single (out_len, in_len) projection over the FULL context."""
    device = train_passages.device
    rng = np.random.default_rng(seed)
    W = nn.Parameter(avg_pool_init(in_len, out_len).to(device))
    opt = torch.optim.Adam([W], lr=lr)
    wte = model.transformer.wte
    for step in range(steps):
        idx = rng.integers(0, train_passages.shape[0], size=batch)
        batch_ids = train_passages[idx]
        ctx = batch_ids[:, :in_len]
        target = batch_ids[:, CONTEXT]
        emb = wte(ctx)  # (B, in_len, D)
        comp = torch.einsum("oi,bid->bod", W, emb)  # (B, out_len, D)
        out = model(inputs_embeds=comp, return_dict=True)
        logits = out.logits[:, -1, :]
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return W


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(f"Loading {MODEL_NAME} (frozen) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load passages
    cache = EXP / "data/passages.npy"
    passages = np.load(cache)
    train_passages = torch.tensor(passages[:N_TRAIN], dtype=torch.long, device=device)
    eval_passages = torch.tensor(passages[N_TRAIN:N_TRAIN + N_EVAL],
                                  dtype=torch.long, device=device)

    # Pre-compute baseline (full-context) logits for all eval passages
    print("Pre-computing baseline (full-context) eval logits ...")
    base_logits_cache = torch.zeros(N_EVAL, model.config.vocab_size,
                                      device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(N_EVAL):
            ctx = eval_passages[i:i+1, :CONTEXT]
            out = model(ctx, return_dict=True)
            base_logits_cache[i] = out.logits[0, -1, :].detach().clone()
            del out
    torch.cuda.empty_cache()
    print(f"  baseline cached: {base_logits_cache.shape}")

    wte = model.transformer.wte

    results = []
    t_total = time.time()

    # Scheme 1: recent_200_only (just truncate to last 200 tokens)
    def fn_recent_200(ids):
        ctx = ids[:, -200:]
        return get_logits_from_ids(model, ctx)
    print("\nScheme: recent_200_only (truncate last 200, no projection)")
    r = eval_scheme(model, "recent_200_only", fn_recent_200,
                     eval_passages, base_logits_cache)
    print(f"  CE_base={r['mean_ce_base']:.3f} CE_comp={r['mean_ce_comp']:.3f} "
          f"gap={r['mean_ce_gap']:+.3f} KL={r['mean_kl']:.3f} "
          f"top1={r['top1_match_rate']:.3f}")
    results.append(r)

    # Scheme 2: recent_255_only (just truncate to last 255 tokens — same
    # length budget as our tiered compressed)
    def fn_recent_255(ids):
        ctx = ids[:, -255:]
        return get_logits_from_ids(model, ctx)
    print("\nScheme: recent_255_only (truncate last 255, no projection)")
    r = eval_scheme(model, "recent_255_only", fn_recent_255,
                     eval_passages, base_logits_cache)
    print(f"  CE_base={r['mean_ce_base']:.3f} CE_comp={r['mean_ce_comp']:.3f} "
          f"gap={r['mean_ce_gap']:+.3f} KL={r['mean_kl']:.3f} "
          f"top1={r['top1_match_rate']:.3f}")
    results.append(r)

    # Scheme 3: recent_400_only
    def fn_recent_400(ids):
        ctx = ids[:, -400:]
        return get_logits_from_ids(model, ctx)
    print("\nScheme: recent_400_only (truncate last 400)")
    r = eval_scheme(model, "recent_400_only", fn_recent_400,
                     eval_passages, base_logits_cache)
    print(f"  CE_base={r['mean_ce_base']:.3f} CE_comp={r['mean_ce_comp']:.3f} "
          f"gap={r['mean_ce_gap']:+.3f} KL={r['mean_kl']:.3f} "
          f"top1={r['top1_match_rate']:.3f}")
    results.append(r)

    # Scheme 4: full_uniform_255 (single linear over 1000 -> 255)
    print("\nScheme: full_uniform_255 (linear 1000 -> 255 over full context, "
          "trained 2000 steps)")
    t0 = time.time()
    W_uniform = train_uniform_projection(model, train_passages,
                                          in_len=1000, out_len=255,
                                          steps=2000, batch=4, lr=1e-3, seed=0)
    print(f"  W trained in {time.time()-t0:.0f}s")
    def fn_uniform(ids):
        emb = wte(ids[:, :CONTEXT])
        comp = torch.einsum("oi,bid->bod", W_uniform, emb)
        return get_logits_from_embeds(model, comp)
    r = eval_scheme(model, "full_uniform_255", fn_uniform,
                     eval_passages, base_logits_cache)
    print(f"  CE_base={r['mean_ce_base']:.3f} CE_comp={r['mean_ce_comp']:.3f} "
          f"gap={r['mean_ce_gap']:+.3f} KL={r['mean_kl']:.3f} "
          f"top1={r['top1_match_rate']:.3f}")
    results.append(r)

    out_path = EXP / "results/ablations.json"
    out_path.write_text(json.dumps({
        "schemes": results,
        "wall_total_s": time.time() - t_total,
    }, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
