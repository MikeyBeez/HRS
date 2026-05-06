"""Three-way comparison: baseline transformer vs compression+attention vs
compression+Mamba on WT103.

Reuses the trained checkpoints from compression_conv (baseline + compressed
attention). Adds the just-trained compression+mamba checkpoint.

Reports:
  1. Validation PPL at training ctx (50 batches each)
  2. Coarsened-anchor accuracy (the same 8 prompts × 8 cycles protocol from
     compression_conv/generation_compare.py).
  3. Inference timing at multiple context lengths.
  4. Top-5 distribution comparison on a few prompts (qualitative).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_mamba"
sys.path.insert(0, str(REPO))

from experiments.compression_conv.model import (
    CompressedConfig, CompressedTransformer,
)
from experiments.compression_conv.train import (
    compressed_targets, baseline_logits_at_targets,
    compressed_logits_at_targets, COMPRESSION_RATIO,
)
from experiments.compression_mamba.model import (
    CompMambaConfig, CompressedMambaTransformer,
)


def load_baseline_or_attn(ckpath, device):
    ck = torch.load(ckpath, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    known = {f for f in CompressedConfig.__dataclass_fields__}
    cfg = CompressedConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = CompressedTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


def load_mamba(ckpath, device):
    ck = torch.load(ckpath, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    known = {f for f in CompMambaConfig.__dataclass_fields__}
    cfg = CompMambaConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = CompressedMambaTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def eval_ppl(model, val_tokens, batch_size, ctx, device, n_batches=50,
              cr=COMPRESSION_RATIO):
    """Same matched-task eval as compression_conv: predict every CR-th token."""
    model.eval()
    losses = []
    g = torch.Generator(); g.manual_seed(123)
    n = val_tokens.shape[0]
    for _ in range(n_batches):
        starts = torch.randint(0, n - ctx - 1, (batch_size,), generator=g)
        x = torch.stack([val_tokens[s:s+ctx] for s in starts]).to(device)
        logits = model(x)
        targets = compressed_targets(x, cr)
        if model.compression_ratio == 1:
            logits = baseline_logits_at_targets(logits, cr)
        else:
            logits = compressed_logits_at_targets(logits, model.compression_ratio)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        )
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


@torch.no_grad()
def time_forward(model, ctx, device, n_warmup=2, n_runs=5):
    """Average forward-pass wall time at given ctx (batch 1)."""
    model.eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, ctx), device=device)
    # Resize positional embedding if ctx > training ctx
    if ctx > model.cfg.ctx_len:
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
    return sum(times) / len(times)


@torch.no_grad()
def coarsened_anchor_acc(model, val_tokens, ctx, n_prompts, n_cycles, device):
    """Coarsened-anchor accuracy: same protocol as
    compression_conv/generation_compare.py."""
    initial_prefix_len = ctx - n_cycles * COMPRESSION_RATIO
    rng = random.Random(0)
    starts = sorted(rng.sample(range(len(val_tokens) - ctx), n_prompts))
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
    return hits, total, hits / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ck",
                    default=str(REPO / "experiments/compression_conv/checkpoints/baseline_wt103_ctx2048.pt"))
    ap.add_argument("--attn-ck",
                    default=str(REPO / "experiments/compression_conv/checkpoints/compressed_wt103_ctx2048.pt"))
    ap.add_argument("--mamba-ck",
                    default=str(EXP / "checkpoints/mamba_wt103_ctx2048.pt"))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", default=str(EXP / "results/three_way_compare.json"))
    args = ap.parse_args()

    device = torch.device("cuda")
    c = torch.load(REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt",
                     weights_only=False)
    val_tokens = c["splits"]["validation"].tokens
    print(f"Val tokens: {len(val_tokens):,}")

    print("\nLoading models...")
    baseline, _ = load_baseline_or_attn(args.baseline_ck, device)
    attn, _ = load_baseline_or_attn(args.attn_ck, device)
    mamba, _ = load_mamba(args.mamba_ck, device)
    print(f"  baseline:           {sum(p.numel() for p in baseline.parameters()):,} params")
    print(f"  comp+attention:     {sum(p.numel() for p in attn.parameters()):,} params")
    print(f"  comp+mamba:         {sum(p.numel() for p in mamba.parameters()):,} params")

    # ---- 1. PPL ----
    print(f"\n{'='*72}\nValidation PPL @ ctx {args.ctx}\n{'='*72}")
    b_loss, b_ppl = eval_ppl(baseline, val_tokens, batch_size=8, ctx=args.ctx,
                                device=device, n_batches=50)
    a_loss, a_ppl = eval_ppl(attn, val_tokens, batch_size=8, ctx=args.ctx,
                                device=device, n_batches=50)
    m_loss, m_ppl = eval_ppl(mamba, val_tokens, batch_size=8, ctx=args.ctx,
                                device=device, n_batches=50)
    print(f"  baseline:        ppl = {b_ppl:7.2f}")
    print(f"  comp+attention:  ppl = {a_ppl:7.2f}  (vs baseline: {a_ppl/b_ppl:.3f}x)")
    print(f"  comp+mamba:      ppl = {m_ppl:7.2f}  (vs baseline: {m_ppl/b_ppl:.3f}x;  "
          f"vs comp+attn: {m_ppl/a_ppl:.3f}x)")

    # ---- 2. Coarsened-anchor accuracy ----
    print(f"\n{'='*72}\nCoarsened-anchor accuracy "
          f"(8 prompts × 8 cycles = 64 predictions each)\n{'='*72}")
    bh, btot, bacc = coarsened_anchor_acc(baseline, val_tokens, args.ctx, 8, 8, device)
    ah, atot, aacc = coarsened_anchor_acc(attn, val_tokens, args.ctx, 8, 8, device)
    mh, mtot, macc = coarsened_anchor_acc(mamba, val_tokens, args.ctx, 8, 8, device)
    print(f"  baseline:        {bh}/{btot} = {bacc:.3f}")
    print(f"  comp+attention:  {ah}/{atot} = {aacc:.3f}")
    print(f"  comp+mamba:      {mh}/{mtot} = {macc:.3f}")

    # ---- 3. Inference timing ----
    print(f"\n{'='*72}\nInference time at varying ctx (batch 1)\n{'='*72}")
    print(f"  {'ctx':>5s}  {'baseline':>10s}  {'comp+attn':>10s}  {'comp+mamba':>11s}  "
          f"{'attn vs base':>13s}  {'mamba vs base':>14s}  {'mamba vs attn':>14s}")
    timing_results = []
    for ctx in [256, 512, 1024, 2048, 4096]:
        if ctx % COMPRESSION_RATIO != 0:
            continue
        b_t = time_forward(baseline, ctx, device)
        a_t = time_forward(attn, ctx, device)
        m_t = time_forward(mamba, ctx, device)
        print(f"  {ctx:5d}  {b_t*1000:8.2f}ms  {a_t*1000:8.2f}ms  "
              f"{m_t*1000:9.2f}ms  "
              f"{b_t/a_t:11.2f}x  {b_t/m_t:12.2f}x  {a_t/m_t:12.2f}x")
        timing_results.append({
            "ctx": ctx,
            "baseline_ms": b_t * 1000, "attn_ms": a_t * 1000,
            "mamba_ms": m_t * 1000,
            "speedup_attn_vs_base": b_t / a_t,
            "speedup_mamba_vs_base": b_t / m_t,
            "speedup_mamba_vs_attn": a_t / m_t,
        })

    # Save
    out = {
        "ppl": {
            "baseline": b_ppl, "comp_attn": a_ppl, "comp_mamba": m_ppl,
            "comp_attn_vs_baseline": a_ppl / b_ppl,
            "comp_mamba_vs_baseline": m_ppl / b_ppl,
            "comp_mamba_vs_comp_attn": m_ppl / a_ppl,
        },
        "anchor_acc": {
            "baseline": bacc, "comp_attn": aacc, "comp_mamba": macc,
            "baseline_hits": bh, "comp_attn_hits": ah, "comp_mamba_hits": mh,
            "total": btot,
        },
        "timing": timing_results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
