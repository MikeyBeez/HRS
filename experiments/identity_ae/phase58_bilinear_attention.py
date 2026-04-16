"""Phase 58: Bilinear vs standard attention at full scale.

Tests whether the full d² Cartesian product of feature interactions
(q^T W k) produces measurably better V-space organization than the
model's standard attention scoring at d=1024.

Two models trained from scratch:
  A: V22 baseline (PerHeadBonsignore exponential kernel)
  B: Bilinear variant (q^T W k with W initialized near identity)

The ONLY difference is the attention score computation. Everything else
(architecture, data, hyperparameters, seed) is identical.

10K steps diagnostic run. If V-space metrics diverge, continue to
convergence. If they don't, they won't.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase58_bilinear_attention.py
"""

import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, build_dataloaders

from experiments.identity_ae.phase22_engram_key import hidden_at_layer
from experiments.identity_ae.phase56_vspace_recon_pretraining import (
    measure_val_ppl, measure_kv_alignment, measure_info_recovery,
)
from train import (
    set_seed, get_lr, get_phase, get_phase_lr_multipliers,
    V1_PARAM_GROUP_INDEX,
)

D = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_STEPS = 43000
LOG_INTERVAL = 100
EVAL_INTERVAL = 1000
MEASURE_CHECKPOINTS = [2000, 5000, 10000, 20000, 43000]


# ================================================================
# Bilinear attention patch
# ================================================================

class BilinearWeights(nn.Module):
    """Learned bilinear matrices W per head, initialized near identity."""
    def __init__(self, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        # Initialize as identity + small noise
        W = torch.eye(head_dim).unsqueeze(0).repeat(n_heads, 1, 1)
        W = W + 0.01 * torch.randn(n_heads, head_dim, head_dim)
        self.W = nn.Parameter(W)  # (H, D_h, D_h)


def patch_bilinear(model, bilinear_weights):
    """Replace the attention score computation in each block with
    bilinear form: score = (q @ W) @ k^T / sqrt(d) instead of the
    standard exponential kernel."""

    for i, block in enumerate(model.blocks):
        attn = block.attn
        original_forward = attn.forward
        bw = bilinear_weights  # shared across layers (or per-layer)

        def make_bilinear_forward(orig_fwd, attn_mod, bw_mod):
            def bilinear_forward(x, return_weights=False, focus_qk=None,
                                 kv_cache=None, start_pos=0):
                B, T, C = x.shape
                n_h = attn_mod.n_heads
                hd = attn_mod.head_dim

                qkv = attn_mod.qkv(x).reshape(B, T, 3, n_h, hd)
                q, k, v = qkv.unbind(dim=2)
                q = q.transpose(1, 2)  # (B, H, T, D_h)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                # RoPE
                cos, sin = attn_mod.rope(start_pos + T)
                cos = cos[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
                sin = sin[start_pos:start_pos + T].unsqueeze(0).unsqueeze(0)
                from model import rotate_half
                q = q * cos + rotate_half(q) * sin
                k = k * cos + rotate_half(k) * sin

                # KV cache
                new_kv_cache = None
                if kv_cache is not None:
                    cached_k, cached_v = kv_cache
                    k = torch.cat([cached_k, k], dim=2)
                    v = torch.cat([cached_v, v], dim=2)
                new_kv_cache = (k, v)

                S = k.shape[2]

                # BILINEAR SCORES: q^T W k / sqrt(d)
                # q: (B, H, T, D_h), W: (H, D_h, D_h), k: (B, H, S, D_h)
                # qW = einsum('bhti,hij->bhtj', q, W)
                qW = torch.einsum('bhti,hij->bhtj', q, bw_mod.W)
                scores = torch.matmul(qW, k.transpose(-2, -1))  # (B, H, T, S)
                scores = scores / math.sqrt(hd)

                # Causal mask
                causal_mask = torch.triu(
                    torch.ones(T, S, device=x.device, dtype=torch.bool),
                    diagonal=S - T + 1)
                scores = scores.masked_fill(causal_mask, float('-inf'))
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = attn_mod.attn_dropout(attn_weights)
                out = attn_weights @ v

                attn_w_out = attn_weights if return_weights else None

                # Keep per-head output scaling from Bonsignore
                head_scales = F.softplus(
                    attn_mod.head_output_scalars).view(1, n_h, 1, 1)
                out = out * head_scales

                out = out.transpose(1, 2).reshape(B, T, C)
                out = attn_mod.resid_dropout(attn_mod.out_proj(out))
                return out, attn_w_out, new_kv_cache
            return bilinear_forward

        attn.forward = make_bilinear_forward(original_forward, attn, bw)


# ================================================================
# Train one model
# ================================================================

def train_model(model, loaders, cfg, device, max_steps, label,
                extra_params=None, val_ds=None):
    """Train a model for max_steps, measuring V-space at checkpoints."""

    param_groups_dict = model.get_param_groups()
    optimizer_groups = []
    group_names = []
    for name, params in param_groups_dict.items():
        if params:
            optimizer_groups.append({
                "params": params,
                "lr": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
            })
            group_names.append(name)
    if extra_params:
        optimizer_groups.append({
            "params": extra_params,
            "lr": cfg.training.learning_rate,
            "weight_decay": 0.0,
        })
        group_names.append("bilinear_W")

    optimizer = torch.optim.AdamW(
        optimizer_groups, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay, betas=(0.9, 0.95))

    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    train_iter = iter(loaders["train"])
    step = 0
    micro_step = 0
    current_phase = get_phase(step, cfg)
    accum_loss = 0.0
    t0 = time.time()
    checkpoint_results = {}

    optimizer.zero_grad()
    model.train()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        x, y = batch[0].to(device), batch[1].to(device)
        B, T = x.shape

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model(x, step=step)
            logits = out.logits
            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                y[:, :-1].reshape(-1)) / GRAD_ACCUM

        loss.backward()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg.training.max_grad_norm)
            if extra_params:
                torch.nn.utils.clip_grad_norm_(extra_params, 1.0)

            new_phase = get_phase(step, cfg)
            if new_phase != current_phase:
                current_phase = new_phase

            base_lr = get_lr(step, cfg.training.warmup_steps,
                             max_steps, cfg.training.learning_rate)
            lr_mults = get_phase_lr_multipliers(current_phase, cfg)
            for i, (pg, name) in enumerate(zip(optimizer.param_groups,
                                                group_names)):
                if name == "bilinear_W":
                    pg["lr"] = base_lr
                else:
                    mult_idx = V1_PARAM_GROUP_INDEX.get(name, 0)
                    pg["lr"] = base_lr * lr_mults[mult_idx]

            optimizer.step()
            optimizer.zero_grad()
            step += 1
            accum_loss += loss.item() * GRAD_ACCUM
        else:
            continue

        if step % LOG_INTERVAL == 0:
            avg = accum_loss / LOG_INTERVAL
            ppl = math.exp(min(avg, 20))
            elapsed = time.time() - t0
            print(f"  [{label}] step {step:5d}  loss {avg:.4f}  "
                  f"ppl {ppl:.1f}  lr {base_lr:.2e}  ({elapsed:.0f}s)")
            accum_loss = 0.0

        if step in MEASURE_CHECKPOINTS:
            print(f"\n  [{label}] --- Measurement at step {step} ---")
            model.eval()

            ppl = measure_val_ppl(model, loaders["validation"], device)
            kv = measure_kv_alignment(model, val_ds, device, n_passages=10)
            info = measure_info_recovery(model, val_ds, device, n_passages=50)

            print(f"    val PPL: {ppl:.2f}")
            for l in sorted(kv.keys()):
                print(f"      L{l}: K={kv[l]['k_cos']:.4f}  "
                      f"V={kv[l]['v_cos']:.4f}")
            print(f"    info recovery: mean={info['mean_recovery']:.1%}")

            checkpoint_results[step] = {
                "ppl": ppl,
                "kv": {str(l): v for l, v in kv.items()},
                "info": info,
            }

            model.train()
            print()

    return checkpoint_results


# ================================================================
# Main
# ================================================================

def main():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results/identity_ae/phase58")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 58: Bilinear vs standard attention")
    print(f"Device: {device}")
    print(f"Steps: {MAX_STEPS}\n")

    # Data (load once, share between models)
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, BATCH_SIZE)
    val_ds = splits["validation"]

    # ============================================================
    # Model A: Standard (Bonsignore exponential kernel)
    # ============================================================
    print("=" * 60)
    print("MODEL A: Standard Bonsignore attention")
    print("=" * 60)

    set_seed(cfg.seed)
    model_a = HRSTransformer(cfg).to(device)
    print(f"  params: {sum(p.numel() for p in model_a.parameters()):,}")

    results_a = train_model(model_a, loaders, cfg, device, MAX_STEPS,
                            "standard", val_ds=val_ds)
    del model_a
    torch.cuda.empty_cache()

    # ============================================================
    # Model B: Bilinear attention
    # ============================================================
    print("=" * 60)
    print("MODEL B: Bilinear attention")
    print("=" * 60)

    set_seed(cfg.seed)
    model_b = HRSTransformer(cfg).to(device)
    n_heads = model_b.blocks[0].attn.n_heads
    head_dim = model_b.blocks[0].attn.head_dim
    bilinear_w = BilinearWeights(n_heads, head_dim).to(device)
    extra_params_count = sum(p.numel() for p in bilinear_w.parameters())
    print(f"  base params: {sum(p.numel() for p in model_b.parameters()):,}")
    print(f"  bilinear W params: {extra_params_count:,}")

    patch_bilinear(model_b, bilinear_w)

    results_b = train_model(model_b, loaders, cfg, device, MAX_STEPS,
                            "bilinear",
                            extra_params=list(bilinear_w.parameters()),
                            val_ds=val_ds)

    # Analyze bilinear weights
    print("\n  Bilinear weight analysis:")
    W = bilinear_w.W.detach().cpu()
    I = torch.eye(head_dim).unsqueeze(0).repeat(n_heads, 1, 1)
    diff = W - I
    for h in range(n_heads):
        frob = diff[h].norm().item()
        off_diag = diff[h].clone()
        off_diag.fill_diagonal_(0)
        off_diag_norm = off_diag.norm().item()
        diag_norm = diff[h].diag().norm().item()
        U, S, Vh = torch.linalg.svd(W[h])
        top1_share = (S[0] ** 2 / (S ** 2).sum()).item()
        eff_rank = (S / S.max()).gt(0.01).sum().item()
        print(f"    head {h}: ||W-I||={frob:.3f}  "
              f"off_diag={off_diag_norm:.3f}  diag={diag_norm:.3f}  "
              f"σ₁%={top1_share:.1%}  eff_rank={eff_rank}")

    del model_b, bilinear_w
    torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 58 SUMMARY: Bilinear vs standard attention")
    print("=" * 72)

    print(f"\n  {'step':>6}  {'std PPL':>8}  {'bil PPL':>8}  "
          f"{'std V-cos':>10}  {'bil V-cos':>10}  "
          f"{'std rec':>8}  {'bil rec':>8}")
    for s in sorted(set(list(results_a.keys()) + list(results_b.keys()))):
        ra = results_a.get(s, {})
        rb = results_b.get(s, {})
        a_ppl = ra.get("ppl", 0)
        b_ppl = rb.get("ppl", 0)
        a_vc = ra.get("kv", {}).get("5", {}).get("v_cos", 0)
        b_vc = rb.get("kv", {}).get("5", {}).get("v_cos", 0)
        a_rec = ra.get("info", {}).get("mean_recovery", 0)
        b_rec = rb.get("info", {}).get("mean_recovery", 0)
        print(f"  {s:>6}  {a_ppl:>8.2f}  {b_ppl:>8.2f}  "
              f"{a_vc:>10.4f}  {b_vc:>10.4f}  "
              f"{a_rec:>7.1%}  {b_rec:>7.1%}")

    out = {
        "max_steps": MAX_STEPS,
        "standard": {str(k): v for k, v in results_a.items()},
        "bilinear": {str(k): v for k, v in results_b.items()},
    }
    with open(results_dir / "bilinear_attention.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
