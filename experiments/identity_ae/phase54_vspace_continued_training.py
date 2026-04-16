"""Phase 54: Saliency-weighted continued training for V-space optimization.

Continue training the base model with saliency-weighted NTP loss to
reorganize V-space around informative tokens. The attention pooling
encoder from Phase 51 provides per-token saliency scores. The model's
full parameters are unfrozen at 1/10th of the pre-training LR.

Stages:
  54a: Saliency-weighted continued training (2K, 5K, 10K step checkpoints)
  54b: Measurement battery at each checkpoint (V-space alignment,
       information recovery, SVD spectrum, passkey retrieval)

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase54_vspace_continued_training.py
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
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, build_dataloaders
from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase32_kv_similarity import (
    install_qkv_hooks, remove_hooks, per_head_cosine,
)
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x
from experiments.identity_ae.phase51_learned_engram import AttentionPoolEncoder

D = 1024
LAYER = 5
BASE_LR = 3e-4   # original pre-training LR
CONT_LR = BASE_LR / 10   # continued training LR
CHECKPOINTS = [2000, 5000, 10000]
LOG_INTERVAL = 200
EVAL_INTERVAL = 500
MAX_CTX_POS = 512
N_EVAL_PASSAGES = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200


# ================================================================
# Saliency computation
# ================================================================

@torch.no_grad()
def compute_batch_saliency(encoder, model, ids_t, device):
    """Compute per-token saliency for a batch. ids_t: (B, T).
    Returns: (B, T) saliency weights."""
    H = hidden_at_layer(model, ids_t, LAYER)  # (B, T, D)
    B, T, _ = H.shape
    q = encoder.query.expand(B, -1, -1)        # (B, 1, D)
    k = encoder.k_proj(H)                       # (B, T, D)
    attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)  # (B, 1, T)
    return F.softmax(attn.squeeze(1), dim=-1)   # (B, T)


# ================================================================
# Measurement functions
# ================================================================

@torch.no_grad()
def measure_kv_alignment(model, tokenizer, device, val_ds, n_passages=10):
    """Phase 32 protocol: K/V cosine between engram and passage centroids."""
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:n_passages].tolist()

    n_layers = len(model.blocks)
    d_model = model.blocks[0].attn.n_heads * model.blocks[0].attn.head_dim
    results = {l: {"cos_k": [], "cos_v": []} for l in range(n_layers)}

    for idx in indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:128]
        ids = ids.unsqueeze(0).to(device)

        # Full passage K/V
        passage_store = {}
        handles = install_qkv_hooks(model, passage_store)
        _ = model(ids, step=0)
        remove_hooks(handles)

        # Engram
        h_L = hidden_at_layer(model, ids, LAYER)
        engram = h_L.mean(dim=1)  # (1, D)

        # Engram K/V
        engram_store = {}
        handles = install_qkv_hooks(model, engram_store)
        x_eng = engram.view(1, 1, d_model)
        _ = forward_from_x(model, x_eng)
        remove_hooks(handles)

        for l in range(n_layers):
            if l not in passage_store or l not in engram_store:
                continue
            pk = passage_store[l]["k"].squeeze(0).mean(dim=0)  # (H, Dh)
            pv = passage_store[l]["v"].squeeze(0).mean(dim=0)
            ek = engram_store[l]["k"].squeeze(0).squeeze(0)    # (H, Dh)
            ev = engram_store[l]["v"].squeeze(0).squeeze(0)
            results[l]["cos_k"].append(per_head_cosine(pk, ek))
            results[l]["cos_v"].append(per_head_cosine(pv, ev))

    summary = {}
    for l in range(n_layers):
        if results[l]["cos_k"]:
            summary[l] = {
                "k_cos": sum(results[l]["cos_k"]) / len(results[l]["cos_k"]),
                "v_cos": sum(results[l]["cos_v"]) / len(results[l]["cos_v"]),
            }
    return summary


@torch.no_grad()
def measure_info_recovery(model, encoder, device, val_ds, n_passages=50):
    """Phase 33 protocol: information recovery with mean and attn pooling."""
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:n_passages].tolist()

    nll_no = []
    nll_full = []
    nll_mean = []
    nll_attn = []

    for idx in indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        ctx = ids[:CONTEXT_LEN]
        cont = ids[CONTEXT_LEN:]
        ctx_t = ctx.unsqueeze(0).to(device)

        H = hidden_at_layer(model, ctx_t, LAYER)
        mean_eng = H.mean(dim=1).squeeze(0).detach()
        attn_eng = encoder(H).squeeze(0).detach()

        nll_no.append(_cont_nll(model, [], cont, device))
        nll_full.append(_cont_nll(model, [("tokens", ctx)], cont, device))
        nll_mean.append(_cont_nll(model, [("hidden", mean_eng)], cont, device))
        nll_attn.append(_cont_nll(model, [("hidden", attn_eng)], cont, device))

    def avg(lst):
        return sum(lst) / max(len(lst), 1)

    no = avg(nll_no)
    full = avg(nll_full)
    gap = no - full
    return {
        "no_context": no, "full_context": full, "gap": gap,
        "mean_recovery": (no - avg(nll_mean)) / gap if gap > 0 else 0,
        "attn_recovery": (no - avg(nll_attn)) / gap if gap > 0 else 0,
    }


@torch.no_grad()
def _forward_segments(model, segments, device):
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)
            parts.append(model.drop(model.tok_emb(ids)))
        elif kind == "hidden":
            parts.append(x.view(1, 1, -1).to(device))
    h = torch.cat(parts, dim=1)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]
    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    return model.lm_head(h)


@torch.no_grad()
def _cont_nll(model, prefix_segments, cont_ids, device):
    M = len(cont_ids)
    segs = list(prefix_segments) + [("tokens", cont_ids[:-1])]
    logits = _forward_segments(model, segs, device)
    plen = sum(len(x) if k == "tokens" else 1 for k, x in prefix_segments)
    pred = logits[:, plen:plen + M - 1, :]
    target = cont_ids[1:].unsqueeze(0).to(device)
    return float(F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                  target.reshape(-1), reduction="mean"))


@torch.no_grad()
def measure_val_ppl(model, val_loader, device, n_batches=20):
    model.eval()
    total = 0.0
    nb = 0
    for batch in val_loader:
        if nb >= n_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x, step=0)
        B, T, V = out.logits.shape
        total += F.cross_entropy(out.logits.reshape(B*T, V),
                                  y.reshape(B*T)).item()
        nb += 1
    return math.exp(min(total / nb, 20))


# ================================================================
# Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase54")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    # Load model (unfrozen for continued training)
    print("Loading base model...")
    model, cfg = load_model(device)
    for p in model.parameters():
        p.requires_grad = True

    # Load encoder (frozen)
    encoder_path = Path("results/identity_ae/phase51/attention_pool.pt")
    encoder = AttentionPoolEncoder(D).to(device)
    encoder.load_state_dict(torch.load(str(encoder_path), map_location=device,
                                       weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("Loaded Phase 51 attention pooling encoder (frozen)")

    # Data
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=2)
    train_loader = loaders["train"]
    train_iter = iter(train_loader)

    # Baseline measurements
    print("\nBaseline measurements (before continued training)...")
    model.eval()
    baseline_ppl = measure_val_ppl(model, loaders["validation"], device)
    print(f"  WikiText val PPL: {baseline_ppl:.2f}")

    val_ds = splits["validation"]
    baseline_kv = measure_kv_alignment(model, tokenizer, device, val_ds, n_passages=10)
    print(f"  K/V alignment:")
    for l in sorted(baseline_kv.keys()):
        print(f"    layer {l}: K={baseline_kv[l]['k_cos']:.4f}  "
              f"V={baseline_kv[l]['v_cos']:.4f}")

    baseline_info = measure_info_recovery(model, encoder, device, val_ds,
                                          n_passages=N_EVAL_PASSAGES)
    print(f"  Info recovery: mean={baseline_info['mean_recovery']:.1%}  "
          f"attn={baseline_info['attn_recovery']:.1%}")

    # Optimizer
    max_steps = CHECKPOINTS[-1]
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONT_LR,
                                   weight_decay=cfg.training.weight_decay,
                                   betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps)

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # ============================================================
    # Phase 54a: Continued training
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE 54a: Saliency-weighted continued training")
    print(f"  LR: {CONT_LR:.1e} (1/10th of pre-training {BASE_LR:.1e})")
    print(f"  Checkpoints at steps: {CHECKPOINTS}")
    print("=" * 60)

    GRAD_ACCUM = 4  # effective batch = 2 * 4 = 8

    model.train()
    step = 0
    accum_loss = 0.0
    t0 = time.time()
    checkpoint_results = {}
    ppl_history = []
    micro_step = 0

    optimizer.zero_grad()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x, y = batch[0].to(device), batch[1].to(device)
        B, T = x.shape

        # Compute saliency (frozen encoder, detached)
        with torch.no_grad():
            sal = compute_batch_saliency(encoder, model, x, device)
            sal = sal.detach()

        # Forward
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model(x, step=0)
            logits = out.logits
            V = logits.shape[-1]

            per_token = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                y[:, :-1].reshape(-1),
                reduction='none'
            ).reshape(B, T - 1)

            sal_shifted = sal[:, 1:]
            sal_norm = sal_shifted / (sal_shifted.sum(dim=-1, keepdim=True) + 1e-8)
            loss = (per_token * sal_norm).sum(dim=-1).mean() / GRAD_ACCUM

        loss.backward()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            accum_loss += loss.item() * GRAD_ACCUM
        else:
            continue  # wait for grad accum

        # Log
        if step % LOG_INTERVAL == 0:
            avg_loss = accum_loss / LOG_INTERVAL
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}  loss {avg_loss:.4f}  "
                  f"lr {lr:.2e}  ({elapsed:.0f}s)")
            accum_loss = 0.0

        # PPL safety check
        if step % EVAL_INTERVAL == 0:
            model.eval()
            ppl = measure_val_ppl(model, loaders["validation"], device)
            ppl_history.append({"step": step, "ppl": ppl})
            drift = (ppl - baseline_ppl) / baseline_ppl * 100
            print(f"    val PPL: {ppl:.2f} ({drift:+.1f}%)")
            if drift > 5.0:
                print(f"    WARNING: PPL drift > 5%, consider stopping")
            model.train()

        # Checkpoint + full measurement
        if step in CHECKPOINTS:
            print(f"\n--- Checkpoint at step {step} ---")
            model.eval()

            # Save checkpoint
            ckpt_path = results_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

            # Full measurement battery
            ppl = measure_val_ppl(model, loaders["validation"], device)
            drift = (ppl - baseline_ppl) / baseline_ppl * 100
            print(f"  val PPL: {ppl:.2f} ({drift:+.1f}%)")

            kv = measure_kv_alignment(model, tokenizer, device, val_ds, n_passages=10)
            print(f"  K/V alignment:")
            for l in sorted(kv.keys()):
                bk = baseline_kv.get(l, {}).get("k_cos", 0)
                bv = baseline_kv.get(l, {}).get("v_cos", 0)
                dk = kv[l]["k_cos"] - bk
                dv = kv[l]["v_cos"] - bv
                print(f"    layer {l}: K={kv[l]['k_cos']:.4f} ({dk:+.4f})  "
                      f"V={kv[l]['v_cos']:.4f} ({dv:+.4f})")

            info = measure_info_recovery(model, encoder, device, val_ds,
                                         n_passages=N_EVAL_PASSAGES)
            dm = info["mean_recovery"] - baseline_info["mean_recovery"]
            da = info["attn_recovery"] - baseline_info["attn_recovery"]
            print(f"  Info recovery: mean={info['mean_recovery']:.1%} ({dm:+.1%})  "
                  f"attn={info['attn_recovery']:.1%} ({da:+.1%})")

            checkpoint_results[step] = {
                "ppl": ppl, "drift_pct": drift,
                "kv": {str(l): v for l, v in kv.items()},
                "info": info,
            }

            model.train()
            print()

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"PHASE 54 SUMMARY: saliency-weighted continued training")
    print(f"{'='*72}")
    print(f"  Total steps: {max_steps}, time: {elapsed/3600:.1f}h")
    print(f"  Baseline val PPL: {baseline_ppl:.2f}")
    print()

    print(f"  {'step':>6}  {'PPL':>7}  {'drift':>7}  "
          f"{'K-cos L5':>9}  {'V-cos L5':>9}  "
          f"{'mean rec':>9}  {'attn rec':>9}")
    # Baseline row
    bk5 = baseline_kv.get(5, {}).get("k_cos", 0)
    bv5 = baseline_kv.get(5, {}).get("v_cos", 0)
    print(f"  {'base':>6}  {baseline_ppl:>7.2f}  {'---':>7}  "
          f"{bk5:>9.4f}  {bv5:>9.4f}  "
          f"{baseline_info['mean_recovery']:>8.1%}  "
          f"{baseline_info['attn_recovery']:>8.1%}")
    for step_k in CHECKPOINTS:
        if step_k in checkpoint_results:
            r = checkpoint_results[step_k]
            kv5 = r["kv"].get("5", {})
            print(f"  {step_k:>6}  {r['ppl']:>7.2f}  "
                  f"{r['drift_pct']:>+6.1f}%  "
                  f"{kv5.get('k_cos', 0):>9.4f}  "
                  f"{kv5.get('v_cos', 0):>9.4f}  "
                  f"{r['info']['mean_recovery']:>8.1%}  "
                  f"{r['info']['attn_recovery']:>8.1%}")

    out = {
        "baseline_ppl": baseline_ppl,
        "baseline_kv": {str(k): v for k, v in baseline_kv.items()},
        "baseline_info": baseline_info,
        "cont_lr": CONT_LR,
        "checkpoints": {str(k): v for k, v in checkpoint_results.items()},
        "ppl_history": ppl_history,
    }
    with open(results_dir / "continued_training.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
