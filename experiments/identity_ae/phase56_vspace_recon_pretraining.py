"""Phase 56: Pre-training with V-space reconstruction loss.

The 30% information recovery ceiling (Phase 55) exists because W_V
discards 70% of hidden-state information during projection. This phase
adds a direct reconstruction signal: an auxiliary decoder at each layer
that maps V back to the pre-attention hidden state. The gradient from
this decoder gives W_V explicit pressure to preserve information,
bypassing the indirect NTP gradient path.

Run B from the spec: reconstruction loss only, no mHC.

Loss = NTP + lambda_recon * sum(MSE(decoder_l(V_l), H_l) for each layer)

Lambda schedule: start at 0.1, cosine decay to 0.01 over training.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase56_vspace_recon_pretraining.py
"""

import json
import math
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, build_dataloaders
from metrics import run_all_metrics

from experiments.identity_ae.phase22_engram_key import hidden_at_layer
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase32_kv_similarity import (
    install_qkv_hooks, remove_hooks, per_head_cosine,
)
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x
from experiments.identity_ae.phase51_learned_engram import AttentionPoolEncoder

from train import set_seed, get_lr, get_phase, get_phase_lr_multipliers, V1_PARAM_GROUP_INDEX

D = 1024
LAMBDA_START = 0.1
LAMBDA_END = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4   # effective batch = 8
MEASURE_CHECKPOINTS = [2000, 5000, 10000, 20000, 43000]
LOG_INTERVAL = 100
EVAL_INTERVAL = 1000
SAVE_INTERVAL = 2500


# ================================================================
# V-space reconstruction decoders (one per layer)
# ================================================================

class VReconDecoders(nn.Module):
    """Per-layer linear decoders: V → H reconstruction."""
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.decoders = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(n_layers)
        ])
        # Initialize near identity so initial recon loss is small
        for dec in self.decoders:
            nn.init.eye_(dec.weight)

    def loss(self, layer_idx, V, H_target):
        """MSE between decoder(V) and the pre-attention hidden state."""
        H_recon = self.decoders[layer_idx](V)
        return F.mse_loss(H_recon, H_target.detach())


# ================================================================
# Hooks to capture pre-attention H and V at each layer
# ================================================================

class VCapture:
    """Forward hooks that capture pre-attention hidden states and V vectors."""
    def __init__(self, model):
        self.model = model
        self.captured = {}  # layer_idx -> {"H": tensor, "V": tensor}
        self.handles = []

    def install(self):
        """Install hooks on every block."""
        # Pre-hook on ln1 to capture pre-attention hidden state
        for i, block in enumerate(self.model.blocks):
            n_heads = block.attn.n_heads
            head_dim = block.attn.head_dim

            def make_ln1_pre_hook(layer_idx):
                def hook(module, args):
                    # args[0] is the input to ln1 = the pre-attention hidden state
                    x = args[0]
                    if layer_idx not in self.captured:
                        self.captured[layer_idx] = {}
                    self.captured[layer_idx]["H"] = x
                return hook

            def make_qkv_hook(layer_idx, H, Dh):
                def hook(module, inp, out):
                    B, T, _ = out.shape
                    qkv = out.reshape(B, T, 3, H, Dh)
                    v = qkv[:, :, 2]  # (B, T, H, Dh)
                    # Flatten heads: (B, T, H*Dh) = (B, T, D)
                    v_flat = v.reshape(B, T, H * Dh)
                    if layer_idx not in self.captured:
                        self.captured[layer_idx] = {}
                    self.captured[layer_idx]["V"] = v_flat
                return hook

            self.handles.append(
                block.ln1.register_forward_pre_hook(make_ln1_pre_hook(i)))
            self.handles.append(
                block.attn.qkv.register_forward_hook(
                    make_qkv_hook(i, n_heads, head_dim)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear(self):
        self.captured = {}


# ================================================================
# Measurement functions (reuse from Phase 54)
# ================================================================

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


@torch.no_grad()
def measure_kv_alignment(model, val_ds, device, n_passages=10):
    n_layers = len(model.blocks)
    d_model = model.blocks[0].attn.n_heads * model.blocks[0].attn.head_dim
    results = {l: {"cos_k": [], "cos_v": []} for l in range(n_layers)}
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:n_passages].tolist()

    for idx in indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:128]
        ids = ids.unsqueeze(0).to(device)

        passage_store = {}
        handles = install_qkv_hooks(model, passage_store)
        _ = model(ids, step=0)
        remove_hooks(handles)

        h_L = hidden_at_layer(model, ids, 5)
        engram = h_L.mean(dim=1)

        engram_store = {}
        handles = install_qkv_hooks(model, engram_store)
        _ = forward_from_x(model, engram.view(1, 1, d_model))
        remove_hooks(handles)

        for l in range(n_layers):
            if l not in passage_store or l not in engram_store:
                continue
            pk = passage_store[l]["k"].squeeze(0).mean(dim=0)
            pv = passage_store[l]["v"].squeeze(0).mean(dim=0)
            ek = engram_store[l]["k"].squeeze(0).squeeze(0)
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
def measure_info_recovery(model, val_ds, device, n_passages=50):
    """Mean-pooling information recovery."""
    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:n_passages].tolist()
    nll_no = []
    nll_full = []
    nll_mean = []

    for idx in indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:256]
        if len(ids) < 256:
            continue
        ctx = ids[:200]
        cont = ids[200:]
        ctx_t = ctx.unsqueeze(0).to(device)
        H = hidden_at_layer(model, ctx_t, 5)
        mean_eng = H.mean(dim=1).squeeze(0).detach()

        nll_no.append(_cont_nll(model, [], cont, device))
        nll_full.append(_cont_nll(model, [("tokens", ctx)], cont, device))
        nll_mean.append(_cont_nll(model, [("hidden", mean_eng)], cont, device))

    no = sum(nll_no) / len(nll_no)
    full = sum(nll_full) / len(nll_full)
    gap = no - full
    mean_nll = sum(nll_mean) / len(nll_mean)
    return {
        "no": no, "full": full, "gap": gap, "mean_nll": mean_nll,
        "mean_recovery": (no - mean_nll) / gap if gap > 0 else 0,
    }


@torch.no_grad()
def _forward_segments(model, segments, device):
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)
            parts.append(model.drop(model.tok_emb(ids)))
        elif kind == "hidden":
            if x.dim() == 1:
                x = x.unsqueeze(0)
            parts.append(x.unsqueeze(0).to(device))
    h = torch.cat(parts, dim=1)
    if h.shape[1] > 512:
        h = h[:, -512:]
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
    plen = sum(len(x) if k == "tokens" else (x.shape[0] if x.dim() >= 2 else 1)
               for k, x in prefix_segments)
    pred = logits[:, plen:plen + M - 1, :]
    target = cont_ids[1:].unsqueeze(0).to(device)
    return float(F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                  target.reshape(-1), reduction="mean"))


# ================================================================
# Main
# ================================================================

def main():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 56: V-space reconstruction pre-training")
    print(f"Device: {device}")

    # Data
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, BATCH_SIZE)

    # Model (fresh, from scratch)
    model = HRSTransformer(cfg).to(device)
    n_layers = len(model.blocks)
    param_counts = model.component_param_counts()
    print(f"Model params: {param_counts['total']:,}")

    # Reconstruction decoders
    recon = VReconDecoders(D, n_layers).to(device)
    recon_params = sum(p.numel() for p in recon.parameters())
    print(f"Recon decoder params: {recon_params:,} (training only)")

    # V capture hooks
    vcap = VCapture(model)
    vcap.install()

    # Optimizer: model + recon decoders
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
    # Add recon decoders
    optimizer_groups.append({
        "params": list(recon.parameters()),
        "lr": cfg.training.learning_rate,
        "weight_decay": 0.0,
    })
    group_names.append("recon_decoders")

    optimizer = torch.optim.AdamW(
        optimizer_groups, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay, betas=(0.9, 0.95))

    max_steps = cfg.training.max_steps
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output
    run_dir = Path("results/identity_ae/phase56")
    run_dir.mkdir(parents=True, exist_ok=True)

    val_ds = splits["validation"]

    # Training loop
    train_iter = iter(loaders["train"])
    step = 0
    micro_step = 0
    current_phase = get_phase(step, cfg)
    accum_ntp = 0.0
    accum_recon_loss = 0.0
    best_val_ppl = float("inf")
    t0 = time.time()
    checkpoint_results = {}

    print(f"\nTraining: {max_steps} steps")
    print(f"Lambda recon: {LAMBDA_START} → {LAMBDA_END} (cosine)")
    print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}\n")

    optimizer.zero_grad()
    model.train()
    recon.train()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        x, y = batch[0].to(device), batch[1].to(device)
        B, T = x.shape

        # Lambda schedule: cosine decay
        progress = step / max_steps
        lambda_recon = LAMBDA_END + 0.5 * (LAMBDA_START - LAMBDA_END) * \
                       (1 + math.cos(math.pi * progress))

        # Forward (hooks capture H and V at each layer)
        vcap.clear()
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model(x, step=step)
            logits = out.logits
            V = logits.shape[-1]

            # NTP loss
            ntp_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                y[:, :-1].reshape(-1))

            # Reconstruction loss across all layers
            total_recon = torch.tensor(0.0, device=device)
            n_recon = 0
            for l in range(n_layers):
                if l in vcap.captured and "H" in vcap.captured[l] and "V" in vcap.captured[l]:
                    H_pre = vcap.captured[l]["H"]
                    V_l = vcap.captured[l]["V"]
                    total_recon = total_recon + recon.loss(l, V_l, H_pre)
                    n_recon += 1
            if n_recon > 0:
                total_recon = total_recon / n_recon

            loss = (ntp_loss + lambda_recon * total_recon) / GRAD_ACCUM

        loss.backward()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg.training.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)

            # Phase LR
            new_phase = get_phase(step, cfg)
            if new_phase != current_phase:
                print(f"\n*** Phase {current_phase} → {new_phase} "
                      f"at step {step} ***\n")
                current_phase = new_phase

            base_lr = get_lr(step, cfg.training.warmup_steps,
                             max_steps, cfg.training.learning_rate)
            lr_mults = get_phase_lr_multipliers(current_phase, cfg)
            for i, (pg, name) in enumerate(zip(optimizer.param_groups,
                                                group_names)):
                if name == "recon_decoders":
                    pg["lr"] = base_lr
                else:
                    mult_idx = V1_PARAM_GROUP_INDEX.get(name, 0)
                    pg["lr"] = base_lr * lr_mults[mult_idx]

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            accum_ntp += ntp_loss.item()
            accum_recon_loss += total_recon.item()
        else:
            continue

        # Log
        if step % LOG_INTERVAL == 0:
            avg_ntp = accum_ntp / LOG_INTERVAL
            avg_recon = accum_recon_loss / LOG_INTERVAL
            ppl = math.exp(min(avg_ntp, 20))
            elapsed = time.time() - t0
            print(f"step {step:6d}  ntp {avg_ntp:.4f}  ppl {ppl:.1f}  "
                  f"recon {avg_recon:.4f}  λ {lambda_recon:.3f}  "
                  f"lr {base_lr:.2e}  P{current_phase}  ({elapsed:.0f}s)")
            accum_ntp = 0.0
            accum_recon_loss = 0.0

        # Eval
        if step % EVAL_INTERVAL == 0:
            model.eval()
            recon.eval()
            vcap.remove()  # remove hooks during eval

            ppl = measure_val_ppl(model, loaders["validation"], device)
            print(f"  val PPL: {ppl:.2f}")

            if ppl < best_val_ppl:
                best_val_ppl = ppl
                torch.save({
                    "step": step, "model_state_dict": model.state_dict(),
                    "recon_state_dict": recon.state_dict(),
                    "val_ppl": ppl,
                }, run_dir / "best.pt")
                print(f"  ** New best: {ppl:.2f}")

            vcap.install()  # reinstall hooks
            model.train()
            recon.train()

        # Full measurement at designated checkpoints
        if step in MEASURE_CHECKPOINTS:
            print(f"\n--- Measurement at step {step} ---")
            model.eval()
            vcap.remove()

            ppl = measure_val_ppl(model, loaders["validation"], device)

            kv = measure_kv_alignment(model, val_ds, device, n_passages=10)
            info = measure_info_recovery(model, val_ds, device, n_passages=50)

            print(f"  val PPL:         {ppl:.2f}")
            print(f"  K/V alignment:")
            for l in sorted(kv.keys()):
                print(f"    L{l}: K={kv[l]['k_cos']:.4f}  V={kv[l]['v_cos']:.4f}")
            print(f"  info recovery:   mean={info['mean_recovery']:.1%}")

            checkpoint_results[step] = {
                "ppl": ppl,
                "kv": {str(l): v for l, v in kv.items()},
                "info": info,
            }

            # Save checkpoint
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "recon_state_dict": recon.state_dict(),
                "val_ppl": ppl,
            }, run_dir / f"checkpoint_{step}.pt")

            # Keep only last 2 checkpoints
            existing = sorted(run_dir.glob("checkpoint_*.pt"),
                              key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]:
                old.unlink()

            vcap.install()
            model.train()
            recon.train()
            print()

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"PHASE 56 SUMMARY: V-space reconstruction pre-training")
    print(f"{'='*72}")
    print(f"  Steps: {step}, time: {elapsed/3600:.1f}h")
    print(f"  Best val PPL: {best_val_ppl:.2f}")
    print(f"  V22 baseline reference: PPL ~21, V-cos 0.65-0.70, "
          f"mean recovery 18%")
    print()

    print(f"  {'step':>6}  {'PPL':>7}  {'V-cos L5':>9}  {'K-cos L5':>9}  "
          f"{'mean rec':>9}")
    for s in sorted(checkpoint_results.keys()):
        r = checkpoint_results[s]
        kv5 = r["kv"].get("5", {})
        print(f"  {s:>6}  {r['ppl']:>7.2f}  "
              f"{kv5.get('v_cos', 0):>9.4f}  "
              f"{kv5.get('k_cos', 0):>9.4f}  "
              f"{r['info']['mean_recovery']:>8.1%}")

    out = {
        "lambda_start": LAMBDA_START, "lambda_end": LAMBDA_END,
        "best_val_ppl": best_val_ppl,
        "checkpoints": {str(k): v for k, v in checkpoint_results.items()},
    }
    with open(run_dir / "recon_pretraining.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
