"""V23 Training: Corrected 50/50 objective weighting + full V22 architecture.

Phase 1 (0-20K): Exponential scaffolding, MLPs frozen, cat_weight=0.5
Phase 2 (20K-43K): MLP co-evolution, calibrated from Phase 1
Phase 3 (43K-63K): Extended co-evolution

Usage:
    python train_v23.py [--device cuda]
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, PerHeadBonsignoreAttention
from data import load_wikitext, build_dataloaders
from losses import CombinedHRSLoss
from metrics import run_all_metrics


def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps: return base_lr * step / warmup_steps
    if step >= max_steps: return base_lr * 0.1
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


PHASE1_END = 20000
PHASE2_START = 20000
PHASE3_START = 43000
MAX_STEPS = 63000
CAT_WEIGHT = 0.5  # Corrected from 0.1 — Nash bargaining discovered 50/50


def calibrate_mlps(model, loaders, device, n_batches=10):
    """Calibrate per-head MLPs to match current exponential scores."""
    print("  Calibrating per-head MLPs...")
    model.eval()
    for block in model.blocks:
        if not isinstance(block.attn, PerHeadBonsignoreAttention):
            continue
        attn = block.attn
        all_q, all_k = [], []
        captured = {}
        def hook_fn(module, input, output):
            h = input[0]
            B, T, C = h.shape
            qkv = module.qkv(h).reshape(B, T, 3, module.n_heads, module.head_dim)
            q, k, _ = qkv.unbind(dim=2)
            captured['q'] = q.transpose(1, 2).detach()
            captured['k'] = k.transpose(1, 2).detach()
        handle = attn.register_forward_hook(hook_fn)
        for bi, batch in enumerate(loaders["train"]):
            if bi >= n_batches: break
            with torch.no_grad(): model(batch[0].to(device), step=0)
            idx = torch.randperm(captured['q'].shape[2])[:32]
            all_q.append(captured['q'][:, :, idx, :])
            all_k.append(captured['k'][:, :, :32, :])
        handle.remove()

        for h in range(attn.n_heads):
            tau_h = attn.log_tau[h].exp().item()
            mlp = attn.head_mlps[h]
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
            for epoch in range(200):
                for qb, kb in zip(all_q, all_k):
                    q_h, k_h = qb[:, h], kb[:, h]
                    q_sq = (q_h**2).sum(-1, keepdim=True)
                    k_sq = (k_h**2).sum(-1, keepdim=True)
                    dot = q_h @ k_h.transpose(-2, -1)
                    dist = q_sq + k_sq.transpose(-2, -1) - 2*dot
                    target = torch.exp(-dist / tau_h)
                    pred = mlp(target.reshape(-1, 1)).reshape(target.shape)
                    loss = F.mse_loss(pred, target)
                    opt.zero_grad(); loss.backward(); opt.step()
        if block.layer_idx == 0:
            print(f"    Layer 0 calibration done")
    print("  Calibration complete")


def train():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    # Override: 8 heads for V23 (same as V20/V22)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"V23: Corrected Objective Weighting (cat_weight={CAT_WEIGHT})")
    print(f"Device: {device}")

    # Data
    print(f"Loading {cfg.training.dataset}...")
    uses_cat = cfg.uses_categorization()
    splits, _ = load_wikitext(
        cfg.training.dataset, cfg.model.max_seq_len,
        with_categories=uses_cat,
        n_categories=cfg.cross_attn_engram.num_categories if uses_cat else 50,
    )
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Model
    model = HRSTransformer(cfg).to(device)
    counts = model.component_param_counts()
    print(f"Parameters: {counts['total']:,}")

    # Disable Layer 3 cross-attention (validated in V21/V22)
    for block in model.blocks:
        if hasattr(block, 'cross_attn') and block.use_cross_attn_engram and block.layer_idx == 3:
            block.use_cross_attn_engram = False
            print(f"  Removed Layer 3 cross-attention")

    # Phase 1: freeze MLPs
    for block in model.blocks:
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            block.attn.freeze_mlps()
    print(f"  Phase 1: MLPs frozen, cat_weight={CAT_WEIGHT}")

    # Loss
    loss_fn = CombinedHRSLoss(locality_cfg=cfg.locality if cfg.locality.enabled else None)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))

    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    run_dir = Path("results/v23_nash_weighted")
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump({"cat_weight": CAT_WEIGHT, "max_steps": MAX_STEPS,
                    "phase1_end": PHASE1_END, "phase2_start": PHASE2_START,
                    "phase3_start": PHASE3_START}, f, indent=2)

    best_val = float("inf")
    step = 0
    phase = 1
    train_iter = iter(loaders["train"])
    log_history = []
    t0 = time.time()
    accum_loss, accum_ce, accum_loc, accum_cat = 0, 0, 0, 0

    print(f"\nTraining: {MAX_STEPS} steps")
    print(f"  Phase 1: 0–{PHASE1_END} | Phase 2: {PHASE2_START}–{PHASE3_START} | Phase 3: {PHASE3_START}–{MAX_STEPS}\n")

    model.train()
    while step < MAX_STEPS:
        optimizer.zero_grad()

        for _ in range(cfg.training.grad_accum_steps):
            try: batch = next(train_iter)
            except StopIteration: train_iter = iter(loaders["train"]); batch = next(train_iter)

            if len(batch) == 3: x, y, ct = batch; ct = ct.to(device)
            else: x, y = batch; ct = None
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(x, step=step, collect_layer_reps=cfg.locality.enabled)
                loss_dict = loss_fn(
                    output.logits, y,
                    layer_representations=output.layer_representations if cfg.locality.enabled else None,
                    categorization_logits=output.categorization_logits if uses_cat else None,
                    categorization_targets=ct if uses_cat else None,
                    categorization_alpha=CAT_WEIGHT if uses_cat else 0.0,
                )
                loss = loss_dict["loss"] / cfg.training.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR scheduling
        base_lr = get_lr(step, 2000, MAX_STEPS, 3e-4)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        optimizer.step()

        # Phase transitions
        if step == PHASE2_START and phase == 1:
            phase = 2
            print(f"\n{'='*60}")
            print(f"PHASE 2: MLP co-evolution (step {step})")
            print(f"{'='*60}")

            # Save Phase 1 checkpoint
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                         "val_ppl": best_val, "phase": 1}, run_dir / "phase1_end.pt")

            # Calibrate and unfreeze
            calibrate_mlps(model, loaders, device)
            for block in model.blocks:
                if isinstance(block.attn, PerHeadBonsignoreAttention):
                    block.attn.unfreeze_mlps()

            # Rebuild optimizer with separate LR groups
            kernel_params, proj_params, scalar_params, other_params = [], [], [], []
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if 'head_mlps' in name or 'head_alphas' in name or 'log_tau' in name:
                    kernel_params.append(param)
                elif 'head_output_scalars' in name or 'gate_scalar' in name:
                    scalar_params.append(param)
                elif 'qkv' in name or 'out_proj' in name:
                    proj_params.append(param)
                else:
                    other_params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": kernel_params, "lr": 1e-3, "weight_decay": 0.01},
                {"params": scalar_params, "lr": 1e-3, "weight_decay": 0.0},
                {"params": proj_params, "lr": 1e-5, "weight_decay": 0.1},
                {"params": other_params, "lr": 3e-5, "weight_decay": 0.1},
            ])
            print(f"  Kernel LR: 1e-3, Proj LR: 1e-5, Other LR: 3e-5\n")

        if step == PHASE3_START and phase == 2:
            phase = 3
            print(f"\n{'='*60}")
            print(f"PHASE 3: Extended co-evolution (step {step})")
            print(f"{'='*60}\n")

        # Engram buffer update
        if cfg.uses_cross_attn_engram() and step % cfg.cross_attn_engram.update_interval == 0:
            model.update_engram_buffer()

        accum_loss += loss_dict["loss"].item()
        accum_ce += loss_dict["ce_loss"].item()
        if "locality_loss" in loss_dict: accum_loc += loss_dict["locality_loss"].item()
        if "categorization_loss" in loss_dict: accum_cat += loss_dict["categorization_loss"].item()

        step += 1

        if step % 100 == 0:
            n = 100
            ppl = math.exp(min(accum_ce / n, 20))
            elapsed = time.time() - t0
            gates = output.cross_attn_gate_values or []
            g = "/".join(f"{g:.3f}" for g in gates) if gates else "N/A"
            print(f"step {step:6d} | CE {accum_ce/n:.4f} | ppl {ppl:.1f} | "
                  f"cat {accum_cat/n:.4f} | P{phase} | gates [{g}] | {elapsed:.0f}s")
            accum_loss, accum_ce, accum_loc, accum_cat = 0, 0, 0, 0

        if step % 1000 == 0:
            blk = model.blocks[0]
            if isinstance(blk.attn, PerHeadBonsignoreAttention):
                diag = blk.attn.get_diagnostics()
                print(f"  Alphas: [{' '.join(f'{a:.3f}' for a in diag['alphas'])}]")
                print(f"  Taus:   [{' '.join(f'{t:.1f}' for t in diag['taus'])}]")
                if hasattr(blk.attn, 'head_output_scalars'):
                    ss = F.softplus(blk.attn.head_output_scalars).detach()
                    print(f"  Scales: [{' '.join(f'{s:.3f}' for s in ss.tolist())}]")
            for block in model.blocks:
                if hasattr(block, 'cross_attn') and block.use_cross_attn_engram:
                    gs = F.softplus(block.cross_attn.gate_scalar).item()
                    gv = torch.sigmoid(block.cross_attn.gate_logit).item()
                    print(f"  Layer {block.layer_idx} gate: {gv:.3f} × {gs:.3f} = {gv*gs:.4f}")

            log_history.append({"step": step, "phase": phase})

        if step % 1000 == 0:
            model.eval()
            metrics = run_all_metrics(model, loaders["validation"], device, amp_dtype, max_batches=10)
            model.train()

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({**metrics, "step": step, "phase": phase}, default=str) + "\n")

            if metrics["val_ppl"] < best_val:
                best_val = metrics["val_ppl"]
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                             "val_ppl": best_val, "phase": phase}, run_dir / "best.pt")
                print(f"  ** val_ppl: {best_val:.2f} *best*\n")
            else:
                print(f"  val_ppl: {metrics['val_ppl']:.2f} (best={best_val:.2f})\n")

        if step % 5000 == 0:
            ckpt_path = run_dir / f"checkpoint_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(), "phase": phase}, ckpt_path)
            existing = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]: old.unlink()

    # Final
    torch.save({"step": step, "model_state_dict": model.state_dict()}, run_dir / "final.pt")
    with open(run_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nV23 Training Complete")
    print(f"  Best val_ppl: {best_val:.2f}")
    print(f"  Cat weight: {CAT_WEIGHT}")
    print(f"  Total steps: {step}")
    print(f"  Results: {run_dir}")


if __name__ == "__main__":
    train()
