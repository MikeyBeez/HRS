"""V20 Training: Scaffolded Bonsignore Kernel with Per-Head MLPs.

Phase 1 (steps 0–20K): MLPs frozen, pure exponential scaffolding.
Phase 2 (steps 20K–43K): MLPs unfrozen, co-evolution with separate LRs.

Logs per-head alpha values and kernel diagnostics every 1000 steps.

Usage:
    python train_v20.py [--resume auto] [--device cuda]
"""

import os
import time
import json
import math
from pathlib import Path
from collections import deque

import torch
import torch.nn.functional as F

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, PerHeadBonsignoreAttention
from data import load_wikitext, build_dataloaders
from losses import CombinedHRSLoss
from metrics import run_all_metrics


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step, warmup_steps, max_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step >= max_steps:
        return base_lr * 0.1
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


PHASE1_END = 20000  # Phase 1: exponential scaffolding
PHASE2_START = 20000  # Phase 2: kernel co-evolution


def train():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"V20 Bonsignore Kernel Training")
    print(f"Device: {device}")

    # Data
    print(f"Loading {cfg.training.dataset}...")
    uses_categories = cfg.uses_categorization()
    splits, tokenizer = load_wikitext(
        cfg.training.dataset, cfg.model.max_seq_len,
        with_categories=uses_categories,
        n_categories=cfg.cross_attn_engram.num_categories if uses_categories else 50,
    )
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Model
    model = HRSTransformer(cfg).to(device)
    param_counts = model.component_param_counts()
    print(f"Parameters: {param_counts['total']:,}")
    for name, count in param_counts.items():
        if count > 0 and name != "total":
            print(f"  {name}: {count:,}")

    # Count per-head MLP params
    mlp_params = sum(
        sum(p.numel() for mlp in block.attn.head_mlps for p in mlp.parameters())
        for block in model.blocks if isinstance(block.attn, PerHeadBonsignoreAttention)
    )
    print(f"  per-head kernel MLPs: {mlp_params:,}")
    print(f"  PEER: {cfg.peer.n_sub_keys}^2 = {cfg.peer.n_sub_keys**2:,} experts")

    # Phase 1: Freeze all per-head MLPs
    for block in model.blocks:
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            block.attn.freeze_mlps()
    print(f"\nPhase 1: Per-head MLPs FROZEN (pure exponential)")

    # Loss
    loss_fn = CombinedHRSLoss(locality_cfg=cfg.locality if cfg.locality.enabled else None)

    # Optimizer: all params at base LR initially
    # Phase 2 will adjust LRs for kernel MLPs
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output
    run_dir = Path(cfg.training.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "ablation": cfg.training.ablation.value,
            "model": cfg.model.__dict__,
            "peer": cfg.peer.__dict__,
            "cross_attn_engram": cfg.cross_attn_engram.__dict__,
            "phase1_end": PHASE1_END,
            "phase2_start": PHASE2_START,
            "total_steps": cfg.training.max_steps,
        }, f, indent=2)

    # Training state
    best_val_ppl = float("inf")
    step = 0
    train_iter = iter(loaders["train"])
    log_history = []
    phase = 1
    t0 = time.time()

    # Accumulators
    accum_loss = 0.0
    accum_ce = 0.0
    accum_loc = 0.0
    accum_cat = 0.0

    print(f"\nTraining: {cfg.training.max_steps} steps")
    print(f"  Phase 1: 0–{PHASE1_END} (exponential scaffolding)")
    print(f"  Phase 2: {PHASE2_START}–{cfg.training.max_steps} (kernel co-evolution)")
    print(f"  Effective batch: {cfg.training.batch_size * cfg.training.grad_accum_steps}")
    print()

    model.train()
    while step < cfg.training.max_steps:
        optimizer.zero_grad()

        for micro_step in range(cfg.training.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                batch = next(train_iter)

            if len(batch) == 3:
                x, y, cat_targets = batch
                cat_targets = cat_targets.to(device)
            else:
                x, y = batch
                cat_targets = None
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(x, step=step, collect_layer_reps=cfg.locality.enabled)

                loss_dict = loss_fn(
                    output.logits, y,
                    layer_representations=output.layer_representations if cfg.locality.enabled else None,
                    categorization_logits=output.categorization_logits if uses_categories else None,
                    categorization_targets=cat_targets if uses_categories else None,
                    categorization_alpha=cfg.cross_attn_engram.categorization_alpha if uses_categories else 0.0,
                )

                loss = loss_dict["loss"] / cfg.training.grad_accum_steps

            loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        # LR scheduling
        base_lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps, cfg.training.learning_rate)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        optimizer.step()

        # Phase transition
        if step == PHASE2_START and phase == 1:
            phase = 2
            print(f"\n{'='*60}")
            print(f"PHASE 2: Unfreezing per-head kernel MLPs (step {step})")
            print(f"{'='*60}\n")

            # Unfreeze MLPs
            for block in model.blocks:
                if isinstance(block.attn, PerHeadBonsignoreAttention):
                    block.attn.unfreeze_mlps()

            # Rebuild optimizer with separate LR groups
            proj_params = []
            mlp_kernel_params = []
            other_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'head_mlps' in name or 'head_alphas' in name:
                    mlp_kernel_params.append(param)
                elif 'qkv' in name or 'out_proj' in name:
                    proj_params.append(param)
                else:
                    other_params.append(param)

            optimizer = torch.optim.AdamW([
                {"params": proj_params, "lr": 1e-5, "weight_decay": 0.1},
                {"params": mlp_kernel_params, "lr": 1e-3, "weight_decay": 0.01},
                {"params": other_params, "lr": 3e-5, "weight_decay": 0.1},
            ])
            print(f"  Projection LR: 1e-5, Kernel MLP LR: 1e-3, Other LR: 3e-5")
            print(f"  Kernel MLP params: {sum(p.numel() for p in mlp_kernel_params):,}")

            # Save Phase 1 checkpoint
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "phase": 1,
                "val_ppl": best_val_ppl,
            }, run_dir / "phase1_end.pt")

        # V18-style engram buffer update
        if cfg.uses_cross_attn_engram() and step % cfg.cross_attn_engram.update_interval == 0:
            model.update_engram_buffer()

        # Accumulate metrics
        loss_val = loss_dict["loss"].item()
        accum_loss += loss_val
        accum_ce += loss_dict["ce_loss"].item()
        if "locality_loss" in loss_dict:
            accum_loc += loss_dict["locality_loss"].item()
        if "categorization_loss" in loss_dict:
            accum_cat += loss_dict["categorization_loss"].item()

        step += 1

        # Logging
        if step % 100 == 0:
            n = 100
            avg_loss = accum_loss / n
            avg_ce = accum_ce / n
            ppl = math.exp(min(avg_ce, 20))
            elapsed = time.time() - t0

            extras = f" | loc {accum_loc/n:.4f}" if cfg.locality.enabled else ""
            if uses_categories:
                extras += f" | cat {accum_cat/n:.4f}"
            if output.cross_attn_gate_values:
                gates = "/".join(f"{g:.3f}" for g in output.cross_attn_gate_values)
                extras += f" | ca_gates [{gates}]"

            print(f"step {step:6d} | loss {avg_loss:.4f} | CE {avg_ce:.4f} | "
                  f"ppl {ppl:.1f} | lr {base_lr:.2e} | P{phase}{extras} | {elapsed:.1f}s")

            accum_loss = 0.0
            accum_ce = 0.0
            accum_loc = 0.0
            accum_cat = 0.0

        # Per-head diagnostics (every 1000 steps)
        if step % 1000 == 0:
            diag = model.blocks[0].attn.get_diagnostics()
            alphas_str = " ".join(f"{a:.3f}" for a in diag["alphas"])
            taus_str = " ".join(f"{t:.1f}" for t in diag["taus"])
            print(f"  Per-head alphas: [{alphas_str}]")
            print(f"  Per-head taus:   [{taus_str}]")

            entry = {
                "step": step, "phase": phase,
                "per_head_alphas": diag["alphas"],
                "per_head_taus": diag["taus"],
                "mean_alpha": diag["mean_alpha"],
            }
            log_history.append(entry)

        # Evaluation
        if step % cfg.training.eval_interval == 0:
            print(f"\n--- Evaluation at step {step} ---")
            model.eval()
            metrics = run_all_metrics(model, loaders["validation"], device, amp_dtype, max_batches=10)
            model.train()

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({**metrics, "step": step, "phase": phase}, default=str) + "\n")

            if metrics["val_ppl"] < best_val_ppl:
                best_val_ppl = metrics["val_ppl"]
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_ppl": best_val_ppl,
                    "phase": phase,
                }, run_dir / "best.pt")
                print(f"  ** New best val_ppl: {best_val_ppl:.2f} (saved best.pt)")

            print(f"  val_ppl: {metrics.get('val_ppl', 'N/A'):.1f}")
            print()

        # Save checkpoint
        if step % cfg.training.save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "phase": phase,
            }, ckpt_path)
            # Keep last 2
            existing = sorted(run_dir.glob("checkpoint_*.pt"),
                              key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]:
                old.unlink()

    # Final save
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, run_dir / "final.pt")

    with open(run_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Best val_ppl: {best_val_ppl:.2f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    train()
