"""V23b: Learnable categorization weight, initialized at 0.5.

Same architecture as V23 but the cat weight is a learnable parameter.
The model negotiates its own LM/categorization tradeoff during training.
Fresh start from random initialization, 63K steps.

Usage:
    python train_v23b.py [--device cuda]
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
from train_v23 import calibrate_mlps


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
PHASE3_START = 43000
MAX_STEPS = 63000


def train():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Learnable cat weight: softplus maps R -> R+
    # Initialize so softplus(cat_weight_raw) ≈ 0.5
    # softplus(x) = log(1 + exp(x)), softplus(0.0) ≈ 0.693
    # We want 0.5, so x = log(exp(0.5) - 1) ≈ -0.19
    cat_weight_raw = nn.Parameter(torch.tensor(-0.19, device=device))

    print(f"V23b: Learnable Categorization Weight")
    print(f"  Initial cat_weight: {F.softplus(cat_weight_raw).item():.4f}")
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

    # Disable Layer 3 cross-attention
    for block in model.blocks:
        if hasattr(block, 'cross_attn') and block.use_cross_attn_engram and block.layer_idx == 3:
            block.use_cross_attn_engram = False

    # Phase 1: freeze MLPs
    for block in model.blocks:
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            block.attn.freeze_mlps()
    print(f"  Phase 1: MLPs frozen, cat_weight learnable from {F.softplus(cat_weight_raw).item():.3f}")

    # Loss
    loss_fn = CombinedHRSLoss(locality_cfg=cfg.locality if cfg.locality.enabled else None)

    # Optimizer — include cat_weight_raw
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [cat_weight_raw],
        lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95),
    )

    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    run_dir = Path("results/v23b_learnable_weight")
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    step = 0
    phase = 1
    train_iter = iter(loaders["train"])
    t0 = time.time()
    accum_ce, accum_cat = 0, 0

    print(f"\nTraining: {MAX_STEPS} steps")
    print(f"  Phase 1: 0–{PHASE1_END} | Phase 2: {PHASE1_END}–{PHASE3_START} | Phase 3: {PHASE3_START}–{MAX_STEPS}\n")

    model.train()
    while step < MAX_STEPS:
        optimizer.zero_grad()

        cat_weight = F.softplus(cat_weight_raw)

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
                    categorization_alpha=cat_weight if uses_cat else 0.0,
                )
                loss = loss_dict["loss"] / cfg.training.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [cat_weight_raw], 1.0)

        # LR scheduling
        base_lr = get_lr(step, 2000, MAX_STEPS, 3e-4)
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

        optimizer.step()

        # Phase transitions
        if step == PHASE1_END and phase == 1:
            phase = 2
            print(f"\n{'='*60}")
            print(f"PHASE 2: MLP co-evolution (step {step})")
            print(f"  cat_weight at transition: {F.softplus(cat_weight_raw).item():.4f}")
            print(f"{'='*60}")

            torch.save({"step": step, "model_state_dict": model.state_dict(),
                         "cat_weight_raw": cat_weight_raw.item(),
                         "val_ppl": best_val}, run_dir / "phase1_end.pt")

            # Unfreeze then calibrate
            for block in model.blocks:
                if isinstance(block.attn, PerHeadBonsignoreAttention):
                    block.attn.unfreeze_mlps()
            calibrate_mlps(model, loaders, device)

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
                {"params": [cat_weight_raw], "lr": 1e-3, "weight_decay": 0.0},
                {"params": proj_params, "lr": 1e-5, "weight_decay": 0.1},
                {"params": other_params, "lr": 3e-5, "weight_decay": 0.1},
            ])
            print(f"  Optimizer rebuilt with separate LR groups\n")

        if step == PHASE3_START and phase == 2:
            phase = 3
            print(f"\n  PHASE 3 (step {step}), cat_weight: {F.softplus(cat_weight_raw).item():.4f}\n")

        if cfg.uses_cross_attn_engram() and step % cfg.cross_attn_engram.update_interval == 0:
            model.update_engram_buffer()

        accum_ce += loss_dict["ce_loss"].item()
        if "categorization_loss" in loss_dict: accum_cat += loss_dict["categorization_loss"].item()
        step += 1

        if step % 100 == 0:
            n = 100; ppl = math.exp(min(accum_ce/n, 20)); elapsed = time.time() - t0
            cw = F.softplus(cat_weight_raw).item()
            gates = output.cross_attn_gate_values or []
            g = "/".join(f"{g:.3f}" for g in gates) if gates else "N/A"
            print(f"step {step:6d} | CE {accum_ce/n:.4f} | ppl {ppl:.1f} | "
                  f"cat {accum_cat/n:.4f} | cw={cw:.3f} | P{phase} | [{g}] | {elapsed:.0f}s")
            accum_ce, accum_cat = 0, 0

        if step % 1000 == 0:
            cw = F.softplus(cat_weight_raw).item()
            blk = model.blocks[0]
            if isinstance(blk.attn, PerHeadBonsignoreAttention):
                diag = blk.attn.get_diagnostics()
                print(f"  cat_weight: {cw:.4f}")
                print(f"  Alphas: [{' '.join(f'{a:.3f}' for a in diag['alphas'])}]")
                print(f"  Taus:   [{' '.join(f'{t:.1f}' for t in diag['taus'])}]")
                ss = F.softplus(blk.attn.head_output_scalars).detach()
                print(f"  Scales: [{' '.join(f'{s:.3f}' for s in ss.tolist())}]")

        if step % 1000 == 0:
            model.eval()
            metrics = run_all_metrics(model, loaders["validation"], device, amp_dtype, max_batches=10)
            model.train()
            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({**metrics, "step": step, "phase": phase,
                                     "cat_weight": F.softplus(cat_weight_raw).item()}, default=str) + "\n")
            if metrics["val_ppl"] < best_val:
                best_val = metrics["val_ppl"]
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                             "cat_weight_raw": cat_weight_raw.item(),
                             "val_ppl": best_val}, run_dir / "best.pt")
                print(f"  ** val_ppl: {best_val:.2f} *best*\n")
            else:
                print(f"  val_ppl: {metrics['val_ppl']:.2f} (best={best_val:.2f})\n")

        if step % 5000 == 0:
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                         "cat_weight_raw": cat_weight_raw.item()}, run_dir / f"checkpoint_{step}.pt")
            existing = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]: old.unlink()

    torch.save({"step": step, "model_state_dict": model.state_dict(),
                 "cat_weight_raw": cat_weight_raw.item()}, run_dir / "final.pt")

    final_cw = F.softplus(cat_weight_raw).item()
    print(f"\nV23b Complete")
    print(f"  Best val_ppl: {best_val:.2f}")
    print(f"  Final cat_weight: {final_cw:.4f} (started at 0.50)")
    print(f"  Results: {run_dir}")


if __name__ == "__main__":
    train()
