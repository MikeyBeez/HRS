"""V21 Training: Full co-evolution from V20 Phase 1 checkpoint.

Restarts from V20's Phase 1 (step 20K) with everything unlocked:
- Per-head kernel MLPs with alpha at 0.5
- Per-head output scalars
- Cross-attention gate scalars
- Learnable categorization loss weight

Usage:
    python train_v21.py [--device cuda]
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


def get_lr(step, warmup, max_steps, base):
    if step < warmup: return base * step / warmup
    if step >= max_steps: return base * 0.1
    progress = (step - warmup) / (max_steps - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def train():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading WikiText-103...")
    uses_cat = cfg.uses_categorization()
    splits, _ = load_wikitext(
        cfg.training.dataset, cfg.model.max_seq_len,
        with_categories=uses_cat,
        n_categories=cfg.cross_attn_engram.num_categories if uses_cat else 50,
    )
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Build model and load Phase 1 checkpoint
    model = HRSTransformer(cfg).to(device)
    ckpt_path = Path("results/v20_bonsignore/phase1_end.pt")
    if not ckpt_path.exists():
        print("ERROR: V20 Phase 1 checkpoint not found!")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # strict=False: V21 has new params (output_scalars, gate_scalar) not in V20 checkpoint
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V20 Phase 1 checkpoint (step {ckpt['step']}, val_ppl {ckpt.get('val_ppl', '?')})")

    # V21 modifications: reset alphas to 0.5, unfreeze MLPs
    for block in model.blocks:
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            # Reset alpha to 0.5 (sigmoid(0) = 0.5)
            with torch.no_grad():
                block.attn.head_alphas.fill_(0.0)
            # Reset output scalars to 0.0 (softplus(0) ≈ 0.693)
            with torch.no_grad():
                block.attn.head_output_scalars.fill_(0.0)
            # Unfreeze everything
            block.attn.unfreeze_mlps()
    print("V21: Alphas reset to 0.5, output scalars reset, MLPs unfrozen")

    # Learnable categorization loss weight (use a 1-element module to keep it a leaf tensor)
    cat_weight_logit = nn.Parameter(torch.tensor(-2.3, device=device))  # softplus(-2.3) ≈ 0.1
    print(f"  Categorization loss weight: {F.softplus(cat_weight_logit).item():.4f}")

    # Loss
    loss_fn = CombinedHRSLoss(locality_cfg=cfg.locality if cfg.locality.enabled else None)

    # Optimizer with separate LR groups
    kernel_params = []
    proj_params = []
    scalar_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
        {"params": [cat_weight_logit], "lr": 1e-4, "weight_decay": 0.0},
        {"params": proj_params, "lr": 1e-5, "weight_decay": 0.1},
        {"params": other_params, "lr": 3e-5, "weight_decay": 0.1},
    ])

    print(f"  Kernel params: {sum(p.numel() for p in kernel_params):,}")
    print(f"  Scalar params: {sum(p.numel() for p in scalar_params):,}")
    print(f"  Projection params: {sum(p.numel() for p in proj_params):,}")
    print(f"  Other params: {sum(p.numel() for p in other_params):,}")

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output
    run_dir = Path("results/v21_learned_scalars")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    START_STEP = 20000
    MAX_STEP = 43000
    best_val = float("inf")
    no_improve = 0
    step = START_STEP
    train_iter = iter(loaders["train"])
    log_history = []
    t0 = time.time()

    accum_loss, accum_ce, accum_loc, accum_cat = 0.0, 0.0, 0.0, 0.0

    print(f"\nV21 Training: steps {START_STEP}–{MAX_STEP}")
    print(f"  Effective batch: {cfg.training.batch_size * cfg.training.grad_accum_steps}\n")

    model.train()
    while step < MAX_STEP:
        optimizer.zero_grad()

        for _ in range(cfg.training.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                batch = next(train_iter)

            if len(batch) == 3:
                x, y, cat_t = batch
                cat_t = cat_t.to(device)
            else:
                x, y = batch
                cat_t = None
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(x, step=step, collect_layer_reps=cfg.locality.enabled)

                loss_dict = loss_fn(
                    output.logits, y,
                    layer_representations=output.layer_representations if cfg.locality.enabled else None,
                    categorization_logits=output.categorization_logits if uses_cat else None,
                    categorization_targets=cat_t if uses_cat else None,
                    categorization_alpha=F.softplus(cat_weight_logit).item() if uses_cat else 0.0,
                )
                loss = loss_dict["loss"] / cfg.training.grad_accum_steps

            loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + [cat_weight_logit], cfg.training.max_grad_norm
        )

        # LR scheduling (cosine decay over Phase 2 range)
        base_lr = get_lr(step, START_STEP + 500, MAX_STEP, 3e-4)
        # Scale each group relative to base
        for i, pg in enumerate(optimizer.param_groups):
            scale = [1e-3/3e-4, 1e-3/3e-4, 1e-4/3e-4, 1e-5/3e-4, 3e-5/3e-4][i]
            pg["lr"] = base_lr * scale

        optimizer.step()

        # Engram buffer update
        if cfg.uses_cross_attn_engram() and step % cfg.cross_attn_engram.update_interval == 0:
            model.update_engram_buffer()

        accum_loss += loss_dict["loss"].item()
        accum_ce += loss_dict["ce_loss"].item()
        if "locality_loss" in loss_dict: accum_loc += loss_dict["locality_loss"].item()
        if "categorization_loss" in loss_dict: accum_cat += loss_dict["categorization_loss"].item()

        step += 1

        # Log every 100 steps
        if step % 100 == 0:
            n = 100
            ppl = math.exp(min(accum_ce / n, 20))
            elapsed = time.time() - t0
            gates = output.cross_attn_gate_values or []
            g_str = "/".join(f"{g:.3f}" for g in gates) if gates else "N/A"
            print(f"step {step:6d} | loss {accum_loss/n:.4f} | CE {accum_ce/n:.4f} | "
                  f"ppl {ppl:.1f} | cat {accum_cat/n:.4f} | gates [{g_str}] | {elapsed:.0f}s")
            accum_loss, accum_ce, accum_loc, accum_cat = 0, 0, 0, 0

        # Detailed diagnostics every 1000 steps
        if step % 1000 == 0:
            blk = model.blocks[0]
            if isinstance(blk.attn, PerHeadBonsignoreAttention):
                diag = blk.attn.get_diagnostics()
                a_str = " ".join(f"{a:.3f}" for a in diag["alphas"])
                t_str = " ".join(f"{t:.1f}" for t in diag["taus"])
                scales = F.softplus(blk.attn.head_output_scalars).detach()
                s_str = " ".join(f"{s:.3f}" for s in scales.tolist())
                print(f"  Alphas:  [{a_str}]")
                print(f"  Taus:    [{t_str}]")
                print(f"  Scales:  [{s_str}]")

            # Gate scalars
            for i, block in enumerate(model.blocks):
                if hasattr(block, 'cross_attn') and block.use_cross_attn_engram:
                    gs = F.softplus(block.cross_attn.gate_scalar).item()
                    gv = torch.sigmoid(block.cross_attn.gate_logit).item()
                    print(f"  Layer {block.layer_idx} gate: {gv:.3f} × {gs:.3f} = {gv*gs:.4f}")

            cat_w = F.softplus(cat_weight_logit).item()
            print(f"  Cat loss weight: {cat_w:.4f}")

            log_history.append({"step": step})

        # Evaluation every 1000 steps
        if step % 1000 == 0:
            model.eval()
            metrics = run_all_metrics(model, loaders["validation"], device, amp_dtype, max_batches=10)
            model.train()

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({**metrics, "step": step}, default=str) + "\n")

            if metrics["val_ppl"] < best_val:
                best_val = metrics["val_ppl"]
                no_improve = 0
                torch.save({
                    "step": step, "model_state_dict": model.state_dict(),
                    "val_ppl": best_val, "cat_weight_logit": cat_weight_logit.item(),
                }, run_dir / "best.pt")
                print(f"  ** val_ppl: {best_val:.2f} *best*\n")
            else:
                no_improve += 1000
                print(f"  val_ppl: {metrics['val_ppl']:.2f} (best={best_val:.2f}, no_improve={no_improve})\n")

            if no_improve >= 5000:
                print(f"Early stopping at step {step}")
                break

        # Checkpoint every 5000
        if step % 5000 == 0:
            torch.save({
                "step": step, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, run_dir / f"checkpoint_{step}.pt")
            existing = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]: old.unlink()

    # Final save
    torch.save({"step": step, "model_state_dict": model.state_dict()}, run_dir / "final.pt")
    with open(run_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nV21 complete. Best val_ppl: {best_val:.2f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    train()
