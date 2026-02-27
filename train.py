"""Training loop for HRS experiments with phased LR schedule."""

import os
import time
import json
import math
from pathlib import Path
from collections import deque

import torch
import torch.nn.functional as F

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, build_dataloaders
from losses import CombinedHRSLoss
from metrics import run_all_metrics


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    if step >= max_steps:
        return base_lr * 0.1
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_phase(step: int, cfg: ExperimentConfig) -> int:
    """Determine current training phase (1-5) based on step count.

    Phase transitions are step-based when phase_N_steps > 0.
    """
    if not cfg.uses_phased_training():
        return 1  # Always phase 1 for non-phased configs

    pc = cfg.phased
    cumulative = 0

    cumulative += pc.phase1_steps
    if step < cumulative:
        return 1

    cumulative += pc.phase2_steps
    if step < cumulative:
        return 2

    cumulative += pc.phase3_steps
    if step < cumulative:
        return 3

    cumulative += pc.phase4_steps
    if step < cumulative:
        return 4

    return 5


def get_phase_lr_multipliers(phase: int, cfg: ExperimentConfig) -> list:
    """Get LR multipliers for current phase.

    Returns list of 9 floats:
    [backbone, gen_head, locality_head, router, conv, expert, attention, sink, engram]

    For non-phased configs, all components get full LR (1.0).
    """
    if not cfg.uses_phased_training():
        return [1.0] * 9

    pc = cfg.phased
    mults = {
        1: pc.phase1_lr_mult,
        2: pc.phase2_lr_mult,
        3: pc.phase3_lr_mult,
        4: pc.phase4_lr_mult,
        5: pc.phase5_lr_mult,
    }
    return mults.get(phase, pc.phase1_lr_mult)


# Map parameter group names to indices in the LR multiplier list
PARAM_GROUP_INDEX = {
    "backbone": 0,
    "gen_head": 1,
    "locality_head": 2,
    "router": 3,
    "conv": 4,
    "expert": 5,
    "attention_tier": 6,
    "sink": 7,
    "engram": 8,
}


def train(cfg: ExperimentConfig, resume_path: str = None):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Ablation: {cfg.training.ablation.value}")

    # Data
    print(f"Loading {cfg.training.dataset}...")
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Model
    model = HRSTransformer(cfg).to(device)
    param_counts = model.component_param_counts()
    print(f"Model parameters: {param_counts['total']:,}")
    for name, count in param_counts.items():
        if count > 0 and name != "total":
            print(f"  {name}: {count:,}")

    # Loss — use label smoothing for routed configs to prevent memorization
    label_smoothing = 0.1 if cfg.uses_router() else 0.0
    loss_fn = CombinedHRSLoss(
        locality_cfg=cfg.locality if cfg.locality.enabled else None,
        label_smoothing=label_smoothing,
    )

    # Optimizer: separate param groups for phased training
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

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output directory
    run_dir = Path(cfg.training.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "ablation": cfg.training.ablation.value,
            "model": cfg.model.__dict__,
            "router": cfg.router.__dict__,
            "tier": cfg.tier.__dict__,
            "engram": cfg.engram.__dict__,
            "phased": {k: v for k, v in cfg.phased.__dict__.items()
                       if not k.startswith("phase") or not k.endswith("_lr_mult")},
            "locality": cfg.locality.__dict__,
            "training": {k: v.value if hasattr(v, 'value') else v
                         for k, v in cfg.training.__dict__.items()},
            "seed": cfg.seed,
            "param_counts": param_counts,
        }, f, indent=2)

    # Resume from checkpoint
    start_step = 0
    log_history = []
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        print(f"Resumed from {resume_path} at step {start_step}")

    # Compile model if requested
    if cfg.training.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # Training loop
    train_loader = loaders["train"]
    train_iter = iter(train_loader)

    model.train()
    step = start_step
    accum_loss = 0.0
    accum_ce = 0.0
    accum_locality = 0.0
    accum_balance = 0.0
    accum_entropy = 0.0
    accum_flops = 0.0
    accum_recon = 0.0
    t0 = time.time()

    # Phase tracking
    current_phase = get_phase(step, cfg)
    val_loss_history = deque(maxlen=10)

    print(f"\nStarting training from step {step}, phase {current_phase}")
    print(f"Max steps: {cfg.training.max_steps}")
    print(f"Effective batch size: {cfg.training.batch_size * cfg.training.grad_accum_steps}")
    print()

    while step < cfg.training.max_steps:
        optimizer.zero_grad()

        for micro_step in range(cfg.training.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Determine what to collect
            needs_layer_reps = cfg.locality.enabled

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(
                    x, step=step,
                    collect_layer_reps=needs_layer_reps,
                )

                loss_dict = loss_fn(
                    output.logits, y,
                    layer_representations=output.layer_representations if needs_layer_reps else None,
                    routing_weights=output.routing_weights if cfg.uses_router() else None,
                    routing_balance_loss_val=output.routing_balance_loss if cfg.uses_router() else None,
                    balance_weight=cfg.router.balance_loss_weight,
                    routing_entropy_loss_val=output.routing_entropy_loss if cfg.uses_router() else None,
                    entropy_weight=cfg.router.entropy_loss_weight if cfg.uses_router() else 0.0,
                    routing_flops_loss_val=output.routing_flops_loss if cfg.uses_router() else None,
                    flops_weight=cfg.router.flops_loss_weight if cfg.uses_router() else 0.0,
                    engram_recon_loss=output.engram_recon_loss if cfg.uses_engrams() else None,
                    recon_weight=cfg.engram.recon_loss_weight if cfg.uses_engrams() else 0.0,
                )

                loss = loss_dict["loss"] / cfg.training.grad_accum_steps

            loss.backward()

        # NaN detection — bail early instead of wasting GPU hours
        loss_val = loss_dict["loss"].item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            nan_count = getattr(train, '_nan_count', 0) + 1
            train._nan_count = nan_count
            if nan_count >= 10:
                print(f"\n!!! Training diverged (NaN for {nan_count} consecutive steps). Stopping. !!!")
                break
        else:
            train._nan_count = 0

        # Accumulate metrics
        accum_loss += loss_val
        accum_ce += loss_dict["ce_loss"].item()
        if "locality_loss" in loss_dict:
            accum_locality += loss_dict["locality_loss"].item()
        if "balance_loss" in loss_dict:
            accum_balance += loss_dict["balance_loss"].item()
        if "entropy_loss" in loss_dict:
            accum_entropy += loss_dict["entropy_loss"].item()
        if "flops_loss" in loss_dict:
            accum_flops += loss_dict["flops_loss"].item()
        if "recon_loss" in loss_dict:
            accum_recon += loss_dict["recon_loss"].item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        # Phase-based LR scheduling
        new_phase = get_phase(step, cfg)
        if new_phase != current_phase:
            print(f"\n*** Phase transition: {current_phase} -> {new_phase} at step {step} ***\n")

            # Engram refinement: freeze encoder + reinit injector at Phase 5 entry
            if new_phase == 5 and cfg.training.ablation == AblationConfig.FULL_HRS_REFINED:
                model.apply_engram_refinement()

            current_phase = new_phase

        # Update LR per parameter group
        base_lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps, cfg.training.learning_rate)
        lr_mults = get_phase_lr_multipliers(current_phase, cfg)

        for i, (pg, name) in enumerate(zip(optimizer.param_groups, group_names)):
            mult_idx = PARAM_GROUP_INDEX.get(name, 0)
            pg["lr"] = base_lr * lr_mults[mult_idx]

        optimizer.step()
        step += 1

        # Logging
        if step % cfg.training.log_interval == 0:
            n = cfg.training.log_interval
            avg_loss = accum_loss / n
            avg_ce = accum_ce / n
            avg_loc = accum_locality / n
            avg_bal = accum_balance / n
            avg_ent = accum_entropy / n
            avg_flops = accum_flops / n
            avg_rec = accum_recon / n
            elapsed = time.time() - t0
            ppl = math.exp(min(avg_ce, 20))

            entry = {
                "step": step,
                "loss": avg_loss,
                "ce_loss": avg_ce,
                "ppl": ppl,
                "lr": base_lr,
                "phase": current_phase,
                "time": elapsed,
            }
            if cfg.locality.enabled:
                entry["locality_loss"] = avg_loc
            if cfg.uses_router():
                entry["balance_loss"] = avg_bal
                entry["entropy_loss"] = avg_ent
                entry["flops_cost"] = avg_flops
            if cfg.uses_engrams():
                entry["recon_loss"] = avg_rec

            log_history.append(entry)

            extras = ""
            if cfg.locality.enabled:
                extras += f" | loc {avg_loc:.4f}"
            if cfg.uses_router():
                extras += f" | bal {avg_bal:.4f} | ent {avg_ent:.4f}"
            if cfg.uses_engrams():
                extras += f" | rec {avg_rec:.4f}"

            print(
                f"step {step:6d} | loss {avg_loss:.4f} | CE {avg_ce:.4f} | "
                f"ppl {ppl:.1f} | lr {base_lr:.2e} | P{current_phase}{extras} | "
                f"{elapsed:.1f}s"
            )

            accum_loss = 0.0
            accum_ce = 0.0
            accum_locality = 0.0
            accum_balance = 0.0
            accum_entropy = 0.0
            accum_flops = 0.0
            accum_recon = 0.0

        # Evaluation
        if step % cfg.training.eval_interval == 0:
            print(f"\n--- Evaluation at step {step} ---")
            model.eval()
            metrics = run_all_metrics(model, loaders["validation"], device, amp_dtype, max_batches=10)
            model.train()

            metrics["step"] = step
            metrics["phase"] = current_phase

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics, default=str) + "\n")

            val_loss_history.append(metrics["val_ppl"])

            print(f"  val_ppl: {metrics.get('val_ppl', 'N/A'):.1f}")
            print(f"  eff_rank_mean: {metrics.get('effective_rank_mean', 'N/A'):.1f}")
            print(f"  cosine_sim_mean: {metrics.get('cosine_sim_mean', 'N/A'):.4f}")
            if "routing_entropy_mean" in metrics:
                print(f"  routing_entropy: {metrics['routing_entropy_mean']:.3f}")
                print(f"  tier_fractions: {metrics.get('tier_fractions_mean', [])}")
            if "magnitude_ratio" in metrics:
                print(f"  magnitude_ratio: {metrics['magnitude_ratio']:.2f}")
            if "flops_ratio" in metrics:
                print(f"  flops_ratio: {metrics['flops_ratio']:.3f}")
            print()

        # Save checkpoint
        if step % cfg.training.save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "phase": current_phase,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Save final
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }, run_dir / "final.pt")

    with open(run_dir / "log_history.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\nTraining complete. Results saved to {run_dir}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HRS model")
    parser.add_argument("--ablation", type=str, default="dense_baseline",
                        choices=[a.value for a in AblationConfig])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--flops-weight", type=float, default=None,
                        help="FLOPs cost penalty weight (0=off, try 0.1-1.0)")
    parser.add_argument("--trc-window", type=int, default=None,
                        help="Enable TRC with given window size (e.g. 8)")
    args = parser.parse_args()

    ablation = AblationConfig(args.ablation)
    cfg = ExperimentConfig.from_ablation(ablation)
    cfg.seed = args.seed
    cfg.training.output_dir = args.output_dir

    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.eval_interval is not None:
        cfg.training.eval_interval = args.eval_interval
    if args.save_interval is not None:
        cfg.training.save_interval = args.save_interval
    if args.compile:
        cfg.training.compile_model = True
    if args.flops_weight is not None:
        cfg.router.flops_loss_weight = args.flops_weight
    if args.trc_window is not None:
        cfg.router.trc_enabled = True
        cfg.router.trc_window = args.trc_window

    # Resolve resume
    resume_path = None
    if args.resume == "auto":
        run_dir = Path(args.output_dir) / cfg.run_name
        ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            resume_path = str(ckpts[-1])
            print(f"Auto-detected checkpoint: {resume_path}")
        else:
            print("No checkpoints found, starting fresh")
    elif args.resume is not None:
        resume_path = args.resume

    train(cfg, resume_path=resume_path)
