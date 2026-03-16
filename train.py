"""Training loop for HRS experiments with phased LR schedule.

Supports v1 (routed), v2 (attention->conv + PEER), and v7 (Memory MLP + V7Router) architectures.
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
    """Determine current training phase based on step count.

    v1: phases 1-5
    v2: phases 1-4
    """
    if not cfg.uses_phased_training():
        return 1

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

    if not cfg.is_v2():
        # v1 has phase 5
        return 5

    return 4  # v2 caps at phase 4


def get_phase_lr_multipliers(phase: int, cfg: ExperimentConfig) -> list:
    """Get LR multipliers for current phase.

    v1/v7 returns 10 floats: [backbone, gen_head, locality_head, router, conv, expert, attention, sink, engram, v7_router]
    v2 returns 5 floats: [backbone, gen_head, locality_head, peer, engram]
    """
    if not cfg.uses_phased_training():
        if cfg.is_v2():
            return [1.0] * 5
        return [1.0] * 10

    pc = cfg.phased

    if cfg.is_v2():
        mults = {
            1: pc.v2_phase1_lr_mult,
            2: pc.v2_phase2_lr_mult,
            3: pc.v2_phase3_lr_mult,
            4: pc.v2_phase4_lr_mult,
        }
        return mults.get(phase, pc.v2_phase1_lr_mult)
    else:
        mults = {
            1: pc.phase1_lr_mult,
            2: pc.phase2_lr_mult,
            3: pc.phase3_lr_mult,
            4: pc.phase4_lr_mult,
            5: pc.phase5_lr_mult,
        }
        return mults.get(phase, pc.phase1_lr_mult)


# v1/v7 parameter group index map
V1_PARAM_GROUP_INDEX = {
    "backbone": 0,
    "gen_head": 1,
    "locality_head": 2,
    "router": 3,
    "conv": 4,
    "expert": 5,
    "attention_tier": 6,
    "sink": 7,
    "engram": 8,
    "v7_router": 9,
}

# v2 parameter group index map
V2_PARAM_GROUP_INDEX = {
    "backbone": 0,
    "gen_head": 1,
    "locality_head": 2,
    "peer": 3,
    "engram": 4,
}


def train(cfg: ExperimentConfig, resume_path: str = None):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_v2 = cfg.is_v2()
    param_group_index = V2_PARAM_GROUP_INDEX if is_v2 else V1_PARAM_GROUP_INDEX

    print(f"Using device: {device}")
    print(f"Ablation: {cfg.training.ablation.value}")
    print(f"Architecture: {'v2 (attn->conv + PEER)' if is_v2 else 'v1 (routed)'}")

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

    if is_v2:
        print(f"  Attention layers: {cfg.n_attention_layers()}, Conv layers: {cfg.n_conv_layers()}")
    if cfg.uses_peer():
        print(f"  PEER: {cfg.peer.n_sub_keys}^2 = {cfg.peer.n_sub_keys**2:,} experts, "
              f"{cfg.peer.n_heads} heads x {cfg.peer.top_k} top-k = {cfg.peer.n_heads * cfg.peer.top_k} active/token")

    # Loss
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

    # v7: Memory MLP SGD optimizer (trained independently, not through Adam)
    memory_mlp_optimizer = None
    if cfg.uses_memory_mlp():
        # Move replay buffer to device (model.to() doesn't move plain tensors)
        rb = model.memory_mlp.replay_buffer
        rb.hidden_states = rb.hidden_states.to(device)
        rb.targets = rb.targets.to(device)
        rb.device = device
        memory_mlp_optimizer = torch.optim.SGD(
            model.memory_mlp.parameters(),
            lr=cfg.memory_mlp.lr,
        )
        print(f"  Memory MLP: {model.memory_mlp.param_count():,} params (trained via SGD, lr={cfg.memory_mlp.lr})")
        print(f"  V7Router: {model.v7_router.param_count():,} params (trained via Adam)")

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output directory
    run_dir = Path(cfg.training.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        "ablation": cfg.training.ablation.value,
        "model": cfg.model.__dict__,
        "locality": cfg.locality.__dict__,
        "training": {k: v.value if hasattr(v, 'value') else v
                     for k, v in cfg.training.__dict__.items()},
        "seed": cfg.seed,
        "param_counts": param_counts,
    }
    if is_v2:
        config_dict["peer"] = cfg.peer.__dict__
    else:
        config_dict["router"] = cfg.router.__dict__
        config_dict["tier"] = cfg.tier.__dict__
    if cfg.uses_peer() and not is_v2:
        config_dict["peer"] = cfg.peer.__dict__
    if cfg.uses_engrams():
        config_dict["engram"] = cfg.engram.__dict__
    if cfg.uses_phased_training():
        config_dict["phased"] = {k: v for k, v in cfg.phased.__dict__.items()
                                  if not k.startswith("phase") or not k.endswith("_lr_mult")}
    if cfg.uses_memory_mlp():
        config_dict["memory_mlp"] = cfg.memory_mlp.__dict__
    if cfg.uses_bdh() or cfg.uses_v10_control():
        config_dict["bdh"] = cfg.bdh.__dict__

    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Resume from checkpoint
    start_step = 0
    log_history = []
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        # v7: restore Memory MLP state
        if cfg.uses_memory_mlp() and "memory_mlp_state" in ckpt:
            mem_state = ckpt["memory_mlp_state"]
            model.memory_mlp.train_step_count = mem_state.get("train_step_count", 0)
            model.memory_mlp.expansion_history = mem_state.get("expansion_history", [])
            model.loss_gate_theta = mem_state.get("loss_gate_theta", 0.0)
            model._loss_gate_calibrated = mem_state.get("loss_gate_calibrated", False)
            if "replay_buffer" in mem_state:
                buf = mem_state["replay_buffer"]
                n = buf["size"]
                if n > 0:
                    model.memory_mlp.replay_buffer.hidden_states[:n] = buf["hidden_states"].to(device)
                    model.memory_mlp.replay_buffer.targets[:n] = buf["targets"].to(device)
                    model.memory_mlp.replay_buffer.size = n
                    model.memory_mlp.replay_buffer.write_idx = n % cfg.memory_mlp.replay_buffer_max
            if memory_mlp_optimizer is not None and "memory_mlp_optimizer" in ckpt:
                memory_mlp_optimizer.load_state_dict(ckpt["memory_mlp_optimizer"])
            print(f"  Restored Memory MLP state (train_steps={model.memory_mlp.train_step_count})")
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
    accum_gate_entropy = 0.0
    accum_gate_mean = 0.0
    accum_v7_router_base_w = 0.0
    accum_v7_mem_loss = 0.0
    accum_v7_flag_rate = 0.0
    accum_v7_router_ent = 0.0
    accum_bdh_focus = 0.0
    accum_bdh_sparsity = 0.0
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

            needs_layer_reps = cfg.locality.enabled

            uses_replacement = cfg.uses_engram_replacement()
            uses_gate = cfg.uses_remember_gate()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                output = model(
                    x, step=step,
                    collect_layer_reps=needs_layer_reps,
                    targets=y if (uses_replacement or uses_gate) else None,
                )

                # Gate values: v5 replacement gates or v6 remember gates
                gate_values = (
                    output.remember_gates if uses_gate
                    else (output.replacement_gates if uses_replacement else None)
                )

                # v7: compute blended logits for loss
                uses_memory_mlp = cfg.uses_memory_mlp()
                if uses_memory_mlp and output.memory_logits is not None:
                    base_probs = F.softmax(output.logits.float(), dim=-1)
                    mem_probs = F.softmax(output.memory_logits.float(), dim=-1)
                    w = output.v7_router_weights  # (B, T, 2)
                    blended_probs = w[:, :, 0:1] * base_probs + w[:, :, 1:2] * mem_probs
                    blended_probs = blended_probs.clamp(min=1e-10)
                    blended_logits = blended_probs.log()
                    logits_for_loss = blended_logits
                else:
                    logits_for_loss = output.logits

                # v9: use learnable scales for hub/entropy/recon losses
                uses_learnable = cfg.uses_learnable_loss_scaling()
                if uses_learnable:
                    ls = model.loss_scaler
                    _balance_w = ls.hub_scale
                    _entropy_w = ls.entropy_scale
                    _recon_w = ls.recon_scale
                else:
                    _balance_w = cfg.router.balance_loss_weight
                    _entropy_w = cfg.router.entropy_loss_weight if cfg.uses_router() else 0.0
                    _recon_w = cfg.engram.recon_loss_weight if cfg.uses_engrams() else 0.0

                loss_dict = loss_fn(
                    logits_for_loss, y,
                    layer_representations=output.layer_representations if needs_layer_reps else None,
                    routing_weights=output.routing_weights if cfg.uses_router() else None,
                    routing_balance_loss_val=output.routing_balance_loss if cfg.uses_router() else None,
                    balance_weight=_balance_w,
                    routing_entropy_loss_val=output.routing_entropy_loss if cfg.uses_router() else None,
                    entropy_weight=_entropy_w,
                    routing_flops_loss_val=output.routing_flops_loss if cfg.uses_router() else None,
                    flops_weight=cfg.router.flops_loss_weight if cfg.uses_router() else 0.0,
                    engram_recon_loss=output.engram_recon_loss if cfg.uses_engrams() else None,
                    recon_weight=_recon_w,
                    gate_values=gate_values,
                    gate_entropy_weight=cfg.engram.gate_entropy_weight if (uses_replacement or uses_gate) else 0.0,
                    v7_router_weights=output.v7_router_weights if uses_memory_mlp else None,
                    v7_router_entropy_weight=cfg.memory_mlp.router_entropy_weight if uses_memory_mlp else 0.0,
                )

                # v9/v10: add keep-alive penalty and placeholder losses
                total_loss = loss_dict["loss"]
                if uses_learnable and output.loss_scaler_penalty is not None:
                    total_loss = total_loss + output.loss_scaler_penalty

                # v10: add scaled placeholder losses
                if cfg.uses_v10_control() and output.placeholder_losses is not None:
                    ls = model.loss_scaler
                    pl = output.placeholder_losses
                    total_loss = total_loss + ls.hub_scale * pl['hub']
                    total_loss = total_loss + ls.entropy_scale * pl['entropy']
                    total_loss = total_loss + ls.recon_scale * pl['recon']

                loss = total_loss / cfg.training.grad_accum_steps

            loss.backward()

        # NaN detection
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
        if "gate_entropy_loss" in loss_dict:
            accum_gate_entropy += loss_dict["gate_entropy_loss"].item()
        if gate_values is not None:
            accum_gate_mean += gate_values.mean().item()
        if "v7_router_entropy" in loss_dict:
            accum_v7_router_ent += loss_dict["v7_router_entropy"].item()
        if output.v7_router_weights is not None:
            accum_v7_router_base_w += output.v7_router_weights[:, :, 0].mean().item()
        if output.bdh_focus_magnitude is not None:
            accum_bdh_focus += output.bdh_focus_magnitude
        if output.bdh_sparsity_level is not None:
            accum_bdh_sparsity += output.bdh_sparsity_level

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

        # Phase-based LR scheduling
        new_phase = get_phase(step, cfg)
        if new_phase != current_phase:
            print(f"\n*** Phase transition: {current_phase} -> {new_phase} at step {step} ***\n")

            # v1/v3/v4: Engram refinement at Phase 5 entry
            if new_phase == 5 and cfg.training.ablation in (
                AblationConfig.FULL_HRS_REFINED, AblationConfig.V3_FULL,
                AblationConfig.V4_FULL,
            ):
                model.apply_engram_refinement()

            current_phase = new_phase

        # Update LR per parameter group
        base_lr = get_lr(step, cfg.training.warmup_steps, cfg.training.max_steps, cfg.training.learning_rate)
        lr_mults = get_phase_lr_multipliers(current_phase, cfg)

        for i, (pg, name) in enumerate(zip(optimizer.param_groups, group_names)):
            if name == "loss_scaler":
                # v9: loss scaler params get 1/10 of base LR, no phase multiplier
                pg["lr"] = base_lr * cfg.bdh.loss_scaler_lr_mult
            else:
                mult_idx = param_group_index.get(name, 0)
                pg["lr"] = base_lr * lr_mults[mult_idx]

        optimizer.step()

        # v7: Train Memory MLP independently via SGD on surprising tokens
        if cfg.uses_memory_mlp() and memory_mlp_optimizer is not None:
            with torch.no_grad():
                B_cur, T_cur, V_cur = output.logits.shape
                per_token_loss = F.cross_entropy(
                    output.logits.float().reshape(-1, V_cur),
                    y.reshape(-1),
                    reduction='none',
                ).reshape(B_cur, T_cur)

            # Auto-calibrate theta from first 100 steps
            if not model._loss_gate_calibrated:
                model._loss_accumulator.append(per_token_loss.mean().item())
                if len(model._loss_accumulator) >= 100:
                    import statistics
                    mu = statistics.mean(model._loss_accumulator)
                    sigma = statistics.stdev(model._loss_accumulator)
                    model.loss_gate_theta = mu + sigma
                    model._loss_gate_calibrated = True
                    print(f"\n  [v7] Auto-calibrated theta = {model.loss_gate_theta:.4f} "
                          f"(mu={mu:.4f}, sigma={sigma:.4f})\n")

            if model._loss_gate_calibrated:
                flag_mask = per_token_loss > model.loss_gate_theta
                flag_rate = flag_mask.float().mean().item()
                accum_v7_flag_rate += flag_rate

                if flag_mask.any() and output.hidden_states is not None:
                    flagged_h = output.hidden_states.detach().float()[flag_mask]
                    flagged_t = y[flag_mask]
                    mem_loss = model.memory_mlp.train_step_with_replay(
                        flagged_h, flagged_t, memory_mlp_optimizer,
                    )
                    accum_v7_mem_loss += mem_loss
                    model.memory_mlp.check_expansion()

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

            if cfg.uses_engram_replacement():
                avg_gate_ent = accum_gate_entropy / n
                avg_gate_mean = accum_gate_mean / n
                entry["gate_entropy_loss"] = avg_gate_ent
                entry["gate_mean"] = avg_gate_mean
                entry["gate_threshold"] = model.engram_replacer.threshold.item()
                entry["gate_sharpness"] = model.engram_replacer.sharpness.item()

            if cfg.uses_remember_gate():
                avg_gate_ent = accum_gate_entropy / n
                avg_gate_mean = accum_gate_mean / n
                entry["gate_entropy_loss"] = avg_gate_ent
                entry["gate_mean"] = avg_gate_mean
                entry["remember_gate_bias"] = model.engram_injector.remember_gate.mlp[-1].bias.item()

            if cfg.uses_memory_mlp():
                avg_v7_base_w = accum_v7_router_base_w / n
                avg_v7_mem_loss = accum_v7_mem_loss / n
                avg_v7_flag_rate = accum_v7_flag_rate / n
                avg_v7_router_ent = accum_v7_router_ent / n
                entry["v7_router_base_w"] = avg_v7_base_w
                entry["v7_router_mem_w"] = 1.0 - avg_v7_base_w
                entry["v7_mem_loss"] = avg_v7_mem_loss
                entry["v7_flag_rate"] = avg_v7_flag_rate
                entry["v7_router_entropy"] = avg_v7_router_ent
                entry["v7_mem_train_steps"] = model.memory_mlp.train_step_count
                entry["v7_mem_d_hidden"] = model.memory_mlp.d_hidden
                entry["v7_replay_buf_size"] = model.memory_mlp.replay_buffer.size

            log_history.append(entry)

            extras = ""
            if cfg.locality.enabled:
                extras += f" | loc {avg_loc:.4f}"
            if cfg.uses_router():
                extras += f" | bal {avg_bal:.4f} | ent {avg_ent:.4f}"
            if cfg.uses_engrams():
                extras += f" | rec {avg_rec:.4f}"
            if cfg.uses_engram_replacement():
                extras += f" | gate {avg_gate_mean:.3f} | θ {model.engram_replacer.threshold.item():.2f}"
            if cfg.uses_remember_gate():
                extras += f" | gate {avg_gate_mean:.3f} | bias {model.engram_injector.remember_gate.mlp[-1].bias.item():.2f}"
            if cfg.uses_memory_mlp():
                extras += f" | w_base {avg_v7_base_w:.3f} | flag {avg_v7_flag_rate:.3f} | mem_loss {avg_v7_mem_loss:.3f}"
            if cfg.uses_bdh():
                avg_bdh_focus = accum_bdh_focus / n
                avg_bdh_sparsity = accum_bdh_sparsity / n
                entry["bdh_focus_magnitude"] = avg_bdh_focus
                entry["bdh_sparsity_level"] = avg_bdh_sparsity
                extras += f" | focus {avg_bdh_focus:.4f} | sparse {avg_bdh_sparsity:.2f}"
            if cfg.uses_learnable_loss_scaling():
                scales = model.loss_scaler.scale_dict()
                entry["hub_scale"] = scales["hub_scale"]
                entry["entropy_scale"] = scales["entropy_scale"]
                entry["recon_scale"] = scales["recon_scale"]
                extras += (f" | h_s {scales['hub_scale']:.4f}"
                          f" | e_s {scales['entropy_scale']:.5f}"
                          f" | r_s {scales['recon_scale']:.4f}")

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
            accum_gate_entropy = 0.0
            accum_gate_mean = 0.0
            accum_v7_router_base_w = 0.0
            accum_v7_mem_loss = 0.0
            accum_v7_flag_rate = 0.0
            accum_v7_router_ent = 0.0
            accum_bdh_focus = 0.0
            accum_bdh_sparsity = 0.0

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
                tier_fracs = metrics.get('tier_fractions_mean', [])
                print(f"  tier_fractions: {tier_fracs}")
                if len(tier_fracs) >= 3:
                    print(f"  sink_pct: {tier_fracs[2]*100:.1f}%")
            if "magnitude_ratio" in metrics:
                print(f"  magnitude_ratio: {metrics['magnitude_ratio']:.2f}")
            if "flops_ratio" in metrics:
                print(f"  flops_ratio: {metrics['flops_ratio']:.3f}")
            if "peer_n_total_experts" in metrics:
                print(f"  peer_experts: {metrics['peer_n_active_per_token']}/{metrics['peer_n_total_experts']} "
                      f"(sparsity {metrics['peer_sparsity']:.4f})")
            if cfg.uses_memory_mlp():
                print(f"  memory_mlp: d_hidden={model.memory_mlp.d_hidden}, "
                      f"train_steps={model.memory_mlp.train_step_count}, "
                      f"replay_buf={model.memory_mlp.replay_buffer.size}")
            if cfg.uses_learnable_loss_scaling():
                scales = model.loss_scaler.scale_dict()
                metrics["loss_scales"] = scales
                print(f"  loss_scales: hub={scales['hub_scale']:.4f} "
                      f"entropy={scales['entropy_scale']:.5f} "
                      f"recon={scales['recon_scale']:.4f}")
            print()

        # Save checkpoint (keep only last 2 to avoid filling disk)
        if step % cfg.training.save_interval == 0:
            ckpt_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "phase": current_phase,
            }
            # v7: save Memory MLP state separately for resumability
            if cfg.uses_memory_mlp():
                ckpt_data["memory_mlp_state"] = {
                    "train_step_count": model.memory_mlp.train_step_count,
                    "expansion_history": model.memory_mlp.expansion_history,
                    "loss_gate_theta": model.loss_gate_theta,
                    "loss_gate_calibrated": model._loss_gate_calibrated,
                    "replay_buffer": {
                        "hidden_states": model.memory_mlp.replay_buffer.hidden_states[:model.memory_mlp.replay_buffer.size].cpu(),
                        "targets": model.memory_mlp.replay_buffer.targets[:model.memory_mlp.replay_buffer.size].cpu(),
                        "size": model.memory_mlp.replay_buffer.size,
                    },
                }
                ckpt_data["memory_mlp_optimizer"] = memory_mlp_optimizer.state_dict()
            ckpt_path = run_dir / f"checkpoint_{step}.pt"
            torch.save(ckpt_data, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # Remove old checkpoints, keeping only last 2
            existing_ckpts = sorted(
                run_dir.glob("checkpoint_*.pt"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
            for old_ckpt in existing_ckpts[:-2]:
                old_ckpt.unlink()
                print(f"Removed old checkpoint {old_ckpt.name}")

    # Save final
    final_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": cfg,
    }
    if cfg.uses_memory_mlp():
        final_data["memory_mlp_state"] = {
            "train_step_count": model.memory_mlp.train_step_count,
            "expansion_history": model.memory_mlp.expansion_history,
            "loss_gate_theta": model.loss_gate_theta,
            "loss_gate_calibrated": model._loss_gate_calibrated,
            "replay_buffer": {
                "hidden_states": model.memory_mlp.replay_buffer.hidden_states[:model.memory_mlp.replay_buffer.size].cpu(),
                "targets": model.memory_mlp.replay_buffer.targets[:model.memory_mlp.replay_buffer.size].cpu(),
                "size": model.memory_mlp.replay_buffer.size,
            },
        }
    torch.save(final_data, run_dir / "final.pt")

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
                        help="FLOPs cost penalty weight (v1 only)")
    parser.add_argument("--trc-window", type=int, default=None,
                        help="Enable TRC with given window size (v1 only)")
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
