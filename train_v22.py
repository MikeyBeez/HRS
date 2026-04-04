"""V22 Training: Learned kernels from V20's mature checkpoint.

No alpha reset. MLPs calibrated to match current exponential, then
co-evolve. Layer 3 cross-attention removed. Per-head output scalars added.

Usage:
    python train_v22.py [--device cuda]
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


def get_lr(step, warmup_end, max_steps, base):
    if step < warmup_end: return base * (step - 43000) / (warmup_end - 43000)
    if step >= max_steps: return base * 0.1
    progress = (step - warmup_end) / (max_steps - warmup_end)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def calibrate_mlps(model, loaders, device, n_batches=10):
    """Calibrate per-head MLPs to match current exponential scores."""
    print("  Calibrating per-head MLPs to match exponential...")
    model.eval()

    for block in model.blocks:
        if not isinstance(block.attn, PerHeadBonsignoreAttention):
            continue

        attn = block.attn
        # Collect q, k pairs from this layer
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
        for batch_idx, batch in enumerate(loaders["train"]):
            if batch_idx >= n_batches:
                break
            x = batch[0].to(device)
            with torch.no_grad():
                model(x, step=0)
            # Sample subset
            q, k = captured['q'], captured['k']
            idx = torch.randperm(q.shape[2])[:32]
            all_q.append(q[:, :, idx, :])
            all_k.append(k[:, :, :32, :])
        handle.remove()

        # For each head, train its MLP to match exponential scores
        for h in range(attn.n_heads):
            tau_h = attn.log_tau[h].exp().item()
            mlp = attn.head_mlps[h]
            opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

            for epoch in range(200):
                total_loss = 0
                for q_batch, k_batch in zip(all_q, all_k):
                    q_h = q_batch[:, h]  # (B, Sq, D)
                    k_h = k_batch[:, h]  # (B, Sk, D)
                    # Exponential scores
                    q_sq = (q_h ** 2).sum(-1, keepdim=True)
                    k_sq = (k_h ** 2).sum(-1, keepdim=True)
                    dot = q_h @ k_h.transpose(-2, -1)
                    dist = q_sq + k_sq.transpose(-2, -1) - 2 * dot
                    exp_scores = torch.exp(-dist / tau_h)  # target

                    # MLP output
                    flat = exp_scores.reshape(-1, 1)
                    mlp_out = mlp(flat).reshape(exp_scores.shape)

                    loss = F.mse_loss(mlp_out, exp_scores)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()

            if h == 0:
                print(f"    Layer {block.layer_idx}, Head 0: final MSE={total_loss/len(all_q):.6f}")

    print("  Calibration complete")


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

    # Build model and load V20 final checkpoint
    model = HRSTransformer(cfg).to(device)
    ckpt_path = Path("results/v20_bonsignore/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V20 best checkpoint (step {ckpt.get('step', '?')}, val_ppl {ckpt.get('val_ppl', '?'):.2f})")

    # V22 modification: disable Layer 3 cross-attention
    for block in model.blocks:
        if hasattr(block, 'cross_attn') and block.use_cross_attn_engram:
            if block.layer_idx == 3:
                block.use_cross_attn_engram = False
                print(f"  Disabled Layer 3 cross-attention")

    # Print current state
    for block in model.blocks:
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            diag = block.attn.get_diagnostics()
            print(f"  Layer {block.layer_idx}: alphas={['%.3f'%a for a in diag['alphas']]}")
            print(f"    taus={['%.1f'%t for t in diag['taus']]}")
            break

    # Calibrate MLPs before unfreezing
    calibrate_mlps(model, loaders, device)

    # Unfreeze MLPs (alpha stays at V20's learned value ~0.72)
    for block in model.blocks:
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            block.attn.unfreeze_mlps()
    print("  MLPs unfrozen (alpha preserved from V20)")

    # Loss
    loss_fn = CombinedHRSLoss(locality_cfg=cfg.locality if cfg.locality.enabled else None)

    # Optimizer with V22 learning rates
    kernel_params = []
    temp_params = []
    scalar_params = []
    proj_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head_mlps' in name or 'head_alphas' in name:
            kernel_params.append(param)
        elif 'log_tau' in name:
            temp_params.append(param)
        elif 'head_output_scalars' in name or 'gate_scalar' in name:
            scalar_params.append(param)
        elif 'qkv' in name or 'out_proj' in name:
            proj_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": kernel_params, "lr": 1e-3, "weight_decay": 0.01},
        {"params": temp_params, "lr": 1e-4, "weight_decay": 0.0},
        {"params": scalar_params, "lr": 1e-3, "weight_decay": 0.0},
        {"params": proj_params, "lr": 1e-5, "weight_decay": 0.1},
        {"params": other_params, "lr": 3e-5, "weight_decay": 0.1},
    ])

    print(f"  Kernel MLP params: {sum(p.numel() for p in kernel_params):,}")
    print(f"  Temperature params: {sum(p.numel() for p in temp_params):,}")
    print(f"  Scalar params: {sum(p.numel() for p in scalar_params):,}")

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output
    run_dir = Path("results/v22_learned_kernel")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    START_STEP = 43000
    MAX_STEP = 53000
    WARMUP_END = 43500
    best_val = 17.34  # V20's best — we need to beat this
    step = START_STEP
    train_iter = iter(loaders["train"])
    t0 = time.time()

    accum_loss, accum_ce, accum_cat = 0.0, 0.0, 0.0

    print(f"\nV22 Training: steps {START_STEP}–{MAX_STEP}")
    print(f"  Target: beat V20's val_ppl {best_val:.2f}\n")

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
                    categorization_alpha=0.1 if uses_cat else 0.0,
                )
                loss = loss_dict["loss"] / cfg.training.grad_accum_steps

            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR scheduling
        base_lr = get_lr(step, WARMUP_END, MAX_STEP, 3e-4)
        scales = [1e-3/3e-4, 1e-4/3e-4, 1e-3/3e-4, 1e-5/3e-4, 3e-5/3e-4]
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = base_lr * scales[i]

        optimizer.step()

        # Engram buffer update
        if cfg.uses_cross_attn_engram() and step % cfg.cross_attn_engram.update_interval == 0:
            model.update_engram_buffer()

        accum_loss += loss_dict["loss"].item()
        accum_ce += loss_dict["ce_loss"].item()
        if "categorization_loss" in loss_dict:
            accum_cat += loss_dict["categorization_loss"].item()

        step += 1

        if step % 100 == 0:
            n = 100
            ppl = math.exp(min(accum_ce / n, 20))
            elapsed = time.time() - t0
            gates = output.cross_attn_gate_values or []
            g_str = "/".join(f"{g:.3f}" for g in gates) if gates else "N/A"
            print(f"step {step:6d} | CE {accum_ce/n:.4f} | ppl {ppl:.1f} | "
                  f"cat {accum_cat/n:.4f} | gates [{g_str}] | {elapsed:.0f}s")
            accum_loss, accum_ce, accum_cat = 0, 0, 0

        # Diagnostics every 1000 steps
        if step % 1000 == 0:
            blk = model.blocks[0]
            diag = blk.attn.get_diagnostics()
            a_str = " ".join(f"{a:.3f}" for a in diag["alphas"])
            t_str = " ".join(f"{t:.1f}" for t in diag["taus"])
            scales_val = F.softplus(blk.attn.head_output_scalars).detach()
            s_str = " ".join(f"{s:.3f}" for s in scales_val.tolist())
            print(f"  Alphas:  [{a_str}]")
            print(f"  Taus:    [{t_str}]")
            print(f"  Scales:  [{s_str}]")

            for block in model.blocks:
                if hasattr(block, 'cross_attn') and block.use_cross_attn_engram:
                    gs = F.softplus(block.cross_attn.gate_scalar).item()
                    gv = torch.sigmoid(block.cross_attn.gate_logit).item()
                    print(f"  Layer {block.layer_idx} gate: {gv:.3f} × {gs:.3f} = {gv*gs:.4f}")

        # Evaluation every 1000 steps
        if step % 1000 == 0:
            model.eval()
            metrics = run_all_metrics(model, loaders["validation"], device, amp_dtype, max_batches=10)
            model.train()

            with open(run_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({**metrics, "step": step}, default=str) + "\n")

            if metrics["val_ppl"] < best_val:
                best_val = metrics["val_ppl"]
                torch.save({
                    "step": step, "model_state_dict": model.state_dict(),
                    "val_ppl": best_val,
                }, run_dir / "best.pt")
                print(f"  ** val_ppl: {best_val:.2f} *BEATS V20*\n")
            else:
                print(f"  val_ppl: {metrics['val_ppl']:.2f} (V20={17.34}, best={best_val:.2f})\n")

    # Final save
    torch.save({"step": step, "model_state_dict": model.state_dict()}, run_dir / "final.pt")

    print(f"\nV22 complete. Best val_ppl: {best_val:.2f}")
    if best_val < 17.34:
        print(f"  BEAT V20 by {17.34 - best_val:.2f} points!")
    else:
        print(f"  Did not beat V20 (delta: {best_val - 17.34:+.2f})")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    train()
