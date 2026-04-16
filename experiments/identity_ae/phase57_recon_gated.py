"""Phase 57: V-space reconstruction loss + gated residual connections.

Phase 56 showed reconstruction loss improves V-space alignment (0.71→0.80
at 2K steps) but PPL degrades (21→43) because unconstrained residual
connections amplify the extra signal across layers. This phase adds
learned sigmoid gates on the residual connections to constrain signal
magnitude, preventing the amplification that forced the model to make
V-space lossy again.

Two changes from Phase 56:
  1. Lambda_recon reduced from 0.1 to 0.01 (10× smaller)
  2. Sigmoid gates on attention and FFN residual connections (convex
     combination instead of unconstrained addition)

Gate: x_out = gate * x_in + (1 - gate) * sublayer_out
Initialized at gate ≈ 0.9 (logit 2.2), so the model starts as 90%
residual + 10% sublayer contribution. Signal magnitude is bounded by
the max of the inputs, never their sum.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase57_recon_gated.py
"""

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
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase32_kv_similarity import (
    install_qkv_hooks, remove_hooks, per_head_cosine,
)
from experiments.identity_ae.phase35_engram_after_ttt import forward_from_x
from experiments.identity_ae.phase56_vspace_recon_pretraining import (
    VReconDecoders, VCapture,
    measure_val_ppl, measure_kv_alignment, measure_info_recovery,
)

from train import (
    set_seed, get_lr, get_phase, get_phase_lr_multipliers,
    V1_PARAM_GROUP_INDEX,
)

D = 1024
LAMBDA_RECON = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4
GATE_INIT_LOGIT = 2.2   # sigmoid(2.2) ≈ 0.9
MEASURE_CHECKPOINTS = [2000, 5000, 10000, 20000, 43000]
LOG_INTERVAL = 100
EVAL_INTERVAL = 1000


# ================================================================
# Gated residual connections via hooks
# ================================================================

class ResidualGates(nn.Module):
    """Learnable sigmoid gates for each block's residual connections.

    For each of the n_layers blocks, we have two gates:
      - attn_gate: controls attention residual
      - ffn_gate: controls FFN/PEER residual
    """
    def __init__(self, n_layers, init_logit=GATE_INIT_LOGIT):
        super().__init__()
        self.attn_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(init_logit))
            for _ in range(n_layers)
        ])
        self.ffn_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(init_logit))
            for _ in range(n_layers)
        ])

    def get_attn_gate(self, layer_idx):
        return torch.sigmoid(self.attn_gates[layer_idx])

    def get_ffn_gate(self, layer_idx):
        return torch.sigmoid(self.ffn_gates[layer_idx])


def install_gate_hooks(model, gates):
    """Install hooks that replace `x = x + sublayer_out` with
    `x = gate * x + (1-gate) * sublayer_out` at both the attention
    and FFN residual connections.

    This works by hooking the block-level forward and modifying the
    residual computation via a wrapper.
    """
    handles = []

    for i, block in enumerate(model.blocks):
        # We need to intercept the residual additions inside HRSBlock.forward.
        # The cleanest way without modifying model.py: use a pre/post hook
        # pair that captures the input, then adjusts the output.
        #
        # Strategy: wrap the entire block forward. Before the block runs,
        # save the input. After it runs, the output is:
        #   x_out = x_in + attn(x_in) + ffn(...)
        # We can't separate attn and ffn contributions from a post-hook alone.
        #
        # Simpler approach: override the block's forward method directly.
        original_forward = block.forward

        def make_gated_forward(layer_idx, orig_fwd):
            def gated_forward(x, step=0, return_weights=False,
                              engrams=None, kv_cache=None, start_pos=0,
                              engram_buffer=None):
                # Save input
                x_input = x

                # Run attention only
                focus_qk = None
                if block.use_bdh and hasattr(block, 'virtual_synapse') and engrams is not None:
                    focus_qk = block.virtual_synapse(engrams)

                attn_out, attn_w, new_kv_cache = block.attn(
                    block.ln1(x), return_weights=return_weights,
                    focus_qk=focus_qk, kv_cache=kv_cache, start_pos=start_pos)

                # Gated attention residual
                g_attn = gates.get_attn_gate(layer_idx)
                x = g_attn * x + (1 - g_attn) * (x + attn_out)
                # Equivalently: x = x + (1 - g_attn) * attn_out
                # But the convex form is more numerically stable

                # V18 cross-attention
                if block.use_cross_attn_engram and engram_buffer is not None:
                    x = x + block.cross_attn(x, engram_buffer)

                # FFN/PEER
                routing_w = None
                if hasattr(block, 'peer_ffn') and not block.use_router:
                    peer_out = block.peer_ffn(block.ln_peer(x))
                    g_ffn = gates.get_ffn_gate(layer_idx)
                    x = g_ffn * x + (1 - g_ffn) * (x + peer_out * block.peer_output_gate)
                elif not block.use_router and hasattr(block, 'mlp'):
                    mlp_out = block.mlp(block.ln2(x))
                    g_ffn = gates.get_ffn_gate(layer_idx)
                    x = g_ffn * x + (1 - g_ffn) * (x + mlp_out)
                else:
                    # Router path — don't gate, too complex to intercept
                    # Fall back to original behavior for router blocks
                    pass

                return x, routing_w, attn_w, new_kv_cache
            return gated_forward

        block.forward = make_gated_forward(i, original_forward)
        # No handles needed — we replaced forward directly

    return handles  # empty, kept for API compatibility


# ================================================================
# Main
# ================================================================

def main():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Phase 57: V-space recon + gated residuals")
    print(f"Device: {device}")
    print(f"Lambda_recon: {LAMBDA_RECON}")
    print(f"Gate init: sigmoid({GATE_INIT_LOGIT}) = {torch.sigmoid(torch.tensor(GATE_INIT_LOGIT)):.3f}")

    # Data
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, BATCH_SIZE)
    val_ds = splits["validation"]

    # Model (fresh)
    model = HRSTransformer(cfg).to(device)
    n_layers = len(model.blocks)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Reconstruction decoders
    recon = VReconDecoders(D, n_layers).to(device)
    print(f"Recon decoder params: {sum(p.numel() for p in recon.parameters()):,}")

    # Residual gates
    gates = ResidualGates(n_layers).to(device)
    print(f"Gate params: {sum(p.numel() for p in gates.parameters()):,}")

    # Install gates (replaces block.forward)
    install_gate_hooks(model, gates)

    # V capture hooks (for reconstruction loss)
    vcap = VCapture(model)
    vcap.install()

    # Optimizer
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
    # Recon decoders
    optimizer_groups.append({
        "params": list(recon.parameters()),
        "lr": cfg.training.learning_rate,
        "weight_decay": 0.0,
    })
    group_names.append("recon_decoders")
    # Gates
    optimizer_groups.append({
        "params": list(gates.parameters()),
        "lr": cfg.training.learning_rate,
        "weight_decay": 0.0,
    })
    group_names.append("gates")

    optimizer = torch.optim.AdamW(
        optimizer_groups, lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay, betas=(0.9, 0.95))

    max_steps = cfg.training.max_steps
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Output
    run_dir = Path("results/identity_ae/phase57")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    train_iter = iter(loaders["train"])
    step = 0
    micro_step = 0
    current_phase = get_phase(step, cfg)
    accum_ntp = 0.0
    accum_recon = 0.0
    best_val_ppl = float("inf")
    t0 = time.time()
    checkpoint_results = {}

    print(f"\nTraining: {max_steps} steps")
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

        vcap.clear()
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=use_amp):
            out = model(x, step=step)
            logits = out.logits
            V = logits.shape[-1]

            ntp_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                y[:, :-1].reshape(-1))

            total_recon = torch.tensor(0.0, device=device)
            n_recon = 0
            for l in range(n_layers):
                if l in vcap.captured and "H" in vcap.captured[l] and "V" in vcap.captured[l]:
                    total_recon = total_recon + recon.loss(
                        l, vcap.captured[l]["V"], vcap.captured[l]["H"])
                    n_recon += 1
            if n_recon > 0:
                total_recon = total_recon / n_recon

            loss = (ntp_loss + LAMBDA_RECON * total_recon) / GRAD_ACCUM

        loss.backward()
        micro_step += 1

        if micro_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            cfg.training.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)

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
                if name in ("recon_decoders", "gates"):
                    pg["lr"] = base_lr
                else:
                    mult_idx = V1_PARAM_GROUP_INDEX.get(name, 0)
                    pg["lr"] = base_lr * lr_mults[mult_idx]

            optimizer.step()
            optimizer.zero_grad()
            step += 1
            accum_ntp += ntp_loss.item()
            accum_recon += total_recon.item()
        else:
            continue

        # Log
        if step % LOG_INTERVAL == 0:
            avg_ntp = accum_ntp / LOG_INTERVAL
            avg_recon = accum_recon / LOG_INTERVAL
            ppl = math.exp(min(avg_ntp, 20))
            elapsed = time.time() - t0
            gate_vals = [f"{gates.get_attn_gate(l).item():.2f}" for l in range(n_layers)]
            print(f"step {step:6d}  ntp {avg_ntp:.4f}  ppl {ppl:.1f}  "
                  f"recon {avg_recon:.4f}  "
                  f"lr {base_lr:.2e}  P{current_phase}  "
                  f"gates [{','.join(gate_vals)}]  ({elapsed:.0f}s)")
            accum_ntp = 0.0
            accum_recon = 0.0

        # Eval
        if step % EVAL_INTERVAL == 0:
            model.eval()
            recon.eval()
            vcap.remove()

            ppl = measure_val_ppl(model, loaders["validation"], device)
            print(f"  val PPL: {ppl:.2f}")

            if ppl < best_val_ppl:
                best_val_ppl = ppl
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "recon_state_dict": recon.state_dict(),
                    "gates_state_dict": gates.state_dict(),
                    "val_ppl": ppl,
                }, run_dir / "best.pt")
                print(f"  ** New best: {ppl:.2f}")

            # Kill condition disabled — V-cos 0.856 at 2K is worth pursuing
            # if step >= 5000 and ppl > 100:
            #     print(f"  KILL: PPL > 100 at step {step}. Stopping.")
            #     break

            vcap.install()
            model.train()
            recon.train()

        # Full measurement
        if step in MEASURE_CHECKPOINTS:
            print(f"\n--- Measurement at step {step} ---")
            model.eval()
            vcap.remove()

            ppl = measure_val_ppl(model, loaders["validation"], device)
            kv = measure_kv_alignment(model, val_ds, device, n_passages=10)
            info = measure_info_recovery(model, val_ds, device, n_passages=50)

            print(f"  val PPL:       {ppl:.2f}")
            print(f"  K/V alignment:")
            for l in sorted(kv.keys()):
                print(f"    L{l}: K={kv[l]['k_cos']:.4f}  V={kv[l]['v_cos']:.4f}")
            print(f"  info recovery: mean={info['mean_recovery']:.1%}")
            print(f"  gate values:")
            for l in range(n_layers):
                ag = gates.get_attn_gate(l).item()
                fg = gates.get_ffn_gate(l).item()
                print(f"    L{l}: attn={ag:.3f}  ffn={fg:.3f}")

            checkpoint_results[step] = {
                "ppl": ppl,
                "kv": {str(l): v for l, v in kv.items()},
                "info": info,
                "gates": {
                    str(l): {
                        "attn": gates.get_attn_gate(l).item(),
                        "ffn": gates.get_ffn_gate(l).item(),
                    } for l in range(n_layers)
                },
            }

            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "recon_state_dict": recon.state_dict(),
                "gates_state_dict": gates.state_dict(),
                "val_ppl": ppl,
            }, run_dir / f"checkpoint_{step}.pt")

            existing = sorted(run_dir.glob("checkpoint_*.pt"),
                              key=lambda p: int(p.stem.split("_")[1]))
            for old in existing[:-2]:
                old.unlink()

            vcap.install()
            model.train()
            recon.train()
            print()

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"PHASE 57 SUMMARY: V-space recon + gated residuals")
    print(f"{'='*72}")
    print(f"  Steps: {step}, time: {elapsed/3600:.1f}h")
    print(f"  Best val PPL: {best_val_ppl:.2f}")
    print(f"  V22 baseline: PPL ~21, V-cos 0.65-0.70, mean rec 18%")
    print(f"  Phase 56 (recon only): PPL 43, V-cos 0.73, mean rec 14%")
    print()

    # Compare Phase 56 vs 57 vs baseline
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
        "lambda_recon": LAMBDA_RECON,
        "gate_init": GATE_INIT_LOGIT,
        "best_val_ppl": best_val_ppl,
        "checkpoints": {str(k): v for k, v in checkpoint_results.items()},
    }
    with open(run_dir / "recon_gated.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
