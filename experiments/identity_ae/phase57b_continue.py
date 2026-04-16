"""Phase 57b: Continue Phase 57 training until PPL plateaus.

Loads the Phase 57 checkpoint and continues training with a reduced LR
and early stopping based on validation PPL improvement.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase57b_continue.py
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
from experiments.identity_ae.phase56_vspace_recon_pretraining import (
    VReconDecoders, VCapture,
    measure_val_ppl, measure_kv_alignment, measure_info_recovery,
)
from experiments.identity_ae.phase57_recon_gated import (
    ResidualGates, install_gate_hooks,
)

from train import set_seed, get_lr, V1_PARAM_GROUP_INDEX

D = 1024
LAMBDA_RECON = 0.01
BATCH_SIZE = 2
GRAD_ACCUM = 4
LOG_INTERVAL = 100
EVAL_INTERVAL = 1000
# Reduced LR for continued training — 1/3 of original
CONT_LR = 1e-4
# Stop if val PPL hasn't improved in this many evals
PATIENCE = 5
# Maximum additional steps
MAX_EXTRA_STEPS = 30000
# Measurement checkpoints (relative to start of continuation)
MEASURE_EVERY = 5000


def main():
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    set_seed(cfg.seed + 1)  # different seed for continuation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path("results/identity_ae/phase57")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Find the latest checkpoint from Phase 57
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        # Try the last numbered checkpoint
        ckpts = sorted(run_dir.glob("checkpoint_*.pt"),
                       key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            ckpt_path = ckpts[-1]
        else:
            print("ERROR: no Phase 57 checkpoint found")
            return

    print(f"Phase 57b: Continue training until PPL plateaus")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    start_step = ckpt["step"]
    prev_ppl = ckpt["val_ppl"]
    print(f"  Resuming from step {start_step}, val PPL {prev_ppl:.2f}")

    # Data
    splits, tokenizer = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, BATCH_SIZE)
    val_ds = splits["validation"]

    # Model
    model = HRSTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_layers = len(model.blocks)

    # Recon decoders
    recon = VReconDecoders(D, n_layers).to(device)
    recon.load_state_dict(ckpt["recon_state_dict"])

    # Gates
    gates = ResidualGates(n_layers).to(device)
    gates.load_state_dict(ckpt["gates_state_dict"])
    install_gate_hooks(model, gates)

    # V capture
    vcap = VCapture(model)
    vcap.install()

    # Optimizer — fresh, reduced LR
    all_params = (list(model.parameters()) +
                  list(recon.parameters()) +
                  list(gates.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=CONT_LR,
                                   weight_decay=cfg.training.weight_decay,
                                   betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, MAX_EXTRA_STEPS)

    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print(f"  LR: {CONT_LR:.1e}")
    print(f"  Max extra steps: {MAX_EXTRA_STEPS}")
    print(f"  Patience: {PATIENCE} evals without improvement\n")

    # Training loop
    train_iter = iter(loaders["train"])
    step = start_step
    micro_step = 0
    accum_ntp = 0.0
    accum_recon = 0.0
    best_ppl = prev_ppl
    patience_counter = 0
    t0 = time.time()
    checkpoint_results = {}

    optimizer.zero_grad()
    model.train()
    recon.train()

    while step < start_step + MAX_EXTRA_STEPS:
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
            torch.nn.utils.clip_grad_norm_(all_params, cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
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
            lr = scheduler.get_last_lr()[0]
            gate_vals = [f"{gates.get_attn_gate(l).item():.2f}"
                         for l in range(n_layers)]
            print(f"step {step:6d}  ntp {avg_ntp:.4f}  ppl {ppl:.1f}  "
                  f"recon {avg_recon:.6f}  lr {lr:.2e}  "
                  f"gates [{','.join(gate_vals)}]  ({elapsed:.0f}s)")
            accum_ntp = 0.0
            accum_recon = 0.0

        # Eval + early stopping
        if step % EVAL_INTERVAL == 0:
            model.eval()
            recon.eval()
            vcap.remove()

            ppl = measure_val_ppl(model, loaders["validation"], device)
            print(f"  val PPL: {ppl:.2f}")

            if ppl < best_ppl - 0.1:  # meaningful improvement
                best_ppl = ppl
                patience_counter = 0
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "recon_state_dict": recon.state_dict(),
                    "gates_state_dict": gates.state_dict(),
                    "val_ppl": ppl,
                }, run_dir / "best.pt")
                print(f"  ** New best: {ppl:.2f}")
            else:
                patience_counter += 1
                print(f"  no improvement ({patience_counter}/{PATIENCE})")
                if patience_counter >= PATIENCE:
                    print(f"  STOPPING: {PATIENCE} evals without improvement")
                    vcap.install()
                    break

            vcap.install()
            model.train()
            recon.train()

        # Measurement
        extra_steps = step - start_step
        if extra_steps > 0 and extra_steps % MEASURE_EVERY == 0:
            print(f"\n--- Measurement at step {step} ---")
            model.eval()
            vcap.remove()

            ppl = measure_val_ppl(model, loaders["validation"], device)
            kv = measure_kv_alignment(model, val_ds, device, n_passages=10)
            info = measure_info_recovery(model, val_ds, device, n_passages=50)

            print(f"  val PPL:       {ppl:.2f}")
            for l in sorted(kv.keys()):
                print(f"    L{l}: K={kv[l]['k_cos']:.4f}  V={kv[l]['v_cos']:.4f}")
            print(f"  info recovery: mean={info['mean_recovery']:.1%}")
            for l in range(n_layers):
                ag = gates.get_attn_gate(l).item()
                fg = gates.get_ffn_gate(l).item()
                print(f"    L{l}: attn={ag:.3f}  ffn={fg:.3f}")

            checkpoint_results[step] = {
                "ppl": ppl,
                "kv": {str(l): v for l, v in kv.items()},
                "info": info,
            }

            vcap.install()
            model.train()
            recon.train()
            print()

    # Final measurement
    print(f"\n--- Final measurement ---")
    model.eval()
    vcap.remove()
    ppl = measure_val_ppl(model, loaders["validation"], device)
    kv = measure_kv_alignment(model, val_ds, device, n_passages=10)
    info = measure_info_recovery(model, val_ds, device, n_passages=50)
    checkpoint_results[step] = {
        "ppl": ppl, "kv": {str(l): v for l, v in kv.items()}, "info": info,
    }

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"PHASE 57b: Continued training complete")
    print(f"{'='*72}")
    print(f"  Start step: {start_step}, end step: {step}")
    print(f"  Extra steps: {step - start_step}, time: {elapsed/3600:.1f}h")
    print(f"  Best val PPL: {best_ppl:.2f}")
    print()
    print(f"  {'step':>6}  {'PPL':>7}  {'V-cos L5':>9}  {'mean rec':>9}")
    for s in sorted(checkpoint_results.keys()):
        r = checkpoint_results[s]
        kv5 = r["kv"].get("5", {})
        print(f"  {s:>6}  {r['ppl']:>7.2f}  "
              f"{kv5.get('v_cos', 0):>9.4f}  "
              f"{r['info']['mean_recovery']:>8.1%}")

    out = {
        "start_step": start_step, "end_step": step,
        "cont_lr": CONT_LR, "best_ppl": best_ppl,
        "checkpoints": {str(k): v for k, v in checkpoint_results.items()},
    }
    with open(run_dir / "continued.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
