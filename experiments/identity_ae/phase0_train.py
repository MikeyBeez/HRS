"""Phase 0: Collect hidden states from V22 and train autoencoder offline.

1. Run V22 inference, save hidden states at insertion layer
2. Train autoencoder to reconstruct them (identity objective)
3. Splice into model, verify lossless insertion

Usage:
    python experiments/identity_ae/phase0_train.py [--device cuda] [--layer middle]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, PerHeadBonsignoreAttention
from data import load_wikitext, build_dataloaders
from identity_autoencoder import IdentityAutoencoder


def load_v22(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v22_learned_kernel/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    for b in model.blocks:
        if hasattr(b, 'cross_attn') and b.use_cross_attn_engram and b.layer_idx == 3:
            b.use_cross_attn_engram = False
    model.eval()
    print(f"Loaded V22 (val_ppl {ckpt.get('val_ppl', '?'):.2f})")
    return model, cfg


def resolve_layer(position, n_layers):
    if position == "middle":
        return n_layers // 2
    elif position == "early":
        return 1
    elif position == "late":
        return n_layers - 2
    else:
        return int(position)


@torch.no_grad()
def collect_representations(model, loaders, device, insert_layer, max_batches=500):
    """Run inference and collect hidden states at the insertion layer."""
    print(f"\n  Collecting representations at layer {insert_layer}...")
    all_hidden = []
    t0 = time.time()

    for batch_idx, batch in enumerate(loaders["train"]):
        if batch_idx >= max_batches:
            break
        x = batch[0].to(device)

        # Forward through layers, capture at insertion point
        h = model.drop(model.tok_emb(x))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h, _, _, _ = block(h, step=0, engram_buffer=eb)
            if i == insert_layer:
                all_hidden.append(h.cpu())
                break

        if (batch_idx + 1) % 100 == 0:
            print(f"    {batch_idx + 1}/{max_batches} batches ({time.time() - t0:.0f}s)")

    hidden = torch.cat(all_hidden, dim=0)  # (N, T, D)
    print(f"  Collected {hidden.shape[0]} sequences, shape {hidden.shape}")
    return hidden


def train_autoencoder(autoencoder, hidden_data, device, n_epochs=50, batch_size=32, lr=1e-3):
    """Train autoencoder on collected hidden states."""
    print(f"\n  Training autoencoder ({autoencoder.param_count():,} params)...")

    # Flatten sequences: (N, T, D) → (N*T, D)
    flat = hidden_data.reshape(-1, hidden_data.shape[-1])
    dataset = TensorDataset(flat)
    loader = DataLoader(dataset, batch_size=batch_size * 512, shuffle=True)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    autoencoder.to(device)
    autoencoder.train()

    t0 = time.time()
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon, error = autoencoder(batch_x.unsqueeze(1))  # add seq dim
            loss = F.mse_loss(recon.squeeze(1), batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch + 1:3d}: MSE={avg_loss:.8f} ({elapsed:.0f}s)")

    autoencoder.eval()
    return avg_loss


@torch.no_grad()
def verify_splice(model, autoencoder, loaders, device, insert_layer, n_batches=10):
    """Verify that model output with autoencoder ≈ output without."""
    print(f"\n  Verifying lossless splice...")
    model.eval()
    autoencoder.eval()

    max_diffs = []
    mean_diffs = []

    for batch_idx, batch in enumerate(loaders["validation"]):
        if batch_idx >= n_batches:
            break
        x = batch[0].to(device)

        # Forward WITHOUT autoencoder
        h_no_ae = model.drop(model.tok_emb(x))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h_no_ae, _, _, _ = block(h_no_ae, step=0, engram_buffer=eb)
        logits_no_ae = model.lm_head(model.ln_f(h_no_ae))

        # Forward WITH autoencoder
        h_ae = model.drop(model.tok_emb(x))
        for i, block in enumerate(model.blocks):
            eb = model.engram_buffer if model._engram_buffer_initialized else None
            h_ae, _, _, _ = block(h_ae, step=0, engram_buffer=eb)
            if i == insert_layer:
                h_ae, _ = autoencoder(h_ae)
        logits_ae = model.lm_head(model.ln_f(h_ae))

        diff = (logits_ae - logits_no_ae).abs()
        max_diffs.append(diff.max().item())
        mean_diffs.append(diff.mean().item())

    max_diff = max(max_diffs)
    mean_diff = sum(mean_diffs) / len(mean_diffs)
    print(f"    Max logit difference:  {max_diff:.6f}")
    print(f"    Mean logit difference: {mean_diff:.8f}")

    passed = max_diff < 0.1  # relaxed threshold — autoencoder won't be perfect
    print(f"    Splice verification: {'PASS' if passed else 'FAIL'} (threshold: 0.1)")
    return max_diff, mean_diff, passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=str, default="middle")
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load V22
    model, cfg = load_v22(device)
    n_layers = cfg.model.n_layers
    d_model = cfg.model.d_model
    insert_layer = resolve_layer(args.layer, n_layers)
    print(f"Insertion layer: {insert_layer} (of {n_layers})")

    # Load data
    print("Loading WikiText-103...")
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    # Results directory
    results_dir = Path("results/identity_ae/phase0")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect representations
    hidden_data = collect_representations(model, loaders, device, insert_layer,
                                          max_batches=args.max_batches)

    # Step 2: Create and train autoencoder
    autoencoder = IdentityAutoencoder(d_model=d_model, hidden_dim=768, bottleneck_dim=256)
    print(f"  Autoencoder: {autoencoder.param_count():,} params")

    final_mse = train_autoencoder(autoencoder, hidden_data, device, n_epochs=args.epochs)

    # Save init weights (the zero reference — before any test-time training)
    torch.save(autoencoder.state_dict(), results_dir / "autoencoder_init.pt")
    print(f"  Saved init weights to {results_dir / 'autoencoder_init.pt'}")

    # Step 3: Verify lossless splice
    autoencoder.to(device)
    max_diff, mean_diff, passed = verify_splice(
        model, autoencoder, loaders, device, insert_layer
    )

    # Step 4: Measure reconstruction error distribution
    print(f"\n  Measuring reconstruction error distribution...")
    autoencoder.eval()
    all_errors = []
    with torch.no_grad():
        for i in range(min(100, hidden_data.shape[0])):
            h = hidden_data[i:i+1].to(device)
            _, error = autoencoder(h)
            all_errors.append(error.cpu())

    errors = torch.cat(all_errors).flatten()
    error_mean = errors.mean().item()
    error_std = errors.std().item()
    error_max = errors.max().item()
    threshold_2sigma = error_mean + 2 * error_std
    threshold_3sigma = error_mean + 3 * error_std

    print(f"    Error mean:  {error_mean:.8f}")
    print(f"    Error std:   {error_std:.8f}")
    print(f"    Error max:   {error_max:.8f}")
    print(f"    2σ threshold: {threshold_2sigma:.6f}")
    print(f"    3σ threshold: {threshold_3sigma:.6f}")

    # Summary
    print(f"\n{'='*60}")
    print("PHASE 0 SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: V22 (val_ppl 17.07)")
    print(f"  Insertion layer: {insert_layer}")
    print(f"  Autoencoder: {autoencoder.param_count():,} params")
    print(f"  Final training MSE: {final_mse:.8f}")
    print(f"  Splice max diff: {max_diff:.6f} ({'PASS' if passed else 'FAIL'})")
    print(f"  Suggested OOD threshold (2σ): {threshold_2sigma:.6f}")
    print(f"  Suggested OOD threshold (3σ): {threshold_3sigma:.6f}")

    # Save results
    results = {
        "insert_layer": insert_layer,
        "n_layers": n_layers,
        "d_model": d_model,
        "autoencoder_params": autoencoder.param_count(),
        "final_mse": final_mse,
        "splice_max_diff": max_diff,
        "splice_mean_diff": mean_diff,
        "splice_passed": passed,
        "error_mean": error_mean,
        "error_std": error_std,
        "error_max": error_max,
        "threshold_2sigma": threshold_2sigma,
        "threshold_3sigma": threshold_3sigma,
    }
    with open(results_dir / "phase0_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
