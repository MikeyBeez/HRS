"""Learned Kernel with Phased Training — Experiment A.

Resume V19 training with a small learned MLP replacing the fixed
exponential kernel. The MLP is initialized to approximate exponential
behavior, then co-evolves with the projection matrices.

Usage:
    python exp_learned_kernel.py [--n-steps 5000] [--device cuda]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, HRSOutput
from data import load_wikitext, build_dataloaders


# ============================================================
# Learned Kernel Module
# ============================================================

class LearnedKernel(nn.Module):
    """Learned attention scoring via decomposed nonlinear projections.

    phi_q(q) and phi_k(k) are independent nonlinear projections.
    Score = phi_q(q) @ phi_k(k).T / sqrt(R).

    This is memory-efficient (no 5D tensor) while still learning
    nonlinear interactions through the composition of projections.
    Can approximate exponential, dot product, or novel functions.
    """

    def __init__(self, d_key, hidden_dim=64):
        super().__init__()
        self.d_key = d_key
        R = hidden_dim

        # Nonlinear projections for q and k
        self.phi_q = nn.Sequential(
            nn.Linear(d_key, R),
            nn.GELU(),
            nn.Linear(R, R),
        )
        self.phi_k = nn.Sequential(
            nn.Linear(d_key, R),
            nn.GELU(),
            nn.Linear(R, R),
        )
        self.scale = 1.0 / math.sqrt(R)

    def forward(self, q, k):
        """Compute pairwise scores.

        Args:
            q: (B, H, Sq, D)
            k: (B, H, Sk, D)

        Returns:
            scores: (B, H, Sq, Sk)
        """
        q_proj = self.phi_q(q)  # (B, H, Sq, R)
        k_proj = self.phi_k(k)  # (B, H, Sk, R)
        return (q_proj @ k_proj.transpose(-2, -1)) * self.scale


def init_kernel_from_exponential(kernel, model, data_loader, device, n_batches=5):
    """Initialize the learned kernel to approximate exponential scoring.

    Collect q,k pairs from V19, compute exponential scores, train MLP to match.
    """
    print("  Initializing learned kernel from exponential...")
    kernel.train()
    optimizer = torch.optim.Adam(kernel.parameters(), lr=1e-3)

    # Collect q,k pairs and exponential scores
    model.eval()
    all_q, all_k, all_exp_scores = [], [], []

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= n_batches:
            break
        x = batch[0].to(device)

        # Hook into first attention layer to get q,k after projection
        captured = {}
        def hook_fn(module, input, output):
            # The attention module processes qkv internally
            # We need to re-derive q,k from the input
            h = input[0]  # pre-norm input to attention
            B, T, C = h.shape
            n_heads = module.n_heads
            head_dim = module.head_dim
            qkv = module.qkv(h).reshape(B, T, 3, n_heads, head_dim)
            q, k, _ = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            captured['q'] = q.detach()
            captured['k'] = k.detach()

        handle = model.blocks[0].attn.register_forward_hook(hook_fn)
        with torch.no_grad():
            model(x, step=0)
        handle.remove()

        q, k = captured['q'], captured['k']
        # Sample subset to fit in memory
        B, H, T, D = q.shape
        idx = torch.randint(0, T, (min(64, T),))
        q_sub = q[:, :, idx, :]  # (B, H, 64, D)

        # Exponential scores for this subset
        temperature = float(D)
        q_sq = (q_sub ** 2).sum(dim=-1, keepdim=True)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)
        dot = q_sub @ k.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot
        exp_scores = -distances / temperature

        all_q.append(q_sub)
        all_k.append(k)
        all_exp_scores.append(exp_scores)

    # Train kernel to match exponential scores
    for epoch in range(500):
        total_loss = 0
        for q_sub, k_full, target in zip(all_q, all_k, all_exp_scores):
            pred = kernel(q_sub, k_full)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            avg = total_loss / len(all_q)
            print(f"    Calibration epoch {epoch+1}: MSE={avg:.6f}")

    kernel.eval()
    # Final R² check
    with torch.no_grad():
        pred = kernel(all_q[0], all_k[0])
        target = all_exp_scores[0]
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"    Final R² vs exponential: {r2.item():.4f}")


# ============================================================
# Modified forward pass with learned kernel
# ============================================================

def forward_with_learned_kernel(model, idx, learned_kernels, step=0):
    """Forward pass replacing exponential attention with learned kernel.

    Args:
        model: V19 model
        idx: (B, T) input token ids
        learned_kernels: list of LearnedKernel modules (one per layer)

    Returns:
        logits: (B, T, V)
    """
    x = model.drop(model.tok_emb(idx))
    B, T, D = x.shape
    device = x.device

    for i, block in enumerate(model.blocks):
        # Pre-norm
        h = block.ln1(x)

        # Manual attention with learned kernel
        attn = block.attn
        qkv = attn.qkv(h).reshape(B, T, 3, attn.n_heads, attn.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # RoPE
        from tiers import rotate_half
        cos, sin = attn.rope(T)
        cos = cos[:T].unsqueeze(0).unsqueeze(0)
        sin = sin[:T].unsqueeze(0).unsqueeze(0)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Learned kernel scores
        scores = learned_kernels[i](q, k)  # (B, H, T, T)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn.attn_dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, D)
        out = attn.resid_dropout(attn.out_proj(out))
        x = x + out

        # Cross-attention engram (if applicable)
        if block.use_cross_attn_engram and model._engram_buffer_initialized:
            x = x + block.cross_attn(x, model.engram_buffer)

        # FFN (PEER)
        h2 = block.ln2(x)
        if hasattr(block, 'peer_ffn'):
            peer_out = block.peer_ffn(h2)
            if isinstance(peer_out, tuple):
                peer_out = peer_out[0]
            x = x + block.peer_output_gate * block.ln_peer(peer_out)
        elif hasattr(block, 'mlp'):
            x = x + block.mlp(h2)

    x = model.ln_f(x)
    logits = model.lm_head(x)
    return logits


# ============================================================
# Training loop
# ============================================================

def train_experiment_a(n_steps=5000, proj_lr=1e-5, kernel_lr=1e-3, device_str="cuda"):
    """Experiment A: Fixed learning rates, learned kernel on V19."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Load V19
    cfg = ExperimentConfig.from_ablation(AblationConfig.V19_EXP_KERNEL)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v19_exp_kernel/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V19 (step {ckpt.get('step', '?')}, val_ppl {ckpt.get('val_ppl', '?'):.2f})")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    n_layers = cfg.model.n_layers
    head_dim = cfg.model.d_model // cfg.model.n_heads

    # Load data
    print("Loading WikiText-103...")
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, cfg.training.batch_size)

    # Create learned kernels (one per layer)
    learned_kernels = nn.ModuleList([
        LearnedKernel(head_dim, hidden_dim=64)
        for _ in range(n_layers)
    ]).to(device)

    kernel_params = sum(p.numel() for p in learned_kernels.parameters())
    print(f"Learned kernel params: {kernel_params:,} ({kernel_params / 1e6:.2f}M)")

    # Initialize kernels to approximate exponential
    init_kernel_from_exponential(learned_kernels[0], model, loaders["train"], device)
    # Copy initialized weights to all layers
    state = learned_kernels[0].state_dict()
    for i in range(1, n_layers):
        learned_kernels[i].load_state_dict(state)
    print("  Copied initialization to all layers")

    # Freeze everything except attention projections and learned kernels
    # This dramatically reduces optimizer state memory
    for name, param in model.named_parameters():
        if 'qkv' in name or 'out_proj' in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    proj_params = [p for p in model.parameters() if p.requires_grad]
    trainable_model = sum(p.numel() for p in proj_params)
    print(f"  Trainable model params (projections only): {trainable_model:,}")

    optimizer = torch.optim.AdamW([
        {"params": proj_params, "lr": proj_lr, "weight_decay": 0.1},
        {"params": learned_kernels.parameters(), "lr": kernel_lr, "weight_decay": 0.01},
    ])

    # Mixed precision
    use_amp = cfg.training.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Training
    results_dir = Path("results/v19_learned_kernel")
    results_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    learned_kernels.train()
    train_iter = iter(loaders["train"])
    losses = []
    val_ppls = []
    best_val = float('inf')
    t0 = time.time()

    print(f"\nTraining: {n_steps} steps, proj_lr={proj_lr}, kernel_lr={kernel_lr}")

    for step in range(n_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        x, y = batch[0].to(device), batch[1].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = forward_with_learned_kernel(model, x, learned_kernels, step=step)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(learned_kernels.parameters()), 1.0
        )
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / 100
            elapsed = time.time() - t0
            ppl = math.exp(min(avg, 20))
            print(f"  step {step+1:5d}: loss={avg:.4f} ppl={ppl:.1f} ({elapsed:.0f}s)")

        if (step + 1) % 500 == 0:
            # Validation
            model.eval()
            learned_kernels.eval()
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for vb in loaders["validation"]:
                    vx, vy = vb[0].to(device), vb[1].to(device)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        vl = forward_with_learned_kernel(model, vx, learned_kernels)
                    B, T, V = vl.shape
                    val_loss += F.cross_entropy(vl.reshape(B*T, V), vy.reshape(B*T)).item()
                    n_val += 1
                    if n_val >= 10:
                        break
            val_loss /= n_val
            val_ppl = math.exp(min(val_loss, 20))
            val_ppls.append((step + 1, val_ppl))

            marker = ""
            if val_ppl < best_val:
                best_val = val_ppl
                torch.save({
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "kernel_state_dict": learned_kernels.state_dict(),
                    "val_ppl": val_ppl,
                }, results_dir / "best.pt")
                marker = " *best*"

            print(f"  → val_ppl={val_ppl:.2f} (best={best_val:.2f}){marker}")
            model.train()
            learned_kernels.train()

    # ============================================================
    # Kernel shape analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("KERNEL SHAPE ANALYSIS")
    print(f"{'='*60}")

    model.eval()
    learned_kernels.eval()

    # Collect q,k pairs and compute scores under different kernels
    captured = {}
    def hook_fn(module, input, output):
        h = input[0]
        B, T, C = h.shape
        qkv = module.qkv(h).reshape(B, T, 3, module.n_heads, module.head_dim)
        q, k, _ = qkv.unbind(dim=2)
        captured['q'] = q.transpose(1, 2).detach()
        captured['k'] = k.transpose(1, 2).detach()

    handle = model.blocks[0].attn.register_forward_hook(hook_fn)
    batch = next(iter(loaders["validation"]))
    with torch.no_grad():
        model(batch[0].to(device), step=0)
    handle.remove()

    q, k = captured['q'], captured['k']
    B, H, T, D = q.shape
    # Sample pairs
    idx_q = torch.randint(0, T, (100,))
    idx_k = torch.randint(0, T, (100,))
    q_sample = q[:1, :1, idx_q, :]  # (1, 1, 100, D)
    k_sample = k[:1, :1, idx_k, :]  # (1, 1, 100, D)

    with torch.no_grad():
        # Learned scores
        learned_scores = learned_kernels[0](q_sample, k_sample).squeeze(0).squeeze(0)  # (100, 100)

        # Dot product scores
        dot_scores = (q_sample @ k_sample.transpose(-2, -1)).squeeze(0).squeeze(0) / math.sqrt(D)

        # Exponential scores
        qs = (q_sample ** 2).sum(-1, keepdim=True)
        ks = (k_sample ** 2).sum(-1, keepdim=True)
        d = q_sample @ k_sample.transpose(-2, -1)
        dist = qs + ks.transpose(-2, -1) - 2 * d
        exp_scores = (-dist / float(D)).squeeze(0).squeeze(0)

    # Flatten and compute correlations
    l_flat = learned_scores.flatten().float().cpu()
    d_flat = dot_scores.flatten().float().cpu()
    e_flat = exp_scores.flatten().float().cpu()

    def r_squared(pred, target):
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return (1 - ss_res / ss_tot).item()

    r2_dot = r_squared(l_flat, d_flat)
    r2_exp = r_squared(l_flat, e_flat)
    corr_dot = torch.corrcoef(torch.stack([l_flat, d_flat]))[0, 1].item()
    corr_exp = torch.corrcoef(torch.stack([l_flat, e_flat]))[0, 1].item()

    print(f"\n  Correlation with dot product:  r={corr_dot:.4f}, R²={r2_dot:.4f}")
    print(f"  Correlation with exponential:  r={corr_exp:.4f}, R²={r2_exp:.4f}")

    if r2_exp > 0.95:
        print("  → Learned kernel ≈ exponential (confirmed exponential was optimal)")
    elif r2_dot > r2_exp:
        print("  → Learned kernel drifted toward dot product")
    elif max(r2_dot, r2_exp) < 0.5:
        print("  → Learned kernel found something novel (low R² with both)")
    else:
        print(f"  → Learned kernel between exponential and dot product")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  V19 baseline val_ppl: {ckpt.get('val_ppl', 'N/A'):.2f}")
    print(f"  Learned kernel best val_ppl: {best_val:.2f}")
    print(f"  Improvement: {ckpt.get('val_ppl', 0) - best_val:+.2f}")
    print(f"  Kernel R² vs exponential: {r2_exp:.4f}")
    print(f"  Kernel R² vs dot product: {r2_dot:.4f}")
    print(f"  Training time: {time.time() - t0:.0f}s")

    # Save
    results = {
        "n_steps": n_steps,
        "proj_lr": proj_lr,
        "kernel_lr": kernel_lr,
        "baseline_val_ppl": ckpt.get("val_ppl"),
        "best_val_ppl": best_val,
        "val_history": val_ppls,
        "kernel_r2_exp": r2_exp,
        "kernel_r2_dot": r2_dot,
        "kernel_corr_exp": corr_exp,
        "kernel_corr_dot": corr_dot,
    }
    with open(results_dir / "experiment_a_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--proj-lr", type=float, default=1e-5)
    parser.add_argument("--kernel-lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_experiment_a(
        n_steps=args.n_steps,
        proj_lr=args.proj_lr,
        kernel_lr=args.kernel_lr,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
