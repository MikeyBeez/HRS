"""Learned Kernel Follow-ups on Tiny Shakespeare.

A3: Kernel shape visualization (analysis of existing trained kernel)
A2: Fully unfrozen training from scratch (4 models compared)

Usage:
    python exp_learned_kernel_followup.py [--device cuda]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from exp_kernel_attention import (
    MiniTransformer, DotProductAttention, ExponentialKernelAttention,
    CharDataset, load_shakespeare, train_model,
    extract_engrams, run_pair_analysis,
)
from torch.utils.data import DataLoader


# ============================================================
# Learned Kernel for Shakespeare (decomposed, memory-efficient)
# ============================================================

class ShakespeareLearnedKernel(nn.Module):
    """Learned attention scoring for the small Shakespeare model."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, hidden_dim=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Decomposed learned kernel: independent nonlinear projections
        R = hidden_dim
        self.phi_q = nn.Sequential(nn.Linear(self.head_dim, R), nn.GELU(), nn.Linear(R, R))
        self.phi_k = nn.Sequential(nn.Linear(self.head_dim, R), nn.GELU(), nn.Linear(R, R))
        self.score_scale = 1.0 / math.sqrt(R)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_proj = self.phi_q(q)
        k_proj = self.phi_k(k)
        scores = (q_proj @ k_proj.transpose(-2, -1)) * self.score_scale

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

    def get_raw_qk(self, x):
        """Extract raw q,k vectors for kernel shape analysis."""
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, _ = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2)

    def compute_learned_scores(self, q, k):
        """Compute learned kernel scores for given q,k."""
        q_proj = self.phi_q(q)
        k_proj = self.phi_k(k)
        return (q_proj @ k_proj.transpose(-2, -1)) * self.score_scale


class ScaffoldedLearnedKernel(ShakespeareLearnedKernel):
    """Same as ShakespeareLearnedKernel but with phase control."""
    pass


# ============================================================
# Shakespeare TransformerBlock with pluggable attention
# ============================================================

class FlexBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, attn_module=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_module
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FlexTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=6,
                 max_seq_len=256, dropout=0.1, attn_cls=None, **attn_kwargs):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            attn = attn_cls(d_model, n_heads, max_seq_len, dropout, **attn_kwargs)
            self.blocks.append(FlexBlock(d_model, n_heads, max_seq_len, dropout, attn))

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x), x


# ============================================================
# A3: Kernel Shape Visualization
# ============================================================

def run_a3_visualization(model, kernel_attn, device, out_dir, label=""):
    """Analyze learned kernel shape by sampling q,k pairs."""
    print(f"\n{'='*60}")
    print(f"A3: Kernel Shape Analysis — {label}")
    print(f"{'='*60}")

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect q,k pairs from all layers
    text, stoi, itos, _ = load_shakespeare()
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    all_q, all_k = [], []
    n_samples = 2000

    with torch.no_grad():
        for start in range(0, min(len(data) - 256, n_samples * 256), 256):
            x = data[start:start+256].unsqueeze(0).to(device)
            # Get q,k from first layer's learned attention
            for block in model.blocks:
                if hasattr(block.attn, 'get_raw_qk'):
                    q, k = block.attn.get_raw_qk(block.ln1(model.drop(model.tok_emb(x) + model.pos_emb(torch.arange(256, device=device)))))
                    all_q.append(q[:, :, :64, :].cpu())  # sample 64 positions
                    all_k.append(k[:, :, :64, :].cpu())
                    break
            if len(all_q) >= 50:
                break

    q_cat = torch.cat(all_q, dim=2)  # (1, H, N, D)
    k_cat = torch.cat(all_k, dim=2)

    # Sample pairs
    N = min(q_cat.shape[2], 500)
    idx_q = torch.randperm(q_cat.shape[2])[:N]
    idx_k = torch.randperm(k_cat.shape[2])[:N]
    q_sample = q_cat[0, 0, idx_q, :].to(device)  # (N, D) — head 0
    k_sample = k_cat[0, 0, idx_k, :].to(device)

    D = q_sample.shape[1]

    with torch.no_grad():
        # Compute all score types for sampled pairs
        results = {"pairs": []}

        for i in range(min(N, 200)):
            q_i = q_sample[i:i+1]  # (1, D)
            for j in range(min(N, 200)):
                k_j = k_sample[j:j+1]  # (1, D)

                # Learned score
                q_exp = q_i.unsqueeze(0).unsqueeze(0)  # (1, 1, 1, D)
                k_exp = k_j.unsqueeze(0).unsqueeze(0)
                learned = kernel_attn.compute_learned_scores(q_exp, k_exp).item()

                # Dot product
                dot = (q_i @ k_j.T).item() / math.sqrt(D)

                # Exponential
                dist = ((q_i - k_j) ** 2).sum().item()
                exp_score = -dist / float(D)

                # Cosine similarity
                cos = F.cosine_similarity(q_i, k_j).item()

                # Euclidean distance
                euc = math.sqrt(dist)

                # Magnitudes
                mag_q = q_i.norm().item()
                mag_k = k_j.norm().item()

                results["pairs"].append({
                    "learned": learned, "dot": dot, "exp": exp_score,
                    "cos": cos, "euc": euc, "mag_q": mag_q, "mag_k": mag_k,
                })

    # Compute correlations
    pairs = results["pairs"]
    learned = torch.tensor([p["learned"] for p in pairs])
    dot = torch.tensor([p["dot"] for p in pairs])
    exp = torch.tensor([p["exp"] for p in pairs])
    cos = torch.tensor([p["cos"] for p in pairs])
    euc = torch.tensor([p["euc"] for p in pairs])

    def r_sq(pred, target):
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return max(0, (1 - ss_res / ss_tot).item())

    r2_dot = r_sq(learned, dot)
    r2_exp = r_sq(learned, exp)
    corr_dot = torch.corrcoef(torch.stack([learned, dot]))[0, 1].item()
    corr_exp = torch.corrcoef(torch.stack([learned, exp]))[0, 1].item()
    corr_cos = torch.corrcoef(torch.stack([learned, cos]))[0, 1].item()
    corr_euc = torch.corrcoef(torch.stack([learned, euc]))[0, 1].item()

    print(f"  Correlations (learned vs):")
    print(f"    Dot product:  r={corr_dot:.4f}, R²={r2_dot:.4f}")
    print(f"    Exponential:  r={corr_exp:.4f}, R²={r2_exp:.4f}")
    print(f"    Cosine sim:   r={corr_cos:.4f}")
    print(f"    Euclidean:    r={corr_euc:.4f}")

    # Residual analysis (learned - exponential)
    residuals = learned - exp
    # Bin by distance and compute mean residual
    euc_bins = torch.linspace(euc.min(), euc.max(), 10)
    print(f"\n  Residual (learned - exponential) by distance:")
    for i in range(len(euc_bins) - 1):
        mask = (euc >= euc_bins[i]) & (euc < euc_bins[i+1])
        if mask.sum() > 5:
            mean_res = residuals[mask].mean().item()
            print(f"    dist [{euc_bins[i]:.2f}, {euc_bins[i+1]:.2f}): "
                  f"mean_residual={mean_res:+.4f} (n={mask.sum().item()})")

    # Per-head analysis
    print(f"\n  Per-head correlations with exponential:")
    for h in range(min(q_cat.shape[1], 6)):
        q_h = q_cat[0, h, idx_q[:100], :].to(device)
        k_h = k_cat[0, h, idx_k[:100], :].to(device)

        h_learned = []
        h_exp = []
        for i in range(50):
            for j in range(50):
                qi = q_h[i:i+1].unsqueeze(0).unsqueeze(0)
                kj = k_h[j:j+1].unsqueeze(0).unsqueeze(0)
                ls = kernel_attn.compute_learned_scores(qi, kj).item()
                d = ((q_h[i] - k_h[j]) ** 2).sum().item()
                es = -d / float(D)
                h_learned.append(ls)
                h_exp.append(es)

        h_l = torch.tensor(h_learned)
        h_e = torch.tensor(h_exp)
        r = torch.corrcoef(torch.stack([h_l, h_e]))[0, 1].item()
        print(f"    Head {h}: r={r:.4f}")

    # Save data
    with open(out_dir / f"kernel_shape_{label}.json", "w") as f:
        json.dump({
            "r2_dot": r2_dot, "r2_exp": r2_exp,
            "corr_dot": corr_dot, "corr_exp": corr_exp,
            "corr_cos": corr_cos, "corr_euc": corr_euc,
            "n_pairs": len(pairs),
        }, f, indent=2)

    return results


# ============================================================
# A2: Four-model comparison
# ============================================================

def run_a2_comparison(device, seed=42):
    """Train four models on Shakespeare and compare."""
    print(f"\n{'='*60}")
    print("A2: Four-Model Comparison")
    print(f"{'='*60}")

    text, stoi, itos, plays = load_shakespeare()
    vocab_size = len(stoi)
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    lines = text.split('\n')
    p1_name, p1_s, p1_e = plays[0]
    p2_name, p2_s, p2_e = plays[1]
    p1_text = '\n'.join(lines[p1_s:p1_e])
    p2_text = '\n'.join(lines[p2_s:p2_e])

    def chunk(t, sz=256):
        return [t[i:i+sz] for i in range(0, len(t)-sz, sz)]

    def elines(t, mn=20, mx=120):
        return [l.strip() for l in t.split('\n') if mn <= len(l.strip()) <= mx and not l.strip().endswith(':')]

    p1_chunks, p2_chunks = chunk(p1_text)[:50], chunk(p2_text)[:50]
    p1_lines, p2_lines = elines(p1_text)[:200], elines(p2_text)[:200]

    models_config = [
        ("1_dot_product", "dot_product"),
        ("2_exponential", "exponential"),
        ("3_learned_scratch", "learned"),
        ("4_scaffolded", "scaffolded"),
    ]

    results = {}

    for name, mode in models_config:
        print(f"\n{'#'*60}")
        print(f"Model: {name}")
        print(f"{'#'*60}")

        torch.manual_seed(seed)

        if mode in ("dot_product", "exponential"):
            model = MiniTransformer(vocab_size, attn_type=mode)
            losses, val_losses, best_step, best_val = train_model(
                model, train_data, val_data,
                n_steps=5000, batch_size=32, lr=3e-4, device=device, label=name,
            )
        elif mode == "learned":
            model = FlexTransformer(
                vocab_size, attn_cls=ShakespeareLearnedKernel, hidden_dim=64,
            )
            losses, val_losses, best_step, best_val = train_model(
                model, train_data, val_data,
                n_steps=5000, batch_size=32, lr=3e-4, device=device, label=name,
            )
        elif mode == "scaffolded":
            # Phase 1: train with exponential for 2500 steps
            model_p1 = MiniTransformer(vocab_size, attn_type="exponential")
            losses_p1, _, _, _ = train_model(
                model_p1, train_data, val_data,
                n_steps=2500, batch_size=32, lr=3e-4, device=device,
                label=f"{name}_phase1",
            )
            # Phase 2: replace with learned kernel, continue training
            model = FlexTransformer(
                vocab_size, attn_cls=ShakespeareLearnedKernel, hidden_dim=64,
            )
            # Copy weights from phase 1 (everything except attention-specific params)
            p1_state = model_p1.state_dict()
            m_state = model.state_dict()
            for key in p1_state:
                if key in m_state and p1_state[key].shape == m_state[key].shape:
                    m_state[key] = p1_state[key]
            model.load_state_dict(m_state)
            del model_p1

            losses, val_losses, best_step, best_val = train_model(
                model, train_data, val_data,
                n_steps=2500, batch_size=32, lr=3e-4, device=device, label=f"{name}_phase2",
            )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}, Best val: {best_val:.4f} @ step {best_step}")

        # Engram analysis
        eng_p1 = extract_engrams(model, p1_chunks, stoi, device)
        eng_p2 = extract_engrams(model, p2_chunks, stoi, device)
        result_play = run_pair_analysis(eng_p1, eng_p2, f"{name} play-level")

        eng_l1 = extract_engrams(model, p1_lines, stoi, device)
        eng_l2 = extract_engrams(model, p2_lines, stoi, device)
        result_line = run_pair_analysis(eng_l1, eng_l2, f"{name} line-level")

        results[name] = {
            "n_params": n_params,
            "best_val": best_val,
            "best_step": best_step,
            "play_level": result_play,
            "line_level": result_line,
        }

        # Kernel shape analysis for learned models
        if mode in ("learned", "scaffolded"):
            kernel_attn = model.blocks[0].attn
            viz = run_a3_visualization(model, kernel_attn, device,
                                       Path("results/learned_kernel_viz"), label=name)
            results[name]["kernel_shape"] = viz

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("A2 SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Model':<25} {'Params':>10} {'Val Loss':>10} {'Play Acc':>10} {'Line Acc':>10}")
    print(f"  {'-'*68}")
    for name, r in results.items():
        play = f"{r['play_level']['accuracy']:.1%}" if r.get('play_level') else "N/A"
        line = f"{r['line_level']['accuracy']:.1%}" if r.get('line_level') else "N/A"
        print(f"  {name:<25} {r['n_params']:>10,} {r['best_val']:>10.4f} {play:>10} {line:>10}")

    out_dir = Path("results/learned_kernel_a2")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    run_a2_comparison(device)


if __name__ == "__main__":
    main()
