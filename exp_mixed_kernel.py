"""Mixed-Kernel Depth Attention for Engram Extraction.

Combines dot product (directional) and exponential kernel (proximity)
in depth-wise attention with a learned interpolation weight.

Three conditions:
E. Learned scalar alpha
F. Fixed alpha = 0.5
G. Per-layer alpha vector

Usage:
    python exp_mixed_kernel.py [--device cuda] [--train-steps 2000]
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
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore, EngramEntry
from niah_egr import NEEDLES, DISTRACTORS
from data import _extract_articles
from exp_attnres import extract_all_layers, eval_niah, eval_grounding


# ============================================================
# Mixed Kernel Depth Attention
# ============================================================

class MixedKernelDepthAttention(nn.Module):
    """Depth-wise attention mixing dot product and exponential kernel.

    score = alpha * dot_score + (1-alpha) * exp_score

    alpha can be: scalar learned, scalar fixed, or per-layer learned.
    """

    def __init__(self, d_model, n_layers, mode="learned_scalar"):
        """
        Args:
            mode: "learned_scalar", "fixed_half", or "per_layer"
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode
        self.total_layers = n_layers + 1  # including embedding

        self.depth_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.layer_scale = nn.Parameter(torch.ones(self.total_layers))

        # Learnable temperature for exponential kernel
        self.log_temperature = nn.Parameter(torch.tensor(math.log(float(d_model))))

        # Mixture weight(s)
        if mode == "learned_scalar":
            self.mix_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        elif mode == "fixed_half":
            self.register_buffer("mix_logit", torch.tensor(0.0))
        elif mode == "per_layer":
            self.mix_logits = nn.Parameter(torch.zeros(self.total_layers))

    def get_alpha(self):
        """Return alpha value(s) for inspection."""
        if self.mode == "per_layer":
            return torch.sigmoid(self.mix_logits).detach()
        else:
            return torch.sigmoid(self.mix_logit).detach()

    def forward(self, hidden_states_list):
        n = len(hidden_states_list)
        D = self.d_model

        stacked = torch.stack(hidden_states_list, dim=1)  # (1, L, T, D)
        scales = self.layer_scale[:n].view(1, n, 1, 1)
        stacked = stacked * scales
        layer_means = stacked.mean(dim=2)  # (1, L, D)

        query = self.depth_query  # (1, 1, D)
        keys = self.key_proj(layer_means)  # (1, L, D)

        # Dot product scores
        dot_scale = 1.0 / math.sqrt(D)
        dot_scores = (query @ keys.transpose(-2, -1)) * dot_scale  # (1, 1, L)

        # Exponential kernel scores
        temperature = self.log_temperature.exp()
        q_sq = (query ** 2).sum(dim=-1, keepdim=True)
        k_sq = (keys ** 2).sum(dim=-1, keepdim=True)
        dot = query @ keys.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot
        exp_scores = -distances / temperature  # (1, 1, L)

        # Mix
        if self.mode == "per_layer":
            alpha = torch.sigmoid(self.mix_logits[:n]).view(1, 1, n)  # (1, 1, L)
        else:
            alpha = torch.sigmoid(self.mix_logit)  # scalar

        mixed_scores = alpha * dot_scores + (1 - alpha) * exp_scores

        weights = F.softmax(mixed_scores, dim=-1).squeeze(0).squeeze(0)  # (L,)

        attended = (weights.unsqueeze(0).unsqueeze(-1) * layer_means).sum(dim=1)
        engram = F.normalize(attended.squeeze(0), dim=0)

        return engram, weights.detach()


# ============================================================
# Training (reused from exp_attnres.py with alpha logging)
# ============================================================

def train_mixed(model, module, tokenizer, articles, device,
                n_steps=2000, lr=1e-3, batch_size=16):
    """Train the mixed-kernel module on contrastive objective."""
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    module.train()

    # Pre-compute hidden states
    print("  Pre-computing hidden states...")
    article_hidden = {}
    for art_idx, (title, text) in enumerate(articles):
        ids_full = tokenizer.encode(text, add_special_tokens=False)
        if len(ids_full) < 64:
            continue
        segments = []
        for start in range(0, len(ids_full), 512):
            seg = ids_full[start:start + 512]
            if len(seg) < 32:
                continue
            seg_ids = torch.tensor(seg, dtype=torch.long)
            hs = extract_all_layers(model, seg_ids, device)
            segments.append(hs)
            if len(segments) >= 3:
                break
        if len(segments) >= 2:
            article_hidden[art_idx] = segments
        if len(article_hidden) >= 200:
            break

    multi_keys = [k for k, v in article_hidden.items() if len(v) >= 2]
    all_keys = list(article_hidden.keys())
    print(f"  {len(article_hidden)} articles, {len(multi_keys)} with 2+ segments")

    losses = []
    t0 = time.time()
    for step in range(n_steps):
        half = batch_size // 2
        loss = torch.tensor(0.0, device=device)

        for _ in range(half):
            art = random.choice(multi_keys)
            i, j = random.sample(range(len(article_hidden[art])), 2)
            eng_a, _ = module(article_hidden[art][i])
            eng_b, _ = module(article_hidden[art][j])
            loss = loss - F.cosine_similarity(eng_a.unsqueeze(0), eng_b.unsqueeze(0))

        for _ in range(half):
            a1, a2 = random.sample(all_keys, 2)
            eng_a, _ = module(random.choice(article_hidden[a1]))
            eng_b, _ = module(random.choice(article_hidden[a2]))
            loss = loss + F.cosine_similarity(eng_a.unsqueeze(0), eng_b.unsqueeze(0))

        loss = loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = sum(losses[-500:]) / 500
            elapsed = time.time() - t0
            with torch.no_grad():
                sample_hs = article_hidden[all_keys[0]][0]
                _, weights = module(sample_hs)
                w_str = " ".join(f"{w:.3f}" for w in weights.tolist())
                alpha = module.get_alpha()
                if alpha.dim() == 0:
                    a_str = f"alpha={alpha.item():.3f}"
                else:
                    a_str = "alpha=[" + " ".join(f"{a:.3f}" for a in alpha.tolist()) + "]"
                temp = module.log_temperature.exp().item()
            print(f"    Step {step+1}: loss={avg:.4f}, {a_str}, temp={temp:.1f}, "
                  f"weights=[{w_str}] ({elapsed:.0f}s)")

    return losses


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train-steps", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded V18 (frozen)")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d_model = cfg.model.d_model
    n_layers = cfg.model.n_layers

    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    articles = _extract_articles(raw["validation"]["text"])

    conditions = {
        "E_mixed_learned": ("learned_scalar", {}),
        "F_mixed_fixed": ("fixed_half", {}),
        "G_mixed_per_layer": ("per_layer", {}),
    }

    all_results = {}

    for cond_name, (mode, _) in conditions.items():
        print(f"\n{'='*60}")
        print(f"CONDITION {cond_name} (mode={mode})")
        print(f"{'='*60}")

        module = MixedKernelDepthAttention(d_model, n_layers, mode=mode).to(device)
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  Trainable params: {n_params:,}")

        train_mixed(model, module, tokenizer, articles, device, n_steps=args.train_steps)

        print("\n  NIAH Retrieval:")
        niah = eval_niah(model, module, tokenizer, device, cond_name)

        print("\n  Generation Grounding:")
        ground = eval_grounding(model, module, tokenizer, device, cond_name)

        # Get final learned values
        with torch.no_grad():
            sample_ids = torch.tensor(
                tokenizer.encode(NEEDLES[0].fact, add_special_tokens=False), dtype=torch.long
            )
            sample_hs = extract_all_layers(model, sample_ids, device)
            _, weights = module(sample_hs)
            alpha = module.get_alpha()
            temp = module.log_temperature.exp().item()

        all_results[cond_name] = {
            "niah": niah,
            "grounding": ground,
            "layer_weights": weights.tolist(),
            "alpha": alpha.tolist() if alpha.dim() > 0 else alpha.item(),
            "temperature": temp,
        }

        layer_names = ["emb"] + [f"L{i+1}" for i in range(n_layers)]
        print(f"\n  Layer weights: {' '.join(f'{w:.3f}' for w in weights.tolist())}")
        if alpha.dim() > 0:
            print(f"  Per-layer alpha: {' '.join(f'{a:.3f}' for a in alpha.tolist())}")
        else:
            print(f"  Alpha: {alpha.item():.3f}")
        print(f"  Temperature: {temp:.1f}")

        del module
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY — ALL CONDITIONS")
    print(f"{'='*60}")

    print(f"\n  Retrieval:")
    print(f"  {'Method':<30} {'Found':>6} {'Rank':>6} {'Sim':>8} {'Alpha':>8}")
    print(f"  {'-'*62}")
    # Baselines
    print(f"  {'Standard engram':<30} {'5/5':>6} {'1.2':>6} {'0.505':>8} {'-':>8}")
    print(f"  {'Delta 5':<30} {'5/5':>6} {'1.0':>6} {'0.539':>8} {'-':>8}")
    print(f"  {'AttnRes dot':<30} {'5/5':>6} {'1.0':>6} {'0.500':>8} {'-':>8}")
    print(f"  {'AttnRes exp':<30} {'5/5':>6} {'1.6':>6} {'0.400':>8} {'-':>8}")

    for name, data in all_results.items():
        niah = data["niah"]
        n_found = sum(1 for r in niah if r["found"])
        ranks = [r["rank"] for r in niah if r["found"]]
        sims = [r["sim"] for r in niah if r["found"]]
        mr = f"{sum(ranks)/len(ranks):.1f}" if ranks else "N/A"
        ms = f"{sum(sims)/len(sims):.4f}" if sims else "N/A"
        a = data["alpha"]
        if isinstance(a, list):
            a_str = "per-L"
        else:
            a_str = f"{a:.3f}"
        print(f"  {name:<30} {n_found:>4}/5 {mr:>6} {ms:>8} {a_str:>8}")

    print(f"\n  Grounding:")
    print(f"  {'Method':<30} {'Recall':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Standard engram':<30} {'3% (1/30)':>10}")
    print(f"  {'Delta 5':<30} {'7% (2/30)':>10}")
    print(f"  {'AttnRes dot':<30} {'3% (1/30)':>10}")
    print(f"  {'AttnRes exp':<30} {'10% (3/30)':>10}")

    for name, data in all_results.items():
        gr = data["grounding"]
        hits = sum(r["hits"] for r in gr)
        total = sum(r["total"] for r in gr)
        pct = hits / total * 100 if total > 0 else 0
        print(f"  {name:<30} {f'{pct:.0f}% ({hits}/{total})':>10}")

    print(f"\n  Layer weights:")
    layer_names = ["emb"] + [f"L{i+1}" for i in range(n_layers)]
    header = "  " + f"{'Method':<25}" + " ".join(f"{n:>6}" for n in layer_names)
    print(header)
    for name, data in all_results.items():
        w = data["layer_weights"]
        print(f"  {name:<25}" + " ".join(f"{v:>6.3f}" for v in w))

    print(f"\n  Per-layer alphas (Condition G):")
    if "G_mixed_per_layer" in all_results:
        alphas = all_results["G_mixed_per_layer"]["alpha"]
        if isinstance(alphas, list):
            for n, a in zip(layer_names, alphas):
                bar = "█" * int(a * 20) + "░" * (20 - int(a * 20))
                label = "dot" if a > 0.5 else "exp"
                print(f"    {n:>4}: {a:.3f} [{bar}] ← {label}")

    # Save
    out_path = Path("results/v18_cross_attn/mixed_kernel_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
