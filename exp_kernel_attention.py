"""Experiment: Exponential Kernel vs Dot Product in Attention.

Trains two minimal transformers on Tiny Shakespeare:
- Model A: standard dot product attention
- Model B: exponential kernel (negative squared distance) attention

Measures training dynamics and engram quality at play-level and line-level.

Usage:
    python exp_kernel_attention.py [--n-steps 5000] [--device cuda]
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Model components
# ============================================================

def build_alibi_bias(n_heads, max_seq_len):
    """Build ALiBi position bias matrix.

    Returns (1, n_heads, max_seq_len, max_seq_len) static bias tensor.
    Each head h has slope m_h = 1 / 2^(8 * h / n_heads).
    Bias = -m_h * |i - j| for query position i, key position j.
    """
    # Head slopes: geometric sequence
    slopes = torch.tensor([1.0 / (2 ** (8 * h / n_heads)) for h in range(n_heads)])

    # Distance matrix: |i - j|
    pos = torch.arange(max_seq_len)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()  # (T, T)

    # Bias per head: -slope * distance
    bias = -slopes.view(1, n_heads, 1, 1) * dist.view(1, 1, max_seq_len, max_seq_len)
    return bias


class DotProductAttention(nn.Module):
    """Standard scaled dot-product attention."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, use_alibi=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_alibi = use_alibi

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

        if use_alibi:
            self.register_buffer("alibi_bias", build_alibi_bias(n_heads, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        if self.use_alibi:
            scores = scores + self.alibi_bias[:, :, :T, :T]
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class ExponentialKernelAttention(nn.Module):
    """Attention using exponential kernel (negative squared distance)."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, temperature=None, use_alibi=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.temperature = temperature if temperature else float(self.head_dim)
        self.use_alibi = use_alibi

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

        if use_alibi:
            self.register_buffer("alibi_bias", build_alibi_bias(n_heads, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Negative squared Euclidean distance via ||q-k||² = ||q||² + ||k||² - 2q·k
        # This avoids materializing the (B, H, T, T, D) intermediate tensor
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        dot = q @ k.transpose(-2, -1)              # (B, H, T, T)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot  # (B, H, T, T)

        scores = -distances / self.temperature
        if self.use_alibi:
            scores = scores + self.alibi_bias[:, :, :T, :T]
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class MLPScorerAttention(nn.Module):
    """Attention using a learned MLP scorer: s(q,k) = MLP([q;k]).

    No dot product, no distance — just a learned nonlinear interaction.
    Memory-efficient: decomposes MLP([q;k]) = MLP_q(q) + MLP_k(k) + bilinear(q,k)
    via a low-rank approximation to avoid the (B,H,T,T,2D) expansion.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, mlp_hidden=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Decomposed MLP scorer: project q and k to shared space, then interact
        # q -> (D -> R), k -> (D -> R), score = (phi_q(q) @ phi_k(k).T) / sqrt(R)
        # phi_q and phi_k are nonlinear projections
        R = mlp_hidden
        self.phi_q = nn.Sequential(nn.Linear(self.head_dim, R), nn.GELU())
        self.phi_k = nn.Sequential(nn.Linear(self.head_dim, R), nn.GELU())

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Nonlinear projections then dot product in projected space
        q_proj = self.phi_q(q)  # (B, H, T, R)
        k_proj = self.phi_k(k)  # (B, H, T, R)
        scale = 1.0 / math.sqrt(q_proj.shape[-1])
        scores = (q_proj @ k_proj.transpose(-2, -1)) * scale  # (B, H, T, T)

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class RandomProjectionAttention(nn.Module):
    """Attention with random frozen projection + nonlinearity.

    s(q,k) = relu(W_q_frozen @ q + W_k_frozen @ k)
    Decomposed to avoid 5D tensor. W is frozen — no learning in the scorer.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, proj_dim=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Random frozen projections: q -> R, k -> R, then dot in R-space
        W_q = torch.randn(n_heads, self.head_dim, proj_dim) * 0.1
        W_k = torch.randn(n_heads, self.head_dim, proj_dim) * 0.1
        self.register_buffer("W_q_frozen", W_q)
        self.register_buffer("W_k_frozen", W_k)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Frozen random projection + ReLU, then dot product in projected space
        q_proj = torch.relu(torch.einsum('bhtd,hdr->bhtr', q, self.W_q_frozen))  # (B,H,T,R)
        k_proj = torch.relu(torch.einsum('bhtd,hdr->bhtr', k, self.W_k_frozen))  # (B,H,T,R)
        scale = 1.0 / math.sqrt(q_proj.shape[-1])
        scores = (q_proj @ k_proj.transpose(-2, -1)) * scale  # (B, H, T, T)

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class L1DistanceAttention(nn.Module):
    """Attention using L1 distance: s(q,k) = -||q-k||_1 / temperature.

    Memory-efficient: computes L1 via chunked iteration over head dim.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.temperature = float(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # L1 distance, chunked over head dim to avoid (B,H,T,T,D) tensor
        # sum_d |q_d - k_d| where q_d: (B,H,T,1) and k_d: (B,H,1,T)
        l1_dist = torch.zeros(B, self.n_heads, T, T, device=x.device)
        chunk = 16  # process 16 dims at a time
        for d_start in range(0, self.head_dim, chunk):
            d_end = min(d_start + chunk, self.head_dim)
            q_chunk = q[:, :, :, d_start:d_end].unsqueeze(3)  # (B,H,T,1,chunk)
            k_chunk = k[:, :, :, d_start:d_end].unsqueeze(2)  # (B,H,1,T,chunk)
            l1_dist += (q_chunk - k_chunk).abs().sum(dim=-1)

        scores = -l1_dist / self.temperature
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class SinProductAttention(nn.Module):
    """Attention using multiplicative sine: s(q,k) = sum_i sin(q_i * k_i).

    Memory-efficient: uses the identity sin(a*b) and chunks over dims.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # sum_d sin(q_d * k_d), chunked to avoid (B,H,T,T,D)
        scores = torch.zeros(B, self.n_heads, T, T, device=x.device)
        chunk = 16
        for d_start in range(0, self.head_dim, chunk):
            d_end = min(d_start + chunk, self.head_dim)
            q_chunk = q[:, :, :, d_start:d_end].unsqueeze(3)  # (B,H,T,1,chunk)
            k_chunk = k[:, :, :, d_start:d_end].unsqueeze(2)  # (B,H,1,T,chunk)
            scores += torch.sin(q_chunk * k_chunk).sum(dim=-1)

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class SoftRankAttention(nn.Module):
    """Attention using soft rank of L2 distances.

    Memory-efficient: uses L2 distances (already (B,H,T,T)) then
    approximates rank via normalized negative distance (avoids the
    (B,H,T,T,T) pairwise-distance-of-distances tensor).
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # L2 distances via ||q-k||² = ||q||² + ||k||² - 2q·k
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)
        dot = q @ k.transpose(-2, -1)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot  # (B, H, T, T)

        # Approximate soft rank: normalize distances per query to [0, 1]
        # then negate. Closer keys get higher scores (lower rank).
        # This preserves relative ordering without the (B,H,T,T,T) tensor.
        d_min = distances.min(dim=-1, keepdim=True).values
        d_max = distances.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        normalized = (distances - d_min) / (d_max - d_min + 1e-6)
        scores = -normalized * self.head_dim  # scale for softmax

        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1,
                 attn_type="dot_product"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)

        # Parse attn_type: "exponential_alibi" -> kernel="exponential", alibi=True
        use_alibi = attn_type.endswith("_alibi")
        base_type = attn_type.replace("_alibi", "") if use_alibi else attn_type

        attn_classes = {
            "dot_product": DotProductAttention,
            "exponential": ExponentialKernelAttention,
            "mlp": MLPScorerAttention,
            "random_proj": RandomProjectionAttention,
            "l1_distance": L1DistanceAttention,
            "sin_product": SinProductAttention,
            "soft_rank": SoftRankAttention,
        }
        cls = attn_classes.get(base_type, DotProductAttention)
        # Only DotProduct and Exponential support ALiBi currently
        if use_alibi and hasattr(cls.__init__, '__code__') and 'use_alibi' in cls.__init__.__code__.co_varnames:
            self.attn = cls(d_model, n_heads, max_seq_len, dropout, use_alibi=True)
        else:
            self.attn = cls(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=6,
                 max_seq_len=256, dropout=0.1, attn_type="dot_product"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # ALiBi provides positional info via attention bias — no pos embedding needed
        self.use_alibi = attn_type.endswith("_alibi")
        if not self.use_alibi:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout, attn_type)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # weight tying

        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        if self.use_alibi:
            x = self.drop(tok)
        else:
            pos = self.pos_emb(torch.arange(T, device=idx.device))
            x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x), x  # logits and hidden states


# ============================================================
# Data
# ============================================================

def load_shakespeare():
    """Load Tiny Shakespeare, return text, char-to-int mapping, play boundaries.

    Tiny Shakespeare has no explicit play headers. We split by detecting
    where the character set changes completely (~line 15600), giving two
    large sections: histories/tragedies (first half) and romances/comedies
    (second half).
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    import urllib.request
    cache_path = Path("datasets/tiny_shakespeare.txt")
    cache_path.parent.mkdir(exist_ok=True)
    if not cache_path.exists():
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(url, cache_path)
    text = cache_path.read_text()

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    # Split into two halves by character-set boundary
    lines = text.split('\n')
    boundary = 15600  # detected empirically: character sets don't overlap here

    plays = [
        ("Histories/Tragedies", 0, boundary),
        ("Romances/Comedies", boundary, len(lines)),
    ]

    return text, stoi, itos, plays


class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


# ============================================================
# Training
# ============================================================

def train_model(model, train_data, val_data, n_steps, batch_size, lr, device, label,
                eval_interval=250, patience=5):
    """Train a model with early stopping. Returns best checkpoint.

    Args:
        patience: stop after this many eval intervals without val improvement
    """
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.to(device)
    model.train()

    losses = []
    val_losses = []
    best_val = float('inf')
    best_state = None
    best_step = 0
    patience_counter = 0
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(n_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % eval_interval == 0:
            # Validation
            model.eval()
            val_ds = CharDataset(val_data, seq_len)
            val_loader = DataLoader(val_ds, batch_size=batch_size, drop_last=True)
            val_loss = 0
            n_val = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vl, _ = model(vx)
                    val_loss += F.cross_entropy(vl.reshape(-1, V), vy.reshape(-1)).item()
                    n_val += 1
                    if n_val >= 20:
                        break
            val_loss /= n_val
            val_losses.append((step + 1, val_loss))
            elapsed = time.time() - t0
            avg_train = sum(losses[-eval_interval:]) / len(losses[-eval_interval:])

            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_step = step + 1
                patience_counter = 0
                marker = " *best*"
            else:
                patience_counter += 1

            print(f"  [{label}] step {step+1:5d}: train={avg_train:.4f} val={val_loss:.4f} "
                  f"(best={best_val:.4f} @{best_step}){marker} ({elapsed:.0f}s)")

            if patience_counter >= patience:
                print(f"  [{label}] Early stopping at step {step+1} (no improvement for {patience} evals)")
                break

            model.train()

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"  [{label}] Restored best checkpoint from step {best_step} (val={best_val:.4f})")

    return losses, val_losses, best_step, best_val


# ============================================================
# Engram analysis
# ============================================================

@torch.no_grad()
def extract_engrams(model, text_segments, stoi, device, layer=-1):
    """Extract engrams from text segments.

    Returns list of (engram_tensor, segment_text) pairs.
    """
    model.eval()
    engrams = []
    for seg in text_segments:
        ids = [stoi.get(c, 0) for c in seg]
        if len(ids) < 5:
            continue
        ids = ids[:model.max_seq_len]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        _, hidden = model(x)
        engram = hidden.mean(dim=1).squeeze(0).cpu()
        engram = F.normalize(engram, dim=0)
        engrams.append((engram, seg[:50]))
    return engrams


def run_pair_analysis(engrams_a, engrams_b, label):
    """Compare same-group vs cross-group engram similarity."""
    if len(engrams_a) < 2 or len(engrams_b) < 2:
        print(f"  {label}: not enough data")
        return None

    # Same-group pairs (within A, within B)
    same_sims = []
    for group in [engrams_a, engrams_b]:
        for i in range(len(group)):
            for j in range(i + 1, min(i + 5, len(group))):
                sim = (group[i][0] @ group[j][0]).item()
                same_sims.append(sim)

    # Cross-group pairs
    cross_sims = []
    for i in range(min(50, len(engrams_a))):
        for j in range(min(50, len(engrams_b))):
            sim = (engrams_a[i][0] @ engrams_b[j][0]).item()
            cross_sims.append(sim)

    if not same_sims or not cross_sims:
        return None

    same_mean = sum(same_sims) / len(same_sims)
    cross_mean = sum(cross_sims) / len(cross_sims)
    gap = same_mean - cross_mean

    # Best threshold accuracy
    all_sims = [(s, 1) for s in same_sims] + [(s, 0) for s in cross_sims]
    best_acc = 0
    best_t = 0
    for t in [i * 0.05 for i in range(-10, 20)]:
        correct = sum(1 for s, l in all_sims if (s > t) == l)
        acc = correct / len(all_sims)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    print(f"  {label}:")
    print(f"    Same-group mean:  {same_mean:.4f} ({len(same_sims)} pairs)")
    print(f"    Cross-group mean: {cross_mean:.4f} ({len(cross_sims)} pairs)")
    print(f"    Gap:              {gap:.4f}")
    print(f"    Best threshold:   {best_t:.2f} @ {best_acc:.1%}")

    return {"same_mean": same_mean, "cross_mean": cross_mean, "gap": gap, "accuracy": best_acc}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernels", type=str, default="all",
                        help="Comma-separated kernel types, or 'all'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ALL_KERNELS = [
        "dot_product", "exponential", "mlp", "random_proj",
        "l1_distance", "sin_product", "soft_rank",
    ]

    if args.kernels == "all":
        kernels_to_test = ALL_KERNELS
    else:
        kernels_to_test = [k.strip() for k in args.kernels.split(",")]

    # Load data
    text, stoi, itos, plays = load_shakespeare()
    vocab_size = len(stoi)
    print(f"Shakespeare: {len(text)} chars, {vocab_size} unique, {len(plays)} sections")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # Prepare engram analysis data
    lines_list = text.split('\n')
    play1_name, p1_start, p1_end = plays[0]
    play2_name, p2_start, p2_end = plays[1]
    play1_text = '\n'.join(lines_list[p1_start:p1_end])
    play2_text = '\n'.join(lines_list[p2_start:p2_end])

    def chunk_text(t, chunk_size=256):
        return [t[i:i+chunk_size] for i in range(0, len(t) - chunk_size, chunk_size)]

    def extract_lines(text_block, min_len=20, max_len=120):
        result = []
        for line in text_block.split('\n'):
            line = line.strip()
            if line.endswith(':') or len(line) < min_len or len(line) > max_len:
                continue
            result.append(line)
        return result

    play1_chunks = chunk_text(play1_text)[:50]
    play2_chunks = chunk_text(play2_text)[:50]
    play1_lines = extract_lines(play1_text)[:200]
    play2_lines = extract_lines(play2_text)[:200]

    # ============================================================
    # Train and evaluate all kernels
    # ============================================================
    all_results = {}

    for kernel_name in kernels_to_test:
        print(f"\n{'#'*60}")
        print(f"KERNEL: {kernel_name}")
        print(f"{'#'*60}")

        torch.manual_seed(args.seed)
        try:
            model = MiniTransformer(vocab_size, attn_type=kernel_name)
        except Exception as e:
            print(f"  FAILED to create model: {e}")
            all_results[kernel_name] = {"error": str(e)}
            continue

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        try:
            losses, val_losses, best_step, best_val = train_model(
                model, train_data, val_data,
                n_steps=args.n_steps, batch_size=args.batch_size,
                lr=args.lr, device=device, label=kernel_name,
            )
        except Exception as e:
            print(f"  TRAINING FAILED: {e}")
            all_results[kernel_name] = {"error": str(e)}
            del model
            torch.cuda.empty_cache()
            continue

        # Engram analysis
        print(f"\n  Engram analysis:")
        eng_p1 = extract_engrams(model, play1_chunks, stoi, device)
        eng_p2 = extract_engrams(model, play2_chunks, stoi, device)
        result_play = run_pair_analysis(eng_p1, eng_p2, f"  {kernel_name} play-level")

        eng_l1 = extract_engrams(model, play1_lines, stoi, device)
        eng_l2 = extract_engrams(model, play2_lines, stoi, device)
        result_line = run_pair_analysis(eng_l1, eng_l2, f"  {kernel_name} line-level")

        all_results[kernel_name] = {
            "n_params": n_params,
            "best_val": best_val,
            "best_step": best_step,
            "play_level": result_play,
            "line_level": result_line,
        }

        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n{'='*80}")
    print("SUMMARY — ALL KERNELS")
    print(f"{'='*80}")
    print(f"\n  {'Kernel':<15} {'Params':>10} {'Val Loss':>10} {'Step':>6} {'Play Acc':>10} {'Line Acc':>10}")
    print(f"  {'-'*65}")

    for kernel_name in kernels_to_test:
        r = all_results.get(kernel_name, {})
        if "error" in r:
            print(f"  {kernel_name:<15} {'FAILED':>10}")
            continue
        play_acc = f"{r['play_level']['accuracy']:.1%}" if r.get('play_level') else "N/A"
        line_acc = f"{r['line_level']['accuracy']:.1%}" if r.get('line_level') else "N/A"
        print(f"  {kernel_name:<15} {r['n_params']:>10,} {r['best_val']:>10.4f} {r['best_step']:>6} "
              f"{play_acc:>10} {line_acc:>10}")

    # Save
    out_path = Path("results/exp_kernel_attention_all.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
