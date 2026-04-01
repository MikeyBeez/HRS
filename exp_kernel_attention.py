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

class DotProductAttention(nn.Module):
    """Standard scaled dot-product attention."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class ExponentialKernelAttention(nn.Module):
    """Attention using exponential kernel (negative squared distance)."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, temperature=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.temperature = temperature if temperature else float(self.head_dim)

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

        # Negative squared Euclidean distance via ||q-k||² = ||q||² + ||k||² - 2q·k
        # This avoids materializing the (B, H, T, T, D) intermediate tensor
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        dot = q @ k.transpose(-2, -1)              # (B, H, T, T)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot  # (B, H, T, T)

        scores = -distances / self.temperature
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1, use_exponential=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if use_exponential:
            self.attn = ExponentialKernelAttention(d_model, n_heads, max_seq_len, dropout)
        else:
            self.attn = DotProductAttention(d_model, n_heads, max_seq_len, dropout)
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
                 max_seq_len=256, dropout=0.1, use_exponential=False):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, dropout, use_exponential)
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
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    text, stoi, itos, plays = load_shakespeare()
    vocab_size = len(stoi)
    print(f"Shakespeare: {len(text)} chars, {vocab_size} unique, {len(plays)} plays")
    for name, start, end in plays[:5]:
        print(f"  {name}: lines {start}-{end}")

    # Encode
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # ============================================================
    # Train Model A: Dot Product
    # ============================================================
    print(f"\n{'='*60}")
    print("MODEL A: Standard Dot Product Attention")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    model_a = MiniTransformer(vocab_size, use_exponential=False)
    n_params = sum(p.numel() for p in model_a.parameters())
    print(f"Parameters: {n_params:,}")

    losses_a, val_losses_a, best_step_a, best_val_a = train_model(
        model_a, train_data, val_data,
        n_steps=args.n_steps, batch_size=args.batch_size,
        lr=args.lr, device=device, label="DotProd",
    )

    # ============================================================
    # Train Model B: Exponential Kernel
    # ============================================================
    print(f"\n{'='*60}")
    print("MODEL B: Exponential Kernel Attention")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    model_b = MiniTransformer(vocab_size, use_exponential=True)
    print(f"Parameters: {sum(p.numel() for p in model_b.parameters()):,}")

    losses_b, val_losses_b, best_step_b, best_val_b = train_model(
        model_b, train_data, val_data,
        n_steps=args.n_steps, batch_size=args.batch_size,
        lr=args.lr, device=device, label="ExpKern",
    )

    # ============================================================
    # Training comparison
    # ============================================================
    print(f"\n{'='*60}")
    print("TRAINING COMPARISON")
    print(f"{'='*60}")
    print(f"  Best val loss — DotProd: {best_val_a:.4f} (step {best_step_a})")
    print(f"  Best val loss — ExpKern: {best_val_b:.4f} (step {best_step_b})")

    # ============================================================
    # Engram analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("ENGRAM ANALYSIS")
    print(f"{'='*60}")

    lines = text.split('\n')

    play1_name, p1_start, p1_end = plays[0]
    play2_name, p2_start, p2_end = plays[1]

    play1_text = '\n'.join(lines[p1_start:p1_end])
    play2_text = '\n'.join(lines[p2_start:p2_end])

    # Play-level: 256-char segments from each half
    def chunk_text(t, chunk_size=256):
        return [t[i:i+chunk_size] for i in range(0, len(t) - chunk_size, chunk_size)]

    play1_chunks = chunk_text(play1_text)[:50]
    play2_chunks = chunk_text(play2_text)[:50]

    print(f"\nPlay-level (256-char segments): {play1_name} ({len(play1_chunks)}) vs {play2_name} ({len(play2_chunks)})")

    eng_a_p1 = extract_engrams(model_a, play1_chunks, stoi, device)
    eng_a_p2 = extract_engrams(model_a, play2_chunks, stoi, device)
    result_a_play = run_pair_analysis(eng_a_p1, eng_a_p2, "DotProd play-level")

    eng_b_p1 = extract_engrams(model_b, play1_chunks, stoi, device)
    eng_b_p2 = extract_engrams(model_b, play2_chunks, stoi, device)
    result_b_play = run_pair_analysis(eng_b_p1, eng_b_p2, "ExpKern play-level")

    # Line-level: individual dialogue lines (more pairs)
    def extract_lines(text_block, min_len=20, max_len=120):
        result = []
        for line in text_block.split('\n'):
            line = line.strip()
            # Skip character names (end with :) and very short/long lines
            if line.endswith(':') or len(line) < min_len or len(line) > max_len:
                continue
            result.append(line)
        return result

    play1_lines = extract_lines(play1_text)[:200]
    play2_lines = extract_lines(play2_text)[:200]

    print(f"\nLine-level ({len(play1_lines)} + {len(play2_lines)} lines):")

    eng_a_l1 = extract_engrams(model_a, play1_lines, stoi, device)
    eng_a_l2 = extract_engrams(model_a, play2_lines, stoi, device)
    result_a_line = run_pair_analysis(eng_a_l1, eng_a_l2, "DotProd line-level")

    eng_b_l1 = extract_engrams(model_b, play1_lines, stoi, device)
    eng_b_l2 = extract_engrams(model_b, play2_lines, stoi, device)
    result_b_line = run_pair_analysis(eng_b_l1, eng_b_l2, "ExpKern line-level")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Training (best checkpoint):")
    print(f"    DotProd: val={best_val_a:.4f} @ step {best_step_a}")
    print(f"    ExpKern: val={best_val_b:.4f} @ step {best_step_b}")

    if result_a_play and result_b_play:
        print(f"\n  Play-level engram separation (256-char segments):")
        print(f"    DotProd: gap={result_a_play['gap']:.4f}, accuracy={result_a_play['accuracy']:.1%}")
        print(f"    ExpKern: gap={result_b_play['gap']:.4f}, accuracy={result_b_play['accuracy']:.1%}")
        play_imp = result_b_play['accuracy'] - result_a_play['accuracy']
        print(f"    ExpKern vs DotProd: {play_imp:+.1%}")

    if result_a_line and result_b_line:
        print(f"\n  Line-level engram separation (individual dialogue):")
        print(f"    DotProd: gap={result_a_line['gap']:.4f}, accuracy={result_a_line['accuracy']:.1%}")
        print(f"    ExpKern: gap={result_b_line['gap']:.4f}, accuracy={result_b_line['accuracy']:.1%}")
        line_imp = result_b_line['accuracy'] - result_a_line['accuracy']
        print(f"    ExpKern vs DotProd: {line_imp:+.1%}")

    # Save
    out_path = Path("results/exp_kernel_attention.json")
    out_path.parent.mkdir(exist_ok=True)
    results = {
        "n_steps": args.n_steps,
        "best_val": {"dot_product": best_val_a, "exponential": best_val_b},
        "best_step": {"dot_product": best_step_a, "exponential": best_step_b},
        "play_level": {
            "dot_product": result_a_play,
            "exponential": result_b_play,
        } if result_a_play else None,
        "line_level": {
            "dot_product": result_a_line,
            "exponential": result_b_line,
        } if result_a_line else None,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
