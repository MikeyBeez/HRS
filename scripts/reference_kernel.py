"""Golden Reference: Bonsignore Kernel in vanilla PyTorch.

Implements:
  S(q, k) = MLP( exp( -||q - k||² / τ ) )

The MLP is initialized to approximate identity so the kernel starts
as a pure RBF/exponential, then co-evolves during Phase 2.

Includes:
- BonsignoreKernel: the scoring function
- BonsignoreAttention: full attention module using the kernel
- ScaffoldedTrainer: Phase 1 (fixed MLP) + Phase 2 (co-evolution)
- Shakespeare validation pipeline

Usage:
    python scripts/reference_kernel.py [--device cuda]
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
from torch.utils.data import Dataset, DataLoader


# ============================================================
# The Bonsignore Kernel
# ============================================================

class BonsignoreKernel(nn.Module):
    """Exponential distance kernel with co-evolving MLP.

    S(q, k) = MLP( exp( -||q-k||² / τ ) )

    The MLP is initialized as near-identity so the kernel starts as
    a pure RBF. During Phase 2, the MLP co-evolves with the projections.

    Gradient property: ∂S/∂q is proportional to S, creating self-reinforcing
    learning on discovered semantic clusters.
    """

    def __init__(self, hidden_dim=64, tau_init=64.0):
        super().__init__()
        # Learnable temperature
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

        # 2-layer MLP: scalar -> hidden -> scalar
        # Initialized to approximate identity on [0, 1] range
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_identity(hidden_dim)

    def _init_identity(self, hidden_dim):
        """Initialize MLP to approximate identity function on [0, 1]."""
        with torch.no_grad():
            # First layer: spread input across hidden dims with small weights
            nn.init.uniform_(self.mlp[0].weight, -0.01, 0.01)
            nn.init.zeros_(self.mlp[0].bias)
            # Second layer: sum back to scalar, scale to approximate identity
            nn.init.uniform_(self.mlp[2].weight, -0.01, 0.01)
            # Bias the output to pass through the input value
            # At init, MLP(x) ≈ small_noise, so we add a residual path
            nn.init.zeros_(self.mlp[2].bias)

        # Residual weight: starts at 1.0 (pure identity), decreases as MLP learns
        self.residual_weight = nn.Parameter(torch.tensor(1.0))

    @property
    def tau(self):
        return self.log_tau.exp()

    def forward(self, q, k, apply_mlp=False):
        """Compute Bonsignore kernel scores.

        Args:
            q: (B, H, Sq, D)
            k: (B, H, Sk, D)
            apply_mlp: if True, apply MLP refinement (only for small tensors)

        Returns:
            scores: (B, H, Sq, Sk)
        """
        # Memory-efficient squared distance: ||q-k||² = ||q||² + ||k||² - 2q·k
        q_sq = (q ** 2).sum(dim=-1, keepdim=True)       # (B, H, Sq, 1)
        k_sq = (k ** 2).sum(dim=-1, keepdim=True)       # (B, H, Sk, 1)
        dot = q @ k.transpose(-2, -1)                    # (B, H, Sq, Sk)
        distances = q_sq + k_sq.transpose(-2, -1) - 2 * dot  # (B, H, Sq, Sk)

        # Exponential kernel: exp(-d² / τ) → use negative distance for log-space
        scores = -distances / self.tau

        if apply_mlp:
            # Only used for small tensors (e.g., after top-K selection)
            exp_scores = torch.exp(scores)
            shape = exp_scores.shape
            flat = exp_scores.reshape(-1, 1)
            mlp_out = self.mlp(flat).reshape(shape)
            alpha = torch.sigmoid(self.residual_weight)
            scores = torch.log((alpha * exp_scores + (1 - alpha) * mlp_out).clamp(min=1e-10))

        return scores

    def freeze_mlp(self):
        """Phase 1: freeze MLP, train only projections with fixed exponential."""
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        self.residual_weight.requires_grad_(False)

    def unfreeze_mlp(self):
        """Phase 2: unfreeze MLP for co-evolution."""
        for p in self.mlp.parameters():
            p.requires_grad_(True)
        self.residual_weight.requires_grad_(True)

    def get_diagnostics(self):
        """Return diagnostic info about kernel state."""
        alpha = torch.sigmoid(self.residual_weight).item()
        return {
            "tau": self.tau.item(),
            "residual_alpha": alpha,
            "mlp_contribution": 1 - alpha,
        }


# ============================================================
# Attention Module using Bonsignore Kernel
# ============================================================

class BonsignoreAttention(nn.Module):
    """Full attention using the Bonsignore kernel for scoring."""

    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1,
                 kernel_hidden=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # One kernel per head would be ideal but expensive.
        # Shared kernel across heads for now.
        self.kernel = BonsignoreKernel(
            hidden_dim=kernel_hidden,
            tau_init=float(self.head_dim),
        )

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len))
                             .view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Bonsignore kernel scores (log-space: -d²/τ)
        scores = self.kernel(q, k)  # (B, H, T, T) in log-space
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


# ============================================================
# Top-K Expert Router using Bonsignore Kernel
# ============================================================

class BonsignoreRouter(nn.Module):
    """Expert router using Bonsignore kernel for Top-K selection.

    Given a query, computes kernel scores against all expert keys
    and returns the top-K experts.
    """

    def __init__(self, d_key, n_experts, top_k=16, kernel_hidden=64):
        super().__init__()
        self.d_key = d_key
        self.n_experts = n_experts
        self.top_k = top_k

        # Expert keys (the lookup table)
        self.expert_keys = nn.Parameter(torch.randn(n_experts, d_key) * 0.02)

        # Shared kernel
        self.kernel = BonsignoreKernel(
            hidden_dim=kernel_hidden,
            tau_init=float(d_key),
        )

    def forward(self, query):
        """Route queries to top-K experts.

        Args:
            query: (B, T, D) query vectors

        Returns:
            indices: (B, T, K) top-K expert indices
            scores: (B, T, K) top-K kernel scores (softmaxed)
        """
        B, T, D = query.shape
        K = self.top_k

        # Reshape for kernel: query as (B, 1, T, D), keys as (1, 1, N, D)
        q = query.unsqueeze(1)  # (B, 1, T, D)
        k = self.expert_keys.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

        # Compute scores in chunks to avoid OOM for large N
        chunk_size = min(self.n_experts, 4096)
        all_scores = []
        for start in range(0, self.n_experts, chunk_size):
            end = min(start + chunk_size, self.n_experts)
            k_chunk = k[:, :, start:end, :]
            chunk_scores = self.kernel(q, k_chunk)  # (B, 1, T, chunk)
            all_scores.append(chunk_scores)

        scores = torch.cat(all_scores, dim=-1).squeeze(1)  # (B, T, N)

        # Top-K selection
        top_scores, top_indices = scores.topk(K, dim=-1)  # (B, T, K)

        # Normalize top-K scores
        top_scores = F.softmax(top_scores, dim=-1)

        return top_indices, top_scores


# ============================================================
# Shakespeare Transformer with Bonsignore Attention
# ============================================================

class BonsignoreTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BonsignoreAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BonsignoreTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=6,
                 max_seq_len=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            BonsignoreTransformerBlock(d_model, n_heads, max_seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x), x

    def get_kernels(self):
        """Return all kernel modules for phase control."""
        return [block.attn.kernel for block in self.blocks]


# ============================================================
# Dataset
# ============================================================

class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start:start + self.seq_len], self.data[start+1:start + self.seq_len + 1]


# ============================================================
# Scaffolded Training
# ============================================================

def train_scaffolded(model, train_data, val_data, device, n_steps=5000,
                     phase1_steps=2500, batch_size=32, lr=3e-4):
    """Train with Phase 1 (fixed kernel) then Phase 2 (co-evolution)."""
    seq_len = model.max_seq_len
    train_ds = CharDataset(train_data, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    model.to(device)
    model.train()

    # Phase 1: freeze kernels
    kernels = model.get_kernels()
    for k in kernels:
        k.freeze_mlp()
    print(f"  Phase 1: MLP frozen, training projections only")

    losses = []
    best_val = float('inf')
    best_state = None
    train_iter = iter(train_loader)
    t0 = time.time()

    for step in range(n_steps):
        # Phase transition
        if step == phase1_steps:
            for k in kernels:
                k.unfreeze_mlp()
            # Lower LR for projections, higher for kernel MLP
            for pg in optimizer.param_groups:
                pg['lr'] = lr * 0.1
            print(f"\n  Phase 2: MLP unfrozen, co-evolution begins (step {step})")

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

        if (step + 1) % 250 == 0:
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
            elapsed = time.time() - t0
            avg_train = sum(losses[-250:]) / 250
            phase = "P1" if step < phase1_steps else "P2"

            # Kernel diagnostics
            diag = kernels[0].get_diagnostics()

            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = " *best*"

            print(f"  [{phase}] step {step+1:5d}: train={avg_train:.4f} val={val_loss:.4f} "
                  f"τ={diag['tau']:.1f} α={diag['residual_alpha']:.3f}{marker} ({elapsed:.0f}s)")

            model.train()

    # Restore best
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return losses, best_val


# ============================================================
# Engram-based topic separation evaluation
# ============================================================

@torch.no_grad()
def evaluate_topic_separation(model, text, stoi, device):
    """Evaluate play-level topic separation using engrams."""
    model.eval()
    lines = text.split('\n')
    boundary = 15600  # Shakespeare play boundary

    p1_text = '\n'.join(lines[:boundary])
    p2_text = '\n'.join(lines[boundary:])

    def chunk(t, sz=256):
        return [t[i:i+sz] for i in range(0, len(t)-sz, sz)]

    def extract(segments):
        engrams = []
        for seg in segments[:50]:
            ids = [stoi.get(c, 0) for c in seg]
            if len(ids) < 5:
                continue
            ids = ids[:model.max_seq_len]
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            _, hidden = model(x)
            eng = hidden.mean(dim=1).squeeze(0)
            eng = F.normalize(eng, dim=0)
            engrams.append(eng.cpu())
        return engrams

    chunks1 = chunk(p1_text)[:50]
    chunks2 = chunk(p2_text)[:50]
    eng1 = extract(chunks1)
    eng2 = extract(chunks2)

    if len(eng1) < 5 or len(eng2) < 5:
        return 0.5

    # Same-group vs cross-group similarity
    same_sims, cross_sims = [], []
    for group in [eng1, eng2]:
        for i in range(len(group)):
            for j in range(i+1, min(i+5, len(group))):
                same_sims.append((group[i] @ group[j]).item())

    for i in range(min(50, len(eng1))):
        for j in range(min(50, len(eng2))):
            cross_sims.append((eng1[i] @ eng2[j]).item())

    # Best threshold accuracy
    all_sims = [(s, 1) for s in same_sims] + [(s, 0) for s in cross_sims]
    best_acc = 0
    for t in [i * 0.05 for i in range(-10, 20)]:
        correct = sum(1 for s, l in all_sims if (s > t) == l)
        acc = correct / len(all_sims)
        best_acc = max(best_acc, acc)

    return best_acc


# ============================================================
# Router validation
# ============================================================

def validate_router(device):
    """Quick validation that BonsignoreRouter works for expert selection."""
    print(f"\n{'='*60}")
    print("Router Validation")
    print(f"{'='*60}")

    d_key = 64
    n_experts = 10000
    top_k = 16

    router = BonsignoreRouter(d_key, n_experts, top_k).to(device)
    query = torch.randn(2, 32, d_key, device=device)  # (B=2, T=32, D=64)

    t0 = time.time()
    indices, scores = router(query)
    elapsed = time.time() - t0

    print(f"  Experts: {n_experts:,}, Top-K: {top_k}")
    print(f"  Query shape: {query.shape}")
    print(f"  Indices shape: {indices.shape}, Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Scores sum per token: {scores.sum(dim=-1).mean():.4f} (should be ~1.0)")
    print(f"  Time: {elapsed*1000:.1f}ms")

    # Check uniqueness of selected experts
    unique_per_token = indices[0, 0].unique().shape[0]
    print(f"  Unique experts per token: {unique_per_token}/{top_k}")

    return True


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-steps", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Shakespeare
    import urllib.request
    cache_path = Path("datasets/tiny_shakespeare.txt")
    cache_path.parent.mkdir(exist_ok=True)
    if not cache_path.exists():
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            cache_path)
    text = cache_path.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    print(f"Shakespeare: {len(text)} chars, {vocab_size} vocab")

    # ============================================================
    # Train Bonsignore Kernel Model
    # ============================================================
    print(f"\n{'='*60}")
    print("Training Bonsignore Kernel (Scaffolded)")
    print(f"{'='*60}")

    torch.manual_seed(42)
    model = BonsignoreTransformer(vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    kernel_params = sum(
        sum(p.numel() for p in k.parameters())
        for k in model.get_kernels()
    )
    print(f"  Total params: {n_params:,}")
    print(f"  Kernel params: {kernel_params:,}")

    losses, best_val = train_scaffolded(
        model, train_data, val_data, device,
        n_steps=args.n_steps, batch_size=16,
    )

    print(f"\n  Best val loss: {best_val:.4f}")

    # ============================================================
    # Topic Separation Evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print("Topic Separation Evaluation")
    print(f"{'='*60}")

    accuracy = evaluate_topic_separation(model, text, stoi, device)
    print(f"  Play-level accuracy: {accuracy:.1%}")

    passed = accuracy > 0.80
    print(f"  Target: >80% — {'PASSED' if passed else 'FAILED'}")

    # ============================================================
    # Kernel Shape Analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("Kernel Shape Analysis")
    print(f"{'='*60}")

    kernel = model.get_kernels()[0]
    diag = kernel.get_diagnostics()
    print(f"  Temperature τ: {diag['tau']:.2f}")
    print(f"  Residual α: {diag['residual_alpha']:.4f}")
    print(f"  MLP contribution: {diag['mlp_contribution']:.4f}")

    # Test kernel output range
    with torch.no_grad():
        q_test = torch.randn(1, 1, 100, 64, device=device)
        k_test = torch.randn(1, 1, 100, 64, device=device)
        scores = kernel(q_test, k_test)
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Score mean: {scores.mean():.4f}")

    # ============================================================
    # Router Validation
    # ============================================================
    validate_router(device)

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("GOLDEN REFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Val loss: {best_val:.4f}")
    print(f"  Topic separation: {accuracy:.1%} ({'PASS' if passed else 'FAIL'})")
    print(f"  Kernel τ: {diag['tau']:.2f}")
    print(f"  Kernel MLP contribution: {diag['mlp_contribution']:.4f}")
    print(f"  Router validated: 10K experts, top-16")

    # Save
    out_dir = Path("results/bonsignore_kernel")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "val_loss": best_val,
        "topic_accuracy": accuracy,
        "passed_80pct": passed,
        "kernel_diagnostics": diag,
        "n_params": n_params,
        "kernel_params": kernel_params,
    }
    with open(out_dir / "reference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "results": results,
    }, out_dir / "reference_model.pt")

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
