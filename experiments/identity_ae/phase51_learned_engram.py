"""Phase 51: Learned optimal engram via attention pooling.

The 18% information recovery floor exists because mean pooling is not
optimized for information recovery. SVD showed V-space has effective rank
~48 — no single direction captures the signal. But mean pooling isn't
trying to find the best direction. It's just an average.

This script trains a small attention-pooling encoder that maps a sequence
of hidden states to a single vector, optimized end-to-end to minimize
continuation NLL when that vector is injected as the sole context.

    Input:  H = hidden states at layer L, shape (1, T, D)
    Encoder: learned query attends over H → single vector e (D,)
    Forward: inject e as position 0, feed continuation through frozen model
    Loss:   NLL of continuation tokens

The encoder learns whatever projection of V-space actually matters for
next-token prediction — not maximum variance (SVD), not the centroid
(mean pooling), but the direction of maximum usefulness.

Three encoder architectures tested:
  1. Attention pooling: one learned query, one cross-attention pass (~3K params)
  2. Weighted mean: learned per-dimension weights on the mean (~2K params)
  3. Multi-head attention pooling: 4 heads, concat + project (~12K params)

Training: 10K WikiText passages, 500 steps of SGD per encoder.
Eval: same 50 passages as Phase 33.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase51_learned_engram.py
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import hidden_at_layer

LAYER = 5
D = 1024
N_TRAIN = 2000
N_EVAL = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200
MAX_CTX_POS = 512
TRAIN_STEPS = 300
BATCH_SIZE = 4
LR = 1e-3


# ================================================================
# Encoder architectures
# ================================================================

class AttentionPoolEncoder(nn.Module):
    """Single learned query attends over hidden states."""
    def __init__(self, d_model=D):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

    def forward(self, H):
        """H: (B, T, D) → engram: (B, D)"""
        B = H.shape[0]
        q = self.query.expand(B, -1, -1)       # (B, 1, D)
        k = self.k_proj(H)                      # (B, T, D)
        v = self.v_proj(H)                      # (B, T, D)
        attn = torch.bmm(q, k.transpose(1, 2))  # (B, 1, T)
        attn = attn / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                 # (B, 1, D)
        return self.out_proj(out.squeeze(1))      # (B, D)


class WeightedMeanEncoder(nn.Module):
    """Learned per-dimension scaling on the mean."""
    def __init__(self, d_model=D):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, H):
        """H: (B, T, D) → engram: (B, D)"""
        return H.mean(dim=1) * self.scale + self.bias


class MultiHeadPoolEncoder(nn.Module):
    """4-head attention pooling, concat, project."""
    def __init__(self, d_model=D, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.queries = nn.Parameter(
            torch.randn(n_heads, 1, self.head_dim) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, H):
        """H: (B, T, D) → engram: (B, D)"""
        B, T, _ = H.shape
        k = self.k_proj(H).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(H).view(B, T, self.n_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, H, T, hd)
        v = v.permute(0, 2, 1, 3)

        q = self.queries.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, 1, hd)
        attn = torch.matmul(q, k.transpose(-1, -2))  # (B, H, 1, T)
        attn = attn / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, 1, hd)
        out = out.squeeze(2)  # (B, H, hd)
        out = out.reshape(B, -1)  # (B, D)
        return self.out_proj(out)


# ================================================================
# Forward through frozen model with engram injection
# ================================================================

def forward_with_engram(model, engram, continuation_ids, device):
    """Forward pass: [engram] + continuation[:-1] → predict continuation[1:].

    engram: (B, D) — injected as the first hidden position
    continuation_ids: (B, M) — the continuation tokens

    Returns: mean NLL over continuation tokens
    """
    B, M = continuation_ids.shape
    # Embed continuation tokens (skip last, predict next)
    cont_emb = model.drop(model.tok_emb(continuation_ids[:, :-1]))  # (B, M-1, D)
    # Prepend engram as position 0
    eng_pos = engram.unsqueeze(1)  # (B, 1, D)
    h = torch.cat([eng_pos, cont_emb], dim=1)  # (B, M, D)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]

    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    logits = model.lm_head(h)  # (B, M, V)

    # Predict continuation[1:] from positions [1..M-1] (after engram)
    pred = logits[:, 1:, :]  # (B, M-2, V) — skip engram position
    target = continuation_ids[:, 1:]  # (B, M-1)
    # Align: pred at position i predicts target at position i
    # But pred has M-2 positions (engram + M-2 cont tokens → M-1 logits, skip first)
    # Actually: h has [eng, cont[0], cont[1], ..., cont[M-2]]
    # logits[0] = from eng, predicts cont[0]... but we want to predict cont[1:]
    # logits[1] = from cont[0], predicts cont[1] ✓
    # So pred = logits[:, 1:, :] has M-1 positions predicting cont[1], cont[2], ...
    # But target = continuation_ids[:, 1:] has M-1 tokens
    # Wait, continuation_ids[:, :-1] has M-1 tokens, so cont_emb is (B, M-1, D)
    # h is (B, M, D) = [eng, cont[0..M-2]]
    # logits is (B, M, V)
    # logits[0] predicts from eng → should predict cont[0]
    # logits[i] predicts from cont[i-1] → should predict cont[i]
    # So target should be continuation_ids[:, :] but we only have M-1 embedded
    # Let me redo: targets for logits are continuation_ids[0..M-1]
    # logits[0] → cont[0], logits[1] → cont[1], ..., logits[M-1] → cont[M-1]
    # But cont[M-1] = continuation_ids[M-1] and we need continuation_ids to have M tokens

    # Simpler: the full target sequence
    target_full = continuation_ids  # (B, M)
    pred_full = logits  # (B, M, V)
    V = pred_full.shape[-1]
    nll = F.cross_entropy(pred_full.reshape(-1, V), target_full.reshape(-1),
                          reduction="mean")
    return nll


@torch.no_grad()
def extract_hidden(model, ids_t, layer):
    """Extract hidden states at a layer."""
    return hidden_at_layer(model, ids_t, layer)


# ================================================================
# Training loop
# ================================================================

def train_encoder(encoder, model, train_data, device, n_steps=TRAIN_STEPS,
                  lr=LR, batch_size=BATCH_SIZE):
    """Train the encoder to minimize continuation NLL.

    train_data: list of (context_ids, continuation_ids) pairs
    """
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    encoder.train()
    # Keep model in eval but allow gradients to flow through it
    model.eval()

    losses = []
    t0 = time.time()

    for step in range(n_steps):
        batch_indices = random.sample(range(len(train_data)), batch_size)

        total_loss = 0.0
        for idx in batch_indices:
            context_ids, continuation_ids = train_data[idx]
            ctx_t = context_ids.unsqueeze(0).to(device)
            cont_t = continuation_ids.unsqueeze(0).to(device)

            # Extract hidden states at target layer (with grad for encoder)
            H = hidden_at_layer(model, ctx_t, LAYER)  # (1, T, D)
            # Encode
            engram = encoder(H)  # (1, D)
            # Forward with engram and get NLL
            nll = forward_with_engram(model, engram, cont_t, device)
            total_loss = total_loss + nll / batch_size

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(total_loss.item())
        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"    step {step+1:4d}  loss {avg:.4f}  "
                  f"lr {scheduler.get_last_lr()[0]:.1e}  ({time.time()-t0:.0f}s)")

    encoder.eval()
    return losses


# ================================================================
# Evaluation
# ================================================================

@torch.no_grad()
def eval_recovery(encoder, model, eval_data, device):
    """Measure information recovery: no_context, full_context, mean_engram,
    learned_engram."""
    from experiments.identity_ae.phase33_engram_context import (
        forward_segments, segments_len, continuation_nll as phase33_nll,
    )

    nll_no = []
    nll_full = []
    nll_mean = []
    nll_learned = []

    encoder.eval()
    for context_ids, continuation_ids in eval_data:
        ctx_t = context_ids.unsqueeze(0).to(device)

        # Mean engram
        H = hidden_at_layer(model, ctx_t, LAYER)
        mean_eng = H.mean(dim=1).squeeze(0).detach()

        # Learned engram
        learned_eng = encoder(H).squeeze(0).detach()

        # No context
        nll_no.append(phase33_nll(model, [], continuation_ids, device))
        # Full context
        nll_full.append(phase33_nll(model, [("tokens", context_ids)],
                                    continuation_ids, device))
        # Mean engram
        nll_mean.append(phase33_nll(model, [("hidden", mean_eng)],
                                    continuation_ids, device))
        # Learned engram
        nll_learned.append(phase33_nll(model, [("hidden", learned_eng)],
                                       continuation_ids, device))

    def avg(lst):
        return sum(lst) / len(lst)

    no = avg(nll_no)
    full = avg(nll_full)
    mean = avg(nll_mean)
    learned = avg(nll_learned)
    gap = no - full

    return {
        "no_context": no,
        "full_context": full,
        "mean_engram": mean,
        "learned_engram": learned,
        "gap": gap,
        "mean_recovery": (no - mean) / gap if gap > 0 else 0,
        "learned_recovery": (no - learned) / gap if gap > 0 else 0,
    }


# ================================================================
# Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase51")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    print("Loading model...")
    model, cfg = load_model(device)
    model.eval()
    # Enable grad flow through frozen model (needed for encoder training)
    for p in model.parameters():
        p.requires_grad = False  # don't update model, but allow grad flow

    # Load WikiText
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    train_ds = splits["train"]
    val_ds = splits["validation"]

    # Prepare training data: (context_ids, continuation_ids) pairs
    print("Preparing training data...")
    torch.manual_seed(42)
    train_indices = torch.randperm(len(train_ds))[:N_TRAIN].tolist()
    train_data = []
    for idx in train_indices:
        item = train_ds[idx]
        ids = item[0] if isinstance(item, (list, tuple)) else item
        ids = ids[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        context_ids = ids[:CONTEXT_LEN]
        continuation_ids = ids[CONTEXT_LEN:]
        train_data.append((context_ids, continuation_ids))
    print(f"  train passages: {len(train_data)}")

    # Prepare eval data
    torch.manual_seed(0)
    eval_indices = torch.randperm(len(val_ds))[:N_EVAL].tolist()
    eval_data = []
    for idx in eval_indices:
        item = val_ds[idx]
        ids = item[0] if isinstance(item, (list, tuple)) else item
        ids = ids[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        context_ids = ids[:CONTEXT_LEN]
        continuation_ids = ids[CONTEXT_LEN:]
        eval_data.append((context_ids, continuation_ids))
    print(f"  eval passages: {len(eval_data)}")

    # ============================================================
    # Baseline: mean pooling
    # ============================================================
    print(f"\n{'='*60}")
    print("BASELINE: mean-pooled engram")
    print("=" * 60)

    class MeanEncoder(nn.Module):
        def forward(self, H):
            return H.mean(dim=1)

    baseline = eval_recovery(MeanEncoder(), model, eval_data, device)
    print(f"  no_context NLL   : {baseline['no_context']:.4f}")
    print(f"  full_context NLL : {baseline['full_context']:.4f}")
    print(f"  mean_engram NLL  : {baseline['mean_engram']:.4f}")
    print(f"  gap              : {baseline['gap']:.4f}")
    print(f"  mean recovery    : {baseline['mean_recovery']:.1%}")

    # ============================================================
    # Train and evaluate each encoder
    # ============================================================
    encoders = [
        ("attention_pool", AttentionPoolEncoder(D)),
        ("weighted_mean", WeightedMeanEncoder(D)),
        ("multihead_pool", MultiHeadPoolEncoder(D, n_heads=4)),
    ]

    all_results = {"baseline": baseline}

    for name, encoder in encoders:
        encoder = encoder.to(device)
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"\n{'='*60}")
        print(f"ENCODER: {name} ({n_params:,} params)")
        print("=" * 60)

        losses = train_encoder(encoder, model, train_data, device)

        print(f"\n  Evaluating...")
        result = eval_recovery(encoder, model, eval_data, device)
        print(f"  no_context NLL      : {result['no_context']:.4f}")
        print(f"  full_context NLL    : {result['full_context']:.4f}")
        print(f"  mean_engram NLL     : {result['mean_engram']:.4f}")
        print(f"  learned_engram NLL  : {result['learned_engram']:.4f}")
        print(f"  gap                 : {result['gap']:.4f}")
        print(f"  mean recovery       : {result['mean_recovery']:.1%}")
        print(f"  learned recovery    : {result['learned_recovery']:.1%}")
        delta = result['learned_recovery'] - result['mean_recovery']
        print(f"  delta vs mean       : {delta:+.1%}")

        all_results[name] = {**result, "n_params": n_params,
                             "final_train_loss": sum(losses[-20:]) / 20}

        # Save encoder
        torch.save(encoder.state_dict(), results_dir / f"{name}.pt")

        # Free
        del encoder
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 51 SUMMARY: learned engram encoders")
    print("=" * 72)
    print(f"  Layer: {LAYER}")
    print(f"  Train passages: {len(train_data)}, steps: {TRAIN_STEPS}")
    print(f"  Eval passages: {len(eval_data)}")
    print(f"  Context: {CONTEXT_LEN} tokens, continuation: {PASSAGE_LEN - CONTEXT_LEN} tokens")
    print()
    print(f"  {'encoder':>20}  {'recovery':>8}  {'vs mean':>8}  {'NLL':>7}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}")
    mean_r = baseline['mean_recovery']
    print(f"  {'mean_pooled':>20}  {mean_r:>7.1%}  {'---':>8}  "
          f"{baseline['mean_engram']:>7.4f}")
    for name in ["attention_pool", "weighted_mean", "multihead_pool"]:
        r = all_results[name]
        delta = r['learned_recovery'] - mean_r
        print(f"  {name:>20}  {r['learned_recovery']:>7.1%}  {delta:>+7.1%}  "
              f"{r['learned_engram']:>7.4f}")

    print(f"\n  full_context NLL: {baseline['full_context']:.4f}  "
          f"(upper bound, {CONTEXT_LEN} tokens)")
    print(f"  no_context NLL:   {baseline['no_context']:.4f}  (lower bound)")

    with open(results_dir / "learned_engram.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
