"""Phase 55: Multi-token engram (1, 2, 5, 10 tokens).

Phase 50 showed V-space has effective rank ~48 — a single vector can't
capture it. Phase 51's attention pooling lifted recovery from 18% to 28%
by choosing the right single vector. This phase tests the obvious next
step: use K tokens instead of 1.

For K tokens, the attention pooling encoder produces K output vectors
(K learned queries, each attending over the hidden states). These are
injected as K prefix positions before the continuation.

Sweep K in {1, 2, 5, 10} and measure information recovery at each.
Also test whether multi-token engrams improve routing accuracy.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase55_multi_token_engram.py
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
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine

LAYER = 5
D = 1024
N_TRAIN = 2000
N_EVAL = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200
MAX_CTX_POS = 512
TRAIN_STEPS = 500
BATCH_SIZE = 4
LR = 1e-3
K_VALUES = [1, 2, 5, 10]


# ================================================================
# Multi-token attention pooling encoder
# ================================================================

class MultiTokenEncoder(nn.Module):
    """K learned queries attend over hidden states → K output vectors."""

    def __init__(self, d_model=D, k=1):
        super().__init__()
        self.k = k
        self.queries = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.k_proj.weight)
        nn.init.eye_(self.v_proj.weight)
        nn.init.eye_(self.out_proj.weight)

    def forward(self, H):
        """H: (B, T, D) → engram: (B, K, D)"""
        B = H.shape[0]
        q = self.queries.expand(B, -1, -1)       # (B, K, D)
        k = self.k_proj(H)                        # (B, T, D)
        v = self.v_proj(H)                        # (B, T, D)
        attn = torch.bmm(q, k.transpose(1, 2))    # (B, K, T)
        attn = attn / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                   # (B, K, D)
        return self.out_proj(out)                   # (B, K, D)


# ================================================================
# Forward with multi-token engram prefix
# ================================================================

def forward_with_engram(model, engram, continuation_ids, device):
    """Forward: [engram_1, ..., engram_K] + continuation[:-1] → logits.

    engram: (B, K, D)
    continuation_ids: (B, M)
    Returns: mean NLL over continuation tokens.
    """
    B, M = continuation_ids.shape
    cont_emb = model.drop(model.tok_emb(continuation_ids[:, :-1]))  # (B, M-1, D)
    h = torch.cat([engram, cont_emb], dim=1)  # (B, K+M-1, D)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]

    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    logits = model.lm_head(h)  # (B, K+M-1, V)

    K = engram.shape[1]
    # logits[K-1] predicts from last engram token → should predict cont[0]
    # logits[K+i-1] predicts from cont[i-1] → should predict cont[i]
    # So target = continuation_ids[:, :] aligned with logits[:, K-1:]
    # But logits has K+M-1 positions total, and we want positions K-1 to K+M-2
    # predicting cont[0] to cont[M-1]
    pred = logits[:, K - 1:K - 1 + M, :]  # (B, M, V)
    target = continuation_ids  # (B, M)
    V = pred.shape[-1]
    nll = F.cross_entropy(pred.reshape(-1, V), target.reshape(-1),
                          reduction="mean")
    return nll


# ================================================================
# Forward segments (for baseline measurements)
# ================================================================

@torch.no_grad()
def forward_segments(model, segments, device):
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)
            parts.append(model.drop(model.tok_emb(ids)))
        elif kind == "hidden":
            # x is (K, D) — multi-token engram
            if x.dim() == 1:
                x = x.unsqueeze(0)  # (1, D)
            parts.append(x.unsqueeze(0).to(device))  # (1, K, D)
    h = torch.cat(parts, dim=1)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]
    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    return model.lm_head(h)


@torch.no_grad()
def continuation_nll(model, prefix_segments, cont_ids, device):
    M = len(cont_ids)
    segs = list(prefix_segments) + [("tokens", cont_ids[:-1])]
    logits = forward_segments(model, segs, device)
    plen = 0
    for kind, x in prefix_segments:
        if kind == "tokens":
            plen += len(x)
        elif kind == "hidden":
            plen += x.shape[0] if x.dim() >= 1 and x.shape[0] != D else 1
    pred = logits[:, plen:plen + M - 1, :]
    target = cont_ids[1:].unsqueeze(0).to(device)
    return float(F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                                  target.reshape(-1), reduction="mean"))


# ================================================================
# Training
# ================================================================

def train_encoder(encoder, model, train_data, device,
                  n_steps=TRAIN_STEPS, lr=LR, batch_size=BATCH_SIZE):
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    encoder.train()
    model.eval()

    losses = []
    t0 = time.time()

    for step in range(n_steps):
        batch_indices = random.sample(range(len(train_data)),
                                      min(batch_size, len(train_data)))
        total_loss = 0.0
        for idx in batch_indices:
            ctx_ids, cont_ids = train_data[idx]
            ctx_t = ctx_ids.unsqueeze(0).to(device)
            cont_t = cont_ids.unsqueeze(0).to(device)

            H = hidden_at_layer(model, ctx_t, LAYER)  # (1, T, D)
            engram = encoder(H)  # (1, K, D)
            nll = forward_with_engram(model, engram, cont_t, device)
            total_loss = total_loss + nll / batch_size

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(total_loss.item())

        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / 100
            print(f"      step {step+1:4d}  loss {avg:.4f}  ({time.time()-t0:.0f}s)")

    encoder.eval()
    return losses


# ================================================================
# Main
# ================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase55")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    print("Loading model...")
    model, cfg = load_model(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Prepare data
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    train_ds = splits["train"]
    val_ds = splits["validation"]

    torch.manual_seed(42)
    train_indices = torch.randperm(len(train_ds))[:N_TRAIN].tolist()
    train_data = []
    for idx in train_indices:
        item = train_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        train_data.append((ids[:CONTEXT_LEN], ids[CONTEXT_LEN:]))
    print(f"  train: {len(train_data)}")

    torch.manual_seed(0)
    eval_indices = torch.randperm(len(val_ds))[:N_EVAL].tolist()
    eval_data = []
    for idx in eval_indices:
        item = val_ds[idx]
        ids = (item[0] if isinstance(item, tuple) else item)[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        eval_data.append((ids[:CONTEXT_LEN], ids[CONTEXT_LEN:]))
    print(f"  eval: {len(eval_data)}")

    # Baselines
    print("\nBaselines...")
    nll_no = []
    nll_full = []
    nll_mean = []
    for ctx, cont in eval_data:
        ctx_t = ctx.unsqueeze(0).to(device)
        H = hidden_at_layer(model, ctx_t, LAYER)
        mean_eng = H.mean(dim=1).squeeze(0).detach()  # (D,)

        nll_no.append(continuation_nll(model, [], cont, device))
        nll_full.append(continuation_nll(model, [("tokens", ctx)],
                                         cont, device))
        nll_mean.append(continuation_nll(model, [("hidden", mean_eng)],
                                         cont, device))

    no = sum(nll_no) / len(nll_no)
    full = sum(nll_full) / len(nll_full)
    gap = no - full
    mean_r = (no - sum(nll_mean) / len(nll_mean)) / gap
    print(f"  no_context:  {no:.4f}")
    print(f"  full_context: {full:.4f}  ({CONTEXT_LEN} tokens)")
    print(f"  mean_engram:  {sum(nll_mean)/len(nll_mean):.4f}  "
          f"recovery {mean_r:.1%}  (1 token)")
    print(f"  gap: {gap:.4f}")

    # Train and evaluate each K
    all_results = {
        "baselines": {
            "no_context": no, "full_context": full, "gap": gap,
            "mean_recovery": mean_r, "context_tokens": CONTEXT_LEN,
        }
    }

    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K = {K} tokens")
        print("=" * 60)

        encoder = MultiTokenEncoder(D, k=K).to(device)
        n_params = sum(p.numel() for p in encoder.parameters())
        print(f"  encoder params: {n_params:,}")

        print(f"  training ({TRAIN_STEPS} steps)...")
        losses = train_encoder(encoder, model, train_data, device)

        print(f"  evaluating...")
        nll_learned = []
        with torch.no_grad():
            for ctx, cont in eval_data:
                ctx_t = ctx.unsqueeze(0).to(device)
                H = hidden_at_layer(model, ctx_t, LAYER)
                engram = encoder(H)  # (1, K, D)
                eng_flat = engram.squeeze(0)  # (K, D)
                nll_learned.append(
                    continuation_nll(model, [("hidden", eng_flat)],
                                     cont, device))

        learned_nll = sum(nll_learned) / len(nll_learned)
        learned_r = (no - learned_nll) / gap if gap > 0 else 0
        compression = CONTEXT_LEN / K

        print(f"  NLL:        {learned_nll:.4f}")
        print(f"  recovery:   {learned_r:.1%}")
        print(f"  compression: {compression:.0f}× ({CONTEXT_LEN} tokens → {K})")
        print(f"  vs mean:    {learned_r - mean_r:+.1%}")

        all_results[f"K={K}"] = {
            "k": K, "nll": learned_nll, "recovery": learned_r,
            "compression": compression, "n_params": n_params,
            "final_loss": sum(losses[-20:]) / 20,
        }

        torch.save(encoder.state_dict(),
                    results_dir / f"encoder_k{K}.pt")
        del encoder
        torch.cuda.empty_cache()

    # ============================================================
    # Routing test with K=5 engrams
    # ============================================================
    print(f"\n{'='*60}")
    print("Routing test: K=5 multi-token engram vs mean L0")
    print("=" * 60)

    tests = stratified_tests()
    encoder_k5 = MultiTokenEncoder(D, k=5).to(device)
    encoder_k5.load_state_dict(
        torch.load(str(results_dir / "encoder_k5.pt"), map_location=device,
                    weights_only=True))
    encoder_k5.eval()

    # Build keys: mean L0 (single vector) and K=5 engram (mean of 5)
    mean_keys = []
    k5_keys = []
    with torch.no_grad():
        for test in tests:
            ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

            # Mean L0
            h0 = model.drop(model.tok_emb(ids_t))
            mean_keys.append([h0.mean(dim=1).squeeze(0).cpu()])

            # K=5 at L5, mean of the 5 vectors for routing
            H = hidden_at_layer(model, ids_t, LAYER)
            eng = encoder_k5(H).squeeze(0)  # (5, D)
            k5_keys.append([eng.mean(dim=0).cpu()])

    def route(q, lib_keys):
        best_a, best_s = -1, -2.0
        for ai, keys in enumerate(lib_keys):
            for kv in keys:
                s = cosine(q, kv)
                if s > best_s:
                    best_s = s
                    best_a = ai
        return best_a

    for label, keys in [("mean L0", mean_keys), ("K=5 mean", k5_keys)]:
        n_routed = 0
        with torch.no_grad():
            for i, test in enumerate(tests):
                for para in held_out_paraphrase(test):
                    ids = tokenizer.encode(para, add_special_tokens=False)
                    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                    if "L0" in label:
                        h0 = model.drop(model.tok_emb(ids_t))
                        q = h0.mean(dim=1).squeeze(0).cpu()
                    else:
                        H = hidden_at_layer(model, ids_t, LAYER)
                        eng = encoder_k5(H).squeeze(0)
                        q = eng.mean(dim=0).cpu()
                    if route(q, keys) == i:
                        n_routed += 1
        print(f"  {label:12s}: {n_routed}/60 ({n_routed/60:.0%})")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 55 SUMMARY: multi-token engram")
    print("=" * 72)
    print(f"  Context: {CONTEXT_LEN} tokens → engram")
    print(f"  Layer: {LAYER}")
    print()
    print(f"  {'tokens':>6}  {'NLL':>7}  {'recovery':>9}  {'compress':>9}  {'vs mean':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*8}")
    print(f"  {'full':>6}  {full:>7.4f}  {'100.0%':>9}  {'1×':>9}  {'':>8}")
    print(f"  {'mean':>6}  {sum(nll_mean)/len(nll_mean):>7.4f}  "
          f"{mean_r:>8.1%}  {CONTEXT_LEN:>8.0f}×  {'---':>8}")
    for K in K_VALUES:
        r = all_results[f"K={K}"]
        delta = r["recovery"] - mean_r
        print(f"  {K:>6}  {r['nll']:>7.4f}  {r['recovery']:>8.1%}  "
              f"{r['compression']:>8.0f}×  {delta:>+7.1%}")

    with open(results_dir / "multi_token.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
