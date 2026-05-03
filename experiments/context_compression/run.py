"""Context-window compression experiment.

Substrate: GPT-2 small (12 layers, 768d, learned absolute position embeddings,
1024 token context). Frozen.

Passage shape: 1001 tokens. The first 1000 are context; token 1000 is the
target.

Compression scheme (last 1000 context tokens, indexed 0..999):
  positions [400..999] = "recent": 200 tokens preserved at 1:1 (wait, that's
    600. Let me recheck the spec.)

Spec says:
  Most recent 200 tokens: 1:1
  Next 200 tokens (positions 200-400 from end): 4:1 → 50 tokens
  Older 600 tokens (positions 400-1000 from end): 128:1 → ~5 tokens

So in 0-indexed positions of the 1000-token context:
  ids[800:1000]  →  recent 200 tokens at 1:1
  ids[600:800]   →  middle 200 tokens compressed 4:1 → 50 tokens
  ids[0:600]     →  older 600 tokens compressed 128:1 → 5 tokens

Compressed context = (5 + 50 + 200) = 255 tokens, then GPT-2 predicts
token 1000 from this 255-token sequence.

Compression: linear projection in the SEQUENCE dimension over token
embeddings. W_128: (5, 600) maps 600 embeddings → 5; W_4: (50, 200)
maps 200 embeddings → 50.

Initialization: average pooling (each compressed token = mean of its
\"bucket\" of consecutive original tokens). This is the natural neutral
starting point.

The base model is frozen; only W_4 and W_128 are trained on next-token
CE on token 1000.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/context_compression"

MODEL_NAME = "gpt2"
TOTAL_TOKENS = 1001
CONTEXT = 1000
RECENT = 200
MIDDLE = 200
OLDER = 600
RECENT_KEEP = 200
MIDDLE_KEEP = 50
OLDER_KEEP = 5
N_TRAIN = 1000
N_EVAL = 100


def avg_pool_init(in_len, out_len):
    """W of shape (out_len, in_len) that averages contiguous in_len/out_len
    chunks. Ratio 4:1 with in=200,out=50 → each output is mean of 4
    inputs. Ratio 128:1 with in=600,out=5 → each output is mean of 120
    inputs (close to 128:1 but the integer math gives 120)."""
    W = torch.zeros(out_len, in_len)
    bucket = in_len / out_len
    for i in range(out_len):
        s = int(round(i * bucket))
        e = int(round((i + 1) * bucket))
        if e <= s: e = s + 1
        W[i, s:e] = 1.0 / (e - s)
    return W


class CompressedContext(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_4 = nn.Parameter(avg_pool_init(MIDDLE, MIDDLE_KEEP))    # (50, 200)
        self.W_128 = nn.Parameter(avg_pool_init(OLDER, OLDER_KEEP))    # (5, 600)

    def forward(self, ids, wte):
        """ids: (B, 1000). Returns inputs_embeds (B, 255, D).

        Splits ids into older[0:600], middle[600:800], recent[800:1000].
        Embeds via wte (frozen). Compresses older with W_128, middle with W_4.
        Concatenates [compressed_older, compressed_middle, embed_recent].
        """
        B = ids.shape[0]
        older = ids[:, :OLDER]                # (B, 600)
        middle = ids[:, OLDER:OLDER + MIDDLE] # (B, 200)
        recent = ids[:, OLDER + MIDDLE:]      # (B, 200)
        emb_older = wte(older)                # (B, 600, D)
        emb_middle = wte(middle)              # (B, 200, D)
        emb_recent = wte(recent)              # (B, 200, D)
        # Apply linear over sequence dim: out = W @ emb (where W is (out_len, in_len))
        comp_older = torch.einsum("oi,bid->bod", self.W_128, emb_older)   # (B, 5, D)
        comp_middle = torch.einsum("oi,bid->bod", self.W_4, emb_middle)   # (B, 50, D)
        return torch.cat([comp_older, comp_middle, emb_recent], dim=1)


def get_logits_baseline(model, ids):
    """Full-context forward. Returns logits at last position. ids: (B, 1000)."""
    out = model(ids, return_dict=True)
    return out.logits[:, -1, :]  # (B, V)


def get_logits_compressed(model, compressor, ids):
    """Compress and run forward via inputs_embeds. ids: (B, 1000).
    Returns logits at last (compressed) position."""
    inputs_embeds = compressor(ids, model.transformer.wte)  # (B, 255, D)
    out = model(inputs_embeds=inputs_embeds, return_dict=True)
    return out.logits[:, -1, :]  # (B, V)


def build_passages(tokenizer, n_passages, total_tokens):
    """Tokenize WT-103 and slice into n_passages non-overlapping chunks."""
    print(f"  loading WT-103 train ...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    needed_tokens = n_passages * total_tokens + 1000
    all_ids = []
    for row in ds:
        t = row["text"]
        if not t.strip():
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= needed_tokens:
            break
    print(f"  collected {len(all_ids):,} tokens")
    chunks = np.array(
        [all_ids[i*total_tokens:(i+1)*total_tokens] for i in range(n_passages)],
        dtype=np.int64,
    )
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    print("=== Context Compression Experiment ===\n")
    print(f"Loading {MODEL_NAME} (fp32) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  {MODEL_NAME}: {n_total/1e6:.0f}M params, all FROZEN")

    print(f"\nBuilding {N_TRAIN + N_EVAL} passages of {TOTAL_TOKENS} tokens ...")
    n_passages = N_TRAIN + N_EVAL
    cache = EXP / "data/passages.npy"
    if cache.exists():
        print(f"  loading cached passages from {cache}")
        passages = np.load(cache)
    else:
        passages = build_passages(tokenizer, n_passages, TOTAL_TOKENS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, passages)
        print(f"  saved to {cache}")
    print(f"  shape: {passages.shape}")
    assert passages.shape == (n_passages, TOTAL_TOKENS)

    train_passages = torch.tensor(passages[:N_TRAIN], dtype=torch.long, device=device)
    eval_passages = torch.tensor(passages[N_TRAIN:N_TRAIN + N_EVAL],
                                  dtype=torch.long, device=device)

    # ---------- Init compressor ----------
    d_model = model.config.n_embd
    compressor = CompressedContext(d_model).to(device)
    print(f"\nCompressor params: "
          f"W_4 {tuple(compressor.W_4.shape)} = {compressor.W_4.numel()}, "
          f"W_128 {tuple(compressor.W_128.shape)} = {compressor.W_128.numel()}")
    print(f"  initialized to average-pool weights "
          f"(each compressed token = mean of contiguous bucket)")

    # ---------- Training ----------
    opt = torch.optim.Adam(compressor.parameters(), lr=args.lr)
    print(f"\n=== Training projections (frozen base, "
          f"{args.steps} steps, batch={args.batch}, lr={args.lr}) ===")
    history = []
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        idx = rng.integers(0, N_TRAIN, size=args.batch)
        batch_ids = train_passages[idx]   # (B, 1001)
        ctx = batch_ids[:, :CONTEXT]      # (B, 1000)
        target = batch_ids[:, CONTEXT]    # (B,)
        logits = get_logits_compressed(model, compressor, ctx)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            history.append({"step": step + 1, "loss": float(loss.item())})
            print(f"  step {step+1:5d}/{args.steps}  loss={loss.item():.3f}  "
                  f"wall={time.time()-t0:.0f}s")
    train_wall = time.time() - t0
    print(f"  total training wall: {train_wall:.0f}s")

    # ---------- Eval ----------
    print(f"\n=== Evaluating on {N_EVAL} held-out passages ===")
    compressor.eval()
    eval_records = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(N_EVAL):
            ids = eval_passages[i:i+1]      # (1, 1001)
            ctx = ids[:, :CONTEXT]
            target = ids[:, CONTEXT]        # (1,)
            # Baseline: full 1000-token context
            base_logits = get_logits_baseline(model, ctx)  # (1, V)
            # Compressed: 255-token context
            comp_logits = get_logits_compressed(model, compressor, ctx)
            # Per-passage metrics
            base_logp = F.log_softmax(base_logits, dim=-1)
            comp_logp = F.log_softmax(comp_logits, dim=-1)
            base_p = base_logp.exp()
            ce_base = -base_logp.gather(-1, target.unsqueeze(-1)).squeeze().item()
            ce_comp = -comp_logp.gather(-1, target.unsqueeze(-1)).squeeze().item()
            kl = (base_p * (base_logp - comp_logp)).sum(-1).item()
            top1_match = (base_logits.argmax(-1) == comp_logits.argmax(-1)).item()
            eval_records.append({
                "i": i,
                "ce_base": ce_base,
                "ce_comp": ce_comp,
                "ce_gap": ce_comp - ce_base,
                "kl": kl,
                "top1_match": top1_match,
            })
    eval_wall = time.time() - t0
    print(f"  eval wall: {eval_wall:.0f}s")

    # Aggregate
    mean_ce_base = float(np.mean([r["ce_base"] for r in eval_records]))
    mean_ce_comp = float(np.mean([r["ce_comp"] for r in eval_records]))
    mean_kl = float(np.mean([r["kl"] for r in eval_records]))
    median_kl = float(np.median([r["kl"] for r in eval_records]))
    p25_kl = float(np.quantile([r["kl"] for r in eval_records], 0.25))
    p75_kl = float(np.quantile([r["kl"] for r in eval_records], 0.75))
    p90_kl = float(np.quantile([r["kl"] for r in eval_records], 0.90))
    top1_rate = float(np.mean([r["top1_match"] for r in eval_records]))

    print(f"\n=== Summary (N_EVAL={N_EVAL}) ===")
    print(f"  Mean CE baseline (full context):    {mean_ce_base:.3f}")
    print(f"  Mean CE compressed (255 tokens):    {mean_ce_comp:.3f}")
    print(f"  Mean CE gap (comp - base):          {mean_ce_comp - mean_ce_base:+.3f}")
    print(f"  Mean KL(base || comp):              {mean_kl:.3f}")
    print(f"  Median KL:                          {median_kl:.3f}")
    print(f"  KL p25 / p75 / p90:                 "
          f"{p25_kl:.3f} / {p75_kl:.3f} / {p90_kl:.3f}")
    print(f"  Top-1 agreement rate:               {top1_rate:.3f}")

    out = {
        "model": MODEL_NAME,
        "config": {
            "context": CONTEXT, "recent": RECENT_KEEP,
            "middle_in": MIDDLE, "middle_keep": MIDDLE_KEEP,
            "older_in": OLDER, "older_keep": OLDER_KEEP,
            "compressed_total": OLDER_KEEP + MIDDLE_KEEP + RECENT_KEEP,
            "n_train": N_TRAIN, "n_eval": N_EVAL,
            "steps": args.steps, "batch": args.batch, "lr": args.lr,
        },
        "training_history": history,
        "training_wall_s": train_wall,
        "eval_wall_s": eval_wall,
        "summary": {
            "mean_ce_base": mean_ce_base,
            "mean_ce_comp": mean_ce_comp,
            "mean_ce_gap": mean_ce_comp - mean_ce_base,
            "mean_kl": mean_kl,
            "median_kl": median_kl,
            "kl_p25": p25_kl, "kl_p75": p75_kl, "kl_p90": p90_kl,
            "top1_match_rate": top1_rate,
        },
        "per_passage": eval_records,
    }
    out_path = EXP / "results/run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
