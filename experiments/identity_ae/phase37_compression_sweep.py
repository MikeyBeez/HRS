"""Phase 37: Engram compression ratio sweep.

Phase 33 showed that engram_then_tokens at 50/50 closes 92% of the
full→none NLL gap. The unanswered question: how far can the engram'd
fraction grow before the recency window can no longer hold up the
prediction quality?

Setup:
  - 50 WikiText validation passages, 448 tokens each
  - Context = first 384 tokens, continuation = last 64 tokens
  - For each split point K, prefix = [engram(first K), tokens(last 384-K)]
  - K sweep: 0, 96, 192, 288, 336, 360, 372, 378, 384
    → compression ratios 1×, 1.33×, 1.99×, 3.96×, 7.84×, 15.4×, 29.5×,
      54.9×, 384×
  - Engrams are L5 mean over the compressed slice, computed in isolation
    under the base model, prepend injection
  - Continuation perplexity measured over the same M−1 = 63 token
    positions in every condition

The shape we're looking for: a flat region where perplexity hugs the
full-context baseline, then a knee where it begins to climb toward the
engram-only ceiling. The location of the knee is the answer.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase37_compression_sweep.py
"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase33_engram_context import (
    continuation_nll, make_engram,
)


LAYER = 5
N_PASSAGES = 50
PASSAGE_LEN = 448
CONTEXT_LEN = 384      # split: 384 ctx + 64 cont
CONT_LEN   = PASSAGE_LEN - CONTEXT_LEN

# K = number of context tokens compressed to a single engram.
# The remaining (CONTEXT_LEN - K) tokens are kept as raw token positions.
# K=0 is the full-context baseline; K=CONTEXT_LEN is engram_only.
K_VALUES = [0, 96, 192, 288, 336, 360, 372, 378, 384]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase37")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    reset_lora_to_zero(model)
    model.eval()

    print(f"Model: V22, layers={len(model.blocks)}")
    print(f"Engram source: L{LAYER}_mean, prepend injection")
    print(f"Setup: {N_PASSAGES} passages, ctx={CONTEXT_LEN}, cont={CONT_LEN}")
    print(f"K sweep: {K_VALUES}\n")

    # ============================================================
    # Sample 50 WikiText validation passages.
    # ============================================================
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]
    print(f"Validation: {len(val_ds)} sequences available, sampling {N_PASSAGES}\n")

    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES].tolist()

    # nll_acc[K] = list of per-passage NLLs
    nll_acc = {K: [] for K in K_VALUES}
    nll_acc["no_context"] = []

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        ids = item[0] if isinstance(item, tuple) else item
        if len(ids) < PASSAGE_LEN:
            continue
        ids = ids[:PASSAGE_LEN]
        context_ids   = ids[:CONTEXT_LEN]
        continuation  = ids[CONTEXT_LEN:]

        # no_context baseline
        nll_acc["no_context"].append(
            continuation_nll(model, [], continuation, device))

        for K in K_VALUES:
            if K == 0:
                # full context, no engram
                prefix = [("tokens", context_ids)]
            elif K == CONTEXT_LEN:
                # engram only, no recent tokens
                eng = make_engram(model, context_ids.unsqueeze(0).to(device))
                prefix = [("hidden", eng)]
            else:
                # engram of first K tokens, then remaining 384-K as tokens
                first_K = context_ids[:K]
                rest    = context_ids[K:]
                eng = make_engram(model, first_K.unsqueeze(0).to(device))
                prefix = [("hidden", eng), ("tokens", rest)]

            nll = continuation_nll(model, prefix, continuation, device)
            nll_acc[K].append(nll)

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1:2d}/{N_PASSAGES}] processed")

    # ============================================================
    # Aggregate.
    # ============================================================
    def avg(xs):
        return sum(xs) / len(xs)

    no_nll  = avg(nll_acc["no_context"])
    no_ppl  = math.exp(no_nll)
    full_nll = avg(nll_acc[0])
    full_ppl = math.exp(full_nll)
    nll_gap = no_nll - full_nll

    rows = []
    for K in K_VALUES:
        nll = avg(nll_acc[K])
        ppl = math.exp(nll)
        recent_tokens = CONTEXT_LEN - K
        positions     = (1 if K > 0 else 0) + recent_tokens
        compression   = CONTEXT_LEN / positions if positions > 0 else float("inf")
        engram_frac   = K / CONTEXT_LEN
        gap_closed    = 1.0 - (nll - full_nll) / nll_gap if nll_gap > 0 else 0.0
        rows.append({
            "K": K,
            "engram_fraction": engram_frac,
            "recent_tokens": recent_tokens,
            "positions": positions,
            "compression": compression,
            "nll": nll,
            "ppl": ppl,
            "gap_closed": gap_closed,
        })

    # ============================================================
    # Print.
    # ============================================================
    print(f"\n{'='*84}")
    print("PHASE 37 SUMMARY (engram compression ratio sweep)")
    print(f"{'='*84}")
    print(f"  In-distribution (WikiText val), {N_PASSAGES} passages")
    print(f"  Context {CONTEXT_LEN} tokens, continuation {CONT_LEN} tokens, "
          f"engram = L{LAYER} mean of first K tokens")
    print()
    print(f"  no_context: NLL {no_nll:.3f}  PPL {no_ppl:.2f}")
    print(f"  full_ctx:   NLL {full_nll:.3f}  PPL {full_ppl:.2f}  (baseline)")
    print(f"  full→none NLL gap: {nll_gap:.3f} nats")
    print()
    print(f"  {'engram%':>8} {'recent':>7} {'pos':>5} {'compr':>7} "
          f"{'NLL':>8} {'PPL':>8} {'Δ vs full':>11} {'gap closed':>11}")
    print(f"  {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*8} {'-'*8} {'-'*11} {'-'*11}")
    for r in rows:
        delta = r["ppl"] - full_ppl
        print(f"  {r['engram_fraction']*100:>7.1f}% {r['recent_tokens']:>7d} "
              f"{r['positions']:>5d} {r['compression']:>6.1f}x "
              f"{r['nll']:>8.3f} {r['ppl']:>8.2f} {delta:>+10.2f} "
              f"{r['gap_closed']:>10.0%}")

    # Find the knee: largest K with gap_closed >= 0.90
    flat_rows = [r for r in rows if r["gap_closed"] >= 0.90]
    if flat_rows:
        knee = max(flat_rows, key=lambda r: r["engram_fraction"])
        print(f"\n  Knee (≥90% gap closed): "
              f"compress {knee['engram_fraction']*100:.0f}% of context "
              f"({knee['compression']:.1f}× compression), "
              f"PPL {knee['ppl']:.2f} vs full {full_ppl:.2f}")
    flat_95 = [r for r in rows if r["gap_closed"] >= 0.95]
    if flat_95:
        knee95 = max(flat_95, key=lambda r: r["engram_fraction"])
        print(f"  Knee (≥95% gap closed): "
              f"compress {knee95['engram_fraction']*100:.0f}% of context "
              f"({knee95['compression']:.1f}× compression), "
              f"PPL {knee95['ppl']:.2f}")

    out = {
        "n_passages":     N_PASSAGES,
        "context_len":    CONTEXT_LEN,
        "continuation":   CONT_LEN,
        "engram_layer":   LAYER,
        "no_context":     {"nll": no_nll, "ppl": no_ppl},
        "full_context":   {"nll": full_nll, "ppl": full_ppl},
        "full_to_none_gap_nll": nll_gap,
        "rows":           rows,
    }
    with open(results_dir / "compression_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
