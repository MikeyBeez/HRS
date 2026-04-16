"""Phase 33b: Engram-as-context perplexity for the L0 mean engram.

Phase 33 measured continuation perplexity for the L5 mean engram across six
prefix configurations and found that engram_then_tokens (compress the
distant half, keep the recent half) closes 92 percent of the full→none NLL
gap at 2x compression. The interpretation: K-space alignment of the L5
engram with the passage's K centroid (Phase 32: 0.82–0.89 past the
bootstrap layer) is sufficient for the recent tokens' attention queries to
select the engram by content match.

Phase 32b found that the L0 engram has K-space alignment 0.988 at layer 0
but degrades to 0.65 by layer 5 — the dual of L5's bootstrap-then-strong
profile. The empirical question this script asks: does that 0.65 late-layer
alignment translate to a measurable degradation in continuation perplexity,
or is it good enough? If L0 injection works as well as L5 injection for
cache compression, the paper simplifies — both applications use the same
L0 mean engram, and the duality of K-space alignment is just a curiosity.
If L0 is meaningfully worse, the duality is a real architectural feature
of the theory and Application 2 keeps L5.

Same setup as Phase 33: 50 WikiText validation passages, 200-token context
+ 56-token continuation, six prefix configurations, perplexity measured
over the same M-1 continuation positions in every condition. Only the
engram extraction changes.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase33b_l0_engram_context.py
"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase33_engram_context import (
    continuation_nll, segments_len,
)


N_PASSAGES = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200
HALF_LEN = 100


# ----------------------------------------------------------------
# L0 mean engram: token embedding mean, no transformer pass.
# ----------------------------------------------------------------
@torch.no_grad()
def make_engram_l0(model, ids_t):
    """L0 mean engram: mean of token embeddings, no block processing."""
    h = model.drop(model.tok_emb(ids_t))   # (1, T, D)
    return h.mean(dim=1).squeeze(0).detach()  # (D,) on device


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase33b")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    reset_lora_to_zero(model)
    model.eval()

    print(f"Engram source: L0 mean (token embedding mean, no transformer pass)")
    print(f"Setup: {N_PASSAGES} passages, ctx={CONTEXT_LEN}, half={HALF_LEN}, "
          f"cont={PASSAGE_LEN - CONTEXT_LEN}\n")

    # Sample 50 WikiText validation passages (same as Phase 33)
    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]
    print(f"Validation: {len(val_ds)} sequences available, sampling {N_PASSAGES}\n")

    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES].tolist()

    conditions = [
        "no_context",
        "full_context",
        "engram_only",
        "engram_then_tokens",
        "tokens_then_engram",
        "two_engrams",
    ]
    nll_acc = {c: [] for c in conditions}

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        ids = item[0] if isinstance(item, tuple) else item
        ids = ids[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        context_ids   = ids[:CONTEXT_LEN]
        continuation  = ids[CONTEXT_LEN:]
        first_half    = context_ids[:HALF_LEN]
        second_half   = context_ids[HALF_LEN:CONTEXT_LEN]

        # L0 engrams (computed in isolation, no information leakage between halves)
        eng_full = make_engram_l0(model, context_ids.unsqueeze(0).to(device))
        eng_h1   = make_engram_l0(model, first_half.unsqueeze(0).to(device))
        eng_h2   = make_engram_l0(model, second_half.unsqueeze(0).to(device))

        nll_acc["no_context"].append(
            continuation_nll(model, [], continuation, device))
        nll_acc["full_context"].append(
            continuation_nll(model, [("tokens", context_ids)], continuation, device))
        nll_acc["engram_only"].append(
            continuation_nll(model, [("hidden", eng_full)], continuation, device))
        nll_acc["engram_then_tokens"].append(
            continuation_nll(model,
                             [("hidden", eng_h1), ("tokens", second_half)],
                             continuation, device))
        nll_acc["tokens_then_engram"].append(
            continuation_nll(model,
                             [("tokens", first_half), ("hidden", eng_h2)],
                             continuation, device))
        nll_acc["two_engrams"].append(
            continuation_nll(model,
                             [("hidden", eng_h1), ("hidden", eng_h2)],
                             continuation, device))

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1:2d}/{N_PASSAGES}] processed")

    # Aggregate
    summary = {}
    for c in conditions:
        nlls = nll_acc[c]
        mean_nll = sum(nlls) / len(nlls)
        ppl = math.exp(mean_nll)
        summary[c] = {"nll": mean_nll, "ppl": ppl, "n": len(nlls)}

    full_ppl = summary["full_context"]["ppl"]
    no_ppl   = summary["no_context"]["ppl"]
    full_nll = summary["full_context"]["nll"]
    no_nll   = summary["no_context"]["nll"]
    nll_gap  = no_nll - full_nll

    print(f"\n{'='*72}")
    print("PHASE 33b SUMMARY (L0 mean engram, KV cache compression test)")
    print(f"{'='*72}")
    print(f"  In-distribution content (WikiText val), {N_PASSAGES} passages")
    print(f"  Context {CONTEXT_LEN} tokens, continuation {PASSAGE_LEN - CONTEXT_LEN} tokens")
    print(f"  Engrams: L0 mean (token embedding mean), prepend injection")
    print()
    print(f"  {'condition':22s} {'NLL':>8} {'PPL':>10} {'/full':>8} "
          f"{'positions':>11} {'gap closed':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*11} {'-'*12}")
    pos_per_cond = {
        "no_context":          0,
        "full_context":      CONTEXT_LEN,
        "engram_only":         1,
        "engram_then_tokens":  1 + HALF_LEN,
        "tokens_then_engram":  HALF_LEN + 1,
        "two_engrams":         2,
    }
    for c in conditions:
        s = summary[c]
        ratio = s["ppl"] / full_ppl
        gap_closed = 1.0 - (s["nll"] - full_nll) / nll_gap if nll_gap > 0 else 0.0
        print(f"  {c:22s} {s['nll']:>8.3f} {s['ppl']:>10.2f} {ratio:>7.2f}x "
              f"{pos_per_cond[c]:>9d}   {gap_closed:>10.0%}")

    print()
    print(f"  Phase 33 reference (L5 mean engram):")
    print(f"    no_context           28.62    1.62x       0           0%")
    print(f"    full_context         17.71    1.00x     200         100%")
    print(f"    engram_only          26.25    1.48x       1          18%")
    print(f"    engram_then_tokens   18.41    1.04x     101          92%")
    print(f"    tokens_then_engram   24.50    1.38x     101          32%")
    print(f"    two_engrams          25.81    1.46x       2          21%")

    out = {
        "n_passages": N_PASSAGES,
        "passage_len": PASSAGE_LEN,
        "context_len": CONTEXT_LEN,
        "half_len": HALF_LEN,
        "engram_source": "L0_mean",
        "summary": summary,
        "positions_per_condition": pos_per_cond,
    }
    with open(results_dir / "l0_engram_context_ppl.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
