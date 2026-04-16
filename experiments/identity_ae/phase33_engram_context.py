"""Phase 33 + 34: Engram-as-context perplexity (pure + hybrid).

Phase 32 showed that the engram lands near the K/V centroid of the
passage past the bootstrap layer. This script asks the next question:
does generation quality survive when we replace tokens with engrams in
the context?

We measure continuation perplexity under six prefix configurations,
with the same passages and the same continuation in every condition,
so all numbers are directly comparable:

  1. no_context              empty prefix                       (lower bound)
  2. full_context            all 200 context tokens             (upper bound)
  3. engram_only             1 engram of full 200-token context (Exp 2)
  4. engram_then_tokens      engram(first 100) + 100 tokens     (Exp 3)
  5. tokens_then_engram      100 tokens + engram(second 100)    (Exp 3)
  6. two_engrams             engram(first 100) + engram(2nd)    (Exp 3)

Setup:
  - 50 WikiText validation passages, 256 tokens each
  - Context = first 200 tokens, continuation = last 56 tokens
  - Engrams are L5 mean-pooled hidden states of the relevant segment,
    computed in isolation under the base model (no information leakage
    between halves in conditions 4–6)
  - Engrams are injected as single hidden-state positions at the model
    input (same prepend method as Phase 35)
  - Perplexity is measured over the same M−1 continuation token positions
    in every condition; we skip continuation[0] so the no_context case
    has the same number of scored tokens as the engram cases.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase33_engram_context.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase22_engram_key import hidden_at_layer, reset_lora_to_zero


LAYER = 5
N_PASSAGES = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200          # split: 200 ctx + 56 cont
HALF_LEN = 100             # 100 + 100 split for hybrid
MAX_CTX_POS = 512


# ----------------------------------------------------------------
# Forward from a list of segments. Each segment is one of:
#   ("tokens", LongTensor[T])    -> get tok_emb, dropout
#   ("hidden", FloatTensor[D])   -> use as a single position directly
# Returns logits at every position.
# ----------------------------------------------------------------
@torch.no_grad()
def forward_segments(model, segments, device):
    parts = []
    for kind, x in segments:
        if kind == "tokens":
            ids = x.unsqueeze(0).to(device)         # (1, T)
            parts.append(model.drop(model.tok_emb(ids)))   # (1, T, D)
        elif kind == "hidden":
            parts.append(x.view(1, 1, -1).to(device))      # (1, 1, D)
        else:
            raise ValueError(kind)
    h = torch.cat(parts, dim=1)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]
    for block in model.blocks:
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    return model.lm_head(h)


def segments_len(segments):
    return sum(len(x) if kind == "tokens" else 1 for kind, x in segments)


@torch.no_grad()
def continuation_nll(model, prefix_segments, continuation_ids, device):
    """NLL averaged over the M-1 token positions [c_1 ... c_{M-1}].

    Input fed to the model: [prefix, continuation[:-1]]
    Logits at position prefix_len + i predict continuation[i+1] for
    i in [0..M-2]. We average cross-entropy over those M-1 positions.

    For prefix_len == 0 the slice starts at position 0, predicting
    c_1..c_{M-1} from inputs c_0..c_{M-2}. Same M-1 scored tokens.
    """
    M = len(continuation_ids)
    full_segments = list(prefix_segments) + [("tokens", continuation_ids[:-1])]
    logits = forward_segments(model, full_segments, device)
    prefix_len = segments_len(prefix_segments)
    pred = logits[:, prefix_len : prefix_len + M - 1, :]   # (1, M-1, V)
    target = continuation_ids[1:].unsqueeze(0).to(device)  # (1, M-1)
    nll = F.cross_entropy(pred.reshape(-1, pred.shape[-1]),
                          target.reshape(-1), reduction="mean")
    return float(nll)


@torch.no_grad()
def make_engram(model, ids_t):
    """L5 mean-pooled hidden state of the segment under the base model."""
    h = hidden_at_layer(model, ids_t, LAYER)   # (1, T, D)
    return h.mean(dim=1).squeeze(0).detach()    # (D,)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase33")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    reset_lora_to_zero(model)
    model.eval()

    n_layers = len(model.blocks)
    n_heads  = model.blocks[0].attn.n_heads
    head_dim = model.blocks[0].attn.head_dim
    d_model  = n_heads * head_dim
    print(f"Model: V22, layers={n_layers}, heads={n_heads}, d_model={d_model}")
    print(f"Engram source: L{LAYER}_mean")
    print(f"Setup: {N_PASSAGES} passages, ctx={CONTEXT_LEN}, half={HALF_LEN}, "
          f"cont={PASSAGE_LEN - CONTEXT_LEN}\n")

    # ============================================================
    # Sample 50 WikiText validation passages.
    # ============================================================
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
        context_ids   = ids[:CONTEXT_LEN]                       # (200,)
        continuation  = ids[CONTEXT_LEN:]                        # (56,)
        first_half    = context_ids[:HALF_LEN]                   # (100,)
        second_half   = context_ids[HALF_LEN:CONTEXT_LEN]        # (100,)

        # Engrams (computed in isolation)
        eng_full = make_engram(model, context_ids.unsqueeze(0).to(device))
        eng_h1   = make_engram(model, first_half.unsqueeze(0).to(device))
        eng_h2   = make_engram(model, second_half.unsqueeze(0).to(device))

        # ----- 1. no_context -----
        nll_acc["no_context"].append(
            continuation_nll(model, [], continuation, device))

        # ----- 2. full_context -----
        nll_acc["full_context"].append(
            continuation_nll(model, [("tokens", context_ids)], continuation, device))

        # ----- 3. engram_only -----
        nll_acc["engram_only"].append(
            continuation_nll(model, [("hidden", eng_full)], continuation, device))

        # ----- 4. engram_then_tokens -----
        nll_acc["engram_then_tokens"].append(
            continuation_nll(model,
                             [("hidden", eng_h1), ("tokens", second_half)],
                             continuation, device))

        # ----- 5. tokens_then_engram -----
        nll_acc["tokens_then_engram"].append(
            continuation_nll(model,
                             [("tokens", first_half), ("hidden", eng_h2)],
                             continuation, device))

        # ----- 6. two_engrams -----
        nll_acc["two_engrams"].append(
            continuation_nll(model,
                             [("hidden", eng_h1), ("hidden", eng_h2)],
                             continuation, device))

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1:2d}/{N_PASSAGES}] processed")

    # ============================================================
    # Aggregate.
    # ============================================================
    summary = {}
    for c in conditions:
        nlls = nll_acc[c]
        mean_nll = sum(nlls) / len(nlls)
        ppl = math.exp(mean_nll)
        summary[c] = {"nll": mean_nll, "ppl": ppl, "n": len(nlls)}

    full_ppl = summary["full_context"]["ppl"]
    no_ppl   = summary["no_context"]["ppl"]

    print(f"\n{'='*72}")
    print("PHASE 33+34 SUMMARY (engram-as-context perplexity)")
    print(f"{'='*72}")
    print(f"  In-distribution content (WikiText val), {N_PASSAGES} passages")
    print(f"  Context {CONTEXT_LEN} tokens, continuation {PASSAGE_LEN - CONTEXT_LEN} tokens")
    print(f"  Engrams: L{LAYER} mean, computed in isolation, prepend injection")
    print()
    print(f"  {'condition':22s} {'NLL':>8} {'PPL':>10} {'/full':>8} {'positions':>11}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*11}")
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
        print(f"  {c:22s} {s['nll']:>8.3f} {s['ppl']:>10.2f} {ratio:>8.2f}x "
              f"{pos_per_cond[c]:>9d}  ")

    # Compression ratios
    print()
    print(f"  Compression ratios (positions used / 200 token baseline):")
    for c in conditions:
        if c in ("no_context", "full_context"):
            continue
        ratio = pos_per_cond[c] / CONTEXT_LEN
        print(f"    {c:22s} {pos_per_cond[c]:3d} pos = {1/ratio:6.1f}x compression")

    # Headline interpretations
    print()
    print(f"  Headline:")
    print(f"    full_context PPL  = {full_ppl:.2f}  (upper bound)")
    print(f"    no_context PPL    = {no_ppl:.2f}  (lower bound)")
    print(f"    engram_only PPL   = {summary['engram_only']['ppl']:.2f}  "
          f"({summary['engram_only']['ppl']/full_ppl:.2f}x full, "
          f"{1 - (summary['engram_only']['ppl'] - full_ppl)/(no_ppl - full_ppl):.0%} of full→none gap closed)")

    out = {
        "n_passages": N_PASSAGES,
        "passage_len": PASSAGE_LEN,
        "context_len": CONTEXT_LEN,
        "half_len": HALF_LEN,
        "engram_layer": LAYER,
        "summary": summary,
        "positions_per_condition": pos_per_cond,
    }
    with open(results_dir / "engram_context_ppl.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
