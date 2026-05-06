"""Needle-in-a-haystack evaluation for compression vs baseline on WT103.

Protocol:
  - Build a long prefix of length T (= training ctx, divisible by COMPRESSION_RATIO)
    by concatenating: [WT103 natural-text fill] + [needle] + [WT103 fill] + [query].
  - The needle is "The magic number is XXX. " inserted at relative depth p.
  - The query "The magic number is" is appended at the end of the prefix
    (last K tokens), so the very next token to predict (= position T) IS
    the answer XXX.
  - Both models read the prefix, take logits at the last predict-able
    position (baseline: T-1; compressed: T/16-1), argmax, and we compare
    to the answer token id.

Variations:
  - Depths p ∈ {0.05, 0.20, 0.40, 0.60, 0.80, 0.95}
  - Multiple distinct "needle answers" (different number tokens)
  - Multiple natural-text fills sampled from val data
  - Total trials = N_NEEDLES × len(depths) × N_FILLS

Reports per-depth retrieval rate (recall accuracy) for both models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_conv"
sys.path.insert(0, str(REPO))

from experiments.compression_conv.model import (
    CompressedConfig, CompressedTransformer,
)
from experiments.compression_conv.train import COMPRESSION_RATIO


WT103_CACHE = REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt"

DEPTHS = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
N_FILLS = 5

QUERY_STRING = " The magic number is"
NEEDLE_TEMPLATE = " The magic number is{answer}. "

# Single-BPE-token answers (each of these is one GPT-2 token, prefixed by space)
NEEDLE_ANSWERS = [" 7", " 13", " 42", " 99", " 256"]


def load_wt103_val():
    c = torch.load(WT103_CACHE, weights_only=False)
    return c["splits"]["validation"].tokens


def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    known = {f for f in CompressedConfig.__dataclass_fields__}
    cfg = CompressedConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = CompressedTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


def make_haystack(val_tokens, fill_start, total_len, needle_tokens,
                   query_tokens, depth, tokenizer):
    """Construct one trial input of length total_len:

      [val_tokens slice] | [needle inserted at depth] | [val_tokens fill] | [query]

    The query occupies the final len(query_tokens) positions. The needle
    is inserted in the slice before the query. Returns input ids tensor
    of shape (1, total_len).
    """
    K = len(query_tokens)
    body_len = total_len - K   # everything before the query
    # Where in the body to place the needle
    insert_pos = max(0, min(body_len - len(needle_tokens),
                              int(round(depth * body_len))))
    # Pull body_len - len(needle) tokens from val data starting at fill_start
    fill_size = body_len - len(needle_tokens)
    if fill_size <= 0:
        raise ValueError("body_len too small for needle")
    fill = val_tokens[fill_start: fill_start + fill_size]
    if len(fill) < fill_size:
        # Wrap around if needed
        fill = torch.cat([fill, val_tokens[:fill_size - len(fill)]])
    fill = fill.tolist()

    body = fill[:insert_pos] + list(needle_tokens) + fill[insert_pos:]
    assert len(body) == body_len, f"body length {len(body)} != {body_len}"

    full = body + list(query_tokens)
    assert len(full) == total_len, f"full length {len(full)} != {total_len}"
    return torch.tensor(full, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def predict_next_token(model, ids):
    """Return argmax token id of the model's prediction for the next token
    after `ids`. For baseline: logits[-1] of full output. For compressed:
    logits at compressed position T/cr - 1."""
    logits = model(ids)               # (1, L, V) where L depends on cr
    next_logits = logits[0, -1, :]    # last predict-able position
    return int(next_logits.argmax().item())


@torch.no_grad()
def top5_tokens(model, ids):
    logits = model(ids)
    next_logits = logits[0, -1, :]
    vals, idxs = torch.topk(next_logits, 5)
    return [(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ck", required=True)
    ap.add_argument("--compressed-ck", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", default=str(EXP / "results/needle_haystack.json"))
    args = ap.parse_args()

    assert args.ctx % COMPRESSION_RATIO == 0, \
        f"ctx must be divisible by {COMPRESSION_RATIO}"

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    val_tokens = load_wt103_val()
    print(f"Val tokens: {len(val_tokens):,}")

    # Tokenize query and verify answer tokens are single tokens
    query_ids = tokenizer.encode(QUERY_STRING)
    print(f"Query '{QUERY_STRING}' → {query_ids} ({len(query_ids)} tokens)")
    answer_ids = []
    for a in NEEDLE_ANSWERS:
        ids = tokenizer.encode(a)
        if len(ids) != 1:
            print(f"  WARN: answer '{a}' tokenizes to {len(ids)} tokens "
                   f"({ids}); skipping")
            continue
        answer_ids.append((a, ids[0]))
    print(f"Single-token answers: {answer_ids}")
    assert len(answer_ids) > 0, "no single-token answers available"

    baseline = load_model(args.baseline_ck, device)
    compressed = load_model(args.compressed_ck, device)
    print(f"Baseline ctx_train={baseline.cfg.ctx_len}  "
          f"compressed ctx_train={compressed.cfg.ctx_len}  "
          f"compression={compressed.compression_ratio}x")

    results = []
    rng = torch.Generator(); rng.manual_seed(0)
    fill_starts = torch.randint(0, len(val_tokens) - args.ctx,
                                  (N_FILLS,), generator=rng).tolist()

    print(f"\nRunning {len(answer_ids)} answers × {len(DEPTHS)} depths × "
          f"{N_FILLS} fills = {len(answer_ids) * len(DEPTHS) * N_FILLS} trials")

    t0 = time.time()
    for ans_str, ans_id in answer_ids:
        needle_text = NEEDLE_TEMPLATE.format(answer=ans_str)
        needle_ids = tokenizer.encode(needle_text)
        for depth in DEPTHS:
            for fill_idx, fs in enumerate(fill_starts):
                ids = make_haystack(val_tokens, fs, args.ctx,
                                      needle_ids, query_ids, depth, tokenizer)
                ids = ids.to(device)

                base_pred = predict_next_token(baseline, ids)
                comp_pred = predict_next_token(compressed, ids)
                base_hit = base_pred == ans_id
                comp_hit = comp_pred == ans_id

                results.append({
                    "answer_str": ans_str, "answer_id": ans_id,
                    "depth": depth, "fill_idx": fill_idx,
                    "baseline_pred_id": base_pred,
                    "baseline_pred_str": tokenizer.decode([base_pred]),
                    "baseline_hit": base_hit,
                    "compressed_pred_id": comp_pred,
                    "compressed_pred_str": tokenizer.decode([comp_pred]),
                    "compressed_hit": comp_hit,
                })

    dt = time.time() - t0
    n = len(results)
    base_acc = sum(r["baseline_hit"] for r in results) / n
    comp_acc = sum(r["compressed_hit"] for r in results) / n
    print(f"\n{n} trials in {dt:.0f}s")
    print(f"Overall: baseline {base_acc:.3f}  compressed {comp_acc:.3f}")

    # Per-depth breakdown
    by_depth_b = defaultdict(list)
    by_depth_c = defaultdict(list)
    for r in results:
        by_depth_b[r["depth"]].append(int(r["baseline_hit"]))
        by_depth_c[r["depth"]].append(int(r["compressed_hit"]))
    print(f"\nPer-depth retrieval (mean of {n // len(DEPTHS)} trials each):")
    print(f"  {'depth':>6s}  {'baseline':>10s}  {'compressed':>11s}")
    per_depth = []
    for d in DEPTHS:
        b = sum(by_depth_b[d]) / len(by_depth_b[d])
        c = sum(by_depth_c[d]) / len(by_depth_c[d])
        print(f"  {d:6.2f}  {b:10.3f}  {c:11.3f}")
        per_depth.append({"depth": d, "baseline": b, "compressed": c})

    # Per-answer breakdown
    by_ans_b = defaultdict(list); by_ans_c = defaultdict(list)
    for r in results:
        by_ans_b[r["answer_str"]].append(int(r["baseline_hit"]))
        by_ans_c[r["answer_str"]].append(int(r["compressed_hit"]))
    print(f"\nPer-answer retrieval:")
    per_answer = []
    for ans_str, _ in answer_ids:
        b = sum(by_ans_b[ans_str]) / max(1, len(by_ans_b[ans_str]))
        c = sum(by_ans_c[ans_str]) / max(1, len(by_ans_c[ans_str]))
        print(f"  {ans_str:>6s}: baseline {b:.3f}  compressed {c:.3f}")
        per_answer.append({"answer": ans_str, "baseline": b, "compressed": c})

    out = {
        "ctx": args.ctx, "n_trials": n,
        "overall": {"baseline": base_acc, "compressed": comp_acc},
        "per_depth": per_depth,
        "per_answer": per_answer,
        "depths": DEPTHS,
        "answers_used": [a for a, _ in answer_ids],
        "n_fills": N_FILLS,
        "wall_s": dt,
        "trials": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
