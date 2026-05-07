"""Mutual information ratio metric per the spec.

Given a tokenized corpus, compute MI(X_t; X_{t+k}) at distances k=1 and k=16
empirically from sample counts on a held-out subset. Report:

  MI(X_t; X_{t+1})    — local mutual information
  MI(X_t; X_{t+16})   — at the compression window
  ratio = MI@16 / MI@1 — high → long-range structure preserved at compression
                          window; low → local-dominated, compression destroys signal

Estimator:
  Sample N positions from the corpus tokens (uniform random).
  At each position t, record the pair (X_t, X_{t+k}).
  Compute MI from empirical joint and marginals using sparse counts:
    MI = sum_{x,y} p(x,y) log( p(x,y) / (p(x) p(y)) )

Plug-in MLE estimator. Slightly biased upward at low N; we use N=200k samples
which keeps bias modest given the 50k vocab.

Also reports H(X_t | X_{t+1}) and H(X_t | X_{t+16}) for cross-check; the ratio
H(X|prev_1)/H(X|prev_16) is a related metric (alternative per spec).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path("/mnt/data/Code/HRS")


def compute_mi(token_pairs):
    """token_pairs: iterable of (a, b) tuples. Returns MI in nats."""
    joint = Counter(token_pairs)
    n = sum(joint.values())
    if n == 0:
        return 0.0, 0.0, 0.0  # mi, h_x, h_y

    marg_x = Counter()
    marg_y = Counter()
    for (x, y), c in joint.items():
        marg_x[x] += c
        marg_y[y] += c

    mi = 0.0
    for (x, y), c_xy in joint.items():
        p_xy = c_xy / n
        p_x = marg_x[x] / n
        p_y = marg_y[y] / n
        mi += p_xy * math.log(p_xy / (p_x * p_y))
    h_x = -sum((c / n) * math.log(c / n) for c in marg_x.values() if c > 0)
    h_y = -sum((c / n) * math.log(c / n) for c in marg_y.values() if c > 0)
    return mi, h_x, h_y


def compute_cond_entropy(token_pairs):
    """H(X | Y) where pairs are (X, Y)."""
    joint = Counter(token_pairs)
    n = sum(joint.values())
    if n == 0:
        return 0.0
    marg_y = Counter()
    for (x, y), c in joint.items():
        marg_y[y] += c
    h_x_given_y = 0.0
    for (x, y), c_xy in joint.items():
        p_xy = c_xy / n
        p_y = marg_y[y] / n
        h_x_given_y += p_xy * math.log(p_y / p_xy)   # = -p_xy log(p_xy/p_y)
    return h_x_given_y


def metric_for_corpus(tokens, distances=(1, 2, 4, 8, 16, 32, 64),
                       n_samples=200_000, seed=0):
    """Return dict of metrics."""
    n = tokens.shape[0]
    g = torch.Generator(); g.manual_seed(seed)

    out = {"corpus_size_tokens": n, "n_samples": n_samples,
           "distances": list(distances), "mi": {}, "h_cond": {}}
    for k in distances:
        max_t = n - k - 1
        if max_t <= 0:
            out["mi"][k] = None
            continue
        # Sample positions
        starts = torch.randint(0, max_t, (min(n_samples, max_t),),
                                  generator=g).tolist()
        # token_pairs: (X_t, X_{t+k})  — we want MI between current and FUTURE token
        pairs = [(int(tokens[t].item()), int(tokens[t + k].item()))
                 for t in starts]
        mi, h_x, h_y = compute_mi(pairs)
        # H(X_t | X_{t+k}) — condition on future, predict current. We want
        # the dual: predict X_t given X_{t-k}? Actually for "info from past"
        # we want H(X_t | X_{t-k}). By stationarity these are equal.
        h_cond = compute_cond_entropy(pairs)
        out["mi"][k] = mi
        out["h_cond"][k] = h_cond

    if 1 in out["mi"] and 16 in out["mi"] and out["mi"][1] is not None:
        out["mi_ratio_16_over_1"] = out["mi"][16] / out["mi"][1]
        out["h_cond_ratio_1_over_16"] = out["h_cond"][1] / out["h_cond"][16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"Loading {args.corpus_path}")
    c = torch.load(args.corpus_path, weights_only=False)
    # Schema flexibility: WT103 has {"splits": {"train": SplitObj(.tokens)}}; code
    # has plain {"train": tensor}; flat TS save would be just a tensor.
    if isinstance(c, torch.Tensor):
        train = c
    elif "splits" in c:
        s = c["splits"]["train"]
        train = s.tokens if hasattr(s, "tokens") else s
    elif "train" in c:
        s = c["train"]
        train = s.tokens if hasattr(s, "tokens") else s
    else:
        raise ValueError(f"unknown corpus schema: keys={list(c.keys())}")
    print(f"Train tokens: {len(train):,}")

    t0 = time.time()
    metrics = metric_for_corpus(train)
    metrics["corpus"] = args.corpus
    print(f"\nCorpus: {args.corpus}")
    print(f"  Tokens: {metrics['corpus_size_tokens']:,}")
    print(f"  MI(X_t; X_{{t+k}}) in nats:")
    for k in metrics["distances"]:
        if metrics["mi"][k] is not None:
            print(f"    k={k:3d}: MI = {metrics['mi'][k]:.4f}    "
                  f"H(X | X_{{t+k}}) = {metrics['h_cond'][k]:.4f}")
    print(f"  MI ratio MI@16 / MI@1 = {metrics.get('mi_ratio_16_over_1', float('nan')):.4f}")
    print(f"  H_cond ratio H@1 / H@16 = {metrics.get('h_cond_ratio_1_over_16', float('nan')):.4f}")
    print(f"  Wall: {time.time() - t0:.0f}s")

    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
