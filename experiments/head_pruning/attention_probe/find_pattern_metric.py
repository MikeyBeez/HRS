"""Finer metric: for each head, measure how strongly it implements the
"find MARKER" pattern — i.e. how much attention mass, from any post-MARKER
query position, lands on the MARKER column.

For each example and head, compute:

  find_strength = mean_{q in (marker_pos, T)} attn[q, marker_pos]

i.e. average of the MARKER-column of the attention matrix, over rows
strictly after MARKER. A head that "finds MARKER" has high values here.

Paired metric:

  read_strength = mean_{q in (marker_pos, T)}
                    attn[q, passkey_start : passkey_end + 1].sum()

"read" = attention mass that lands on the passkey span, across post-MARKER
queries. Heads that copy the passkey to the output should have high read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
BASELINE_IMPORTANCE = Path(
    "/mnt/data/Code/HRS/experiments/head_pruning/results/head_importance.json"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default=str(RESULTS_DIR / "attention_tensors.pt"))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--importance", default=str(BASELINE_IMPORTANCE))
    ap.add_argument("--out",
                     default=str(RESULTS_DIR / "find_read_metrics.json"))
    args = ap.parse_args()

    attn_blob = torch.load(args.attn, map_location="cpu", weights_only=False)
    examples = torch.load(args.examples, map_location="cpu", weights_only=False)
    attn = attn_blob["attention"]  # [N, L, H, T, T]

    N, L, H, T, _ = attn.shape
    marker = examples["marker_pos"].numpy()
    passkey_start = examples["passkey_start"].numpy()
    passkey_end = examples["passkey_end"].numpy()  # inclusive
    K = examples["pkcfg"]["passkey_len"]

    find_strength = np.zeros((L, H))
    read_strength = np.zeros((L, H))
    post_count = 0
    for i in range(N):
        m = int(marker[i])
        ps = int(passkey_start[i])
        pe = int(passkey_end[i])
        # Post-marker query rows: (marker, T).  Use marker+1 to exclude the
        # marker row itself (which doesn't participate in "finding" from
        # after; it's the target).
        post_rows = slice(m + 1, T)
        # For each (L, H), grab attn[i, :, :, post_rows, m] and post_rows, ps..pe+1.
        mcol = attn[i, :, :, post_rows, m]                 # [L, H, n_post]
        pspan = attn[i, :, :, post_rows, ps : pe + 1].sum(-1)  # [L, H, n_post]
        find_strength += mcol.mean(-1).numpy()
        read_strength += pspan.mean(-1).numpy()
        post_count += 1
    find_strength /= post_count
    read_strength /= post_count

    # Also compute the same thing from just the final query position
    # (query_pos) for comparison, so we see what the earlier metric was.
    to_marker_from_qp = np.zeros((L, H))
    to_passkey_from_qp = np.zeros((L, H))
    for i in range(N):
        m = int(marker[i])
        ps = int(passkey_start[i])
        pe = int(passkey_end[i])
        q = int(examples["query_pos"][i])
        to_marker_from_qp += attn[i, :, :, q, m].numpy()
        to_passkey_from_qp += attn[i, :, :, q, ps : pe + 1].sum(-1).numpy()
    to_marker_from_qp /= N
    to_passkey_from_qp /= N

    # Load head importance.
    imp = json.loads(Path(args.importance).read_text())
    per_imp = {(r["layer"], r["head"]): r["importance_passkey_exact"]
                for r in imp["per_head"]}

    rows = []
    print(f"\n{'L':>2} {'H':>2} {'Δpk':>6} "
          f"{'find':>8} {'read':>8} "
          f"{'to_marker(qp)':>14} {'to_passkey(qp)':>15}")
    for l in range(L):
        for h in range(H):
            rows.append({
                "layer": l, "head": h,
                "delta_passkey": per_imp.get((l, h), 0.0),
                "find_strength": float(find_strength[l, h]),
                "read_strength": float(read_strength[l, h]),
                "to_marker_from_query": float(to_marker_from_qp[l, h]),
                "to_passkey_from_query": float(to_passkey_from_qp[l, h]),
            })
            print(f"{l:>2} {h:>2} {per_imp.get((l, h), 0.0):>+6.2f} "
                  f"{find_strength[l, h]:>8.3f} {read_strength[l, h]:>8.3f} "
                  f"{to_marker_from_qp[l, h]:>14.3f} "
                  f"{to_passkey_from_qp[l, h]:>15.3f}")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
