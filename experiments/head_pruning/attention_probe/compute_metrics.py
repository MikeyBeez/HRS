"""Phase B: per-(layer, head) attention metrics averaged across examples.

For each example, the "query position" is the QUERY token index — the first
position where the model must emit an answer token. We read the attention
distribution of that query over all keys, then score:

  to_marker      = sum of mass on the MARKER position
  to_passkey     = sum of mass on the passkey digit span
  to_post_marker = sum on positions immediately after MARKER (post-marker
                     window of length passkey_len; overlaps passkey for
                     baseline passkey task — we still report it separately
                     so the metric is well-defined under generalizations)
  entropy        = Shannon entropy of the row
  max_offset     = (argmax(attn) - marker_pos); positive = attending to
                     something after MARKER, negative = before, 0 = on.

The 5 metrics are averaged across examples per (layer, head).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def compute_head_metrics(attn: torch.Tensor, examples: dict) -> dict:
    """
    attn: [N, L, H, T, T]  softmax'd attention.
    examples: dict from generate_batch.py.
    """
    N, L, H, T, _ = attn.shape
    query_pos = examples["query_pos"]        # [N]
    marker_pos = examples["marker_pos"]      # [N]
    passkey_start = examples["passkey_start"]
    passkey_end = examples["passkey_end"]    # inclusive
    K = examples["pkcfg"]["passkey_len"]

    # Extract the query row for every example and every (layer, head).
    # Indexing: attn[i, :, :, query_pos[i], :]  -> [L, H, T]
    idx = torch.arange(N)
    q_rows = attn[idx, :, :, query_pos, :]   # [N, L, H, T]

    # Also precompute marker + passkey + post-marker indicator masks (N, T).
    marker_mask = torch.zeros(N, T, dtype=torch.bool)
    passkey_mask = torch.zeros(N, T, dtype=torch.bool)
    postmarker_mask = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        m = int(marker_pos[i])
        ps = int(passkey_start[i])
        pe = int(passkey_end[i])  # inclusive
        marker_mask[i, m] = True
        passkey_mask[i, ps : pe + 1] = True
        # post-marker window: positions (m+1 ... m+K), same as passkey in this task.
        postmarker_mask[i, m + 1 : m + 1 + K] = True

    # to_marker: sum of attention mass on the MARKER position.
    # q_rows [N,L,H,T] * marker_mask [N,T] broadcast over L,H.
    to_marker = (q_rows * marker_mask[:, None, None, :]).sum(-1)          # [N, L, H]
    to_passkey = (q_rows * passkey_mask[:, None, None, :]).sum(-1)
    to_postmarker = (q_rows * postmarker_mask[:, None, None, :]).sum(-1)

    # Entropy per row (masking zeros to avoid log(0)).
    eps = 1e-12
    ent = -(q_rows.clamp(min=eps) * (q_rows.clamp(min=eps)).log()).sum(-1)  # [N, L, H]

    # Argmax → offset from marker_pos.
    max_idx = q_rows.argmax(-1)                                   # [N, L, H]
    max_offset = max_idx - marker_pos[:, None, None]

    # Aggregate across examples (mean).
    per_head = []
    for l in range(L):
        for h in range(H):
            per_head.append({
                "layer": l,
                "head": h,
                "to_marker": float(to_marker[:, l, h].mean()),
                "to_passkey": float(to_passkey[:, l, h].mean()),
                "to_postmarker": float(to_postmarker[:, l, h].mean()),
                "entropy": float(ent[:, l, h].mean()),
                "max_offset_mean": float(max_offset[:, l, h].float().mean()),
                "max_offset_median": float(max_offset[:, l, h].float().median()),
            })
    return {
        "per_head": per_head,
        "n_examples": N,
        "n_layers": L,
        "n_heads": H,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default=str(RESULTS_DIR / "attention_tensors.pt"))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "head_metrics.json"))
    args = ap.parse_args()

    attn_blob = torch.load(args.attn, map_location="cpu", weights_only=False)
    examples = torch.load(args.examples, map_location="cpu", weights_only=False)

    metrics = compute_head_metrics(attn_blob["attention"], examples)

    Path(args.out).write_text(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")

    # Print a quick summary table.
    print(f"\n{'L':>2} {'H':>2} {'to_marker':>11} {'to_passkey':>11} "
          f"{'to_postmk':>11} {'entropy':>8} {'max_off':>8}")
    for r in metrics["per_head"]:
        print(f"{r['layer']:>2} {r['head']:>2} "
              f"{r['to_marker']:>11.3f} {r['to_passkey']:>11.3f} "
              f"{r['to_postmarker']:>11.3f} "
              f"{r['entropy']:>8.3f} {r['max_offset_median']:>8.1f}")


if __name__ == "__main__":
    main()
