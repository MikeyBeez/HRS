"""Extended Phase C: plot attention from positions OTHER than query.

The initial probe showed L3 does the "read" at query position with ~1.0
attention on the passkey. L0 retrieval heads look inactive at query — their
work must happen earlier. Specifically, during the passage, L0 heads
probably attend from the passkey tokens (positions marker+1..marker+K)
backward to MARKER, annotating the passkey tokens' residual stream with
"I sit right after the MARKER."

This script plots attention rows at:
  - marker_pos (what MARKER attends to as query)
  - passkey_start, i.e. marker_pos + 1 (what p1 attends to as query)
  - query_pos (for comparison to the original figure)

Averaged over the 200 examples (aligned to MARKER center). The resulting
grid shows, per head, where it attends from each of these three "probe"
positions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def _aligned_rows(attn: torch.Tensor, query_positions: torch.Tensor,
                    marker_positions: torch.Tensor) -> torch.Tensor:
    """Extract attn row at query_positions[i], align so MARKER is at T/2."""
    N, L, H, T, _ = attn.shape
    idx = torch.arange(N)
    rows = attn[idx, :, :, query_positions, :]  # [N, L, H, T]
    center = T // 2
    aligned = torch.zeros_like(rows)
    for i in range(N):
        shift = int(center - marker_positions[i].item())
        aligned[i] = rows[i].roll(shifts=shift, dims=-1)
    return aligned.mean(0)  # [L, H, T]


def plot_multi_query(attn, examples, out_path: Path):
    N, L, H, T, _ = attn.shape
    marker = examples["marker_pos"]
    pk_start = examples["passkey_start"]
    query = examples["query_pos"]
    K = examples["pkcfg"]["passkey_len"]

    mean_at_marker = _aligned_rows(attn, marker, marker)        # row @ MARKER
    mean_at_passkey = _aligned_rows(attn, pk_start, marker)     # row @ p1
    mean_at_query = _aligned_rows(attn, query, marker)          # row @ QUERY

    center = T // 2
    probes = [("attn @ MARKER", mean_at_marker, "#00a000"),
               ("attn @ passkey start (p1)", mean_at_passkey, "#0030a0"),
               ("attn @ QUERY", mean_at_query, "#e07000")]

    # Three panels stacked, each a 4x4 head grid. We build a single tall figure.
    fig, axes = plt.subplots(L * 3, H, figsize=(3.0 * H, 1.6 * L * 3),
                              sharex=True)

    for probe_idx, (label, mat, color) in enumerate(probes):
        for l in range(L):
            for h in range(H):
                ax = axes[probe_idx * L + l, h]
                ax.plot(mat[l, h].numpy(), lw=1.0, color=color)
                ax.axvline(center, color="#00a000", lw=0.8, linestyle="--")
                ax.axvspan(center + 1, center + 1 + K,
                            color="#0030a0", alpha=0.15)
                ax.set_yscale("log")
                ax.set_ylim(1e-5, 1.0)
                ax.set_xlim(0, T)
                ax.grid(alpha=0.2)
                if l == 0:
                    ax.set_title(f"H{h}", fontsize=10)
                if h == 0:
                    ax.set_ylabel(f"L{l}\n{label}" if l == 0
                                     else f"L{l}", fontsize=9)

    fig.suptitle("Attention row at three probe positions "
                   "(aligned MARKER@T/2; dashed=MARKER, shaded=passkey)",
                   fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_full_matrix_retrieval(attn, examples, out_path: Path,
                                  retrieval_heads: list):
    """Plot the full 256x256 attention matrix for each retrieval head,
    averaged over examples (each aligned so MARKER@T/2)."""
    N, L, H, T, _ = attn.shape
    marker = examples["marker_pos"]
    center = T // 2

    # Average attention matrices aligned in BOTH query and key dims.
    aligned = torch.zeros_like(attn)
    for i in range(N):
        shift = int(center - marker[i].item())
        # shift rows (query dim) and cols (key dim) so MARKER -> T/2 on both.
        aligned[i] = attn[i].roll(shifts=(shift, shift), dims=(-2, -1))
    mean_mat = aligned.mean(0)  # [L, H, T, T]

    nh = len(retrieval_heads)
    fig, axes = plt.subplots(1, nh, figsize=(4.5 * nh, 4.5))
    if nh == 1:
        axes = [axes]
    for i, (l, h) in enumerate(retrieval_heads):
        ax = axes[i]
        mat = mean_mat[l, h].numpy()
        # Clip log for visualization.
        im = ax.imshow(np.log10(mat + 1e-6), aspect="equal",
                         cmap="viridis", vmin=-5, vmax=0)
        ax.axhline(center, color="#00ff00", lw=0.6, alpha=0.7)
        ax.axvline(center, color="#00ff00", lw=0.6, alpha=0.7)
        ax.set_title(f"L{l} H{h} (avg, aligned)")
        ax.set_xlabel("key (col)")
        ax.set_ylabel("query (row)")
        plt.colorbar(im, ax=ax, label="log10(attn)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default=str(RESULTS_DIR / "attention_tensors.pt"))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--importance", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    attn_blob = torch.load(args.attn, map_location="cpu", weights_only=False)
    examples = torch.load(args.examples, map_location="cpu", weights_only=False)
    attn = attn_blob["attention"]

    plot_multi_query(attn, examples, RESULTS_DIR / "fig4_multi_query.png")

    # Full-matrix for a handful of heads spanning the interesting layers.
    if args.importance:
        import json
        imp = json.loads(Path(args.importance).read_text())
        retr = sorted(
            [(r["layer"], r["head"]) for r in imp["per_head"]
             if r["importance_passkey_exact"] > args.threshold],
            key=lambda k: -next(r["importance_passkey_exact"]
                                   for r in imp["per_head"]
                                   if r["layer"] == k[0] and r["head"] == k[1]),
        )
    else:
        retr = [(0, 3), (2, 0), (0, 2)]
    # Also include one L3 head for comparison.
    retr = retr + [(3, 0)]
    plot_full_matrix_retrieval(attn, examples,
                                 RESULTS_DIR / "fig5_full_matrix.png",
                                 retrieval_heads=retr)


if __name__ == "__main__":
    main()
