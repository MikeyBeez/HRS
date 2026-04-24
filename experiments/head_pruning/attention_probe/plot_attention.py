"""Phase C: three figures.

  Fig 1 — 4x4 grid of heads, averaged attention heatmap across 200 examples.
  Fig 2 — 4x4 grid, one cherry-picked successful example with MARKER ~mid.
  Fig 3 — retrieval-head zoom: L0 H2, L0 H3, L2 H0 averaged + single-example +
          marginal bars of to-marker / to-passkey / to-elsewhere.

For Figs 1 and 2, showing a 256x256 attention heatmap per cell in a 4x4 grid
is too dense. Instead we plot the query row (attention from QUERY position
over all keys) stacked across examples as a [N, T] image — that's the
relevant slice for passkey retrieval anyway.
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


def _query_rows(attn: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
    """attn: [N, L, H, T, T] -> [N, L, H, T] the row at query_pos per example."""
    N = attn.shape[0]
    idx = torch.arange(N)
    return attn[idx, :, :, query_pos, :]


def _annotate(ax, marker_pos, passkey_start, passkey_end, query_pos, T):
    for p, c in [(marker_pos, "#00a000"), (query_pos, "#e07000")]:
        ax.axvline(p, color=c, lw=0.6, alpha=0.8)
    # Passkey span as a shaded rectangle (subtle).
    ax.axvspan(passkey_start, passkey_end + 1, color="#0030a0", alpha=0.10)


def fig1_gestalt_grid(attn: torch.Tensor, examples: dict, out_path: Path):
    """Average the query row across examples, aligned to MARKER position.

    Because MARKER position varies per example, we align each row to a common
    reference (center the MARKER at T/2) before averaging.
    """
    N, L, H, T, _ = attn.shape
    q = _query_rows(attn, examples["query_pos"])  # [N, L, H, T]
    marker = examples["marker_pos"].numpy()

    # Align to MARKER center by rolling.
    center = T // 2
    aligned = torch.zeros_like(q)
    for i in range(N):
        shift = int(center - marker[i])
        aligned[i] = q[i].roll(shifts=shift, dims=-1)
    mean_q = aligned.mean(0)  # [L, H, T]

    fig, axes = plt.subplots(L, H, figsize=(3 * H, 2.1 * L), sharex=True)
    for l in range(L):
        for h in range(H):
            ax = axes[l, h] if L > 1 else axes[h]
            ax.plot(mean_q[l, h].numpy(), lw=1.0)
            ax.set_title(f"L{l} H{h}", fontsize=10)
            ax.axvline(center, color="#00a000", lw=0.8)
            ax.axvspan(center + 1, center + 1 + 4, color="#0030a0", alpha=0.15)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.0)
            ax.set_xlim(0, T)
            ax.grid(alpha=0.2)
    fig.suptitle("QUERY-position attention across all examples "
                   "(aligned so MARKER is at T/2, log-scale)", fontsize=11)
    fig.supxlabel("key position (MARKER at dashed line; passkey shaded)")
    fig.supylabel("attention mass (log)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def fig2_single_example(attn: torch.Tensor, examples: dict, out_path: Path,
                          pick_idx: int | None = None):
    """4x4 grid, one example, QUERY row attention plotted with annotations."""
    N, L, H, T, _ = attn.shape
    marker = examples["marker_pos"].numpy()

    if pick_idx is None:
        # Prefer an example with marker somewhere near the middle.
        want_lo, want_hi = int(T * 0.35), int(T * 0.55)
        candidates = [i for i in range(N) if want_lo <= marker[i] <= want_hi]
        pick_idx = candidates[len(candidates) // 2] if candidates else 0

    q = _query_rows(attn, examples["query_pos"])[pick_idx]  # [L, H, T]
    m = int(examples["marker_pos"][pick_idx])
    ps = int(examples["passkey_start"][pick_idx])
    pe = int(examples["passkey_end"][pick_idx])
    qp = int(examples["query_pos"][pick_idx])

    fig, axes = plt.subplots(L, H, figsize=(3 * H, 2.1 * L), sharex=True)
    for l in range(L):
        for h in range(H):
            ax = axes[l, h] if L > 1 else axes[h]
            ax.plot(q[l, h].numpy(), lw=1.0)
            ax.set_title(f"L{l} H{h}", fontsize=10)
            _annotate(ax, m, ps, pe, qp, T)
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1.0)
            ax.set_xlim(0, T)
            ax.grid(alpha=0.2)
    fig.suptitle(f"Example #{pick_idx}: MARKER @ {m}, passkey {ps}-{pe}, "
                   f"QUERY @ {qp}", fontsize=11)
    fig.supxlabel("key position (green=MARKER, orange=QUERY, shaded=passkey)")
    fig.supylabel("attention mass (log)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path} (example {pick_idx})")


def fig3_retrieval_zoom(attn: torch.Tensor, examples: dict, out_path: Path,
                           retrieval_heads: list, pick_idx: int | None = None):
    """Zoom on the retrieval heads.

    Two rows per head:
      (a) average aligned + single-example query-row curves.
      (b) marginal bar: to-marker vs to-passkey vs to-elsewhere.
    """
    N, L, H, T, _ = attn.shape
    q = _query_rows(attn, examples["query_pos"])  # [N, L, H, T]
    marker = examples["marker_pos"].numpy()
    K = examples["pkcfg"]["passkey_len"]

    if pick_idx is None:
        want_lo, want_hi = int(T * 0.35), int(T * 0.55)
        cands = [i for i in range(N) if want_lo <= marker[i] <= want_hi]
        pick_idx = cands[len(cands) // 2] if cands else 0

    # Averaged aligned curves.
    center = T // 2
    aligned = torch.zeros_like(q)
    for i in range(N):
        shift = int(center - marker[i])
        aligned[i] = q[i].roll(shifts=shift, dims=-1)
    mean_q = aligned.mean(0)  # [L, H, T]

    # Per-head marginals.
    nh = len(retrieval_heads)
    fig, axes = plt.subplots(2, nh, figsize=(4.5 * nh, 6.5))
    if nh == 1:
        axes = axes[:, None]

    for col, (l, h) in enumerate(retrieval_heads):
        # Top: avg aligned curve, plus the cherry-picked example.
        ax = axes[0, col]
        ax.plot(mean_q[l, h].numpy(), label="avg (aligned)", color="black",
                 lw=1.2)
        example_q = q[pick_idx, l, h].numpy()
        # For apples-to-apples, align the example too.
        shift = int(center - marker[pick_idx])
        example_q = np.roll(example_q, shift)
        ax.plot(example_q, label=f"ex #{pick_idx}", color="#c04040",
                 alpha=0.6, lw=0.8)
        ax.axvline(center, color="#00a000", lw=0.8)
        ax.axvspan(center + 1, center + 1 + K, color="#0030a0", alpha=0.15)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.0)
        ax.set_xlim(0, T)
        ax.set_title(f"L{l} H{h}", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        ax.set_xlabel("key position (aligned)")
        ax.set_ylabel("attention (log)")

        # Bottom: marginal bars.
        ax = axes[1, col]
        # Compute marker / passkey / elsewhere means across examples.
        # q[:, l, h, :] with masks from examples.
        to_marker_vals = np.zeros(N)
        to_passkey_vals = np.zeros(N)
        to_post_vals = np.zeros(N)
        for i in range(N):
            m = int(examples["marker_pos"][i])
            ps = int(examples["passkey_start"][i])
            pe = int(examples["passkey_end"][i])
            row = q[i, l, h].numpy()
            to_marker_vals[i] = row[m]
            to_passkey_vals[i] = row[ps : pe + 1].sum()
            to_post_vals[i] = row[m + 1 : m + 1 + K].sum()
        elsewhere = 1.0 - (to_marker_vals + to_passkey_vals)
        # Use stacked bar summarizing means.
        labels = ["MARKER", "passkey", "elsewhere"]
        values = [to_marker_vals.mean(), to_passkey_vals.mean(),
                   elsewhere.mean()]
        ax.bar(labels, values, color=["#00a000", "#0030a0", "#aaaaaa"])
        ax.set_ylim(0, max(1.0, max(values) * 1.1))
        ax.set_title(f"L{l} H{h}: mass split", fontsize=10)
        ax.grid(alpha=0.2, axis="y")
        for j, v in enumerate(values):
            ax.text(j, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default=str(RESULTS_DIR / "attention_tensors.pt"))
    ap.add_argument("--examples", default=str(RESULTS_DIR / "probe_examples.pt"))
    ap.add_argument("--importance", default=None,
                    help="Path to head_importance.json; used to select retrieval heads.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--pick", type=int, default=None,
                    help="Force a specific example index for Fig 2/3.")
    args = ap.parse_args()

    attn_blob = torch.load(args.attn, map_location="cpu", weights_only=False)
    examples = torch.load(args.examples, map_location="cpu", weights_only=False)
    attn = attn_blob["attention"]

    # Identify retrieval heads for Fig 3.
    if args.importance:
        import json
        imp = json.loads(Path(args.importance).read_text())
        retrieval = sorted(
            [(r["layer"], r["head"]) for r in imp["per_head"]
             if r["importance_passkey_exact"] > args.threshold],
            key=lambda k: -next(r["importance_passkey_exact"]
                                   for r in imp["per_head"]
                                   if r["layer"] == k[0] and r["head"] == k[1]),
        )
    else:
        # Default: baseline's three top retrieval heads.
        retrieval = [(0, 3), (2, 0), (0, 2)]

    fig1_gestalt_grid(attn, examples, RESULTS_DIR / "fig1_gestalt_grid.png")
    fig2_single_example(attn, examples,
                          RESULTS_DIR / "fig2_single_example.png",
                          pick_idx=args.pick)
    fig3_retrieval_zoom(attn, examples,
                          RESULTS_DIR / "fig3_retrieval_heads_zoom.png",
                          retrieval_heads=retrieval,
                          pick_idx=args.pick)


if __name__ == "__main__":
    main()
