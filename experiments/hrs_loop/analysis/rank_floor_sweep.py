"""Test 2: SVD-truncate Variant B's MPAR_project (up-projection) at
inference time and re-evaluate perplexity.

We truncate the up-projector (rank_m -> d) which controls the bias
signal that reaches the residual stream. Ranks swept:
  {256, 128, 64, 32, 16, 8, 4}

256 > rank_m=128 tests whether padding above rank is a no-op (expected).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.hrs_loop.analysis._shared import (
    CKPT_DIR, RESULTS_DIR, eval_ppl_with_T, load_checkpoint, load_val_loader,
)


RANKS = [256, 128, 64, 32, 16, 8, 4]


@torch.no_grad()
def truncate_projector(model, rank: int):
    """In-place SVD truncation of MPAR_project.up.weight to `rank`.

    The up-projector has shape (d_model, rank_m). We truncate the SVD
    so only the top-`rank` singular components are preserved. If rank
    exceeds min(d_model, rank_m), truncation is a no-op.
    """
    W = model.recurrent.project_up.up.weight.data  # (d, rank_m)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    keep = min(rank, S.shape[0])
    S_new = S.clone()
    if keep < S.shape[0]:
        S_new[keep:] = 0
    W_new = U @ torch.diag(S_new) @ Vh
    model.recurrent.project_up.up.weight.data.copy_(W_new)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    ap.add_argument("--n-batches", type=int, default=30)
    ap.add_argument("--out", default=str(RESULTS_DIR / "test2_rank_floor.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, _tok = load_val_loader(batch_size=16)

    # Reload a fresh copy per rank (truncation is destructive).
    records = []
    for rank in RANKS:
        model, cfg, ckpt = load_checkpoint(args.variant, device)
        truncate_projector(model, rank)
        ppl = eval_ppl_with_T(model, val_loader, device, T=cfg.T_default,
                                n_batches=args.n_batches)
        print(f"  rank={rank:>3}: val_ppl={ppl:.3f}")
        records.append({"rank": rank, "val_ppl": ppl})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "variant": args.variant,
        "records": records,
        "baseline_full_rank": records[0]["val_ppl"],
    }, indent=2))
    print(f"wrote {out_path}")

    # Plot.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["rank"] for r in records]
    ys = [r["val_ppl"] for r in records]
    ax.plot(xs, ys, marker="o", lw=1.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlabel("SVD-truncated rank of MPAR_project (up)")
    ax.set_ylabel("val PPL")
    ax.set_title(f"Variant {args.variant}: rank floor under MPAR_project truncation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = RESULTS_DIR / "test2_rank_floor.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
