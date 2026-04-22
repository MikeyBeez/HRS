"""Test 3: MPAR cosine structure across loops.

For Variant B on held-out val batches, capture m_1..m_T and compute:
  - same-input cross-loop cosine: cos(m_t, m_{t+1}) averaged over batch.
  - random-batch baseline: cos(m_t[i], m_t[j]) for i != j at same loop index.

Success criterion (from plan): same-input cross-loop cos >= 0.5; random
cross-batch cos < 0.05.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.hrs_loop.analysis._shared import (
    RESULTS_DIR, load_checkpoint, load_val_loader,
)


@torch.no_grad()
def collect_mpars(model, val_loader, device, n_batches: int = 20) -> torch.Tensor:
    """Return (N, T, rank_m) stack of all captured MPARs across batches."""
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    all_m = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                             enabled=(device.type == "cuda")):
            _logits, captured = model(x[:, :-1], mpar_capture=True)
        stacked = torch.stack(captured, dim=1)  # (B, T, rank_m)
        all_m.append(stacked.to(torch.float32).cpu())
    return torch.cat(all_m, dim=0)


def pairwise_cos_same_input(m_all: torch.Tensor):
    """m_all: (N, T, R). Returns list of mean cos(m[:, t], m[:, t+1]) across N."""
    N, T, R = m_all.shape
    results = []
    for t in range(T - 1):
        a = F.normalize(m_all[:, t], dim=-1)
        b = F.normalize(m_all[:, t + 1], dim=-1)
        cos = (a * b).sum(-1)              # (N,)
        results.append({
            "loop_pair": [t + 1, t + 2],   # 1-indexed label for plot
            "mean_cos": float(cos.mean()),
            "std_cos": float(cos.std()),
            "min_cos": float(cos.min()),
            "max_cos": float(cos.max()),
        })
    return results


def pairwise_cos_cross_batch(m_all: torch.Tensor, n_pairs: int = 2000):
    """Random cross-batch pairs at the SAME loop index."""
    import random
    rng = random.Random(1234)
    N, T, R = m_all.shape
    results = []
    for t in range(T):
        cos_vals = []
        for _ in range(n_pairs):
            i, j = rng.randrange(N), rng.randrange(N)
            if i == j:
                continue
            a = F.normalize(m_all[i, t], dim=-1)
            b = F.normalize(m_all[j, t], dim=-1)
            cos_vals.append(float((a * b).sum()))
        import statistics
        results.append({
            "loop": t + 1,
            "mean_cos": statistics.mean(cos_vals),
            "std_cos": statistics.pstdev(cos_vals),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--out", default=str(RESULTS_DIR / "test3_mpar_cosine.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(args.variant, device)
    val_loader, _tok = load_val_loader(batch_size=16)

    m_all = collect_mpars(model, val_loader, device, n_batches=args.n_batches)
    print(f"collected MPARs shape: {tuple(m_all.shape)}  (N, T, rank_m)")

    same = pairwise_cos_same_input(m_all)
    cross = pairwise_cos_cross_batch(m_all)

    print("\nSame-input cross-loop cosine:")
    for r in same:
        print(f"  m_{r['loop_pair'][0]} ↔ m_{r['loop_pair'][1]}: "
              f"mean={r['mean_cos']:.3f}  std={r['std_cos']:.3f}  "
              f"range [{r['min_cos']:.3f}, {r['max_cos']:.3f}]")
    print("\nCross-batch cosine at same loop index (random baseline):")
    for r in cross:
        print(f"  loop {r['loop']}: mean={r['mean_cos']:+.3f}  "
              f"std={r['std_cos']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "variant": args.variant,
        "same_input_cross_loop": same,
        "cross_batch_same_loop": cross,
        "n_sequences": m_all.shape[0],
    }, indent=2))
    print(f"wrote {Path(args.out)}")

    # Plot.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs_same = [r["loop_pair"][0] + 0.5 for r in same]
    ys_same = [r["mean_cos"] for r in same]
    ys_same_err = [r["std_cos"] for r in same]
    xs_cross = [r["loop"] for r in cross]
    ys_cross = [r["mean_cos"] for r in cross]
    ys_cross_err = [r["std_cos"] for r in cross]
    ax.errorbar(xs_same, ys_same, yerr=ys_same_err, marker="o",
                 lw=1.8, label="same input, consecutive loops")
    ax.errorbar(xs_cross, ys_cross, yerr=ys_cross_err, marker="s",
                 lw=1.8, label="cross-batch at same loop (random)")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.6,
                label="target (>= 0.5)")
    ax.set_xlabel("loop index")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(f"Variant {args.variant}: MPAR cosine structure")
    ax.set_ylim(-0.2, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    png = RESULTS_DIR / "test3_mpar_cosine.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
