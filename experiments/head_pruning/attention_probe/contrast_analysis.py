"""Phase D: hypothesis-specific statistical checks.

1. Retrieval heads vs non-retrieval heads: contrast on to_marker.
2. L0 retrieval heads (H2, H3) vs L2 H0: attending to MARKER (find) vs
   post-MARKER/passkey (compose).

Retrieval heads are read from the existing baseline importance ranking
(Δpasskey > 0.5 when zeroed). For baseline seed 0, these are
L0 H1 (0.53), L0 H2 (0.96), L0 H3 (1.00), L2 H0 (1.00).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
BASELINE_IMPORTANCE = Path(
    "/mnt/data/Code/HRS/experiments/head_pruning/results/head_importance.json"
)


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U, returning (U, p approx via normal)."""
    a = sorted(a)
    b = sorted(b)
    na, nb = len(a), len(b)
    combined = sorted([(x, "a") for x in a] + [(x, "b") for x in b])
    # Rank (midranks for ties)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = mean_rank
        i = j + 1
    ranksum_a = sum(ranks[k] for k, (x, lbl) in enumerate(combined) if lbl == "a")
    U_a = ranksum_a - na * (na + 1) / 2
    U_b = na * nb - U_a
    U = min(U_a, U_b)
    # Normal approximation
    mu = na * nb / 2
    sigma2 = na * nb * (na + nb + 1) / 12
    if sigma2 <= 0:
        return U, float("nan")
    z = (U - mu) / (sigma2 ** 0.5)
    # Two-sided p via erf approximation.
    import math
    p = math.erfc(abs(z) / math.sqrt(2))
    return U, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(RESULTS_DIR / "head_metrics.json"))
    ap.add_argument("--importance", default=str(BASELINE_IMPORTANCE))
    ap.add_argument("--out", default=str(RESULTS_DIR / "contrast_analysis.txt"))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Δpasskey threshold to count a head as retrieval.")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    importance = json.loads(Path(args.importance).read_text())

    # Retrieval heads from importance ranking.
    per_imp = {(r["layer"], r["head"]): r for r in importance["per_head"]}
    retrieval = set()
    for (l, h), r in per_imp.items():
        if r["importance_passkey_exact"] > args.threshold:
            retrieval.add((l, h))

    # Map metrics by (layer, head).
    per_m = {(r["layer"], r["head"]): r for r in metrics["per_head"]}

    ret_to_marker = [per_m[k]["to_marker"] for k in retrieval]
    non_to_marker = [per_m[k]["to_marker"] for k in per_m if k not in retrieval]

    U, p = mann_whitney_u(ret_to_marker, non_to_marker)

    lines = ["Contrast analysis", "=================", ""]
    lines.append(f"Retrieval heads (Δpasskey > {args.threshold}):")
    for l, h in sorted(retrieval):
        m = per_m[(l, h)]
        delta = per_imp[(l, h)]["importance_passkey_exact"]
        lines.append(f"  L{l}H{h}: Δpasskey={delta:+.2f}  "
                      f"to_marker={m['to_marker']:.3f}  "
                      f"to_passkey={m['to_passkey']:.3f}  "
                      f"to_postmk={m['to_postmarker']:.3f}  "
                      f"entropy={m['entropy']:.3f}  "
                      f"max_off={m['max_offset_median']:+.1f}")
    lines.append("")
    lines.append("Non-retrieval heads:")
    for k in sorted(per_m):
        if k in retrieval:
            continue
        l, h = k
        m = per_m[k]
        delta = per_imp[k]["importance_passkey_exact"]
        lines.append(f"  L{l}H{h}: Δpasskey={delta:+.2f}  "
                      f"to_marker={m['to_marker']:.3f}  "
                      f"to_passkey={m['to_passkey']:.3f}  "
                      f"to_postmk={m['to_postmarker']:.3f}  "
                      f"entropy={m['entropy']:.3f}  "
                      f"max_off={m['max_offset_median']:+.1f}")
    lines.append("")

    def _summary(vals):
        if not vals:
            return "n=0"
        return (f"n={len(vals)} mean={sum(vals)/len(vals):.3f} "
                 f"min={min(vals):.3f} max={max(vals):.3f}")

    lines.append(f"to_marker (retrieval):      {_summary(ret_to_marker)}")
    lines.append(f"to_marker (non-retrieval):  {_summary(non_to_marker)}")
    lines.append(f"Mann-Whitney U: U={U:.1f}  two-sided p ≈ {p:.4f}")
    lines.append("")

    # L0 retrieval vs L2 retrieval distinctness.
    l0_ret = [k for k in retrieval if k[0] == 0]
    l2_ret = [k for k in retrieval if k[0] == 2]
    lines.append(f"L0 retrieval heads: {l0_ret}")
    for k in l0_ret:
        m = per_m[k]
        lines.append(f"  L{k[0]}H{k[1]}: to_marker={m['to_marker']:.3f} "
                      f"to_postmk={m['to_postmarker']:.3f} "
                      f"ratio={m['to_marker']/max(1e-9, m['to_postmarker']):.2f}")
    lines.append(f"L2 retrieval heads: {l2_ret}")
    for k in l2_ret:
        m = per_m[k]
        lines.append(f"  L{k[0]}H{k[1]}: to_marker={m['to_marker']:.3f} "
                      f"to_postmk={m['to_postmarker']:.3f} "
                      f"ratio={m['to_marker']/max(1e-9, m['to_postmarker']):.2f}")

    # Interpretation: find hypothesis predicts L0 to_marker >> to_postmarker
    # and L2 to_postmarker > to_marker.
    if l0_ret and l2_ret:
        l0_marker = sum(per_m[k]["to_marker"] for k in l0_ret) / len(l0_ret)
        l0_post = sum(per_m[k]["to_postmarker"] for k in l0_ret) / len(l0_ret)
        l2_marker = sum(per_m[k]["to_marker"] for k in l2_ret) / len(l2_ret)
        l2_post = sum(per_m[k]["to_postmarker"] for k in l2_ret) / len(l2_ret)
        lines.append("")
        lines.append(f"L0 retrieval mean: to_marker={l0_marker:.3f} "
                      f"to_postmk={l0_post:.3f}")
        lines.append(f"L2 retrieval mean: to_marker={l2_marker:.3f} "
                      f"to_postmk={l2_post:.3f}")
        lines.append("")
        finds_predict = l0_marker > l0_post
        composes_predict = l2_post > l2_marker
        lines.append(f"Predicted by find/compose hypothesis:")
        lines.append(f"  L0 attends marker > post-marker: "
                      f"{'✓' if finds_predict else '✗'}")
        lines.append(f"  L2 attends post-marker > marker: "
                      f"{'✓' if composes_predict else '✗'}")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
