"""Stage 2 analysis: aggregate best_val_ppl across seeds {0,1,2,3} for each
variant, compute pairwise Welch's t-test and Cohen's d for the claimed
gaps (B−A, C−B, D−B).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

SEEDS = [0, 1, 2, 3]
VARIANTS = ["A", "B", "C", "D"]


def welch_t(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    se2 = va / na + vb / nb
    if se2 <= 0:
        return (float("inf") if ma != mb else 0.0,
                0.0 if ma != mb else 1.0,
                float("nan"))
    t = (ma - mb) / math.sqrt(se2)
    # Welch–Satterthwaite df
    df_num = se2 ** 2
    df_den = (va / na) ** 2 / max(1, na - 1) + (vb / nb) ** 2 / max(1, nb - 1)
    df = df_num / df_den if df_den > 0 else float("inf")
    # Normal approximation for p (two-sided) is fine at these df
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p, df


def cohens_d(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) if na > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) if nb > 1 else 0.0
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("inf") if ma != mb else 0.0
    return (ma - mb) / pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "stage2_seed_variance.json"))
    args = ap.parse_args()

    per_variant = {}
    for v in VARIANTS:
        bests, bests_steps, finals = [], [], []
        for s in SEEDS:
            suffix = "" if s == 0 else f"_seed{s}"
            log_path = RESULTS_DIR / f"train_log_{v}{suffix}.json"
            if not log_path.exists():
                print(f"  missing {log_path}")
                continue
            d = json.loads(log_path.read_text())
            bests.append(d["best_val_ppl"])
            bests_steps.append(d["best_step"])
            finals.append(d["final_val_ppl"])
        per_variant[v] = {
            "n": len(bests),
            "best_val_ppl": bests,
            "best_step": bests_steps,
            "final_val_ppl": finals,
            "best_mean": statistics.mean(bests) if bests else None,
            "best_std": statistics.stdev(bests) if len(bests) > 1 else None,
            "best_step_mean": statistics.mean(bests_steps) if bests_steps else None,
            "final_mean": statistics.mean(finals) if finals else None,
            "final_std": statistics.stdev(finals) if len(finals) > 1 else None,
            "overfit_ratio_mean": (statistics.mean(finals) / statistics.mean(bests)
                                     if bests and finals else None),
        }

    # Pairwise significance tests.
    pairs = [("B", "A"), ("C", "B"), ("D", "B"), ("D", "A"), ("C", "A")]
    compare = {}
    for v1, v2 in pairs:
        a = per_variant[v1]["best_val_ppl"]
        b = per_variant[v2]["best_val_ppl"]
        if not a or not b:
            continue
        t, p, df = welch_t(a, b)
        d = cohens_d(a, b)
        ma, mb = statistics.mean(a), statistics.mean(b)
        compare[f"{v1}_vs_{v2}"] = {
            "mean_diff_v1_minus_v2": ma - mb,
            "relative_diff_pct": 100.0 * (ma - mb) / mb,
            "cohen_d": d,
            "welch_t": t,
            "welch_p_two_sided": p,
            "welch_df": df,
            "significant_at_p05": p < 0.05,
            "v1_mean_std": (ma, statistics.stdev(a) if len(a) > 1 else 0.0),
            "v2_mean_std": (mb, statistics.stdev(b) if len(b) > 1 else 0.0),
            "v1_n": len(a), "v2_n": len(b),
        }

    out = {"per_variant": per_variant, "comparisons": compare}
    Path(args.out).write_text(json.dumps(out, indent=2))

    # Pretty table.
    print(f"\n{'Variant':>7}  {'n':>2}  {'mean ± std':>18}  {'min':>8}  "
          f"{'max':>8}  {'best step mean':>14}  {'overfit ratio':>14}")
    print("-" * 80)
    for v in VARIANTS:
        r = per_variant[v]
        if r["best_mean"] is None:
            continue
        ms = f"{r['best_mean']:.2f} ± {r['best_std']:.2f}" if r['best_std'] is not None else f"{r['best_mean']:.2f}"
        print(f"{v:>7}  {r['n']:>2}  {ms:>18}  "
              f"{min(r['best_val_ppl']):>8.2f}  {max(r['best_val_ppl']):>8.2f}  "
              f"{r['best_step_mean']:>14.0f}  "
              f"{r['overfit_ratio_mean']:>14.2f}")

    print(f"\nPairwise comparisons (best val PPL):")
    for key, c in compare.items():
        marker = " **" if c["significant_at_p05"] else "   "
        print(f"  {key:>8}  Δ={c['mean_diff_v1_minus_v2']:+7.2f} "
              f"({c['relative_diff_pct']:+6.2f}%)  "
              f"d={c['cohen_d']:+5.2f}  "
              f"t={c['welch_t']:+5.2f}  p={c['welch_p_two_sided']:.3f}{marker}")

    print(f"\nwrote {Path(args.out)}")


if __name__ == "__main__":
    main()
