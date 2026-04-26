"""Analyze saliency-compression-then-attention sweep vs prior variants.

Quality + speed comparison. The user explicitly asked for timing data so
the speedup story is reportable.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from experiments.head_aggregation.analyze import welch_t_ci


def load_run(p: Path) -> dict:
    return json.loads(p.read_text())


def collect(dir_: Path, variant: str) -> list[dict]:
    return [load_run(p) for p in sorted(dir_.glob(f"{variant}_seed*.json"))]


def stats(runs: list[dict]) -> dict:
    if not runs:
        return {"n": 0}
    ppls = [r["final_val_ppl"] for r in runs]
    n = len(ppls)
    mean = statistics.mean(ppls)
    std = statistics.stdev(ppls) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n, "ppls": ppls,
        "mean": mean, "std": std, "se": se,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "wall_s_mean": statistics.mean([r["wall_seconds"] for r in runs]),
        "total_params_mean": int(statistics.mean([r["total_params"] for r in runs])),
        "step_time_ms_median_mean": statistics.mean(
            [r["step_time_ms_median"] for r in runs]
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compression-dir",
                    default="experiments/saliency_pool/results_compression_causal")
    ap.add_argument("--shakespeare-dir",
                    default="experiments/saliency_pool/results")
    ap.add_argument("--cumulative-dir",
                    default="experiments/saliency_pool/results_cumulative_mean")
    ap.add_argument("--dual-dir",
                    default="experiments/saliency_pool/results_dual_projection")
    args = ap.parse_args()

    cp_dir = Path(args.compression_dir)
    sh_dir = Path(args.shakespeare_dir)
    cm_dir = Path(args.cumulative_dir)
    dp_dir = Path(args.dual_dir)

    bl_runs = collect(sh_dir, "baseline")
    d_runs = collect(sh_dir, "D")
    cm_runs = collect(cm_dir, "cumulative_mean")
    dp_runs = collect(dp_dir, "dual_projection")
    dpc_runs = collect(dp_dir, "dual_projection_with_cumulative")
    c4_runs = collect(cp_dir, "compress_4")
    c8_runs = collect(cp_dir, "compress_8")
    c16_runs = collect(cp_dir, "compress_16")

    s = {
        "baseline": stats(bl_runs),
        "D": stats(d_runs),
        "cumulative_mean": stats(cm_runs),
        "dual_projection": stats(dp_runs),
        "dual_projection_with_cumulative": stats(dpc_runs),
        "compress_4": stats(c4_runs),
        "compress_8": stats(c8_runs),
        "compress_16": stats(c16_runs),
    }

    bl_ppls = [r["final_val_ppl"] for r in bl_runs]
    dpc_ppls = [r["final_val_ppl"] for r in dpc_runs]
    d_ppls = [r["final_val_ppl"] for r in d_runs]
    c4_ppls = [r["final_val_ppl"] for r in c4_runs]
    c8_ppls = [r["final_val_ppl"] for r in c8_runs]
    c16_ppls = [r["final_val_ppl"] for r in c16_runs]

    cmp = {
        "c4_vs_baseline": welch_t_ci(bl_ppls, c4_ppls) if (bl_ppls and c4_ppls) else None,
        "c8_vs_baseline": welch_t_ci(bl_ppls, c8_ppls) if (bl_ppls and c8_ppls) else None,
        "c16_vs_baseline": welch_t_ci(bl_ppls, c16_ppls) if (bl_ppls and c16_ppls) else None,
        "c4_vs_dpc": welch_t_ci(dpc_ppls, c4_ppls) if (dpc_ppls and c4_ppls) else None,
        "c8_vs_dpc": welch_t_ci(dpc_ppls, c8_ppls) if (dpc_ppls and c8_ppls) else None,
        "c16_vs_dpc": welch_t_ci(dpc_ppls, c16_ppls) if (dpc_ppls and c16_ppls) else None,
        "c4_vs_d": welch_t_ci(d_ppls, c4_ppls) if (d_ppls and c4_ppls) else None,
        "c4_vs_c8": welch_t_ci(c8_ppls, c4_ppls) if (c4_ppls and c8_ppls) else None,
        "c8_vs_c16": welch_t_ci(c16_ppls, c8_ppls) if (c8_ppls and c16_ppls) else None,
    }

    print("\n=== Compression sweep — quality + speed summary ===\n")
    header = (f"{'variant':>34s}  {'n':>2s}  {'mean':>8s}  {'std':>6s}  "
              f"{'95% CI':>20s}  {'params':>10s}  {'step_ms':>8s}  "
              f"{'wall_s':>7s}  {'speedup':>8s}")
    print(header)
    print("-" * len(header))

    bl_step = s["baseline"]["step_time_ms_median_mean"] if s["baseline"]["n"] else float("nan")
    bl_wall = s["baseline"]["wall_s_mean"] if s["baseline"]["n"] else float("nan")
    rows = [
        ("baseline (V22 Bonsignore)", s["baseline"]),
        ("D (V/W_O + uniform attn)", s["D"]),
        ("cumulative_mean (no proj)", s["cumulative_mean"]),
        ("dual_projection", s["dual_projection"]),
        ("dual_projection_with_cumulative", s["dual_projection_with_cumulative"]),
        ("compress_4 (m=64)", s["compress_4"]),
        ("compress_8 (m=32)", s["compress_8"]),
        ("compress_16 (m=16)", s["compress_16"]),
    ]
    for label, st in rows:
        if st["n"] == 0:
            print(f"{label:>34s}  -- no runs --")
            continue
        speedup = bl_step / st["step_time_ms_median_mean"]
        print(f"{label:>34s}  {st['n']:>2d}  {st['mean']:>8.3f}  {st['std']:>6.3f}  "
              f"[{st['ci95'][0]:>7.3f},{st['ci95'][1]:>7.3f}]  "
              f"{st['total_params_mean']:>10,d}  "
              f"{st['step_time_ms_median_mean']:>7.1f}  "
              f"{st['wall_s_mean']:>7.0f}  "
              f"{speedup:>7.2f}x")

    def _fmt_t(t):
        if t is None:
            return "—"
        return (f"Δ={t['mean_diff']:+.3f}  "
                f"d={t['cohens_d']:+.2f}  "
                f"p={t['p']:.4g}  "
                f"95% CI Δ=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}]")

    print("\n=== Pairwise (Welch's t) ===")
    print(f"  compress_4   vs baseline:   {_fmt_t(cmp['c4_vs_baseline'])}")
    print(f"  compress_8   vs baseline:   {_fmt_t(cmp['c8_vs_baseline'])}")
    print(f"  compress_16  vs baseline:   {_fmt_t(cmp['c16_vs_baseline'])}")
    print(f"  compress_4   vs dpc:        {_fmt_t(cmp['c4_vs_dpc'])}")
    print(f"  compress_8   vs dpc:        {_fmt_t(cmp['c8_vs_dpc'])}")
    print(f"  compress_16  vs dpc:        {_fmt_t(cmp['c16_vs_dpc'])}")
    print(f"  compress_4   vs variant_D:  {_fmt_t(cmp['c4_vs_d'])}")
    print(f"  compress_4   vs compress_8: {_fmt_t(cmp['c4_vs_c8'])}")
    print(f"  compress_8   vs compress_16:{_fmt_t(cmp['c8_vs_c16'])}")

    out_json = cp_dir / "report.json"
    out_json.write_text(json.dumps({
        "stats": s,
        "comparisons": cmp,
        "baseline_step_ms": bl_step,
        "baseline_wall_s": bl_wall,
    }, indent=2))
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
