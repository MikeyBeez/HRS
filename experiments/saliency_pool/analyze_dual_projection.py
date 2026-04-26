"""Analyze the dual-projection experiment vs baseline / variant_D / cumulative_mean.

Loads:
  - dual_projection                  seeds 0..3 from `results_dual_projection/`
  - dual_projection_with_cumulative  seeds 0..3 from `results_dual_projection/`
  - baseline                         seeds 0..3 from `results/`
  - D                                seeds 0..3 from `results/`
  - cumulative_mean                  seeds 0..3 from `results_cumulative_mean/`

Writes:
  - `results_dual_projection/report.json` — machine-readable summary
  - prints a textual summary suitable for the REPORT.md
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
    ap.add_argument("--dual-dir",
                    default="experiments/saliency_pool/results_dual_projection")
    ap.add_argument("--shakespeare-dir",
                    default="experiments/saliency_pool/results")
    ap.add_argument("--cumulative-dir",
                    default="experiments/saliency_pool/results_cumulative_mean")
    args = ap.parse_args()

    dp_dir = Path(args.dual_dir)
    sh_dir = Path(args.shakespeare_dir)
    cm_dir = Path(args.cumulative_dir)

    dp_runs = collect(dp_dir, "dual_projection")
    dpc_runs = collect(dp_dir, "dual_projection_with_cumulative")
    bl_runs = collect(sh_dir, "baseline")
    d_runs = collect(sh_dir, "D")
    cm_runs = collect(cm_dir, "cumulative_mean")

    s = {
        "baseline": stats(bl_runs),
        "D": stats(d_runs),
        "cumulative_mean": stats(cm_runs),
        "dual_projection": stats(dp_runs),
        "dual_projection_with_cumulative": stats(dpc_runs),
    }

    bl_ppls = [r["final_val_ppl"] for r in bl_runs]
    d_ppls = [r["final_val_ppl"] for r in d_runs]
    cm_ppls = [r["final_val_ppl"] for r in cm_runs]
    dp_ppls = [r["final_val_ppl"] for r in dp_runs]
    dpc_ppls = [r["final_val_ppl"] for r in dpc_runs]

    cmp = {
        "dp_vs_baseline": welch_t_ci(bl_ppls, dp_ppls) if (bl_ppls and dp_ppls) else None,
        "dpc_vs_baseline": welch_t_ci(bl_ppls, dpc_ppls) if (bl_ppls and dpc_ppls) else None,
        "dp_vs_d": welch_t_ci(d_ppls, dp_ppls) if (d_ppls and dp_ppls) else None,
        "dpc_vs_d": welch_t_ci(d_ppls, dpc_ppls) if (d_ppls and dpc_ppls) else None,
        "dp_vs_cm": welch_t_ci(cm_ppls, dp_ppls) if (cm_ppls and dp_ppls) else None,
        "dpc_vs_cm": welch_t_ci(cm_ppls, dpc_ppls) if (cm_ppls and dpc_ppls) else None,
        "dpc_vs_dp": welch_t_ci(dp_ppls, dpc_ppls) if (dp_ppls and dpc_ppls) else None,
    }

    print("\n=== Dual-projection sweep — summary ===\n")
    header = f"{'variant':>34s}  {'n':>2s}  {'mean':>8s}  {'std':>6s}  {'95% CI':>20s}  {'params':>10s}  {'step_ms':>8s}  {'wall_s':>7s}"
    print(header)
    print("-" * len(header))
    rows = [
        ("baseline (V22 Bonsignore)", s["baseline"]),
        ("D (V/W_O + uniform attn)", s["D"]),
        ("cumulative_mean (no proj)", s["cumulative_mean"]),
        ("dual_projection", s["dual_projection"]),
        ("dual_projection_with_cumulative", s["dual_projection_with_cumulative"]),
    ]
    for label, st in rows:
        if st["n"] == 0:
            print(f"{label:>34s}  -- no runs --")
            continue
        print(f"{label:>34s}  {st['n']:>2d}  {st['mean']:>8.3f}  {st['std']:>6.3f}  "
              f"[{st['ci95'][0]:>7.3f},{st['ci95'][1]:>7.3f}]  "
              f"{st['total_params_mean']:>10,d}  "
              f"{st['step_time_ms_median_mean']:>7.1f}  "
              f"{st['wall_s_mean']:>7.0f}")

    def _fmt_t(t):
        if t is None:
            return "—"
        return (f"Δ={t['mean_diff']:+.3f}  "
                f"d={t['cohens_d']:+.2f}  "
                f"p={t['p']:.4g}  "
                f"95% CI Δ=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}]")

    print("\n=== Pairwise (Welch's t) ===")
    print(f"  dual_projection                  vs baseline:        {_fmt_t(cmp['dp_vs_baseline'])}")
    print(f"  dual_projection_with_cumulative  vs baseline:        {_fmt_t(cmp['dpc_vs_baseline'])}")
    print(f"  dual_projection                  vs variant_D:       {_fmt_t(cmp['dp_vs_d'])}")
    print(f"  dual_projection_with_cumulative  vs variant_D:       {_fmt_t(cmp['dpc_vs_d'])}")
    print(f"  dual_projection                  vs cumulative_mean: {_fmt_t(cmp['dp_vs_cm'])}")
    print(f"  dual_projection_with_cumulative  vs cumulative_mean: {_fmt_t(cmp['dpc_vs_cm'])}")
    print(f"  dual_projection_with_cumulative  vs dual_projection: {_fmt_t(cmp['dpc_vs_dp'])}")

    out_json = dp_dir / "report.json"
    out_json.write_text(json.dumps({
        "stats": s,
        "comparisons": cmp,
    }, indent=2))
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
