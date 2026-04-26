"""Analyze the causal cumulative mean experiment vs baseline + variant D.

Loads:
  - cumulative_mean seeds 0..3 from `results_cumulative_mean/`
  - baseline   seeds 0..3 from `results/`     (existing Shakespeare baseline)
  - D          seeds 0..3 from `results/`     (existing pure-mean-pool floor)

Writes:
  - `results_cumulative_mean/report.json` — machine-readable summary
  - `results_cumulative_mean/REPORT.md`   — human-readable report
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
    ap.add_argument("--cumulative-mean-dir",
                    default="experiments/saliency_pool/results_cumulative_mean")
    ap.add_argument("--shakespeare-dir",
                    default="experiments/saliency_pool/results")
    args = ap.parse_args()

    cm_dir = Path(args.cumulative_mean_dir)
    sh_dir = Path(args.shakespeare_dir)

    cm_runs = collect(cm_dir, "cumulative_mean")
    bl_runs = collect(sh_dir, "baseline")
    d_runs = collect(sh_dir, "D")

    cm_s = stats(cm_runs)
    bl_s = stats(bl_runs)
    d_s = stats(d_runs)

    bl_ppls = [r["final_val_ppl"] for r in bl_runs]
    cm_ppls = [r["final_val_ppl"] for r in cm_runs]
    d_ppls = [r["final_val_ppl"] for r in d_runs]

    cm_vs_bl = welch_t_ci(bl_ppls, cm_ppls) if (bl_ppls and cm_ppls) else None
    d_vs_bl = welch_t_ci(bl_ppls, d_ppls) if (bl_ppls and d_ppls) else None
    cm_vs_d = welch_t_ci(d_ppls, cm_ppls) if (d_ppls and cm_ppls) else None

    print("\n=== Cumulative-mean vs baseline + variant D — summary ===\n")
    print(f"{'variant':>20s}  {'n':>2s}  {'mean':>8s}  {'std':>6s}  "
          f"{'95% CI':>18s}  {'params':>10s}  {'step_ms':>8s}  {'wall_s':>7s}")
    print("-" * 100)
    for label, s in [("baseline (V22 Bonsignore)", bl_s),
                     ("D (V/W_O + mean pool)", d_s),
                     ("cumulative_mean", cm_s)]:
        if s["n"] == 0:
            print(f"{label:>20s}  -- no runs --")
            continue
        print(f"{label:>20s}  {s['n']:>2d}  {s['mean']:>8.3f}  {s['std']:>6.3f}  "
              f"[{s['ci95'][0]:>7.3f},{s['ci95'][1]:>7.3f}]  "
              f"{s['total_params_mean']:>10,d}  "
              f"{s['step_time_ms_median_mean']:>7.1f}  "
              f"{s['wall_s_mean']:>7.0f}")

    # Pairwise comparisons
    def _fmt_t(t):
        if t is None:
            return "—"
        return (f"Δ={t['mean_diff']:+.3f}  "
                f"d={t['cohens_d']:+.2f}  "
                f"p={t['p']:.4f}  "
                f"95% CI Δ=[{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}]")

    print("\n=== Pairwise (Welch's t) ===")
    print(f"  cumulative_mean vs baseline:     {_fmt_t(cm_vs_bl)}")
    print(f"  D                vs baseline:    {_fmt_t(d_vs_bl)}")
    print(f"  cumulative_mean vs D:            {_fmt_t(cm_vs_d)}")

    out_json = cm_dir / "report.json"
    out_json.write_text(json.dumps({
        "baseline": bl_s, "D": d_s, "cumulative_mean": cm_s,
        "cm_vs_baseline": cm_vs_bl, "d_vs_baseline": d_vs_bl,
        "cm_vs_d": cm_vs_d,
    }, indent=2))
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
