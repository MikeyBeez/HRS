"""Aggregate engram_dropout sweep results into the deliverable table.

Per spec:
  - val PPL with engram on
  - val PPL with engram off (ablation gap)
  - per-head kernel specialization (variance across heads of taus / alphas /
    scales). Reported separately for the engram-injection layer and the
    average across all layers.
  - training stability flag
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_runs(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("p*_seed*.json"))]


def head_var(diags: list[dict], layer_idx: int | None = None) -> dict[str, float]:
    """For each parameter (taus, alphas, scales), compute variance across heads.
    If layer_idx is None, average over all layers; else just that layer."""
    if layer_idx is not None:
        layers = [layer_idx]
    else:
        layers = list(range(len(diags)))
    out = {}
    for key in ["taus", "alphas", "scales"]:
        per_layer = []
        for li in layers:
            vals = diags[li][key]
            if len(vals) >= 2:
                per_layer.append(statistics.variance(vals))
        out[key] = statistics.mean(per_layer) if per_layer else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/engram_dropout/results")
    args = ap.parse_args()
    rd = Path(args.results_dir)
    runs = load_runs(rd)
    if not runs:
        print(f"No runs found in {rd}")
        return

    # Group by p
    by_p: dict[float, list[dict]] = {}
    for r in runs:
        by_p.setdefault(r["engram_dropout_p"], []).append(r)

    rows = []
    for p in sorted(by_p):
        rs = by_p[p]
        ppl_on = [r["final_val_ppl_on"] for r in rs if not r["diverged"]]
        ppl_off = [r["final_val_ppl_off"] for r in rs if not r["diverged"]]
        gaps = [r["ablation_gap"] for r in rs if not r["diverged"]]
        n_div = sum(1 for r in rs if r["diverged"])
        engram_layer = rs[0].get("history", [{}])[0].get("step")  # placeholder
        injection_layer = 2  # cfg.engram_layer

        # Per-head kernel specialization at the engram-injection layer
        spec_inj = []
        spec_all = []
        for r in rs:
            if r["diverged"]:
                continue
            diags = r["final_diagnostics"]["layers"]
            spec_inj.append(head_var(diags, injection_layer))
            spec_all.append(head_var(diags, None))

        def agg(metric, by="injection"):
            data = spec_inj if by == "injection" else spec_all
            vals = [d[metric] for d in data]
            if not vals:
                return None
            return {"mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}

        row = {
            "p": p,
            "n_runs": len(rs),
            "n_diverged": n_div,
            "ppl_on_mean": statistics.mean(ppl_on) if ppl_on else float("nan"),
            "ppl_on_std": statistics.stdev(ppl_on) if len(ppl_on) > 1 else 0.0,
            "ppl_off_mean": statistics.mean(ppl_off) if ppl_off else float("nan"),
            "ppl_off_std": statistics.stdev(ppl_off) if len(ppl_off) > 1 else 0.0,
            "gap_mean": statistics.mean(gaps) if gaps else float("nan"),
            "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
            "head_var_taus_inj": agg("taus", "injection"),
            "head_var_alphas_inj": agg("alphas", "injection"),
            "head_var_scales_inj": agg("scales", "injection"),
            "head_var_taus_avg": agg("taus", "all"),
            "head_var_alphas_avg": agg("alphas", "all"),
            "head_var_scales_avg": agg("scales", "all"),
            "stability": "diverged" if n_div else "converged",
        }
        rows.append(row)

    # Print table
    print("\n=== Engram dropout sweep — summary ===\n")
    header = (f"{'p':>5}  {'n':>2}  {'ppl_on':>14}  {'ppl_off':>14}  "
              f"{'gap':>10}  {'τ var (inj)':>12}  {'α var (inj)':>12}  "
              f"{'scale var (inj)':>15}  stability")
    print(header)
    print("-" * len(header))
    for r in rows:
        ppl_on_s = f"{r['ppl_on_mean']:.3f}±{r['ppl_on_std']:.3f}"
        ppl_off_s = f"{r['ppl_off_mean']:.3f}±{r['ppl_off_std']:.3f}"
        gap_s = f"{r['gap_mean']:+.3f}"
        tau_v = r["head_var_taus_inj"]["mean"] if r["head_var_taus_inj"] else 0
        a_v = r["head_var_alphas_inj"]["mean"] if r["head_var_alphas_inj"] else 0
        s_v = r["head_var_scales_inj"]["mean"] if r["head_var_scales_inj"] else 0
        print(f"{r['p']:>5.2f}  {r['n_runs']:>2}  {ppl_on_s:>14}  {ppl_off_s:>14}  "
              f"{gap_s:>10}  {tau_v:>12.3e}  {a_v:>12.3e}  {s_v:>15.3e}  "
              f"{r['stability']}")

    out_json = Path(args.results_dir) / "report.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
