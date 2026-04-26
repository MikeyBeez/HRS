"""Aggregate all variant×seed results into a comparison table.

Reports for each variant:
  n, mean val PPL, std, 95% CI, Welch's t-test vs baseline, Cohen's d,
  final per-head-MLP Frobenius norms (mean across layers, for
  hypothesis-mechanism diagnostics).

Writes a short markdown REPORT.md and a detailed report.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from experiments.head_aggregation.config import VARIANTS


def welch_t_ci(a: list[float], b: list[float]) -> dict:
    """Welch's t-test, Cohen's d, and 95% CI on mean difference.

    a = baseline, b = variant. Returns dict with t, df, p (2-sided approx),
    cohens_d, mean_diff, ci95.
    """
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan"),
                "cohens_d": float("nan"), "mean_diff": float("nan"),
                "ci95": [float("nan"), float("nan")]}

    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a) * na / (na - 1), statistics.pvariance(b) * nb / (nb - 1)
    se = math.sqrt(va / na + vb / nb)
    t = (mb - ma) / se if se > 0 else float("inf")
    # Welch-Satterthwaite df
    df_num = (va / na + vb / nb) ** 2
    df_den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = df_num / df_den if df_den > 0 else float("inf")

    # 2-sided p-value via Student's t-distribution survival function.
    # Avoid scipy — approximate with Wilson–Hilferty transform.
    def t_sf(tval, df):
        # 2-sided tail probability using Satterthwaite-style approximation.
        # For screening purposes this is adequate (exact p not critical).
        x = abs(tval)
        # Transform t to approximately normal via Wilson-Hilferty-like method.
        # For df > 5, t is approximately normal with slight correction.
        if not math.isfinite(x):
            return 0.0
        if df >= 30:
            # Treat as standard normal.
            return math.erfc(x / math.sqrt(2))
        # Rough t-to-normal approximation (Fisher's): z ≈ sqrt(df * log(1 + t^2/df))
        z_approx = math.sqrt(df * math.log(1 + x * x / df))
        # Sign-preserving
        return math.erfc(z_approx / math.sqrt(2))

    p = t_sf(t, df)

    # Cohen's d with pooled SD
    sd_pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (mb - ma) / sd_pooled if sd_pooled > 0 else float("inf")

    # 95% CI on mean diff (using approximate t quantile; 1.96 for large df)
    z = 1.96 if df >= 30 else 2.45  # crude for small df
    ci_lo = (mb - ma) - z * se
    ci_hi = (mb - ma) + z * se

    return {"t": t, "df": df, "p": p, "cohens_d": d,
            "mean_diff": mb - ma, "ci95": [ci_lo, ci_hi]}


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    """Load all {variant}_seed{seed}.json files, group by variant."""
    out: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for p in sorted(results_dir.glob("*_seed*.json")):
        if p.name == "sweep_summary.json":
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception as e:
            print(f"  warn: couldn't parse {p}: {e}")
            continue
        v = rec.get("variant")
        if v in out:
            out[v].append(rec)
    return out


def summary_line(runs: list[dict]) -> dict:
    ppls = [r["final_val_ppl"] for r in runs]
    n = len(ppls)
    if n == 0:
        return {"n": 0}
    mean = statistics.mean(ppls)
    std = statistics.stdev(ppls) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    # Extract mean per-head MLP norm at final step (for variants that have it)
    mlp_norms = []
    for r in runs:
        diag_list = r.get("final_diagnostics", []) or []
        for layer_d in diag_list:
            if "Wh_in_frobenius" in layer_d:
                mlp_norms.append(layer_d["Wh_in_frobenius"] + layer_d["Wh_out_frobenius"])
    return {
        "n": n,
        "mean": mean, "std": std, "se": se, "ci95": list(ci),
        "raw_ppls": ppls,
        "mlp_frob_sum_mean": statistics.mean(mlp_norms) if mlp_norms else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/head_aggregation/results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs_by_variant = load_runs(results_dir)

    print("\n=== Head-aggregation ablation: variant × seed summary ===\n")
    print(f"{'variant':>8s}  {'n':>2s}  {'mean':>8s}  {'std':>6s}  "
          f"{'95% CI':>18s}  {'vs baseline (Δ)':>18s}  {'Cohen d':>8s}  {'p':>7s}  "
          f"{'mlp Frob':>10s}")
    print("-" * 110)

    baseline_runs = runs_by_variant.get("baseline", [])
    baseline_ppls = [r["final_val_ppl"] for r in baseline_runs]
    base_stats = summary_line(baseline_runs) if baseline_runs else None

    report_rows = {}
    for v in VARIANTS:
        stats = summary_line(runs_by_variant[v])
        if stats["n"] == 0:
            print(f"{v:>8s}  -- no runs --")
            continue
        if v == "baseline" or not baseline_ppls:
            tstat = {"mean_diff": 0.0, "cohens_d": 0.0, "p": float("nan"),
                     "ci95": [0.0, 0.0]}
        else:
            variant_ppls = [r["final_val_ppl"] for r in runs_by_variant[v]]
            tstat = welch_t_ci(baseline_ppls, variant_ppls)
        mlp_str = f"{stats['mlp_frob_sum_mean']:.2f}" if stats.get("mlp_frob_sum_mean") else "  —"
        print(f"{v:>8s}  {stats['n']:>2d}  {stats['mean']:>8.3f}  {stats['std']:>6.3f}  "
              f"[{stats['ci95'][0]:>7.3f},{stats['ci95'][1]:>7.3f}]  "
              f"{tstat['mean_diff']:+8.3f}          {tstat['cohens_d']:+6.2f}  "
              f"{tstat['p']:>6.3f}  {mlp_str:>10s}")
        report_rows[v] = {"stats": stats, "ttest": tstat}

    # Write machine-readable + human-readable
    out_json = results_dir / "report.json"
    out_json.write_text(json.dumps({"rows": report_rows}, indent=2))
    print(f"\nSaved machine-readable summary to {out_json}")

    # Markdown report
    md = ["# Head-aggregation ablation — Shakespeare screening\n"]
    md.append(f"**Scaffold**: TinyBonsignoreTransformer at d=384, n_heads=8, "
              f"n_layers=6, d_ff=1536, ctx=256. Tiny Shakespeare, 2000 steps, "
              f"batch 32, AdamW lr=3e-4 cosine.\n")
    if base_stats:
        md.append(f"**Baseline**: n={base_stats['n']}, "
                  f"mean val PPL = {base_stats['mean']:.3f} ± {base_stats['std']:.3f}.\n")
    md.append("| variant | n | mean PPL | std | 95% CI | Δ vs baseline | Cohen's d | p | per-head MLP Frob (Wh_in+Wh_out) |")
    md.append("|:-------:|:-:|:--------:|:---:|:------:|:-------------:|:---------:|:-:|:--------------------------------:|")
    for v in VARIANTS:
        row = report_rows.get(v)
        if not row:
            continue
        s = row["stats"]
        t = row["ttest"]
        mlp = f"{s['mlp_frob_sum_mean']:.2f}" if s.get("mlp_frob_sum_mean") else "—"
        md.append(f"| {v} | {s['n']} | {s['mean']:.3f} | {s['std']:.3f} | "
                  f"[{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                  f"{t['mean_diff']:+.3f} | {t['cohens_d']:+.2f} | "
                  f"{t['p']:.3f} | {mlp} |")
    md.append("")
    md.append("**Per-head MLP Frobenius column**: sum of Wh_in and Wh_out Frobenius "
              "norms, averaged over all layers × seeds. At init this is ~2 × sqrt(H×dh×d_inter×0.02²). "
              "If the hypothesis mechanism is active, this should grow above init.")
    md.append("")
    (results_dir / "REPORT.md").write_text("\n".join(md))
    print(f"Saved markdown report to {results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
