"""Aggregate variant × seed results into a comparison table.

Outputs:
  results/report.json — machine-readable
  results/REPORT.md   — human-readable

Reuses the Welch's t-test + Cohen's d implementation from head_aggregation.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from experiments.saliency_pool.config import VARIANTS
from experiments.head_aggregation.analyze import welch_t_ci, summary_line


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
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


def avg_attn_entropy(runs: list[dict]) -> float | None:
    """Average final-step attn entropy across all layers × all seeds."""
    vals = []
    for r in runs:
        diags = r.get("final_diagnostics", []) or []
        for d in diags:
            v = d.get("last_attn_entropy_mean")
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    return statistics.mean(vals)


def avg_sal_mlp_frob(runs: list[dict]) -> float | None:
    vals = []
    for r in runs:
        diags = r.get("final_diagnostics", []) or []
        for d in diags:
            v = d.get("sal_mlp_frob")
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    return statistics.mean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="experiments/saliency_pool/results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    runs_by_variant = load_runs(results_dir)

    print("\n=== Saliency-pooling ablation: variant × seed summary ===\n")
    print(f"{'variant':>9s}  {'n':>2s}  {'mean':>8s}  {'std':>6s}  "
          f"{'95% CI':>18s}  {'long_ppl':>9s}  {'Δ vs base':>10s}  "
          f"{'Cohen d':>8s}  {'p':>7s}  {'attn_ent':>9s}  {'sal_frob':>9s}")
    print("-" * 130)

    baseline_runs = runs_by_variant.get("baseline", [])
    baseline_ppls = [r["final_val_ppl"] for r in baseline_runs]

    report_rows = {}
    for v in VARIANTS:
        runs = runs_by_variant[v]
        stats = summary_line(runs)
        if stats["n"] == 0:
            print(f"{v:>9s}  -- no runs --")
            continue
        long_ppls = [r["long_ctx_val_ppl"] for r in runs
                     if r.get("long_ctx_val_ppl") is not None]
        long_mean = statistics.mean(long_ppls) if long_ppls else float("nan")
        if v == "baseline" or not baseline_ppls:
            tstat = {"mean_diff": 0.0, "cohens_d": 0.0, "p": float("nan"),
                     "ci95": [0.0, 0.0]}
        else:
            variant_ppls = [r["final_val_ppl"] for r in runs]
            tstat = welch_t_ci(baseline_ppls, variant_ppls)
        ent = avg_attn_entropy(runs)
        frob = avg_sal_mlp_frob(runs)
        ent_str = f"{ent:.3f}" if ent is not None else "  —"
        frob_str = f"{frob:.2f}" if frob is not None else "  —"
        long_str = f"{long_mean:.3f}" if not math.isnan(long_mean) else "  —"
        print(f"{v:>9s}  {stats['n']:>2d}  {stats['mean']:>8.3f}  "
              f"{stats['std']:>6.3f}  "
              f"[{stats['ci95'][0]:>7.3f},{stats['ci95'][1]:>7.3f}]  "
              f"{long_str:>9s}  {tstat['mean_diff']:+8.3f}  "
              f"{tstat['cohens_d']:+6.2f}  {tstat['p']:>6.3f}  "
              f"{ent_str:>9s}  {frob_str:>9s}")
        report_rows[v] = {
            "stats": stats, "ttest": tstat, "long_ppl": long_mean,
            "attn_entropy": ent, "sal_mlp_frob": frob,
            "n_runs": stats["n"],
        }

    out_json = results_dir / "report.json"
    out_json.write_text(json.dumps({"rows": report_rows}, indent=2))
    print(f"\nSaved {out_json}")

    md = ["# Saliency-pooling ablation — Shakespeare screening\n"]
    md.append("**Scaffold**: TinyTransformer at d=384, n_heads=6, n_layers=6, "
              "d_ff=1536, ctx=256. Tiny Shakespeare, 2000 steps, batch 32, "
              "AdamW lr=3e-4 cosine.\n")
    md.append("| variant | aggregation | n | mean PPL | std | 95% CI | long-ctx PPL | Δ vs baseline | Cohen's d | p | attn entropy (last layer mean) | sal MLP Frob |")
    md.append("|:-------:|:-----------:|:-:|:--------:|:---:|:------:|:------------:|:-------------:|:---------:|:-:|:------------------------------:|:------------:|")
    descs = {
        "baseline": "V22 Bonsignore Q-K attention",
        "A": "shared global saliency (single MLP, single pattern)",
        "B": "per-pair saliency MLP (2d→d→1)",
        "C": "context-summary saliency (causal cumulative mean)",
        "D": "pure causal mean pool (no saliency)",
    }
    for v in VARIANTS:
        row = report_rows.get(v)
        if not row:
            continue
        s = row["stats"]; t = row["ttest"]
        ent = row.get("attn_entropy"); frob = row.get("sal_mlp_frob")
        long_str = (f"{row['long_ppl']:.3f}"
                    if not math.isnan(row['long_ppl']) else "—")
        ent_s = f"{ent:.3f}" if ent is not None else "—"
        frob_s = f"{frob:.2f}" if frob is not None else "—"
        md.append(f"| {v} | {descs[v]} | {s['n']} | {s['mean']:.3f} | "
                  f"{s['std']:.3f} | [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] | "
                  f"{long_str} | {t['mean_diff']:+.3f} | {t['cohens_d']:+.2f} | "
                  f"{t['p']:.3f} | {ent_s} | {frob_s} |")
    md.append("")
    md.append("**Diagnostics:** attn entropy is the mean entropy of the "
              "post-softmax attention distribution at each query position, "
              "averaged across (B, T_q, layers, seeds). Uniform over a "
              "T-position prefix gives entropy log(T) ~ 5.5 at T=256. Lower "
              "= more peaked attention. sal MLP Frob is the L2 norm of all "
              "saliency MLP parameters (where applicable), averaged across "
              "layers × seeds.")
    md.append("")
    (results_dir / "REPORT.md").write_text("\n".join(md))
    print(f"Saved {results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
