"""Run all four attention variants on both tasks, then emit a comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.diagonal_attention.config import TrainConfig, VARIANTS
from experiments.diagonal_attention.train import train_lm
from experiments.diagonal_attention.eval_passkey import train_passkey


def _fmt(v, width, spec=""):
    try:
        return format(v, spec).rjust(width)
    except (TypeError, ValueError):
        return str(v).rjust(width)


def render_table(lm: dict, pk: dict) -> str:
    lines = []
    header = (
        f"{'Variant':<10} {'Val PPL':>9} {'Passkey Exact':>14} "
        f"{'Passkey Digit':>14} {'Score Params':>13} {'Step (ms)':>10} "
        f"{'Peak Mem MB':>12}"
    )
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    for v in VARIANTS:
        l = lm.get(v, {})
        p = pk.get(v, {})
        lines.append(
            f"{v:<10} "
            f"{_fmt(l.get('final_val_ppl'), 9, '.2f')} "
            f"{_fmt(p.get('final', {}).get('overall_exact_acc'), 14, '.3f')} "
            f"{_fmt(p.get('final', {}).get('overall_digit_acc'), 14, '.3f')} "
            f"{_fmt(l.get('score_params'), 13, ',d')} "
            f"{_fmt(l.get('step_time_ms_median'), 10, '.2f')} "
            f"{_fmt(l.get('peak_mem_mb'), 12, '.0f')}"
        )
    return "\n".join(lines)


def plot_passkey(pk: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in VARIANTS:
        rec = pk.get(v)
        if not rec:
            continue
        buckets = rec["final"]["by_bucket"]
        xs = [b["position_frac"] for b in buckets]
        ys = [b["exact_acc"] for b in buckets]
        ax.plot(xs, ys, marker="o", label=v)
    ax.set_xlabel("passkey position (fraction of max)")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Passkey retrieval vs. marker position")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000,
                    help="LM steps (also default for passkey if --passkey-steps not set)")
    ap.add_argument("--passkey-steps", type=int, default=None)
    ap.add_argument("--out", default="experiments/diagonal_attention/results")
    ap.add_argument("--variants", nargs="+", default=VARIANTS,
                    choices=VARIANTS)
    ap.add_argument("--skip-lm", action="store_true")
    ap.add_argument("--skip-passkey", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_tcfg = TrainConfig(steps=args.steps)
    pk_steps = args.passkey_steps if args.passkey_steps is not None else args.steps
    pk_tcfg = TrainConfig(steps=pk_steps)

    lm_records: dict = {}
    pk_records: dict = {}

    for v in args.variants:
        print(f"\n=== LM / {v} ===")
        if not args.skip_lm:
            lm_records[v] = train_lm(v, out_dir, lm_tcfg)
            torch.cuda.empty_cache()

    for v in args.variants:
        print(f"\n=== Passkey / {v} ===")
        if not args.skip_passkey:
            pk_records[v] = train_passkey(v, out_dir, pk_tcfg)
            torch.cuda.empty_cache()

    summary = {
        "lm": lm_records,
        "passkey": pk_records,
        "lm_steps": args.steps,
        "passkey_steps": pk_steps,
        "variants": args.variants,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    table = render_table(lm_records, pk_records)
    (out_dir / "comparison.txt").write_text(table + "\n")
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(table)

    plot_passkey(pk_records, out_dir / "passkey_accuracy.png")
    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
