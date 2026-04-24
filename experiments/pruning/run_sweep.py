"""Run a (sparsity × strategy × scope) sweep and summarize.

Each run:
  1. Load both baselines (LM, passkey) from checkpoints.
  2. Build PruneState on each and apply the strategy (one-shot or iterative)
     to reach target sparsity.
  3. Optionally fine-tune each model on its own task.
  4. Evaluate val PPL (LM model) and passkey accuracy (passkey model).
  5. Record sparsity report (from passkey model since both are the same shape).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.diagonal_attention.config import ModelConfig, PasskeyConfig
from experiments.diagonal_attention.model import TinyTransformer

from experiments.pruning.eval import eval_lm, eval_passkey
from experiments.pruning.finetune import finetune_lm, finetune_passkey
from experiments.pruning.prune import (
    SCOPES,
    apply_masks,
    init_prune_state,
    prune_to_sparsity,
    sparsity_report,
)


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


STRATEGIES = ("oneshot", "ft_short", "ft_long", "iterative")

DEFAULT_SPARSITIES = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)

# Strategy parameters.
FT_SHORT_STEPS = 500
FT_LONG_STEPS = 2000
ITER_STEP_FRAC = 0.20   # advance sparsity by 20% per iteration
ITER_FT_STEPS = 1000

LR_FT = 1e-4


@dataclass
class Baselines:
    lm_ckpt: dict
    pk_ckpt: dict

    @classmethod
    def load(cls, device: torch.device) -> "Baselines":
        lm = torch.load(CKPT_DIR / "mha_lm.pt", map_location=device, weights_only=False)
        pk = torch.load(CKPT_DIR / "mha_passkey.pt", map_location=device, weights_only=False)
        return cls(lm_ckpt=lm, pk_ckpt=pk)


def _fresh_lm(bl: Baselines, device):
    mcfg = ModelConfig(**bl.lm_ckpt["mcfg"])
    m = TinyTransformer(mcfg).to(device)
    m.load_state_dict(bl.lm_ckpt["state_dict"])
    return m, mcfg


def _fresh_pk(bl: Baselines, device):
    mcfg = ModelConfig(**bl.pk_ckpt["mcfg"])
    m = TinyTransformer(mcfg).to(device)
    m.load_state_dict(bl.pk_ckpt["state_dict"])
    pkcfg = PasskeyConfig(**bl.pk_ckpt["pkcfg"])
    return m, mcfg, pkcfg


def _apply_strategy(model, mcfg, st, strategy: str, target_sparsity: float,
                      *, task: str, pkcfg=None):
    """Run the pruning strategy, returning total fine-tune steps used."""
    if strategy == "oneshot":
        prune_to_sparsity(st, target_sparsity)
        return 0
    if strategy == "ft_short":
        prune_to_sparsity(st, target_sparsity)
        ft_steps = FT_SHORT_STEPS
    elif strategy == "ft_long":
        prune_to_sparsity(st, target_sparsity)
        ft_steps = FT_LONG_STEPS
    elif strategy == "iterative":
        # Advance sparsity by ITER_STEP_FRAC each round until reaching target.
        current = 0.0
        ft_steps_total = 0
        while current + 1e-9 < target_sparsity:
            current = min(current + ITER_STEP_FRAC, target_sparsity)
            prune_to_sparsity(st, current)
            if task == "lm":
                finetune_lm(model, mcfg, st, ITER_FT_STEPS, lr=LR_FT)
            else:
                finetune_passkey(model, mcfg, pkcfg, st, ITER_FT_STEPS, lr=LR_FT)
            ft_steps_total += ITER_FT_STEPS
        return ft_steps_total
    else:
        raise ValueError(f"unknown strategy {strategy}")

    if task == "lm":
        finetune_lm(model, mcfg, st, ft_steps, lr=LR_FT)
    else:
        finetune_passkey(model, mcfg, pkcfg, st, ft_steps, lr=LR_FT)
    return ft_steps


def run_one(bl: Baselines, scope: str, target_sparsity: float, strategy: str,
             device: torch.device) -> dict:
    t0 = time.time()

    # LM side.
    lm_model, lm_mcfg = _fresh_lm(bl, device)
    lm_st = init_prune_state(lm_model, scope)
    apply_masks(lm_st)
    lm_ft_steps = _apply_strategy(
        lm_model, lm_mcfg, lm_st, strategy, target_sparsity, task="lm"
    )
    val_ppl = eval_lm(lm_model, lm_mcfg)
    lm_rep = sparsity_report(lm_st, lm_model)
    lm_st.release()
    del lm_model

    # Passkey side.
    pk_model, pk_mcfg, pkcfg = _fresh_pk(bl, device)
    pk_st = init_prune_state(pk_model, scope)
    apply_masks(pk_st)
    pk_ft_steps = _apply_strategy(
        pk_model, pk_mcfg, pk_st, strategy, target_sparsity, task="passkey",
        pkcfg=pkcfg,
    )
    pk_metrics = eval_passkey(pk_model, pkcfg)
    pk_rep = sparsity_report(pk_st, pk_model)
    pk_st.release()
    del pk_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    rec = {
        "scope": scope,
        "target_sparsity": target_sparsity,
        "strategy": strategy,
        "val_ppl": val_ppl,
        "passkey_exact": pk_metrics["overall_exact_acc"],
        "passkey_digit": pk_metrics["overall_digit_acc"],
        "passkey_by_bucket": pk_metrics["by_bucket"],
        "scope_sparsity_lm": lm_rep["scope_sparsity"],
        "scope_sparsity_pk": pk_rep["scope_sparsity"],
        "effective_params_lm": lm_rep["effective_total_params"],
        "effective_params_pk": pk_rep["effective_total_params"],
        "scope_total_params": pk_rep["scope_total_params"],
        "lm_ft_steps": lm_ft_steps,
        "pk_ft_steps": pk_ft_steps,
        "wall_seconds": time.time() - t0,
    }
    return rec


def render_table(records: list[dict]) -> str:
    lines = []
    header = (f"{'scope':<6} {'sparsity':>9} {'strategy':<10} "
              f"{'val_ppl':>9} {'pk_exact':>9} {'pk_digit':>9} "
              f"{'actual_sp':>10} {'eff_params':>11} {'wall_s':>7}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in records:
        lines.append(
            f"{r['scope']:<6} {r['target_sparsity']:>9.2f} {r['strategy']:<10} "
            f"{r['val_ppl']:>9.2f} {r['passkey_exact']:>9.3f} {r['passkey_digit']:>9.3f} "
            f"{r['scope_sparsity_pk']:>10.3f} {r['effective_params_pk']:>11,d} "
            f"{r['wall_seconds']:>7.1f}"
        )
    return "\n".join(lines)


def plot_curves(records: list[dict], out_dir: Path, tag: str = ""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    # Degradation curves: sparsity vs val_ppl per strategy.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1, ax2 = axes
    strategies = sorted({r["strategy"] for r in records},
                         key=lambda s: ("oneshot", "ft_short", "ft_long",
                                         "iterative").index(s))
    for strat in strategies:
        rs = sorted([r for r in records if r["strategy"] == strat],
                     key=lambda r: r["target_sparsity"])
        xs = [r["target_sparsity"] for r in rs]
        ppl = [r["val_ppl"] for r in rs]
        exact = [r["passkey_exact"] for r in rs]
        ax1.plot(xs, ppl, marker="o", label=strat)
        ax2.plot(xs, exact, marker="o", label=strat)
    ax1.set_xlabel("target sparsity")
    ax1.set_ylabel("val PPL (lower = better)")
    ax1.set_title("LM: val PPL vs sparsity")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("target sparsity")
    ax2.set_ylabel("passkey exact-match acc")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("Passkey: exact-match vs sparsity")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    prefix = f"{tag}_" if tag else ""
    p = out_dir / f"{prefix}degradation_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")

    # Scatter: passkey exact vs val PPL.
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for strat in strategies:
        rs = [r for r in records if r["strategy"] == strat]
        xs = [r["val_ppl"] for r in rs]
        ys = [r["passkey_exact"] for r in rs]
        sizes = [30 + 120 * r["target_sparsity"] for r in rs]
        ax.scatter(xs, ys, s=sizes, alpha=0.8, label=strat)
    ax.set_xlabel("val PPL")
    ax.set_ylabel("passkey exact-match acc")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Passkey vs PPL (marker size = sparsity)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"{prefix}passkey_vs_ppl.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scopes", nargs="+", default=["mlp"], choices=list(SCOPES))
    ap.add_argument("--sparsities", nargs="+", type=float,
                    default=list(DEFAULT_SPARSITIES))
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                    choices=list(STRATEGIES))
    ap.add_argument("--out", default=str(RESULTS_DIR / "results.json"))
    ap.add_argument("--tag", default="",
                    help="Suffix added to output filenames (e.g., 'short').")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = Baselines.load(device)
    print(f"baseline LM val_ppl = {bl.lm_ckpt.get('final_val_ppl', 'n/a')}")
    pk_sum = bl.pk_ckpt.get("final_passkey", {})
    print(f"baseline passkey exact = {pk_sum.get('overall_exact_acc', 'n/a')} "
          f"digit = {pk_sum.get('overall_digit_acc', 'n/a')}")

    combos = [(scope, sp, strat)
               for scope in args.scopes
               for sp in args.sparsities
               for strat in args.strategies]
    print(f"{len(combos)} runs queued")

    records = []
    for i, (scope, sp, strat) in enumerate(combos):
        # oneshot at 0% sparsity is equivalent across strategies; skip the
        # redundant 0-sparsity rows for non-oneshot to save time.
        if sp == 0.0 and strat != "oneshot":
            continue
        print(f"\n[{i+1}/{len(combos)}] scope={scope} sparsity={sp:.2f} strategy={strat}")
        rec = run_one(bl, scope, sp, strat, device)
        print(f"  -> val_ppl={rec['val_ppl']:.2f} "
              f"passkey_exact={rec['passkey_exact']:.3f} "
              f"actual_sp={rec['scope_sparsity_pk']:.3f} "
              f"wall={rec['wall_seconds']:.1f}s")
        records.append(rec)
        # Save incrementally so a crash doesn't lose prior runs.
        tag = args.tag + "_" if args.tag else ""
        out_path = RESULTS_DIR / f"{tag}results.json"
        out_path.write_text(json.dumps(records, indent=2))

    table = render_table(records)
    tag = args.tag + "_" if args.tag else ""
    (RESULTS_DIR / f"{tag}comparison.txt").write_text(table + "\n")
    print("\n" + table)
    plot_curves(records, RESULTS_DIR, tag=args.tag)


if __name__ == "__main__":
    main()
