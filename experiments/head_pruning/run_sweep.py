"""Phase 2: prune heads in {ascending, descending, random} order of passkey
importance. At each count sweep point, measure (no-FT, 500-step FT) on both
metrics.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from experiments.diagonal_attention.config import ModelConfig, PasskeyConfig
from experiments.pruning.eval import eval_lm, eval_passkey
from experiments.pruning.finetune import finetune_lm, finetune_passkey
from experiments.pruning.run_sweep import Baselines, _fresh_lm, _fresh_pk

from experiments.head_pruning.prune_heads import build_head_prune_state


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

FT_STEPS = 500
FT_LR = 1e-4
SWEEP_COUNTS = (0, 2, 4, 6, 8, 10, 12, 14)


def load_importance():
    p = RESULTS_DIR / "head_importance.json"
    return json.loads(p.read_text())


def _orders(importance: dict, seed: int = 0):
    asc = [tuple(x) for x in importance["order_least_passkey_first"]]
    desc = [tuple(x) for x in importance["order_most_passkey_first"]]
    rng = random.Random(seed)
    rnd = asc.copy()
    rng.shuffle(rnd)
    return {
        "least_passkey_first": asc,
        "random": rnd,
        "most_passkey_first": desc,
    }


def run_point(bl: Baselines, pruned: list, device: torch.device,
               do_ft: bool) -> dict:
    """Evaluate (LM PPL, passkey) after pruning `pruned` heads, optionally FT."""
    t0 = time.time()

    # LM side.
    lm_model, lm_mcfg = _fresh_lm(bl, device)
    lm_st = build_head_prune_state(lm_model, pruned)
    if do_ft:
        finetune_lm(lm_model, lm_mcfg, lm_st, FT_STEPS, lr=FT_LR)
    ppl = eval_lm(lm_model, lm_mcfg)
    lm_st.release()
    del lm_model

    # Passkey side.
    pk_model, pk_mcfg, pkcfg = _fresh_pk(bl, device)
    pk_st = build_head_prune_state(pk_model, pruned)
    if do_ft:
        finetune_passkey(pk_model, pk_mcfg, pkcfg, pk_st, FT_STEPS, lr=FT_LR)
    metrics = eval_passkey(pk_model, pkcfg)
    pk_st.release()
    del pk_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "n_pruned": len(pruned),
        "heads": [list(h) for h in pruned],
        "do_ft": do_ft,
        "val_ppl": ppl,
        "passkey_exact": metrics["overall_exact_acc"],
        "passkey_digit": metrics["overall_digit_acc"],
        "wall_seconds": time.time() - t0,
    }


def run_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = Baselines.load(device)
    imp = load_importance()
    orders = _orders(imp)

    records = []
    total = len(orders) * len(SWEEP_COUNTS) * 2
    i = 0
    for order_name, ordered_heads in orders.items():
        for n in SWEEP_COUNTS:
            pruned = ordered_heads[:n]
            for do_ft in (False, True):
                i += 1
                print(f"[{i}/{total}] order={order_name} n_pruned={n} "
                      f"ft={'yes' if do_ft else 'no'}")
                rec = run_point(bl, pruned, device, do_ft)
                rec["order"] = order_name
                print(f"    -> val_ppl={rec['val_ppl']:.2f} "
                      f"passkey={rec['passkey_exact']:.3f} "
                      f"wall={rec['wall_seconds']:.1f}s")
                records.append(rec)
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                (RESULTS_DIR / "sweep_results.json").write_text(
                    json.dumps(records, indent=2))

    return records


def plot_sweep(records: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_pk_noft, ax_pk_ft), (ax_ppl_noft, ax_ppl_ft) = axes

    for do_ft, col in [(False, 0), (True, 1)]:
        for order in ("least_passkey_first", "random", "most_passkey_first"):
            rs = sorted(
                [r for r in records if r["order"] == order and r["do_ft"] == do_ft],
                key=lambda r: r["n_pruned"],
            )
            xs = [r["n_pruned"] for r in rs]
            pk = [r["passkey_exact"] for r in rs]
            ppl = [r["val_ppl"] for r in rs]
            axes[0][col].plot(xs, pk, marker="o", label=order)
            axes[1][col].plot(xs, ppl, marker="o", label=order)

    ax_pk_noft.set_title("Passkey exact — no FT")
    ax_pk_ft.set_title("Passkey exact — 500 step FT")
    ax_ppl_noft.set_title("Val PPL — no FT")
    ax_ppl_ft.set_title("Val PPL — 500 step FT")

    for row in axes[:1]:
        for ax in row:
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("passkey exact")
    for row in axes[1:]:
        for ax in row:
            ax.set_ylabel("val PPL (lower=better)")
    for row in axes:
        for ax in row:
            ax.set_xlabel("heads pruned (of 16)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.tight_layout()
    p = RESULTS_DIR / "sweep_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true",
                    help="Skip the sweep and just re-plot from existing results.")
    args = ap.parse_args()
    if args.skip_run:
        records = json.loads((RESULTS_DIR / "sweep_results.json").read_text())
    else:
        records = run_sweep()
    plot_sweep(records)


if __name__ == "__main__":
    main()
