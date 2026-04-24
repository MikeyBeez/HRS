"""Verify that a given layer's heads are genuinely prunable across seeds.

For each seed, zero all heads in the target layer and measure:
  - val PPL (LM baseline) and passkey exact (passkey baseline), no FT
  - both after 500-step fine-tuning (same recipe as Phase 2)

This is the "jointly zero a whole layer" test, distinct from Phase 1 which
measured single-head ablation.

Usage:
    PYTHONPATH=. .venv/bin/python -m \\
        experiments.head_pruning.seeds.verify_layer_prunable --layer 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import experiments.pruning.train_baseline as _train_base
import experiments.pruning.run_sweep as _run_sweep

from experiments.pruning.eval import eval_lm, eval_passkey
from experiments.pruning.finetune import finetune_lm, finetune_passkey
from experiments.pruning.run_sweep import Baselines, _fresh_lm, _fresh_pk

from experiments.head_pruning.prune_heads import build_head_prune_state


ROOT = Path(__file__).resolve().parent
ORIGINAL_CKPT_DIR = Path("/mnt/data/Code/HRS/experiments/pruning/checkpoints")
SEED_ROOT = ROOT  # experiments/head_pruning/seeds


def _set_ckpt_dir(seed: int):
    """Point Baselines.load at this seed's checkpoints."""
    if seed == 0:
        ckpt_dir = ORIGINAL_CKPT_DIR
    else:
        ckpt_dir = SEED_ROOT / f"seed_{seed}" / "checkpoints"
    _train_base.CKPT_DIR = ckpt_dir
    _run_sweep.CKPT_DIR = ckpt_dir


def _restore():
    _train_base.CKPT_DIR = ORIGINAL_CKPT_DIR
    _run_sweep.CKPT_DIR = ORIGINAL_CKPT_DIR


def test_seed(seed: int, layer: int, ft_steps: int = 500) -> dict:
    """Zero every head in `layer` on both baselines; measure PPL + passkey."""
    _set_ckpt_dir(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = Baselines.load(device)
    n_heads = bl.lm_ckpt["mcfg"]["n_heads"]
    heads_to_zero = [(layer, h) for h in range(n_heads)]

    # LM side
    lm_model, lm_mcfg = _fresh_lm(bl, device)
    lm_st = build_head_prune_state(lm_model, heads_to_zero)
    ppl_noft = eval_lm(lm_model, lm_mcfg)
    if ft_steps > 0:
        finetune_lm(lm_model, lm_mcfg, lm_st, ft_steps, lr=1e-4)
    ppl_ft = eval_lm(lm_model, lm_mcfg)
    lm_st.release()
    del lm_model

    # Passkey side
    pk_model, pk_mcfg, pkcfg = _fresh_pk(bl, device)
    pk_st = build_head_prune_state(pk_model, heads_to_zero)
    pk_noft = eval_passkey(pk_model, pkcfg)
    if ft_steps > 0:
        finetune_passkey(pk_model, pk_mcfg, pkcfg, pk_st, ft_steps, lr=1e-4)
    pk_ft = eval_passkey(pk_model, pkcfg)
    pk_st.release()
    del pk_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Baseline for reference
    bl_ppl = bl.lm_ckpt.get("final_val_ppl")
    bl_pk = bl.pk_ckpt["final_passkey"]["overall_exact_acc"]

    return {
        "seed": seed,
        "layer_zeroed": layer,
        "heads_zeroed": [list(h) for h in heads_to_zero],
        "baseline_val_ppl": bl_ppl,
        "baseline_passkey_exact": bl_pk,
        "val_ppl_no_ft": ppl_noft,
        "val_ppl_ft": ppl_ft,
        "passkey_exact_no_ft": pk_noft["overall_exact_acc"],
        "passkey_digit_no_ft": pk_noft["overall_digit_acc"],
        "passkey_exact_ft": pk_ft["overall_exact_acc"],
        "passkey_digit_ft": pk_ft["overall_digit_acc"],
        "delta_ppl_no_ft": ppl_noft - bl_ppl,
        "delta_ppl_ft": ppl_ft - bl_ppl,
        "delta_passkey_no_ft": bl_pk - pk_noft["overall_exact_acc"],
        "delta_passkey_ft": bl_pk - pk_ft["overall_exact_acc"],
    }


def render_table(records: list[dict], ft_steps: int = 500) -> str:
    lines = []
    layer = records[0]["layer_zeroed"]
    lines.append(f"Zero all heads of layer {layer}; {ft_steps}-step FT.\n")
    header = (f"{'seed':>4} {'bl_ppl':>7} {'ppl_noft':>9} {'ppl_ft':>8} "
               f"{'bl_pk':>6} {'pk_noft':>8} {'pk_ft':>7} "
               f"{'Δppl_ft':>9} {'Δpk_ft':>9}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in records:
        lines.append(
            f"{r['seed']:>4} "
            f"{r['baseline_val_ppl']:>7.2f} "
            f"{r['val_ppl_no_ft']:>9.2f} "
            f"{r['val_ppl_ft']:>8.2f} "
            f"{r['baseline_passkey_exact']:>6.2f} "
            f"{r['passkey_exact_no_ft']:>8.3f} "
            f"{r['passkey_exact_ft']:>7.3f} "
            f"{r['delta_ppl_ft']:>+9.2f} "
            f"{r['delta_passkey_ft']:>+9.3f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=3,
                    help="Layer index whose heads will all be zeroed.")
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--ft-steps", type=int, default=500)
    ap.add_argument("--out", default=None,
                    help="JSON output path; default: seeds/layer_{N}_prunable.json")
    args = ap.parse_args()

    tag = f"_ft{args.ft_steps}" if args.ft_steps != 500 else ""
    out_path = Path(args.out) if args.out else (
        ROOT / f"layer_{args.layer}_prunable{tag}.json")

    records = []
    for seed in args.seeds:
        ckpt_path = (ORIGINAL_CKPT_DIR if seed == 0
                       else SEED_ROOT / f"seed_{seed}" / "checkpoints")
        if not (ckpt_path / "mha_passkey.pt").exists():
            print(f"  seed {seed}: missing checkpoint, skipping")
            continue
        print(f"=== seed {seed}: zero layer {args.layer} ===")
        t0 = time.time()
        rec = test_seed(seed, args.layer, args.ft_steps)
        rec["wall_seconds"] = time.time() - t0
        records.append(rec)
        print(f"  ppl: {rec['baseline_val_ppl']:.2f} -> "
              f"{rec['val_ppl_no_ft']:.2f} (noft) -> "
              f"{rec['val_ppl_ft']:.2f} (ft)")
        print(f"  passkey: {rec['baseline_passkey_exact']:.2f} -> "
              f"{rec['passkey_exact_no_ft']:.3f} (noft) -> "
              f"{rec['passkey_exact_ft']:.3f} (ft) "
              f"[wall={rec['wall_seconds']:.1f}s]")
        # Save incrementally.
        out_path.write_text(json.dumps(records, indent=2))

    _restore()

    table = render_table(records, ft_steps=args.ft_steps)
    (ROOT / f"layer_{args.layer}_prunable{tag}.txt").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
