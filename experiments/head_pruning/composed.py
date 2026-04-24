"""Phase 3: compose head pruning + MLP magnitude pruning.

Takes the best Phase-2 head config (largest n_pruned that still gives
passkey_exact == 1.0 after FT, by least-passkey-first order), then on top of
that applies 90% MLP magnitude pruning + FT.

Reports: combined effective param count, PPL, passkey accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.pruning.eval import eval_lm, eval_passkey
from experiments.pruning.finetune import finetune_lm, finetune_passkey
from experiments.pruning.prune import (
    apply_masks,
    init_prune_state,
    prune_to_sparsity,
    sparsity_report,
)
from experiments.pruning.run_sweep import Baselines, _fresh_lm, _fresh_pk

from experiments.head_pruning.prune_heads import (
    build_head_prune_state,
    compose_with_mlp_mask_state,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def pick_best_head_set() -> list:
    """Find the largest head-count that keeps passkey_exact==1.0 after FT under
    the least-passkey-first ordering. Falls back to the highest-count config
    if none are exactly 1.0."""
    recs = json.loads((RESULTS_DIR / "sweep_results.json").read_text())
    candidates = [r for r in recs
                   if r["order"] == "least_passkey_first"
                   and r["do_ft"]
                   and r["n_pruned"] > 0]
    candidates.sort(key=lambda r: -r["n_pruned"])
    for r in candidates:
        if r["passkey_exact"] >= 1.0 - 1e-6:
            return [tuple(h) for h in r["heads"]]
    # Fallback: take the highest-count candidate regardless.
    return [tuple(h) for h in candidates[0]["heads"]]


def run_composed(pruned_heads: list, mlp_sparsity: float,
                  ft_steps: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = Baselines.load(device)

    t0 = time.time()

    # ---------- LM side ----------
    lm_model, lm_mcfg = _fresh_lm(bl, device)
    lm_head_st = build_head_prune_state(lm_model, pruned_heads)
    lm_mlp_st = init_prune_state(lm_model, "mlp")
    prune_to_sparsity(lm_mlp_st, mlp_sparsity)
    apply_masks(lm_mlp_st)
    lm_combined = compose_with_mlp_mask_state(lm_head_st, lm_mlp_st)
    finetune_lm(lm_model, lm_mcfg, lm_combined, ft_steps, lr=1e-4)
    ppl = eval_lm(lm_model, lm_mcfg)
    lm_report = sparsity_report(lm_combined, lm_model)
    lm_combined.release()
    del lm_model

    # ---------- Passkey side ----------
    pk_model, pk_mcfg, pkcfg = _fresh_pk(bl, device)
    pk_head_st = build_head_prune_state(pk_model, pruned_heads)
    pk_mlp_st = init_prune_state(pk_model, "mlp")
    prune_to_sparsity(pk_mlp_st, mlp_sparsity)
    apply_masks(pk_mlp_st)
    pk_combined = compose_with_mlp_mask_state(pk_head_st, pk_mlp_st)
    finetune_passkey(pk_model, pk_mcfg, pkcfg, pk_combined, ft_steps, lr=1e-4)
    metrics = eval_passkey(pk_model, pkcfg)
    pk_report = sparsity_report(pk_combined, pk_model)
    pk_combined.release()
    del pk_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Attention FLOPs savings are proportional to heads remaining.
    total_heads = 16
    heads_remaining = total_heads - len(pruned_heads)
    attn_flops_frac = heads_remaining / total_heads

    # MLP param compression on the pruned scope.
    mlp_nonzero_frac = 1.0 - pk_report["scope_sparsity"]

    rec = {
        "pruned_heads": [list(h) for h in pruned_heads],
        "n_heads_pruned": len(pruned_heads),
        "heads_remaining": heads_remaining,
        "attn_flops_fraction_of_baseline": attn_flops_frac,
        "mlp_target_sparsity": mlp_sparsity,
        "mlp_actual_sparsity_lm": lm_report["scope_sparsity"],
        "mlp_actual_sparsity_pk": pk_report["scope_sparsity"],
        "mlp_nonzero_fraction": mlp_nonzero_frac,
        "val_ppl": ppl,
        "passkey_exact": metrics["overall_exact_acc"],
        "passkey_digit": metrics["overall_digit_acc"],
        "effective_params_lm": lm_report["effective_total_params"],
        "effective_params_pk": pk_report["effective_total_params"],
        "total_params": lm_report["all_params"],
        "ft_steps": ft_steps,
        "wall_seconds": time.time() - t0,
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-sparsity", type=float, default=0.9)
    ap.add_argument("--ft-steps", type=int, default=500)
    args = ap.parse_args()

    pruned_heads = pick_best_head_set()
    print(f"best head set: {pruned_heads}  ({len(pruned_heads)} of 16 pruned)")
    rec = run_composed(pruned_heads, args.mlp_sparsity, args.ft_steps)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "composed_result.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
