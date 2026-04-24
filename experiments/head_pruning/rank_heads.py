"""Phase 1: Ablate each of 16 heads individually; record importance.

For each (layer, head):
  - Load LM baseline, zero that head, measure val PPL. importance_ppl = ppl - baseline_ppl.
  - Load passkey baseline, zero that head, measure passkey exact. importance_passkey = baseline_exact - pruned_exact.

Outputs a JSON record and a side-by-side heatmap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pruning.eval import eval_lm, eval_passkey
from experiments.pruning.run_sweep import Baselines, _fresh_lm, _fresh_pk

from experiments.head_pruning.prune_heads import build_head_prune_state


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def rank_all_heads() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = Baselines.load(device)
    baseline_ppl = bl.lm_ckpt.get("final_val_ppl")
    baseline_exact = bl.pk_ckpt["final_passkey"]["overall_exact_acc"]
    baseline_digit = bl.pk_ckpt["final_passkey"]["overall_digit_acc"]

    print(f"baseline LM val_ppl = {baseline_ppl:.3f}")
    print(f"baseline passkey exact = {baseline_exact:.3f} digit = {baseline_digit:.3f}")

    # Recompute baselines at eval settings to be comparable to the ablated
    # runs (same eval_lm / eval_passkey functions).
    lm_model, lm_mcfg = _fresh_lm(bl, device)
    ref_ppl = eval_lm(lm_model, lm_mcfg)
    del lm_model
    pk_model, pk_mcfg, pkcfg = _fresh_pk(bl, device)
    ref_metrics = eval_passkey(pk_model, pkcfg)
    ref_exact = ref_metrics["overall_exact_acc"]
    ref_digit = ref_metrics["overall_digit_acc"]
    del pk_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"ref (re-eval) LM val_ppl = {ref_ppl:.3f} "
          f"passkey exact = {ref_exact:.3f} digit = {ref_digit:.3f}")

    n_layers = 4
    n_heads = 4
    per_head = []
    for layer in range(n_layers):
        for head in range(n_heads):
            # LM side: ablate on LM model, measure PPL.
            lm_model, lm_mcfg = _fresh_lm(bl, device)
            lm_st = build_head_prune_state(lm_model, [(layer, head)])
            ppl = eval_lm(lm_model, lm_mcfg)
            lm_st.release()
            del lm_model

            # Passkey side: ablate on passkey model, measure passkey.
            pk_model, pk_mcfg, pkcfg = _fresh_pk(bl, device)
            pk_st = build_head_prune_state(pk_model, [(layer, head)])
            metrics = eval_passkey(pk_model, pkcfg)
            pk_st.release()
            del pk_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            rec = {
                "layer": layer,
                "head": head,
                "pruned_ppl": ppl,
                "pruned_passkey_exact": metrics["overall_exact_acc"],
                "pruned_passkey_digit": metrics["overall_digit_acc"],
                "importance_ppl": ppl - ref_ppl,
                "importance_passkey_exact": ref_exact - metrics["overall_exact_acc"],
                "importance_passkey_digit": ref_digit - metrics["overall_digit_acc"],
            }
            per_head.append(rec)
            print(f"  L{layer} H{head}: ppl={ppl:.2f} (Δ={rec['importance_ppl']:+.2f}) "
                  f"pk_exact={metrics['overall_exact_acc']:.3f} "
                  f"(Δ={rec['importance_passkey_exact']:+.3f})")

    # Ranks (ascending by passkey importance = most-safe first).
    by_pk_asc = sorted(per_head, key=lambda r: r["importance_passkey_exact"])
    by_pk_desc = sorted(per_head, key=lambda r: -r["importance_passkey_exact"])
    by_ppl_asc = sorted(per_head, key=lambda r: r["importance_ppl"])

    out = {
        "baseline_ppl": ref_ppl,
        "baseline_passkey_exact": ref_exact,
        "baseline_passkey_digit": ref_digit,
        "per_head": per_head,
        "order_least_passkey_first": [(r["layer"], r["head"]) for r in by_pk_asc],
        "order_most_passkey_first": [(r["layer"], r["head"]) for r in by_pk_desc],
        "order_least_ppl_first": [(r["layer"], r["head"]) for r in by_ppl_asc],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "head_importance.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {RESULTS_DIR / 'head_importance.json'}")
    return out


def plot_heatmap(data: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping heatmap")
        return

    per_head = data["per_head"]
    n_layers = max(r["layer"] for r in per_head) + 1
    n_heads = max(r["head"] for r in per_head) + 1
    ppl_mat = np.zeros((n_layers, n_heads))
    pk_mat = np.zeros((n_layers, n_heads))
    for r in per_head:
        ppl_mat[r["layer"], r["head"]] = r["importance_ppl"]
        pk_mat[r["layer"], r["head"]] = r["importance_passkey_exact"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(ppl_mat, aspect="auto", cmap="Reds")
    axes[0].set_title("ΔPPL from zeroing single head\n(higher = more important for LM)")
    axes[0].set_xlabel("head index")
    axes[0].set_ylabel("layer")
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            axes[0].text(h, l, f"{ppl_mat[l, h]:+.1f}", ha="center", va="center",
                          color="black", fontsize=9)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pk_mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Δpasskey exact from zeroing single head\n(higher = more important for retrieval)")
    axes[1].set_xlabel("head index")
    axes[1].set_ylabel("layer")
    axes[1].set_xticks(range(n_heads))
    axes[1].set_yticks(range(n_layers))
    for l in range(n_layers):
        for h in range(n_heads):
            axes[1].text(h, l, f"{pk_mat[l, h]:+.2f}", ha="center", va="center",
                          color="black", fontsize=9)
    plt.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = ap.parse_args()
    data = rank_all_heads()
    plot_heatmap(data, Path(args.out_dir) / "importance_heatmap.png")


if __name__ == "__main__":
    main()
