"""Train 3 fresh MHA baselines with seeds 1/2/3, rank heads on each,
and compare to the seed-0 baseline.

Implementation detail: the existing `train_baseline.py` and `rank_heads.py`
modules read their checkpoint/results paths from module-level `CKPT_DIR` /
`RESULTS_DIR` constants at call time. This driver monkey-patches those
constants for each seed so the baseline code runs unchanged, writes to
per-seed directories, and then restores the originals. That preserves
"identical training procedure" — we're only varying the seed and the
output path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import experiments.pruning.train_baseline as _train_base
import experiments.pruning.run_sweep as _run_sweep
import experiments.head_pruning.rank_heads as _rank


ROOT = Path(__file__).resolve().parent  # experiments/head_pruning/seeds
BASELINE_IMPORTANCE_PATH = ROOT.parent / "results" / "head_importance.json"
SEEDS = [1, 2, 3]


def _set_paths(seed_dir: Path):
    """Point both modules at the per-seed dir."""
    seed_ckpt_dir = seed_dir / "checkpoints"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
    _train_base.CKPT_DIR = seed_ckpt_dir
    _run_sweep.CKPT_DIR = seed_ckpt_dir
    _rank.RESULTS_DIR = seed_dir


def _restore_paths():
    # Reset to the values the modules had at import time.
    _train_base.CKPT_DIR = _train_base.ROOT / "checkpoints"
    _run_sweep.CKPT_DIR = _train_base.ROOT / "checkpoints"  # same dir
    _rank.RESULTS_DIR = _rank.ROOT / "results"


def train_seed(seed: int, lm_steps: int, pk_steps: int) -> dict:
    """Train both baselines for one seed. Returns quality summary."""
    seed_dir = ROOT / f"seed_{seed}"
    _set_paths(seed_dir)

    lm_ckpt_path = _train_base.train_lm_baseline(lm_steps, seed=seed)
    lm_blob = torch.load(lm_ckpt_path, map_location="cpu", weights_only=False)
    lm_ppl = lm_blob["final_val_ppl"]

    pk_ckpt_path = _train_base.train_passkey_baseline(pk_steps, seed=seed)
    pk_blob = torch.load(pk_ckpt_path, map_location="cpu", weights_only=False)
    pk_exact = pk_blob["final_passkey"]["overall_exact_acc"]
    pk_digit = pk_blob["final_passkey"]["overall_digit_acc"]

    converged = (pk_exact >= 0.95) and (lm_ppl <= 5.5)
    return {
        "seed": seed,
        "lm_val_ppl": lm_ppl,
        "passkey_exact": pk_exact,
        "passkey_digit": pk_digit,
        "converged": converged,
        "ckpt_dir": str(seed_dir / "checkpoints"),
    }


def rank_seed(seed: int) -> dict:
    """Run Phase-1 head ranking for one seed. Returns the importance dict
    (same schema as the baseline run)."""
    seed_dir = ROOT / f"seed_{seed}"
    _set_paths(seed_dir)
    data = _rank.rank_all_heads()
    _rank.plot_heatmap(data, seed_dir / "importance_heatmap.png")
    return data


def top_retrieval_heads(data: dict, threshold: float = 0.5):
    """Heads with Δpasskey_exact > threshold, sorted descending.
    Always returns at least the top 3 regardless of threshold."""
    per = data["per_head"]
    sorted_heads = sorted(per, key=lambda r: -r["importance_passkey_exact"])
    chosen = [h for h in sorted_heads if h["importance_passkey_exact"] > threshold]
    if len(chosen) < 3:
        chosen = sorted_heads[:3]
    return [{"layer": h["layer"], "head": h["head"],
              "delta": h["importance_passkey_exact"]} for h in chosen]


def compute_overlap(all_imps: dict, converged_seeds: set | None = None) -> dict:
    """Given {seed: importance_dict}, compute cross-seed position overlap.

    If `converged_seeds` is provided, only those seeds contribute to the
    summary statistics. Non-converged seeds are still included in per_seed
    with a `converged=False` flag so they appear in the plot + raw JSON.
    """
    if converged_seeds is None:
        converged_seeds = set(all_imps.keys())

    top = {s: top_retrieval_heads(d) for s, d in all_imps.items()}
    baseline_positions = {(h["layer"], h["head"]) for h in top[0]}
    baseline_layers = {h["layer"] for h in top[0]}

    per_seed = {}
    for s, heads in top.items():
        pos = {(h["layer"], h["head"]) for h in heads}
        layers = {h["layer"] for h in heads}
        per_seed[s] = {
            "converged": s in converged_seeds,
            "top_heads": heads,
            "count_above_threshold": len(heads),
            "positions": sorted(pos),
            "layers": sorted(layers),
            "exact_position_match_with_baseline": sorted(pos & baseline_positions),
            "n_exact_match": len(pos & baseline_positions),
            "layer_match_with_baseline": sorted(layers & baseline_layers),
            "n_layer_match": len(layers & baseline_layers),
        }

    non_baseline_converged = [s for s in per_seed
                                if s != 0 and s in converged_seeds]
    n_bl = len(per_seed[0]["positions"])
    n_bl_layers = len(baseline_layers)
    summary = {
        "baseline_top_heads": top[0],
        "converged_seeds": sorted(converged_seeds),
        "excluded_seeds": sorted(set(all_imps.keys()) - converged_seeds),
        "n_baseline_heads": len(top[0]),
        "baseline_layers": sorted(baseline_layers),
        # Fraction of baseline positions covered by each converged non-baseline seed.
        "position_coverage_per_seed": {
            s: per_seed[s]["n_exact_match"] / max(1, n_bl)
            for s in non_baseline_converged
        },
        "layer_coverage_per_seed": {
            s: per_seed[s]["n_layer_match"] / max(1, n_bl_layers)
            for s in non_baseline_converged
        },
        "all_seeds_cover_baseline_layers": all(
            per_seed[s]["n_layer_match"] == n_bl_layers
            for s in non_baseline_converged
        ),
        "all_seeds_cover_baseline_positions": all(
            per_seed[s]["n_exact_match"] == n_bl
            for s in non_baseline_converged
        ),
        "count_consistency": (
            len({per_seed[s]["count_above_threshold"]
                  for s in converged_seeds}) == 1
        ),
        "retrieval_head_counts": {s: per_seed[s]["count_above_threshold"]
                                    for s in converged_seeds},
    }
    return {"per_seed": per_seed, "summary": summary}


def plot_grid(all_imps: dict, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping grid")
        return

    seeds = sorted(all_imps.keys())
    n = len(seeds)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(9, 4.2 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, s in enumerate(seeds):
        data = all_imps[s]
        per = data["per_head"]
        n_layers = max(r["layer"] for r in per) + 1
        n_heads = max(r["head"] for r in per) + 1
        mat = np.zeros((n_layers, n_heads))
        for r in per:
            mat[r["layer"], r["head"]] = r["importance_passkey_exact"]

        ax = axes[i]
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"seed {s}: Δpasskey per head")
        ax.set_xlabel("head index")
        ax.set_ylabel("layer")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        for l in range(n_layers):
            for h in range(n_heads):
                ax.text(h, l, f"{mat[l, h]:+.2f}", ha="center", va="center",
                         color="black", fontsize=8)
        plt.colorbar(im, ax=ax)

    for j in range(n, rows * cols):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def render_overlap_text(overlap: dict, all_imps: dict,
                          convergence: dict | None = None) -> str:
    s = overlap["summary"]
    lines = ["Overlap analysis", "================", ""]
    if s["excluded_seeds"]:
        lines.append(f"Seeds excluded (did not converge to passkey>=0.95): "
                      f"{s['excluded_seeds']}")
        lines.append("")

    lines.append("Retrieval heads per seed (Δpasskey > 0.5, or top-3):")
    for seed in sorted(all_imps.keys()):
        info = overlap["per_seed"][seed]
        heads = info["top_heads"]
        joined = ", ".join(f"L{h['layer']}H{h['head']}(Δ={h['delta']:+.2f})"
                             for h in heads)
        tag = "(baseline)" if seed == 0 else ""
        mark = "  " if info["converged"] else "X "
        lines.append(f"  {mark}seed {seed} {tag}: {joined}")
    lines.append("")

    lines.append(f"baseline top heads: "
                  f"{[(h['layer'], h['head']) for h in s['baseline_top_heads']]}")
    lines.append(f"baseline layer set: {s['baseline_layers']}")
    lines.append("")

    lines.append("Per converged non-baseline seed:")
    for seed, info in overlap["per_seed"].items():
        if seed == 0 or not info["converged"]:
            continue
        n_bl = s["n_baseline_heads"]
        n_bl_layers = len(s["baseline_layers"])
        lines.append(f"  seed {seed}: "
                      f"n_retrieval_heads={info['count_above_threshold']}, "
                      f"positions={info['positions']}, "
                      f"layers={info['layers']}, "
                      f"baseline-position-hits={info['n_exact_match']}/{n_bl}, "
                      f"baseline-layer-hits={info['n_layer_match']}/{n_bl_layers}")
    lines.append("")

    lines.append(f"all_seeds_cover_baseline_positions: "
                  f"{s['all_seeds_cover_baseline_positions']}")
    lines.append(f"all_seeds_cover_baseline_layers: "
                  f"{s['all_seeds_cover_baseline_layers']}")
    lines.append(f"count_consistency (all converged seeds have same # retrieval heads): "
                  f"{s['count_consistency']}")
    lines.append(f"retrieval head counts (converged seeds): "
                  f"{s['retrieval_head_counts']}")
    lines.append(f"position_coverage_per_seed: {s['position_coverage_per_seed']}")
    lines.append(f"layer_coverage_per_seed: {s['layer_coverage_per_seed']}")
    return "\n".join(lines)


def interpret(overlap: dict) -> str:
    s = overlap["summary"]
    exact_all = s["all_seeds_cover_baseline_positions"]
    layer_all = s["all_seeds_cover_baseline_layers"]
    if exact_all and s["count_consistency"]:
        return ("Outcome 1: **Exact same positions across all seeds.** "
                "The (layer, head) addresses of the retrieval circuit are "
                "structural — the architecture has natural retrieval slots.")
    if layer_all:
        return ("Outcome 2: **Baseline layers are hit by every converged seed.** "
                "Retrieval is layer-localized — the specific layers are "
                "consistent, but the head indices within those layers vary. "
                "Head counts may also vary.")
    return ("Outcome 3 (or mixed): **Positions are not stable across seeds.** "
            "Retrieval is a function the network allocates wherever gradient "
            "flow favors it, not a property of specific (layer, head) "
            "positions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--lm-steps", type=int, default=5000)
    ap.add_argument("--passkey-steps", type=int, default=20000)
    ap.add_argument("--skip-train", action="store_true",
                    help="Assume checkpoints already exist; only rank + analyze.")
    args = ap.parse_args()

    t0 = time.time()

    convergence = {}
    if not args.skip_train:
        for s in args.seeds:
            print(f"\n=== Training seed {s} ===")
            conv = train_seed(s, args.lm_steps, args.passkey_steps)
            convergence[s] = conv
            print(f"  seed {s}: lm_ppl={conv['lm_val_ppl']:.3f} "
                  f"passkey_exact={conv['passkey_exact']:.3f} "
                  f"converged={conv['converged']}")

    all_imps: dict = {}
    # Include the existing baseline (seed 0).
    print("\n=== Loading baseline (seed 0) importance ===")
    all_imps[0] = json.loads(BASELINE_IMPORTANCE_PATH.read_text())

    for s in args.seeds:
        print(f"\n=== Ranking heads for seed {s} ===")
        if args.skip_train:
            _set_paths(ROOT / f"seed_{s}")
            imp = json.loads(
                (ROOT / f"seed_{s}" / "head_importance.json").read_text())
        else:
            imp = rank_seed(s)
        all_imps[s] = imp

    _restore_paths()

    # Rebuild convergence from saved checkpoints when skipping training.
    if args.skip_train and not convergence:
        for s in args.seeds:
            ckpt = torch.load(
                ROOT / f"seed_{s}" / "checkpoints" / "mha_passkey.pt",
                map_location="cpu", weights_only=False,
            )
            lm_ckpt = torch.load(
                ROOT / f"seed_{s}" / "checkpoints" / "mha_lm.pt",
                map_location="cpu", weights_only=False,
            )
            pk_exact = ckpt["final_passkey"]["overall_exact_acc"]
            lm_ppl = lm_ckpt["final_val_ppl"]
            convergence[s] = {
                "seed": s,
                "lm_val_ppl": lm_ppl,
                "passkey_exact": pk_exact,
                "passkey_digit": ckpt["final_passkey"]["overall_digit_acc"],
                "converged": (pk_exact >= 0.95) and (lm_ppl <= 5.5),
            }

    # Baseline (seed 0) is always considered converged (it's the reference).
    converged_seeds = {0} | {
        s for s, c in convergence.items() if c.get("converged")
    }

    print("\n=== Computing overlap + plots ===")
    overlap = compute_overlap(all_imps, converged_seeds=converged_seeds)
    comparison_path = ROOT / "comparison.json"
    comparison_path.write_text(json.dumps({
        "convergence": convergence,
        "overlap": overlap,
        "seeds_included": sorted(all_imps.keys()),
        "converged_seeds": sorted(converged_seeds),
    }, indent=2))
    print(f"wrote {comparison_path}")

    txt = render_overlap_text(overlap, all_imps, convergence)
    txt += "\n\n" + interpret(overlap) + "\n"
    (ROOT / "overlap_analysis.txt").write_text(txt)
    print(f"wrote {ROOT / 'overlap_analysis.txt'}")
    print("\n" + txt)

    plot_grid(all_imps, ROOT / "importance_grid.png")

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
