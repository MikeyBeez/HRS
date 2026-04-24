"""Run the full α-sweep (phases 1, 2a, 2b, 3) on an arbitrary seed's
checkpoints. Intended for seed transfer: does the sharp-threshold result
hold for other initializations whose retrieval heads landed at different
indices?

For seed 2, the retrieval heads (Δpasskey > 0.5, from the seed transfer
experiment):
  L0 H2 (1.00), L2 H0 (0.98), L0 H3 (0.89), L0 H1 (0.88),
  L1 H2 (0.85), L1 H3 (0.80), L0 H0 (0.71)

So the direct analogues of baseline's {beacon, find, read} are:
  beacon = L2 H0 (same index as baseline)
  find   = L0 H2 (top find head — baseline's analogue is L0 H3 at 1.00)
  read   = L3 H0 (L3 layer is redundant across seeds; pick H0)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import experiments.amplitude_sweep._common as _common
from experiments.amplitude_sweep._common import eval_point, load_baselines
from experiments.amplitude_sweep.alpha_hooks import (
    fit_sigmoid,
    transition_width,
)


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results"

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
BETAS = [1, 2, 3, 5, 10, 20]


def set_seed_ckpt_dir(seed: int):
    if seed == 0:
        ckpt_dir = Path("/mnt/data/Code/HRS/experiments/pruning/checkpoints")
    else:
        ckpt_dir = Path(
            f"/mnt/data/Code/HRS/experiments/head_pruning/seeds/seed_{seed}/"
            f"checkpoints"
        )
    _common.CKPT_DIR = ckpt_dir
    print(f"using checkpoints from {ckpt_dir}")
    return ckpt_dir


def phase1(bl, beacon, out_dir):
    print(f"\n=== Phase 1: α-sweep on beacon L{beacon[0]}H{beacon[1]} ===")
    records = []
    t0 = time.time()
    for alpha in ALPHAS:
        rec = eval_point(bl, {beacon: alpha}, measure_lm=True)
        rec["alpha"] = alpha
        rec["target"] = list(beacon)
        records.append(rec)
        print(f"  α={alpha:>4.2f}  passkey={rec['passkey_exact']:.3f} "
              f"digit={rec['passkey_digit']:.3f} ppl={rec['val_ppl']:.3f}")

    alphas = [r["alpha"] for r in records]
    passkeys = [r["passkey_exact"] for r in records]
    astar, k, r2 = fit_sigmoid(alphas, passkeys)
    tw = transition_width(alphas, passkeys)

    out = {
        "target_layer": beacon[0],
        "target_head": beacon[1],
        "records": records,
        "fit_sigmoid": {"alpha_star": astar, "k": k, "r2": r2},
        "transition": {"alpha_at_hi": tw[0], "alpha_at_lo": tw[1],
                        "width": tw[2]},
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "phase1_beacon.json").write_text(json.dumps(out, indent=2))
    print(f"  fit: α*={astar:.3f} k={k:.2f} r²={r2:.3f}  width={tw[2]}")
    return out


def phase2a(bl, beacon, siblings, out_dir):
    print(f"\n=== Phase 2a: zero {beacon}, amplify one sibling at a time ===")
    records = []
    t0 = time.time()
    for sib in siblings:
        for beta in BETAS:
            scales = {beacon: 0.0, (beacon[0], sib): beta}
            rec = eval_point(bl, scales, measure_lm=False)
            rec["sibling"] = sib
            rec["beta"] = beta
            records.append(rec)
            print(f"  H0=0, H{sib}×{beta}: "
                  f"passkey={rec['passkey_exact']:.3f} "
                  f"digit={rec['passkey_digit']:.3f}")

    best = max(records, key=lambda r: r["passkey_exact"])
    (out_dir / "phase2a_siblings.json").write_text(json.dumps({
        "records": records,
        "best": best,
        "wall_seconds": time.time() - t0,
    }, indent=2))
    print(f"  best: sibling H{best['sibling']} × β={best['beta']} "
          f"-> passkey={best['passkey_exact']:.3f}")


def phase2b(bl, beacon, siblings, out_dir):
    print(f"\n=== Phase 2b: zero {beacon}, amplify all siblings together ===")
    records = []
    t0 = time.time()
    for beta in BETAS:
        scales = {beacon: 0.0}
        for s in siblings:
            scales[(beacon[0], s)] = beta
        rec = eval_point(bl, scales, measure_lm=False)
        rec["beta"] = beta
        records.append(rec)
        print(f"  H0=0, all siblings × {beta}: "
              f"passkey={rec['passkey_exact']:.3f}")

    best = max(records, key=lambda r: r["passkey_exact"])
    (out_dir / "phase2b_all_siblings.json").write_text(json.dumps({
        "records": records,
        "best": best,
        "wall_seconds": time.time() - t0,
    }, indent=2))
    print(f"  best: β={best['beta']} -> passkey={best['passkey_exact']:.3f}")


def phase3(bl, find, read, out_dir):
    print(f"\n=== Phase 3: cross-layer α-sweep on find {find} + read {read} ===")
    results = {}
    t0 = time.time()
    for tag, target in [(f"L{find[0]}H{find[1]}", find),
                          (f"L{read[0]}H{read[1]}", read)]:
        print(f"  -- {tag}")
        records = []
        for alpha in ALPHAS:
            rec = eval_point(bl, {target: alpha}, measure_lm=False)
            rec["alpha"] = alpha
            rec["target"] = list(target)
            records.append(rec)
            print(f"    α={alpha:>4.2f}  passkey={rec['passkey_exact']:.3f} "
                  f"digit={rec['passkey_digit']:.3f}")
        alphas = [r["alpha"] for r in records]
        passkeys = [r["passkey_exact"] for r in records]
        try:
            astar, k, r2 = fit_sigmoid(alphas, passkeys)
        except Exception:
            astar, k, r2 = None, None, None
        tw = transition_width(alphas, passkeys)
        results[tag] = {
            "records": records,
            "fit_sigmoid": {"alpha_star": astar, "k": k, "r2": r2},
            "transition": {
                "alpha_at_hi": tw[0], "alpha_at_lo": tw[1], "width": tw[2]
            },
        }
        print(f"    α*={astar} k={k} r²={r2} width={tw[2]}")

    results["wall_seconds"] = time.time() - t0
    (out_dir / "phase3_cross_layer.json").write_text(
        json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--beacon", nargs=2, type=int, default=[2, 0],
                    help="(layer, head) of beacon to sweep in phase 1.")
    ap.add_argument("--find", nargs=2, type=int, default=[0, 2],
                    help="(layer, head) of find head for phase 3.")
    ap.add_argument("--read", nargs=2, type=int, default=[3, 0],
                    help="(layer, head) of read head for phase 3.")
    ap.add_argument("--siblings", nargs="+", type=int, default=[1, 2, 3],
                    help="Head indices to amplify in phase 2.")
    args = ap.parse_args()

    set_seed_ckpt_dir(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bl = load_baselines(device)

    out_dir = RESULTS_ROOT / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    beacon = tuple(args.beacon)
    find = tuple(args.find)
    read = tuple(args.read)

    t_start = time.time()
    phase1(bl, beacon, out_dir)
    phase2a(bl, beacon, args.siblings, out_dir)
    phase2b(bl, beacon, args.siblings, out_dir)
    phase3(bl, find, read, out_dir)
    elapsed = time.time() - t_start

    summary = {
        "seed": args.seed,
        "beacon": list(beacon),
        "find": list(find),
        "read": list(read),
        "wall_seconds": elapsed,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== done in {elapsed:.0f}s. Outputs in {out_dir} ===")


if __name__ == "__main__":
    main()
