"""Aggregate supernet results: per-mechanism gates × layers × seeds."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/multi_mechanism_supernet"

SEEDS = [0, 1, 2]
N_LAYERS = 6


def load(tag):
    p = EXP / f"results/train_{tag}.json"
    return json.loads(p.read_text())


def gate_key_to_li_name(k):
    m = re.match(r"L(\d+)_(.+)", k)
    return int(m.group(1)), m.group(2)


def main():
    print(f"\n{'='*92}\nMULTI-MECHANISM SUPERNET (3 seeds floor-released, 1 floor-pinned control)\n{'='*92}\n")

    # Released runs
    rel_runs = [load(f"wt103_released_seed{s}") for s in SEEDS]
    pinned_run = load("wt103_pinned_seed0")

    print(f"Released-schedule overall metrics:")
    ppls = [r["final_val_ppl"] for r in rel_runs]
    ancs = [r["anchor_acc"] for r in rel_runs]
    walls = [r["wall_s"] for r in rel_runs]
    print(f"  val PPL:      {mean(ppls):.2f} ± {stdev(ppls):.2f}")
    print(f"  anchor acc:   {mean(ancs):.3f} ± {stdev(ancs):.3f}")
    print(f"  wall time:    {mean(walls):.0f}s per run")

    print(f"\nFloor-pinned control:")
    print(f"  val PPL:      {pinned_run['final_val_ppl']:.2f}")
    print(f"  anchor acc:   {pinned_run['anchor_acc']:.3f}")
    print(f"  wall time:    {pinned_run['wall_s']:.0f}s")
    print(f"  pinned vs released gap: "
          f"{(pinned_run['final_val_ppl'] - mean(ppls))/mean(ppls)*100:+.2f}%")

    # Per-mechanism gates aggregated across seeds × layers
    print(f"\n{'='*92}\nGATE MAGNITUDES — released, mean across 3 seeds × 6 layers\n{'='*92}\n")
    # Build dict: mechanism_name -> list of all gate values across seeds and layers
    by_mech = defaultdict(list)
    by_mech_layer = defaultdict(list)   # (mech, layer) -> list across seeds
    for r in rel_runs:
        for k, v in r["final_gates"].items():
            li, name = gate_key_to_li_name(k)
            by_mech[name].append(v)
            by_mech_layer[(name, li)].append(v)

    print(f"  {'mechanism':>22s}  {'mean':>6s}  {'std':>6s}  {'n':>4s}  range across (seed, layer)")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*4}  {'-'*40}")
    # Sort by mean gate descending
    sorted_mechs = sorted(by_mech.items(), key=lambda kv: -mean(kv[1]))
    for name, vals in sorted_mechs:
        m = mean(vals); s = stdev(vals)
        lo = min(vals); hi = max(vals)
        print(f"  {name:>22s}  {m:6.3f}  {s:6.3f}  {len(vals):4d}  [{lo:.3f}, {hi:.3f}]")

    # Per-layer breakdown for top mechanisms
    print(f"\n{'='*92}\nPER-LAYER GATE MEANS (across 3 seeds)\n{'='*92}\n")
    print(f"  {'mechanism':>22s} | " + " ".join(f"L{li:2d}    " for li in range(N_LAYERS)))
    for name, _ in sorted_mechs:
        row = [f"{name:>22s} |"]
        for li in range(N_LAYERS):
            vals = by_mech_layer[(name, li)]
            row.append(f"{mean(vals):.3f}")
        print("  " + " ".join(row))

    # Per-spec classification
    print(f"\n{'='*92}\nPER-SPEC CLASSIFICATION (high: mean>0.7 std<0.15; low: mean<0.2 std<0.1)\n{'='*92}\n")
    for name, vals in sorted_mechs:
        m = mean(vals); s = stdev(vals)
        if m > 0.7 and s < 0.15:
            cls = "HIGH"
        elif m < 0.2 and s < 0.1:
            cls = "LOW"
        else:
            cls = "AMBIGUOUS"
        print(f"  {name:>22s}: mean={m:.3f} std={s:.3f}  → {cls}")

    # Pinned vs released per-mechanism gate (sanity: pinned should be uniform at floor)
    print(f"\n{'='*92}\nFLOOR-PINNED CONTROL — final gates (should be ≥0.5 floor)\n{'='*92}\n")
    pinned_by_mech = defaultdict(list)
    for k, v in pinned_run["final_gates"].items():
        li, name = gate_key_to_li_name(k)
        pinned_by_mech[name].append(v)
    print(f"  {'mechanism':>22s}  {'mean':>6s}  range")
    for name, _ in sorted_mechs:
        vals = pinned_by_mech[name]
        m = mean(vals)
        print(f"  {name:>22s}  {m:6.3f}  [{min(vals):.3f}, {max(vals):.3f}]")

    # Save
    out = {
        "released": {
            "val_ppl_mean": mean(ppls), "val_ppl_std": stdev(ppls),
            "anchor_acc_mean": mean(ancs), "anchor_acc_std": stdev(ancs),
            "wall_mean": mean(walls),
        },
        "pinned": {
            "val_ppl": pinned_run["final_val_ppl"],
            "anchor_acc": pinned_run["anchor_acc"],
            "wall": pinned_run["wall_s"],
        },
        "by_mechanism_released": {
            name: {"mean": mean(vals), "std": stdev(vals),
                    "min": min(vals), "max": max(vals), "n": len(vals)}
            for name, vals in by_mech.items()
        },
        "by_mechanism_pinned": {
            name: {"mean": mean(vals), "min": min(vals), "max": max(vals)}
            for name, vals in pinned_by_mech.items()
        },
        "per_layer_released": {
            f"{name}_L{li}": mean(vals)
            for (name, li), vals in by_mech_layer.items()
        },
    }
    out_path = EXP / "results/aggregate.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
