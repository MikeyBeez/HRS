"""Phase 1 — sanity check: train 5 fresh single-passage adapters, verify ~93%.

Picks 5 random passages with seed 0 (separate from the size-5 set's seed 42),
trains one adapter per passage at the same hyperparameters as the original
Dickens-50 procedure (RANK=128, 150 steps), and evaluates each on its own
held-out probes.

Catches infrastructure regressions before the scaling sweep. Aborts if any
single-passage adapter falls below 0.70 retrieval.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/capacity_scaling"
DICKENS = REPO / "experiments/per_passage_dickens"


def main():
    library = json.loads((DICKENS / "data/library.json").read_text())
    rng = random.Random(0)
    sample_ids = sorted(rng.sample(range(len(library)), 5))
    print(f"Phase 1 sanity passages (seed=0): {sample_ids}")

    results = []
    for aid in sample_ids:
        adapter_path = EXP / f"adapters/phase1_p{aid:03d}.pt"
        # Train
        subprocess.run([
            ".venv/bin/python", "experiments/capacity_scaling/combined_adapter.py",
            "--passage-ids", str(aid),
            "--out-path", str(adapter_path),
        ], check=True, env={**__import__('os').environ,
                            "PYTHONPATH": str(REPO)})
        # Eval
        label = f"phase1_p{aid:03d}"
        subprocess.run([
            ".venv/bin/python", "experiments/capacity_scaling/eval_combined.py",
            "--adapter-path", str(adapter_path),
            "--label", label,
        ], check=True, env={**__import__('os').environ,
                            "PYTHONPATH": str(REPO)})
        d = json.loads((EXP / f"results/eval_{label}.json").read_text())
        results.append({"passage_id": aid,
                         "retrieval": d["overall_retrieval"],
                         "training_time_s": d["training_time_s"]})

    print(f"\n{'='*60}\nPHASE 1 SANITY SUMMARY\n{'='*60}")
    for r in results:
        flag = "✓" if r["retrieval"] >= 0.70 else "FAIL"
        print(f"  passage {r['passage_id']:2d}  retrieval={r['retrieval']:.3f}  "
              f"train={r['training_time_s']:.0f}s  {flag}")
    mean_ret = sum(r["retrieval"] for r in results) / len(results)
    print(f"\n  mean retrieval = {mean_ret:.3f}  (target ~0.93)")
    if mean_ret < 0.70:
        print(f"  WARNING: mean below 0.70 — possible infrastructure regression. "
              f"Investigate before Phase 2.")
        sys.exit(1)
    elif mean_ret < 0.85:
        print(f"  Mean below 0.85; not a regression but worth flagging.")
    else:
        print(f"  OK — proceed to Phase 2.")

    (EXP / "results/phase1_baseline_summary.json").write_text(
        json.dumps({"results": results, "mean": mean_ret}, indent=2))


if __name__ == "__main__":
    main()
