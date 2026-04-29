"""Part A: 5 passages × 3 conditions × 5 incremental steps.

Conditions:
  A1 = retrain-from-scratch on cumulative training data (data-as-storage approach)
  A2 = sequential fine-tune (passage 1 -> tune on 2 -> tune on 3 ...)
  A3 = independent per-passage adapters (no merging)

After step N (passages 1..N absorbed), evaluate held-out recall on all 5.

Output: results/part_a.json with per-step per-condition per-entry recall.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

# Use the harness selected by --harness flag (default: harness2 multi-layer).
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--harness", choices=["1", "2"], default="2")
parser.add_argument("--rank", type=int, default=64)
parser.add_argument("--n_steps_per_passage", type=int, default=400)
parser.add_argument("--high_lr", type=float, default=5e-3)
parser.add_argument("--base_lr", type=float, default=1e-4)
parser.add_argument("--out", default="results/part_a.json")
args = parser.parse_args()

if args.harness == "1":
    from experiments.data_storage_adapter.harness import (
        load_base, reset_lora, train_adapter, eval_retrieval,
        make_training_sources, lora_state_dict, load_lora_state,
    )
else:
    from experiments.data_storage_adapter.harness2 import (
        load_base, reset_lora, train_adapter, eval_retrieval,
        make_training_sources, lora_state_dict, load_lora_state,
    )


def main():
    device = torch.device("cuda")
    out_path = REPO / "experiments/data_storage_adapter" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())
    pip = groups["groups"]["G1_pip_childhood"][:5]
    print(f"Part A: 5 Pip passages, harness{args.harness}, rank={args.rank}, "
          f"steps_per_passage={args.n_steps_per_passage}")
    for e in pip:
        print(f"  [{e['id']}] '{e['fact']}' answer='{e['answer']}'")

    HIGH_LR = args.high_lr
    BASE_LR = args.base_lr
    EVAL_KW = {"temperature": 0.6, "top_k": 20, "seeds": (0, 1, 2)}

    results = {
        "config": {
            "harness": args.harness, "rank": args.rank,
            "n_steps_per_passage": args.n_steps_per_passage,
            "high_lr": HIGH_LR, "base_lr": BASE_LR,
        },
        "A1_retrain_from_scratch": [],
        "A2_sequential_fine_tune": [],
        "A3_independent": [],
    }

    # --- A1: retrain from scratch on cumulative data ---
    print("\n[A1] retrain-from-scratch on cumulative data")
    t0 = time.time()
    for k in range(1, 6):
        entries = pip[:k]
        model, cfg, tok = load_base(args.rank, device)
        reset_lora(model)
        sources = make_training_sources(entries, tok, cfg.ctx_len, device)
        n_steps = args.n_steps_per_passage * k
        info = train_adapter(model, sources, n_steps=n_steps,
                              high_lr=HIGH_LR, base_lr=BASE_LR, seed=0)
        ev_full = eval_retrieval(model, pip, tok, device, **EVAL_KW)
        per_passage_rates = [pe["rate"] for pe in ev_full["per_entry"]]
        rec = {
            "step": k, "trained_on": list(range(k)),
            "n_steps": n_steps, "n_sources": len(sources),
            "loss_init": info["loss_init"],
            "loss_final": info["loss_mean_last10"],
            "mean_rate": ev_full["mean_rate"],
            "per_passage_rate": per_passage_rates,
        }
        results["A1_retrain_from_scratch"].append(rec)
        print(f"  k={k} steps={n_steps} loss {info['loss_init']:.2f}->"
              f"{info['loss_mean_last10']:.2f}  mean_rate={ev_full['mean_rate']:.3f}  "
              f"per-passage={[f'{r:.2f}' for r in per_passage_rates]}")
    print(f"  A1 wall: {time.time()-t0:.0f}s")

    # --- A2: sequential fine-tune ---
    print("\n[A2] sequential fine-tune")
    t0 = time.time()
    model, cfg, tok = load_base(args.rank, device)
    reset_lora(model)
    for k in range(1, 6):
        # Train on passage k-1 only (additional fine-tune step)
        sources = make_training_sources([pip[k - 1]], tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=args.n_steps_per_passage,
                              high_lr=HIGH_LR, base_lr=BASE_LR, seed=k)
        ev_full = eval_retrieval(model, pip, tok, device, **EVAL_KW)
        per_passage_rates = [pe["rate"] for pe in ev_full["per_entry"]]
        rec = {
            "step": k, "trained_on": [k - 1],
            "n_steps": args.n_steps_per_passage, "n_sources": len(sources),
            "loss_init": info["loss_init"],
            "loss_final": info["loss_mean_last10"],
            "mean_rate": ev_full["mean_rate"],
            "per_passage_rate": per_passage_rates,
        }
        results["A2_sequential_fine_tune"].append(rec)
        print(f"  k={k} added={k-1} loss {info['loss_init']:.2f}->"
              f"{info['loss_mean_last10']:.2f}  mean_rate={ev_full['mean_rate']:.3f}  "
              f"per-passage={[f'{r:.2f}' for r in per_passage_rates]}")
    print(f"  A2 wall: {time.time()-t0:.0f}s")

    # --- A3: independent per-passage adapters; eval each on its own passage,
    # but we report per-passage rates on the full 5-passage set when each
    # adapter is loaded separately.
    print("\n[A3] independent per-passage adapters")
    t0 = time.time()
    a3_per_passage = []
    for i, e in enumerate(pip):
        model, cfg, tok = load_base(args.rank, device)
        reset_lora(model)
        sources = make_training_sources([e], tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=args.n_steps_per_passage,
                              high_lr=HIGH_LR, base_lr=BASE_LR, seed=0)
        ev_self = eval_retrieval(model, [e], tok, device, **EVAL_KW)
        ev_others = eval_retrieval(model, [x for x in pip if x["id"] != e["id"]],
                                    tok, device, **EVAL_KW)
        a3_per_passage.append({
            "passage_id": e["id"], "answer": e["answer"],
            "n_steps": args.n_steps_per_passage,
            "loss_init": info["loss_init"],
            "loss_final": info["loss_mean_last10"],
            "self_rate": ev_self["mean_rate"],
            "other_rate": ev_others["mean_rate"],
        })
        print(f"  passage {e['id']} self_rate={ev_self['mean_rate']:.3f} "
              f"other_rate={ev_others['mean_rate']:.3f}")
    results["A3_independent"] = a3_per_passage
    print(f"  A3 wall: {time.time()-t0:.0f}s")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
