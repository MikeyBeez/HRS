"""Phase 20: Sweep three remediations for the LoRA-cumulative-forgetting issue.

Phase 19 showed cumulative retrieval drops to 10/20 (50%) because:
  - High LR (3e-4 -> 1e-4) overshoots and corrupts older encodings
  - 80/20 new/old split gives only ~20 maintenance steps
  - Rank 512 may not have enough capacity for 20 distinct memories

Test three orthogonal fixes:
  A) lower_lr:        1e-4 -> 5e-5, 200 steps, 80/20 rehearsal
  B) equal_rehearsal: 3e-4 -> 1e-4, 200 steps, 50/50 rehearsal
  C) more_capacity:   3e-4 -> 1e-4, 100 steps, 80/20, RANK=1024

All configs use L4-5 LoRA + dual gate with replay (phase 18).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase20_sweep.py
"""

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_gated, val_ppl_ungated, get_hidden_at_layer, BASE_THRESHOLD,
)
from experiments.identity_ae.phase19_rehearsal import run_rehearsal_ttt
from experiments.identity_ae.lora_wrapper import apply_lora
from experiments.identity_ae.dual_gate import DualGate


def stratified_tests():
    all_tests = generate_passkeys(50)
    by_type = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][:5] + by_type["entity"][:5]
            + by_type["technical"][:5] + by_type["fact"][:5])


def run_config(name, rank, n_steps, high_lr, base_lr, new_prob,
                tokenizer, device, results_dir):
    print(f"\n{'='*60}")
    print(f"CONFIG {name}")
    print(f"  rank={rank}  steps={n_steps}  {high_lr:.1e}->{base_lr:.1e}  "
          f"new_prob={new_prob}")
    print(f"{'='*60}")

    random.seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)
    print(f"  LoRA params: {n_lora:,}")

    dual_gate = DualGate(d_model=1024, base_threshold=BASE_THRESHOLD,
                         novel_threshold=BASE_THRESHOLD)
    dual_gate.load_base_gate("results/identity_ae/phase0/autoencoder_init_20ep.pt")
    dual_gate.to(device)
    for p in dual_gate.novel_gate.parameters():
        p.requires_grad = True

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    baseline_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
    baseline_ungated = val_ppl_ungated(model, loaders["validation"], device)
    print(f"  Baseline val PPL (gated): {baseline_gated:.3f}")

    tests = stratified_tests()
    accumulated_ids = []
    replay_buffer = []
    cumulative_trace = []
    history = []

    t0 = time.time()
    for i, test in enumerate(tests):
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        accumulated_ids.append(ids_t)

        run_rehearsal_ttt(model, dual_gate, accumulated_ids, replay_buffer,
                          tokenizer, device, n_steps, high_lr, base_lr,
                          new_prob=new_prob)

        with torch.no_grad():
            h_new = get_hidden_at_layer(model, ids_t).detach().clone()
        replay_buffer.append(h_new)

        # Cumulative retrieval over all absorbed
        retrieved = 0
        per_test = []
        for j, prev in enumerate(tests[: i + 1]):
            gen = generate_greedy(model, prev["prompt"], tokenizer, device, 50)
            found = check_passkey(gen, prev["passkey"])
            if found:
                retrieved += 1
            per_test.append({"i": j, "type": prev["type"], "found": found})
        cumulative_trace.append(retrieved)

        v_gated = val_ppl_gated(model, dual_gate, loaders["validation"], device)
        d_gated = (v_gated - baseline_gated) / baseline_gated * 100

        history.append({
            "i": i + 1, "new_type": test["type"], "retrieved": retrieved,
            "of_total": i + 1, "per_test": per_test,
            "val_gated": v_gated, "delta_gated": d_gated,
        })

        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/20] new={test['type']:9s}  "
              f"retrieved={retrieved:2d}/{i+1:<2d}  "
              f"gated_delta={d_gated:+.3f}%  ({elapsed:.0f}s)")

    # Final per-type
    final_per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    for p in history[-1]["per_test"]:
        if p["found"]:
            final_per_type[p["type"]] += 1

    final_gated = history[-1]["val_gated"]
    final_delta = history[-1]["delta_gated"]
    final_retrieved = history[-1]["retrieved"]

    print(f"\n  {name} FINAL:")
    print(f"    Cumulative retrieval: {final_retrieved}/20 ({final_retrieved/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"      {ptype}: {final_per_type[ptype]}/5")
    print(f"    Trace: {cumulative_trace}")
    print(f"    Val PPL (gated): {baseline_gated:.3f} -> {final_gated:.3f} ({final_delta:+.3f}%)")

    summary = {
        "name": name,
        "config": {"rank": rank, "n_steps": n_steps, "high_lr": high_lr,
                   "base_lr": base_lr, "new_prob": new_prob,
                   "n_lora_params": n_lora},
        "final_retrieval": final_retrieved / 20,
        "final_per_type": {k: v / 5 for k, v in final_per_type.items()},
        "cumulative_trace": cumulative_trace,
        "baseline_gated": baseline_gated,
        "final_gated": final_gated,
        "delta_gated_pct": final_delta,
        "history": history,
    }
    with open(results_dir / f"{name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    del model, dual_gate
    torch.cuda.empty_cache()
    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase20")
    results_dir.mkdir(parents=True, exist_ok=True)

    # (name, rank, n_steps, high_lr, base_lr, new_prob)
    configs = [
        ("A_lower_lr",        512,  200, 1e-4, 5e-5, 0.8),
        ("B_equal_rehearsal", 512,  200, 3e-4, 1e-4, 0.5),
        ("C_more_capacity",   1024, 100, 3e-4, 1e-4, 0.8),
    ]

    all_results = {}
    for name, rank, n_steps, high_lr, base_lr, new_prob in configs:
        all_results[name] = run_config(name, rank, n_steps, high_lr, base_lr,
                                        new_prob, tokenizer, device, results_dir)

    print(f"\n{'='*60}")
    print("PHASE 20 SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 19 baseline: 10/20 (50%)  trace=[1,1,1,1,1,2,1,2,3,6,3,3,4,2,8,8,8,8,11,10]")
    print()
    print(f"  {'Config':22s} {'Retrieval':>10} {'Drift':>10}")
    print(f"  {'-'*44}")
    for name, summary in all_results.items():
        print(f"  {name:22s} {summary['final_retrieval']:>10.0%} "
              f"{summary['delta_gated_pct']:>+9.3f}%")

    with open(results_dir / "summary.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "history"}
                    for k, v in all_results.items()}, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
