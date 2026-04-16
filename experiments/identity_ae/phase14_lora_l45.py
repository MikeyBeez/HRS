"""Phase 14: LoRA on layers 4-5 only, post-gate, scheduled LR.

Gate sits at layer 3, so layers 0-3 must stay pristine. Layers 4-5 are
post-gate and can be modified freely. This script applies LoRA to attention
(qkv, out_proj) and PEER FFN (input_proj, output_proj) on blocks 4 and 5,
with the warmup-then-decay schedule from phase11.

Goal: 100% passkey retrieval with zero base forgetting.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase14_lora_l45.py
"""

import copy
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase13_lora_scheduled import run_ttt_lora_scheduled
from experiments.identity_ae.lora_wrapper import apply_lora, reset_lora


L45_TARGETS = [
    'blocks.4.attn.qkv',
    'blocks.4.attn.out_proj',
    'blocks.4.peer_ffn.input_proj',
    'blocks.4.peer_ffn.output_proj',
    'blocks.5.attn.qkv',
    'blocks.5.attn.out_proj',
    'blocks.5.peer_ffn.input_proj',
    'blocks.5.peer_ffn.output_proj',
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase14")
    results_dir.mkdir(parents=True, exist_ok=True)

    tests = generate_passkeys(50)

    # (name, rank, n_steps, high_lr, base_lr)
    configs = [
        ("l45_r256_50",  256, 50,  3e-4, 1e-4),
        ("l45_r512_50",  512, 50,  3e-4, 1e-4),
        ("l45_r512_100", 512, 100, 3e-4, 1e-4),
        ("l45_r768_50",  768, 50,  3e-4, 1e-4),
    ]

    all_results = {}

    for name, rank, n_steps, high_lr, base_lr in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {name}")
        print(f"  rank={rank}  steps={n_steps}  {high_lr:.1e} -> {base_lr:.1e}")
        print(f"{'='*60}")

        model, cfg = load_model(device)
        n_lora = apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)
        print(f"  LoRA params: {n_lora:,} on {len(L45_TARGETS)} layers")

        base_state = copy.deepcopy(model.state_dict())

        results = {"name": name, "rank": rank, "targets": L45_TARGETS,
                   "n_steps": n_steps, "high_lr": high_lr, "base_lr": base_lr,
                   "n_lora_params": n_lora, "tests": []}
        n_found = 0

        t0 = time.time()
        for ti, test in enumerate(tests):
            model.load_state_dict(base_state)
            reset_lora(model)

            run_ttt_lora_scheduled(model, test["passage"], tokenizer, device,
                                   n_steps, high_lr, base_lr)

            gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
            found = check_passkey(gen, test["passkey"])
            if found:
                n_found += 1

            results["tests"].append({
                "id": test["id"],
                "type": test["type"],
                "passkey": test["passkey"],
                "weights_found": found,
                "gen_weights": gen[:100],
            })

            if (ti + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{ti+1:2d}/50] WEIGHTS={n_found} ({elapsed:.0f}s)")

        results["summary"] = {
            "weights_rate": n_found / len(tests),
            "elapsed_s": time.time() - t0,
        }
        for ptype in ["numeric", "entity", "technical", "fact"]:
            type_tests = [t for t in results["tests"] if t["type"] == ptype]
            if type_tests:
                w = sum(1 for t in type_tests if t["weights_found"]) / len(type_tests)
                results["summary"][f"{ptype}_weights_rate"] = w

        print(f"\n  {name} RESULT: {n_found}/50 ({n_found/50:.0%})")
        for ptype in ["numeric", "entity", "technical", "fact"]:
            k = f"{ptype}_weights_rate"
            if k in results["summary"]:
                print(f"    {ptype}: {results['summary'][k]:.0%}")

        with open(results_dir / f"{name}.json", "w") as f:
            json.dump(results, f, indent=2)
        all_results[name] = results["summary"]

        del model
        torch.cuda.empty_cache()

        if n_found == 50:
            print(f"\n  100% reached with {name}, stopping here.")
            break

    print(f"\n{'='*60}")
    print("LORA L4-5 SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':18s} {'Retrieval':>10} {'Time':>8}")
    print(f"  {'-'*38}")
    for name, summary in all_results.items():
        print(f"  {name:18s} {summary['weights_rate']:>10.0%} "
              f"{summary['elapsed_s']:>7.0f}s")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
