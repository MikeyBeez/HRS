"""Phase 11: LR-scheduled TTT for faster passkey absorption.

Hypothesis: warmup-high then decay should let us hit ~98% in 50 steps
instead of needing 100-200 steps at constant lr=5e-5.

Schedule: first half at high_lr (2-3x baseline), second half at base_lr (5e-5).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase11_lr_schedule.py
"""

import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)


def run_ttt_scheduled(model, passage, tokenizer, device, n_steps, high_lr, base_lr):
    """Two-phase LR via StepLR: high_lr for first half, base_lr for second half."""
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    for _ in range(n_steps):
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase11")
    results_dir.mkdir(parents=True, exist_ok=True)

    tests = generate_passkeys(50)
    print(f"Generated {len(tests)} passkey tests")

    # (name, n_steps, high_lr, base_lr)
    schedules = [
        ("sched_50_2x",   50, 1.0e-4, 5e-5),
        ("sched_50_3x",   50, 1.5e-4, 5e-5),
        ("sched_60_2x",   60, 1.0e-4, 5e-5),
        ("sched_40_3x",   40, 1.5e-4, 5e-5),
    ]

    all_results = {}

    for name, n_steps, high_lr, base_lr in schedules:
        print(f"\n{'='*60}")
        print(f"SCHEDULE: {name}  steps={n_steps}  {high_lr:.0e} -> {base_lr:.0e}")
        print(f"{'='*60}")

        model, cfg = load_model(device)
        for p in model.parameters():
            p.requires_grad = True
        base_state = copy.deepcopy(model.state_dict())

        results = {"name": name, "n_steps": n_steps, "high_lr": high_lr,
                   "base_lr": base_lr, "tests": []}
        n_weights_found = 0

        t0 = time.time()
        for ti, test in enumerate(tests):
            model.load_state_dict(base_state)
            passage = test["passage"]
            passkey = test["passkey"]
            prompt = test["prompt"]

            run_ttt_scheduled(model, passage, tokenizer, device, n_steps, high_lr, base_lr)

            gen_weights = generate_greedy(model, prompt, tokenizer, device, 50)
            weights_found = check_passkey(gen_weights, passkey)
            if weights_found:
                n_weights_found += 1

            results["tests"].append({
                "id": test["id"],
                "type": test["type"],
                "passkey": passkey,
                "weights_found": weights_found,
                "gen_weights": gen_weights[:100],
            })

            if (ti + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{ti+1:2d}/50] WEIGHTS={n_weights_found} ({elapsed:.0f}s)")

        results["summary"] = {
            "weights_rate": n_weights_found / len(tests),
            "elapsed_s": time.time() - t0,
        }
        for ptype in ["numeric", "entity", "technical", "fact"]:
            type_tests = [t for t in results["tests"] if t["type"] == ptype]
            if type_tests:
                w = sum(1 for t in type_tests if t["weights_found"]) / len(type_tests)
                results["summary"][f"{ptype}_weights_rate"] = w

        print(f"\n  {name} RESULT: {n_weights_found}/50 ({n_weights_found/50:.0%})")
        for ptype in ["numeric", "entity", "technical", "fact"]:
            k = f"{ptype}_weights_rate"
            if k in results["summary"]:
                print(f"    {ptype}: {results['summary'][k]:.0%}")

        with open(results_dir / f"{name}.json", "w") as f:
            json.dump(results, f, indent=2)
        all_results[name] = results["summary"]

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("LR SCHEDULE SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Schedule':16s} {'Retrieval':>10} {'Time':>8}")
    print(f"  {'-'*36}")
    for name, summary in all_results.items():
        print(f"  {name:16s} {summary['weights_rate']:>10.0%} "
              f"{summary['elapsed_s']:>7.0f}s")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
