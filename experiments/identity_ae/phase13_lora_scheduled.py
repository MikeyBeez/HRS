"""Phase 13: LR-scheduled TTT on LoRA adapters only.

Phase 11 showed warmup-then-decay hits 100% retrieval at 40 steps —
but on the full model, costing ~9% val PPL per passage.

LoRA freezes the base weights, so the same aggressive schedule applied to
the adapter should give us the speedup with zero base forgetting.

Phase 9 baseline: rank 128, last-layer, constant lr=1e-4, 100 steps -> 62%.

This script tries scheduled LR on LoRA configs to find one that hits ~100%.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase13_lora_scheduled.py
"""

import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.lora_wrapper import apply_lora, reset_lora


def run_ttt_lora_scheduled(model, passage, tokenizer, device, n_steps, high_lr, base_lr):
    """Two-phase LR via StepLR, on LoRA params only."""
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()


@torch.no_grad()
def compute_val_perplexity(model, val_loader, device, max_batches=20):
    model.eval()
    total_loss = 0
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        out = model(x, step=0)
        B, T, V = out.logits.shape
        total_loss += F.cross_entropy(out.logits.reshape(B*T, V), y.reshape(B*T)).item()
        n += 1
    return math.exp(min(total_loss / n, 20))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase13")
    results_dir.mkdir(parents=True, exist_ok=True)

    tests = generate_passkeys(50)

    # Configs: (name, rank, target_modules, n_steps, high_lr, base_lr)
    # Start with last-layer-only (phase9 baseline) then escalate.
    configs = [
        ("lora128_last_50_3x",  128, ['blocks.5.attn.qkv', 'blocks.5.attn.out_proj'],
         50, 3e-4, 1e-4),
        ("lora128_last_60_5x",  128, ['blocks.5.attn.qkv', 'blocks.5.attn.out_proj'],
         60, 5e-4, 1e-4),
        ("lora128_all_50_3x",   128, ['qkv', 'out_proj'],
         50, 3e-4, 1e-4),
        ("lora256_all_50_3x",   256, ['qkv', 'out_proj'],
         50, 3e-4, 1e-4),
    ]

    all_results = {}

    for name, rank, target_modules, n_steps, high_lr, base_lr in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {name}")
        print(f"  rank={rank}  targets={target_modules}")
        print(f"  steps={n_steps}  {high_lr:.1e} -> {base_lr:.1e}")
        print(f"{'='*60}")

        # Fresh model + LoRA
        model, cfg = load_model(device)
        n_lora = apply_lora(model, rank=rank, alpha=rank * 2, target_modules=target_modules)
        print(f"  LoRA params: {n_lora:,}")

        base_state = copy.deepcopy(model.state_dict())
        lora_param_names = [n for n, p in model.named_parameters() if 'lora_' in n]

        results = {"name": name, "rank": rank, "target_modules": target_modules,
                   "n_steps": n_steps, "high_lr": high_lr, "base_lr": base_lr,
                   "n_lora_params": n_lora, "tests": []}
        n_weights_found = 0

        t0 = time.time()
        for ti, test in enumerate(tests):
            # Reset to baseline (full state restore — also resets LoRA)
            model.load_state_dict(base_state)
            reset_lora(model)

            run_ttt_lora_scheduled(model, test["passage"], tokenizer, device,
                                   n_steps, high_lr, base_lr)

            gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
            found = check_passkey(gen, test["passkey"])
            if found:
                n_weights_found += 1

            results["tests"].append({
                "id": test["id"],
                "type": test["type"],
                "passkey": test["passkey"],
                "weights_found": found,
                "gen_weights": gen[:100],
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
    print("LORA SCHEDULED SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':24s} {'Retrieval':>10} {'Time':>8}")
    print(f"  {'-'*44}")
    for name, summary in all_results.items():
        print(f"  {name:24s} {summary['weights_rate']:>10.0%} "
              f"{summary['elapsed_s']:>7.0f}s")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
