"""Phase 12: Forgetting cost of the LR-scheduled TTT.

The phase11 result showed 100% retrieval at 40 steps with a 1.5e-4 -> 5e-5
schedule (5x speedup). But the high-LR warmup may cost more catastrophic
forgetting per passage than the constant lr=5e-5 baseline.

This script absorbs N passkeys with sched_40_3x and measures WikiText val PPL
after every passage. Compare to the paper's ~1-2% per passage at constant 5e-5.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase12_forgetting.py
"""

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
from experiments.identity_ae.phase11_lr_schedule import run_ttt_scheduled


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

    results_dir = Path("results/identity_ae/phase12")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model + val loader
    model, cfg = load_model(device)
    for p in model.parameters():
        p.requires_grad = True

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    # Schedule under test
    N_STEPS = 40
    HIGH_LR = 1.5e-4
    BASE_LR = 5e-5
    N_PASSAGES = 10  # absorb 10 passkeys cumulatively

    tests = generate_passkeys(50)[:N_PASSAGES]

    val_ppl_baseline = compute_val_perplexity(model, loaders["validation"], device)
    print(f"Baseline WikiText val PPL: {val_ppl_baseline:.3f}")
    print(f"Schedule: {N_STEPS} steps, {HIGH_LR:.1e} -> {BASE_LR:.1e}")
    print(f"Absorbing {N_PASSAGES} passkeys cumulatively (no model reload)")
    print()

    history = [{"i": 0, "val_ppl": val_ppl_baseline, "delta_pct": 0.0,
                "passkey_found": None, "passkey_id": None}]

    t0 = time.time()
    for i, test in enumerate(tests):
        run_ttt_scheduled(model, test["passage"], tokenizer, device,
                          N_STEPS, HIGH_LR, BASE_LR)

        # Did we learn this passkey?
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        found = check_passkey(gen, test["passkey"])

        # Val PPL after absorption
        ppl = compute_val_perplexity(model, loaders["validation"], device)
        delta = (ppl - val_ppl_baseline) / val_ppl_baseline * 100

        history.append({
            "i": i + 1,
            "val_ppl": ppl,
            "delta_pct": delta,
            "passkey_found": found,
            "passkey_id": test["id"],
            "passkey_type": test["type"],
        })

        elapsed = time.time() - t0
        flag = "OK" if found else "MISS"
        print(f"  [{i+1:2d}/{N_PASSAGES}] {test['type']:9s} {flag:4s}  "
              f"val_ppl={ppl:.3f} ({delta:+.2f}%)  ({elapsed:.0f}s)")

    # Summary
    final = history[-1]
    n_found = sum(1 for h in history[1:] if h["passkey_found"])
    per_passage = final["delta_pct"] / N_PASSAGES

    print()
    print(f"{'='*60}")
    print(f"FORGETTING SUMMARY (sched_40_3x, {N_PASSAGES} passages)")
    print(f"{'='*60}")
    print(f"  Retrieval:           {n_found}/{N_PASSAGES} ({n_found/N_PASSAGES:.0%})")
    print(f"  Val PPL:             {val_ppl_baseline:.3f} -> {final['val_ppl']:.3f}")
    print(f"  Total degradation:   {final['delta_pct']:+.2f}%")
    print(f"  Per-passage average: {per_passage:+.2f}%")
    print(f"  Paper baseline:      ~1-2% per passage at constant lr=5e-5")

    summary = {
        "schedule": {"n_steps": N_STEPS, "high_lr": HIGH_LR, "base_lr": BASE_LR},
        "n_passages": N_PASSAGES,
        "retrieval": n_found / N_PASSAGES,
        "val_ppl_baseline": val_ppl_baseline,
        "val_ppl_final": final["val_ppl"],
        "total_delta_pct": final["delta_pct"],
        "per_passage_delta_pct": per_passage,
        "history": history,
    }
    with open(results_dir / "sched_40_3x_forgetting.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
