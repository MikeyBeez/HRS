"""Phase 15: Cumulative forgetting check for L4-5 LoRA scheduled TTT.

Absorbs 20 passkeys sequentially using the phase14 winning config
(rank 512, layers 4-5 attn+peer_ffn, 100 steps, 3e-4 -> 1e-4) WITHOUT
resetting the LoRA adapter between passages. Measures WikiText val PPL
after each absorption.

The base weights are frozen so they can't drift. The question is whether
the *active* LoRA adapter, after absorbing 20 passkeys, leaks into general
WikiText predictions.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase15_l45_forgetting.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase12_forgetting import compute_val_perplexity
from experiments.identity_ae.phase13_lora_scheduled import run_ttt_lora_scheduled
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.lora_wrapper import apply_lora


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase15")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Winning config from phase14
    RANK = 512
    N_STEPS = 100
    HIGH_LR = 3e-4
    BASE_LR = 1e-4
    N_PASSAGES = 20

    # Load model + LoRA + val loader
    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params: {n_lora:,} on {len(L45_TARGETS)} layers")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    tests = generate_passkeys(50)[:N_PASSAGES]

    val_ppl_baseline = compute_val_perplexity(model, loaders["validation"], device)
    print(f"Baseline WikiText val PPL: {val_ppl_baseline:.3f}")
    print(f"Schedule: {N_STEPS} steps, {HIGH_LR:.1e} -> {BASE_LR:.1e}")
    print(f"Absorbing {N_PASSAGES} passkeys cumulatively (no reset between)")
    print()

    history = [{"i": 0, "val_ppl": val_ppl_baseline, "delta_pct": 0.0,
                "passkey_found": None, "passkey_id": None}]

    n_found = 0
    t0 = time.time()
    for i, test in enumerate(tests):
        run_ttt_lora_scheduled(model, test["passage"], tokenizer, device,
                                N_STEPS, HIGH_LR, BASE_LR)

        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        found = check_passkey(gen, test["passkey"])
        if found:
            n_found += 1

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

    final = history[-1]
    per_passage = final["delta_pct"] / N_PASSAGES

    print()
    print(f"{'='*60}")
    print(f"L45 LORA CUMULATIVE FORGETTING ({N_PASSAGES} passages)")
    print(f"{'='*60}")
    print(f"  Retrieval (cumulative): {n_found}/{N_PASSAGES} ({n_found/N_PASSAGES:.0%})")
    print(f"  Val PPL:                {val_ppl_baseline:.3f} -> {final['val_ppl']:.3f}")
    print(f"  Total degradation:      {final['delta_pct']:+.2f}%")
    print(f"  Per-passage average:    {per_passage:+.3f}%")
    print(f"  Phase 12 (full-model):  +44.12% per passage")

    summary = {
        "config": {"rank": RANK, "n_steps": N_STEPS, "high_lr": HIGH_LR,
                   "base_lr": BASE_LR, "targets": L45_TARGETS},
        "n_passages": N_PASSAGES,
        "retrieval": n_found / N_PASSAGES,
        "val_ppl_baseline": val_ppl_baseline,
        "val_ppl_final": final["val_ppl"],
        "total_delta_pct": final["delta_pct"],
        "per_passage_delta_pct": per_passage,
        "history": history,
    }
    with open(results_dir / "l45_r512_cumulative.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
