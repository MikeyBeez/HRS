"""Phase 38a: Per-passage adapter rank sweep.

The current adapter library uses rank 512 across 8 target modules on layers
4-5, totaling ~10.5M parameters per adapter. A passkey passage is 50–80
tokens; the structured content has at most a handful of degrees of freedom
(entity name, passkey value, relation). We are using ~150,000 parameters
per token of absorbed content. The rank-1024 negative result in Phase 20
shows we're past the useful capacity, but we never swept *down* from 512.
This script does the downward sweep.

For each rank R in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}:
  - Reload the base model fresh (LoRA wrapping changes structure, so we
    cannot reuse a model wrapped at a different rank without surgery)
  - Apply LoRA with rank R, alpha=2*R (preserves the alpha/rank ratio
    used in the rest of the paper)
  - For each of the 20 stratified passkey passages:
      - Reset the LoRA to zero
      - Train for 150 steps with the Phase 24 winner schedule (3e-4 → 1e-4)
      - Generate from the absorption prompt (same-prompt retrieval)
      - Score against the expected passkey

We report per-rank: trainable parameter count, total training time, total
retrieval (out of 20), per-type breakdown, and storage size (bytes).

This is the *upper bound* test — same prompt as absorption, no library, no
routing. If a rank fails here, no downstream architecture can save it. The
follow-up (Phase 38b) is to take the smallest rank that holds 100% same-
prompt and run the held-out paraphrase test against it.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase38_rank_sweep.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase21_per_passage_adapters import train_passage_adapter
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)


N_STEPS = 150
GEN_TOKENS = 50

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def per_type_zeros():
    return {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}


def state_dict_bytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase38")
    results_dir.mkdir(parents=True, exist_ok=True)

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")
    print(f"Steps per adapter: {N_STEPS}")
    print(f"Ranks to sweep: {RANKS}\n")

    summary_rows = []

    for rank in RANKS:
        # ---- Fresh model + LoRA structure ----
        model, _ = load_model(device)
        n_lora = apply_lora(model, rank=rank, alpha=rank * 2,
                            target_modules=L45_TARGETS)

        # Sanity-snapshot a fresh state dict to know its byte size
        reset_lora_to_zero(model)
        sample_sd = {k: v.detach().cpu().clone()
                     for k, v in get_lora_state_dict(model).items()}
        bytes_per_adapter = state_dict_bytes(sample_sd)
        print(f"{'='*64}")
        print(f"RANK {rank:4d}  |  {n_lora:>11,} params  |  "
              f"{bytes_per_adapter/1e6:6.2f} MB / adapter")
        print(f"{'='*64}")

        # ---- Per-passage absorption + same-prompt retrieval ----
        n_correct = 0
        per_type = per_type_zeros()
        per_passage = []
        t0 = time.time()

        for ti, test in enumerate(tests):
            reset_lora_to_zero(model)
            train_passage_adapter(model, test["passage"], tokenizer, device,
                                  n_steps=N_STEPS,
                                  high_lr=HIGH_LR, base_lr=BASE_LR)
            gen = generate_greedy(model, test["prompt"], tokenizer, device,
                                  GEN_TOKENS)
            hit = check_passkey(gen, test["passkey"])
            if hit:
                n_correct += 1
                per_type[test["type"]] += 1
            per_passage.append({
                "id": test["id"], "type": test["type"],
                "passkey": test["passkey"], "hit": hit,
                "gen": gen[:120],
            })

        elapsed = time.time() - t0
        per_passage_time = elapsed / len(tests)

        print(f"  retrieval: {n_correct}/20 ({n_correct/20:.0%})  "
              f"per-type: num={per_type['numeric']}/5 ent={per_type['entity']}/5 "
              f"tech={per_type['technical']}/5 fact={per_type['fact']}/5")
        print(f"  total training time: {elapsed:.1f}s  "
              f"({per_passage_time*1000:.0f} ms/passage)\n")

        summary_rows.append({
            "rank": rank,
            "alpha": rank * 2,
            "params_per_adapter": n_lora,
            "bytes_per_adapter": bytes_per_adapter,
            "n_correct": n_correct,
            "retrieval_rate": n_correct / 20,
            "per_type": dict(per_type),
            "total_time_s": elapsed,
            "ms_per_passage": per_passage_time * 1000,
            "per_passage": per_passage,
        })

        # Free GPU memory before next iteration
        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Summary table
    # ============================================================
    print(f"{'='*72}")
    print("PHASE 38a SUMMARY: rank sweep, single-passage retrieval (n=20)")
    print(f"{'='*72}")
    print(f"  {'rank':>5} {'params':>13} {'MB/ad':>8} "
          f"{'retr':>8} {'num':>5} {'ent':>5} {'tech':>5} {'fact':>5} {'ms/pa':>8}")
    print(f"  {'-'*5} {'-'*13} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*8}")
    for r in summary_rows:
        pt = r["per_type"]
        print(f"  {r['rank']:>5} {r['params_per_adapter']:>13,} "
              f"{r['bytes_per_adapter']/1e6:>7.2f} "
              f"{r['n_correct']:>3}/20 "
              f"{pt['numeric']:>3}/5 {pt['entity']:>3}/5 "
              f"{pt['technical']:>3}/5 {pt['fact']:>3}/5 "
              f"{r['ms_per_passage']:>7.0f}")

    # Find the smallest rank that holds 100%
    perfect = [r for r in summary_rows if r["n_correct"] == 20]
    if perfect:
        smallest = min(perfect, key=lambda r: r["rank"])
        baseline_512 = next((r for r in summary_rows if r["rank"] == 512), None)
        print(f"\n  Smallest rank with 100% retrieval: rank {smallest['rank']} "
              f"({smallest['params_per_adapter']:,} params, "
              f"{smallest['bytes_per_adapter']/1e6:.2f} MB / adapter)")
        if baseline_512:
            ratio = baseline_512["params_per_adapter"] / smallest["params_per_adapter"]
            print(f"  Compression vs rank-512 baseline: {ratio:.0f}× smaller")
    else:
        # Find the best rank if none reach 100
        best = max(summary_rows, key=lambda r: r["n_correct"])
        print(f"\n  No rank achieved 100%. Best: rank {best['rank']} "
              f"({best['n_correct']}/20).")

    # ============================================================
    # Save
    # ============================================================
    with open(results_dir / "rank_sweep.json", "w") as f:
        json.dump({
            "n_steps": N_STEPS,
            "ranks": RANKS,
            "rows": summary_rows,
        }, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
