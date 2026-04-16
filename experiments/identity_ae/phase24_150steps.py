"""Phase 24: Phase 23 with 150 training steps per adapter.

Phase 23 hit 100% routing but 90% retrieval — two adapters didn't fully
encode their passkey at 100 steps. Bump to 150 steps to clear the stragglers.

Single config: L5_mean_cosine (all configs tied in phase 23).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase24_150steps.py
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
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase21_per_passage_adapters import train_passage_adapter
from experiments.identity_ae.phase22_engram_key import (
    make_key, cosine_match, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


N_STEPS = 150
SOURCE = "L5_mean"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase24")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Steps per adapter: {N_STEPS}")
    print(f"Key source: {SOURCE}")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # Absorption
    print(f"{'='*60}")
    print("ABSORPTION PHASE (150 steps each)")
    print(f"{'='*60}")
    library = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_passage_adapter(model, test["passage"], tokenizer, device,
                               n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        prompt_ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append((prompt_t, sd, dict(test)))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")
    print(f"  Library: {len(library)} adapters  ({time.time()-t0:.0f}s total)")

    # Key extraction (base model, prompts)
    reset_lora_to_zero(model)
    library_keys = [make_key(model, prompt_t, SOURCE) for prompt_t, _, _ in library]

    # Routing + retrieval (one pass)
    print(f"\n{'='*60}")
    print("ROUTING + RETRIEVAL")
    print(f"{'='*60}")

    n_correct = 0
    n_retrieved = 0
    routed_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    found_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    per_test = []

    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        q = library_keys[i]  # query = same prompt under base model
        best_idx, score = cosine_match(q, library_keys)

        correct = (best_idx == i)
        if correct:
            n_correct += 1
            routed_by_type[test["type"]] += 1

        _, sd, _ = library[best_idx]
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)

        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        found = check_passkey(gen, test["passkey"])
        if found:
            n_retrieved += 1
            found_by_type[test["type"]] += 1

        per_test.append({
            "id": test["id"], "type": test["type"], "passkey": test["passkey"],
            "best_match_idx": best_idx, "score": score,
            "correct_route": correct, "found": found, "gen": gen[:120],
        })

    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n{'='*60}")
    print(f"PHASE 24 SUMMARY (prompt-keyed, 150 steps)")
    print(f"{'='*60}")
    print(f"  Routing:     {n_correct}/20 ({n_correct/20:.0%})")
    print(f"  Retrieval:   {n_retrieved}/20 ({n_retrieved/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: routed {routed_by_type[ptype]}/5  retrieved {found_by_type[ptype]}/5")
    print(f"  Val PPL drift: {drift:+.3f}%")

    # If any retrieval misses, show the failing cases
    misses = [t for t in per_test if not t["found"]]
    if misses:
        print(f"\n  {len(misses)} retrieval misses:")
        for m in misses:
            print(f"    [{m['type']}] passkey={m['passkey']!r}  gen={m['gen']!r}")

    summary = {
        "n_steps": N_STEPS, "source": SOURCE,
        "routing": n_correct / 20, "retrieval": n_retrieved / 20,
        "routed_by_type": {k: v / 5 for k, v in routed_by_type.items()},
        "found_by_type": {k: v / 5 for k, v in found_by_type.items()},
        "baseline_ppl": baseline_ppl, "drift_pct": drift,
        "per_test": per_test,
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
