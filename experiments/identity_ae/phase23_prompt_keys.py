"""Phase 23: Per-passage adapter library keyed on PROMPT engrams.

Phases 21-22 stored library keys derived from passages but compared them
against query engrams derived from prompts — completely different text
distributions. The fix: at absorption time, train the adapter on the passage
but store the *prompt's* engram as the key. At retrieval time, the incoming
prompt's engram matches against stored prompt engrams. Same distribution.

One pass. Compute prompt engram, match to library, load adapter, generate.

Tests a few key sources for ablation but the headline config is
L5_mean_cosine — best in phase 22.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase23_prompt_keys.py
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, N_STEPS, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase21_per_passage_adapters import train_passage_adapter
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, make_key, l2_match, cosine_match, reset_lora_to_zero,
    stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase23")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA structure applied: {n_lora:,} params per adapter")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL (zero adapter): {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION: train adapter on PASSAGE, store key from PROMPT
    # ============================================================
    print(f"{'='*60}")
    print("ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []  # list of (prompt_ids, lora_state_dict_cpu, test_meta)

    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_passage_adapter(model, test["passage"], tokenizer, device,
                               n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)

        # Store PROMPT tokens (not passage) for key computation
        prompt_ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append((prompt_t, sd, dict(test)))

        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    print(f"  Library: {len(library)} adapters")

    # ============================================================
    # KEY EXTRACTION: under base model, compute prompt keys for both
    # library entries and incoming queries (same prompt for each test).
    # ============================================================
    sources = ["L5_mean", "L5_last", "L3_mean", "L3_last"]
    metrics = ["cosine", "L2"]

    reset_lora_to_zero(model)

    print(f"\n{'='*60}")
    print("KEY EXTRACTION (base model, prompts only)")
    print(f"{'='*60}")

    library_keys = {src: [] for src in sources}
    for prompt_t, _, _ in library:
        for src in sources:
            library_keys[src].append(make_key(model, prompt_t, src))

    # Query keys are computed from the same prompts at retrieval time —
    # for this benchmark, the test prompts equal the library prompts, so
    # under the base model, query and library keys are identical. The
    # interesting failure mode would be paraphrased queries — for now we
    # measure the upper bound: same prompt, same model, same key.
    query_keys = library_keys  # by construction

    # ============================================================
    # ROUTING + RETRIEVAL: one pass each
    # ============================================================
    print(f"\n{'='*60}")
    print("ROUTING + RETRIEVAL (one pass)")
    print(f"{'='*60}")

    all_results = {}
    for src in sources:
        for metric in metrics:
            label = f"{src}_{metric}"
            n_correct = 0
            n_retrieved = 0
            routed_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
            found_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
            per_test = []

            for i, test in enumerate(tests):
                reset_lora_to_zero(model)
                q = query_keys[src][i]
                if metric == "L2":
                    best_idx, score = l2_match(q, library_keys[src])
                else:
                    best_idx, score = cosine_match(q, library_keys[src])

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
                    "id": test["id"], "type": test["type"],
                    "best_match_idx": best_idx, "score": score,
                    "correct_route": correct, "found": found,
                })

            print(f"  {label:18s}  routing={n_correct:2d}/20 ({n_correct/20:.0%})  "
                  f"retrieval={n_retrieved:2d}/20 ({n_retrieved/20:.0%})")
            for ptype in ["numeric", "entity", "technical", "fact"]:
                print(f"      {ptype}: routed {routed_by_type[ptype]}/5  "
                      f"retrieved {found_by_type[ptype]}/5")

            all_results[label] = {
                "routing": n_correct / 20,
                "retrieval": n_retrieved / 20,
                "routed_by_type": {k: v / 5 for k, v in routed_by_type.items()},
                "found_by_type": {k: v / 5 for k, v in found_by_type.items()},
                "per_test": per_test,
            }

    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n{'='*60}")
    print("PHASE 23 SUMMARY (prompt-keyed library)")
    print(f"{'='*60}")
    print(f"  {'Config':22s} {'Routing':>10} {'Retrieval':>10}")
    print(f"  {'-'*44}")
    for label, r in all_results.items():
        print(f"  {label:22s} {r['routing']:>10.0%} {r['retrieval']:>10.0%}")
    print(f"\n  Val PPL drift (zero adapter): {drift:+.3f}%")

    with open(results_dir / "prompt_keys.json", "w") as f:
        json.dump({"baseline_ppl": baseline_ppl, "drift_pct": drift,
                    "results": all_results}, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
