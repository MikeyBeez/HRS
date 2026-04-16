"""Phase 27: Held-out paraphrase generalization test.

Phase 26 hit 100%/100% but the test paraphrases were the same templates
used at training. This script tests whether the engram routing actually
generalizes: train with the SAME 3 paraphrases as phase 26, but query with
3 entirely new templates the system never saw at absorption.

The training distribution (slots 0-2) and the test distribution (slots 3-5)
are disjoint surface forms of the same underlying questions.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase27_held_out.py
"""

import json
import random
import re
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
from experiments.identity_ae.phase22_engram_key import (
    make_key, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


N_STEPS = 150
SOURCE = "L5_mean"


def held_out_paraphrase(test):
    """Three entirely new templates, disjoint from train_paraphrase()."""
    p = test["prompt"]
    t = test["type"]

    if t == "numeric":
        m = re.search(r"the (\w+) facility", p)
        fac = m.group(1)
        return [
            f"I need the access code for the {fac} facility.",
            f"Look up the {fac} facility's access code.",
            f"Access code, {fac} facility?",
        ]

    if t == "entity":
        m = re.search(r"When did (.+?) make", p)
        name = m.group(1)
        return [
            f"I'm trying to find when {name} made the discovery.",
            f"Date of breakthrough by {name}?",
            f"Tell me the date {name} made the breakthrough.",
        ]

    if t == "technical":
        m = re.search(r"the (\w+) in (\w+)", p)
        thing, unit = m.group(1), m.group(2)
        return [
            f"I need the {thing}'s critical threshold in {unit}.",
            f"Look up the critical {unit} for the {thing}.",
            f"Critical {thing} value, in {unit}?",
        ]

    if t == "fact":
        m = re.search(r"the (\w+) Protocol", p)
        proto = m.group(1)
        return [
            f"I'm looking up the signatory count for the {proto} Protocol.",
            f"Required signatories on the {proto} Protocol?",
            f"Tell me the signatory requirement of the {proto} Protocol.",
        ]

    raise ValueError(t)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase27")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Steps per adapter: {N_STEPS}, source: {SOURCE}")
    print("Train paraphrases (slots 0-2): used at absorption")
    print("Held-out paraphrases (slots 3-5): used ONLY at retrieval\n")

    tests = stratified_tests()

    # Sanity print for one test
    print(f"Sample for: '{tests[0]['prompt']}'")
    print("  Train paraphrases (used at absorption):")
    for p in [tests[0]["prompt"]] + train_paraphrase(tests[0]):
        print(f"    - {p}")
    print("  Held-out paraphrases (used at query):")
    for p in held_out_paraphrase(tests[0]):
        print(f"    - {p}")
    print()

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION (same as phase 26: train + key on slots 0-2)
    # ============================================================
    print(f"{'='*60}")
    print("ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []

    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)

        train_prompts = [test["prompt"]] + train_paraphrase(test)  # slots 0..3 (4 total)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)

        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}

        reset_lora_to_zero(model)
        keys = []
        for p in train_prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys.append(make_key(model, ids_t, SOURCE))

        library.append({"keys": keys, "sd": sd, "meta": dict(test)})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    print(f"  Library: {len(library)} adapters × 4 train-keys each")

    # ============================================================
    # HELD-OUT RETRIEVAL
    # ============================================================
    print(f"\n{'='*60}")
    print("HELD-OUT PARAPHRASE RETRIEVAL")
    print(f"{'='*60}")

    def route(query_key):
        best_a, best_score = -1, -2.0
        q_n = query_key / (query_key.norm() + 1e-8)
        for ai, entry in enumerate(library):
            for k in entry["keys"]:
                k_n = k / (k.norm() + 1e-8)
                sim = float(torch.dot(q_n, k_n))
                if sim > best_score:
                    best_score = sim
                    best_a = ai
        return best_a, best_score

    per_slot = [{"routing": 0, "retrieval": 0, "by_type": {}} for _ in range(3)]
    for slot in per_slot:
        for ptype in ["numeric", "entity", "technical", "fact"]:
            slot["by_type"][ptype] = {"routed": 0, "retrieved": 0}

    detailed = []
    for i, test in enumerate(tests):
        ho_paras = held_out_paraphrase(test)
        for slot_idx, para in enumerate(ho_paras):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key(model, ids_t, SOURCE)
            best_a, score = route(q)

            correct = (best_a == i)
            if correct:
                per_slot[slot_idx]["routing"] += 1
                per_slot[slot_idx]["by_type"][test["type"]]["routed"] += 1

            sd_gpu = {k: v.to(device) for k, v in library[best_a]["sd"].items()}
            load_lora_state_dict(model, sd_gpu)

            gen = generate_greedy(model, para, tokenizer, device, 50)
            found = check_passkey(gen, test["passkey"])
            if found:
                per_slot[slot_idx]["retrieval"] += 1
                per_slot[slot_idx]["by_type"][test["type"]]["retrieved"] += 1

            detailed.append({
                "test_id": test["id"], "type": test["type"], "slot": slot_idx,
                "paraphrase": para, "passkey": test["passkey"],
                "best_adapter": best_a, "correct_route": correct,
                "found": found, "gen": gen[:120],
            })

    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n{'='*60}")
    print(f"PHASE 27 SUMMARY (HELD-OUT paraphrase generalization)")
    print(f"{'='*60}")
    print(f"  20 originals × 3 held-out paraphrases each = 60 query trials")
    print()
    print(f"  {'Slot':6s} {'Routing':>10} {'Retrieval':>10}")
    print(f"  {'-'*28}")
    for i, slot in enumerate(per_slot):
        print(f"  {i+4:>4d}   {slot['routing']/20:>10.0%} {slot['retrieval']/20:>10.0%}")
    total_r = sum(s["routing"] for s in per_slot)
    total_ret = sum(s["retrieval"] for s in per_slot)
    print(f"  {'-'*28}")
    print(f"  Total  {total_r:>4d}/60   {total_ret:>4d}/60")
    print(f"  Avg    {total_r/60:>10.0%} {total_ret/60:>10.0%}")

    print(f"\n  Per-type retrieval (across 3 held-out paraphrases):")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        ret = sum(s["by_type"][ptype]["retrieved"] for s in per_slot)
        rou = sum(s["by_type"][ptype]["routed"] for s in per_slot)
        print(f"    {ptype:9s}: routed {rou:2d}/15  retrieved {ret:2d}/15")

    print(f"\n  Val PPL drift: {drift:+.3f}%")

    misses = [d for d in detailed if not d["found"]]
    if misses:
        print(f"\n  {len(misses)}/60 retrieval misses (showing first 8):")
        for m in misses[:8]:
            print(f"    [{m['type']}] '{m['paraphrase']}'")
            print(f"       routed_to={m['best_adapter']} (correct={m['correct_route']})")
            print(f"       gen={m['gen']!r}")

    summary = {
        "n_steps": N_STEPS, "source": SOURCE,
        "held_out": True,
        "n_originals": 20, "n_paraphrases_per": 3, "n_trials": 60,
        "per_slot": per_slot,
        "total_routing_pct": total_r / 60,
        "total_retrieval_pct": total_ret / 60,
        "drift_pct": drift,
        "detailed": detailed,
    }
    with open(results_dir / "held_out.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
