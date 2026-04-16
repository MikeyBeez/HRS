"""Phase 38b: Held-out paraphrase test at rank 128.

Phase 38a found that rank 128 is the smallest that holds 100% same-prompt
retrieval — a 4× reduction from the rank 512 used in the rest of the paper.
This script asks the binding follow-up question: does the *generalization*
result also survive at rank 128, or was the rank-512 basin doing useful
work for paraphrase coverage?

We replay the Phase 26 (training-distribution paraphrase) and Phase 31
(held-out paraphrase, nonstop_mean pooling) protocols at rank 128 with no
other changes. The library is built the same way (multi-prompt training,
multi-key extraction); the held-out test uses the same disjoint paraphrase
templates and the same nonstop_mean pooling that won Phase 31.

Numbers to compare:
  Training-distribution paraphrases (60 trials):
    rank 512 (Phase 26):       60/60 (100%) routing / 60/60 (100%) retrieval
    rank 128 (this script):    ?
  Held-out paraphrases (60 trials, nonstop_mean):
    rank 512 (Phase 31):       46/60 (77%) routing / 45/60 (75%) retrieval
    rank 128 (this script):    ?

If both numbers hold, the paper picks up a free 4× storage reduction
(10.5 MB → 2.6 MB per adapter) with no quality cost. If they drop, we
learn how much of the rank-512 result was generalization that the smaller
adapter can't reproduce.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase38b_rank128_heldout.py
"""

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import (
    make_key_weighted, cosine,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128                # Phase 38a smallest 100% rank
ALPHA = RANK * 2          # preserve alpha/rank ratio
N_STEPS = 150
LAYER = 5
STRATEGY = "nonstop_mean"  # Phase 31 winner


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase38b")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}  (rank {RANK}, alpha {ALPHA})")
    print(f"Steps per adapter: {N_STEPS}, layer: {LAYER}, pooling: {STRATEGY}\n")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION (multi-prompt training, identical to Phase 26/31)
    # ============================================================
    print(f"{'='*60}")
    print("ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append({"sd": sd, "test": dict(test), "train_prompts": train_prompts})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    # ============================================================
    # KEY EXTRACTION (nonstop_mean only)
    # ============================================================
    reset_lora_to_zero(model)
    print(f"\nExtracting keys with strategy={STRATEGY}")
    library_keys = []
    for entry in library:
        keys_for_entry = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_for_entry.append(make_key_weighted(model, tokenizer, ids_t, STRATEGY))
        library_keys.append(keys_for_entry)

    def route(query_key):
        best_a, best_score = -1, -2.0
        for ai, keys in enumerate(library_keys):
            for k in keys:
                s = cosine(query_key, k)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    # ============================================================
    # PART 1: Same-prompt retrieval (upper bound sanity)
    # ============================================================
    print(f"\n{'='*60}")
    print("PART 1: SAME-PROMPT RETRIEVAL")
    print(f"{'='*60}")
    n_routed_sp, n_retr_sp = 0, 0
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
        best_a = route(q)
        if best_a == i:
            n_routed_sp += 1

        sd = library[best_a]["sd"]
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        if check_passkey(gen, test["passkey"]):
            n_retr_sp += 1

    print(f"  same-prompt: routing {n_routed_sp}/20 ({n_routed_sp/20:.0%})  "
          f"retrieval {n_retr_sp}/20 ({n_retr_sp/20:.0%})")

    # ============================================================
    # PART 2: Training-distribution paraphrase (Phase 26 protocol)
    # ============================================================
    print(f"\n{'='*60}")
    print("PART 2: TRAINING-DISTRIBUTION PARAPHRASE (60 trials)")
    print(f"{'='*60}")
    n_routed_td, n_retr_td = 0, 0
    per_type_td_routed = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    per_type_td_retr   = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    for i, test in enumerate(tests):
        # train_paraphrase returns the 3 training paraphrase templates
        for para in train_paraphrase(test):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
            best_a = route(q)
            if best_a == i:
                n_routed_td += 1
                per_type_td_routed[test["type"]] += 1

            sd = library[best_a]["sd"]
            sd_gpu = {k: v.to(device) for k, v in sd.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, para, tokenizer, device, 50)
            if check_passkey(gen, test["passkey"]):
                n_retr_td += 1
                per_type_td_retr[test["type"]] += 1

    print(f"  training-distribution: routing {n_routed_td}/60 ({n_routed_td/60:.0%})  "
          f"retrieval {n_retr_td}/60 ({n_retr_td/60:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype:9s}: routed {per_type_td_routed[ptype]:2d}/15  "
              f"retrieved {per_type_td_retr[ptype]:2d}/15")

    # ============================================================
    # PART 3: Held-out paraphrase (Phase 31 protocol)
    # ============================================================
    print(f"\n{'='*60}")
    print("PART 3: HELD-OUT PARAPHRASE (60 trials, nonstop_mean)")
    print(f"{'='*60}")
    n_routed_ho, n_retr_ho = 0, 0
    per_type_ho_routed = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    per_type_ho_retr   = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    per_slot_ho_routed = [0, 0, 0]
    per_slot_ho_retr   = [0, 0, 0]
    for i, test in enumerate(tests):
        ho_paras = held_out_paraphrase(test)
        for slot_idx, para in enumerate(ho_paras):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key_weighted(model, tokenizer, ids_t, STRATEGY)
            best_a = route(q)
            if best_a == i:
                n_routed_ho += 1
                per_type_ho_routed[test["type"]] += 1
                per_slot_ho_routed[slot_idx] += 1

            sd = library[best_a]["sd"]
            sd_gpu = {k: v.to(device) for k, v in sd.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, para, tokenizer, device, 50)
            if check_passkey(gen, test["passkey"]):
                n_retr_ho += 1
                per_type_ho_retr[test["type"]] += 1
                per_slot_ho_retr[slot_idx] += 1

    print(f"  held-out: routing {n_routed_ho}/60 ({n_routed_ho/60:.0%})  "
          f"retrieval {n_retr_ho}/60 ({n_retr_ho/60:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype:9s}: routed {per_type_ho_routed[ptype]:2d}/15  "
              f"retrieved {per_type_ho_retr[ptype]:2d}/15")
    print(f"    slots:   routed {per_slot_ho_routed}  retrieved {per_slot_ho_retr}")

    # ============================================================
    # DRIFT
    # ============================================================
    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    # ============================================================
    # SUMMARY: side-by-side with Phase 26 / Phase 31 (rank 512)
    # ============================================================
    print(f"\n{'='*68}")
    print(f"PHASE 38b SUMMARY: rank {RANK} vs rank 512 (paper baseline)")
    print(f"{'='*68}")
    print(f"  Adapter size: {n_lora:,} params  (vs 10,485,760 at rank 512, "
          f"{10_485_760/n_lora:.1f}× smaller)\n")
    print(f"  {'Test':32s} {'rank 512':>14}  {'rank 128':>14}")
    print(f"  {'-'*32} {'-'*14}  {'-'*14}")
    print(f"  {'Same-prompt routing':32s} {'20/20 (100%)':>14}  "
          f"{f'{n_routed_sp}/20 ({n_routed_sp/20:.0%})':>14}")
    print(f"  {'Same-prompt retrieval':32s} {'20/20 (100%)':>14}  "
          f"{f'{n_retr_sp}/20 ({n_retr_sp/20:.0%})':>14}")
    print(f"  {'Training-distrib routing':32s} {'60/60 (100%)':>14}  "
          f"{f'{n_routed_td}/60 ({n_routed_td/60:.0%})':>14}")
    print(f"  {'Training-distrib retrieval':32s} {'60/60 (100%)':>14}  "
          f"{f'{n_retr_td}/60 ({n_retr_td/60:.0%})':>14}")
    print(f"  {'Held-out routing (nonstop)':32s} {'46/60 (77%)':>14}  "
          f"{f'{n_routed_ho}/60 ({n_routed_ho/60:.0%})':>14}")
    print(f"  {'Held-out retrieval (nonstop)':32s} {'45/60 (75%)':>14}  "
          f"{f'{n_retr_ho}/60 ({n_retr_ho/60:.0%})':>14}")
    print(f"\n  Val PPL drift: {drift:+.3f}%")

    out = {
        "rank": RANK,
        "alpha": ALPHA,
        "params_per_adapter": n_lora,
        "n_steps": N_STEPS,
        "strategy": STRATEGY,
        "drift_pct": drift,
        "same_prompt": {
            "routing": n_routed_sp, "retrieval": n_retr_sp, "n": 20,
        },
        "training_distribution": {
            "routing": n_routed_td, "retrieval": n_retr_td, "n": 60,
            "per_type_routed": per_type_td_routed,
            "per_type_retrieved": per_type_td_retr,
        },
        "held_out": {
            "routing": n_routed_ho, "retrieval": n_retr_ho, "n": 60,
            "per_type_routed": per_type_ho_routed,
            "per_type_retrieved": per_type_ho_retr,
            "per_slot_routed": per_slot_ho_routed,
            "per_slot_retrieved": per_slot_ho_retr,
        },
    }
    with open(results_dir / "rank128_heldout.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
