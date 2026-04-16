"""Phase 26: Multi-key routing + multi-paraphrase adapter training.

Phase 25 found two failures with single-key per-passage adapters:
  1. Routing collapses to wrong adapter under paraphrase (50% accuracy)
  2. Even when loaded, adapters trained only on the passage fail on
     paraphrased prompts they never saw (slot-3 garbage generation)

Two coupled fixes:
  A) At absorption time, generate K paraphrases of the prompt and store
     all K+1 engram keys for the SAME adapter. Route by min cosine distance
     to any of an adapter's stored keys.
  B) Train the adapter on the passage AND on each paraphrased prompt
     followed by the passkey, so the adapter generalizes across surface
     forms.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase26_multikey.py
"""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, RANK, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.phase22_engram_key import (
    make_key, cosine_match, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


N_STEPS = 150
SOURCE = "L5_mean"


def train_adapter_multipara(model, passage, prompts_with_answers, tokenizer, device,
                             n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR):
    """Train one adapter on the passage AND on prompt-answer pairs.

    At each step, sample one of:
        - the passage tokens
        - one of the (prompt + " " + passkey) tokens
    uniformly, and run an LM-loss step.
    """
    # Pre-tokenize all training sources
    sources = []
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    sources.append(torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device))
    for pa in prompts_with_answers:
        ids = tokenizer.encode(pa, add_special_tokens=False)
        sources.append(torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device))

    params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    for _ in range(n_steps):
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase26")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Steps per adapter: {N_STEPS}")
    print(f"Key source: {SOURCE}")
    print(f"Multi-key: original prompt + 3 paraphrases (4 keys per adapter)")
    print(f"Multi-train: passage + 4 prompt+answer pairs per adapter\n")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION
    # ============================================================
    print(f"{'='*60}")
    print("ABSORPTION PHASE")
    print(f"{'='*60}")

    # library entries:
    #   keys: list of K+1 engram tensors (CPU)
    #   sd:   adapter state dict (CPU)
    #   meta: original test dict
    library = []

    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)

        # Generate paraphrases (3) — same function used for the test
        para_prompts = paraphrase(test)
        all_prompts = [test["prompt"]] + para_prompts  # 4 total

        # Multi-paraphrase training: train on passage + each (prompt + answer)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in all_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)

        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}

        # Multi-key: store keys for all 4 prompt forms
        # Compute under BASE model (zero adapter) for distribution consistency
        reset_lora_to_zero(model)
        keys = []
        for p in all_prompts:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys.append(make_key(model, ids_t, SOURCE))

        library.append({"keys": keys, "sd": sd, "meta": dict(test),
                         "stored_prompts": all_prompts})

        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    print(f"  Library: {len(library)} adapters × 4 keys each")

    # ============================================================
    # RETRIEVAL: route by min cosine distance to any stored key
    # ============================================================
    print(f"\n{'='*60}")
    print("PARAPHRASE RETRIEVAL (multi-key routing)")
    print(f"{'='*60}")

    def route(query_key):
        """Find adapter with the highest cosine sim across any of its keys."""
        best_adapter = -1
        best_score = -2.0
        best_key_in_adapter = -1
        q_n = query_key / (query_key.norm() + 1e-8)
        for ai, entry in enumerate(library):
            for ki, k in enumerate(entry["keys"]):
                k_n = k / (k.norm() + 1e-8)
                sim = float(torch.dot(q_n, k_n))
                if sim > best_score:
                    best_score = sim
                    best_adapter = ai
                    best_key_in_adapter = ki
        return best_adapter, best_key_in_adapter, best_score

    per_slot = [{"routing": 0, "retrieval": 0, "by_type": {}} for _ in range(3)]
    for slot in per_slot:
        for ptype in ["numeric", "entity", "technical", "fact"]:
            slot["by_type"][ptype] = {"routed": 0, "retrieved": 0}

    detailed = []
    reset_lora_to_zero(model)

    for i, test in enumerate(tests):
        paraphrases = paraphrase(test)
        for slot_idx, para in enumerate(paraphrases):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key(model, ids_t, SOURCE)
            best_adapter, _, score = route(q)

            correct = (best_adapter == i)
            if correct:
                per_slot[slot_idx]["routing"] += 1
                per_slot[slot_idx]["by_type"][test["type"]]["routed"] += 1

            sd = library[best_adapter]["sd"]
            sd_gpu = {k: v.to(device) for k, v in sd.items()}
            load_lora_state_dict(model, sd_gpu)

            gen = generate_greedy(model, para, tokenizer, device, 50)
            found = check_passkey(gen, test["passkey"])
            if found:
                per_slot[slot_idx]["retrieval"] += 1
                per_slot[slot_idx]["by_type"][test["type"]]["retrieved"] += 1

            detailed.append({
                "test_id": test["id"], "type": test["type"], "slot": slot_idx,
                "paraphrase": para, "passkey": test["passkey"],
                "best_adapter": best_adapter, "correct_route": correct,
                "found": found, "gen": gen[:120],
            })

    # Final drift
    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n{'='*60}")
    print(f"PHASE 26 SUMMARY (multi-key + multi-paraphrase training)")
    print(f"{'='*60}")
    print(f"  20 originals × 3 paraphrases each = 60 query trials")
    print(f"  Phase 25 baseline: routing 50%, retrieval 48%")
    print()
    print(f"  {'Slot':6s} {'Routing':>10} {'Retrieval':>10}")
    print(f"  {'-'*28}")
    for i, slot in enumerate(per_slot):
        print(f"  {i+1:>4d}   {slot['routing']/20:>10.0%} {slot['retrieval']/20:>10.0%}")
    total_r = sum(s["routing"] for s in per_slot)
    total_ret = sum(s["retrieval"] for s in per_slot)
    print(f"  {'-'*28}")
    print(f"  Total  {total_r:>4d}/60   {total_ret:>4d}/60")
    print(f"  Avg    {total_r/60:>10.0%} {total_ret/60:>10.0%}")

    print(f"\n  Per-type retrieval (across 3 paraphrases):")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        ret = sum(s["by_type"][ptype]["retrieved"] for s in per_slot)
        rou = sum(s["by_type"][ptype]["routed"] for s in per_slot)
        print(f"    {ptype:9s}: routed {rou:2d}/15  retrieved {ret:2d}/15")

    print(f"\n  Val PPL drift: {drift:+.3f}%")

    misses = [d for d in detailed if not d["found"]]
    if misses:
        print(f"\n  {len(misses)}/60 retrieval misses:")
        for m in misses[:8]:
            print(f"    [{m['type']}] '{m['paraphrase']}'")
            print(f"       passkey={m['passkey']!r}  routed_to={m['best_adapter']} "
                  f"(correct={m['correct_route']})")
            print(f"       gen={m['gen']!r}")

    summary = {
        "n_steps": N_STEPS, "source": SOURCE,
        "n_originals": 20, "n_paraphrases_per": 3, "n_trials": 60,
        "per_slot": per_slot,
        "total_routing_pct": total_r / 60,
        "total_retrieval_pct": total_ret / 60,
        "drift_pct": drift,
        "detailed": detailed,
    }
    with open(results_dir / "multikey.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
