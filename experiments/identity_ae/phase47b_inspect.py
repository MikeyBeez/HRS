"""Phase 47b: Inspect Phase 47's actual generations for substring false positives.

Phase 47 reported 100% routing / 97% retrieval for the L0 → L5 projection
routing strategy on held-out paraphrases. The check_passkey function used by
that report is a substring match: if the passkey appears anywhere in the
generation, it counts as a hit. This script replays Phase 47's projection
routing path with full generation logging so we can manually verify the
generations are real answers to the questions and not substring artifacts.

For each of the 60 held-out paraphrase trials we record:
  - the held-out query (paraphrase)
  - the expected passkey
  - the routed adapter index (must equal the true adapter index)
  - the full 80-token generation
  - whether check_passkey returns True
  - a manual judgment heuristic: does the generation start with or near
    the passkey, in an answer-like context?

Output: a JSON file with all 60 trials and a printed summary that groups
trials by (routing correct, passkey found, looks-like-answer) so we can
spot any false positives by inspection.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase47b_inspect.py
"""

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
ALPHA = 256
N_STEPS = 150
PROJ_STEPS = 500
PROJ_LR = 1e-3
PROJ_TEMP = 0.05
GEN_TOKENS = 80
D = 1024


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def l5_mean(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h.mean(dim=1).squeeze(0).detach().cpu()


def looks_like_answer(gen, passkey, ptype):
    """Heuristic: does the generation contain the passkey in an answer-like
    context? Returns one of:
      - 'answer'  : passkey is near the front and surrounded by answer-like text
      - 'present' : passkey appears but not in obvious answer position
      - 'absent'  : passkey not present at all
    """
    if not check_passkey(gen, passkey):
        return "absent"

    # Find the position of the passkey (with normalization)
    if passkey in gen:
        pos = gen.find(passkey)
    else:
        clean_pk = passkey.replace(',', '').replace(' ', '')
        clean_gen = gen.replace(',', '').replace(' ', '')
        pos = clean_gen.find(clean_pk)
        # Map back roughly — if it's in the first 40 chars of the cleaned gen
        if pos < 40:
            return "answer"
        return "present"

    # Position heuristic: if passkey appears in first 30 chars of the gen, it's
    # in answer position. Otherwise it's somewhere else.
    if pos < 30:
        return "answer"
    return "present"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase47b")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: {len(tests)} passages\n")

    # ============================================================
    # Build the library
    # ============================================================
    model, _ = load_model(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    print("=" * 60)
    print("BUILD LIBRARY")
    print("=" * 60)
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
    # Extract base-model L0 and L5 keys (no adapter loaded)
    # ============================================================
    print("\nExtracting L0 and L5 keys (base model, no adapter loaded)...")
    reset_lora_to_zero(model)
    library_keys_l0 = []
    library_keys_l5 = []
    for entry in library:
        keys_l0 = []
        keys_l5 = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_l0.append(l0_mean(model, ids_t))
            keys_l5.append(l5_mean(model, ids_t))
        library_keys_l0.append(keys_l0)
        library_keys_l5.append(keys_l5)

    # Build flat training pairs
    flat_l0 = []
    flat_l5 = []
    flat_adapter = []
    for ai, (keys_l0, keys_l5) in enumerate(zip(library_keys_l0, library_keys_l5)):
        for k_l0, k_l5 in zip(keys_l0, keys_l5):
            flat_l0.append(k_l0)
            flat_l5.append(k_l5)
            flat_adapter.append(ai)
    flat_l0 = torch.stack(flat_l0).to(device)
    flat_l5 = torch.stack(flat_l5).to(device)
    flat_adapter = torch.tensor(flat_adapter, device=device)

    # Train projection
    print("\nTraining L0 → L5 projection (InfoNCE, 500 steps)...")
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    optimizer = torch.optim.Adam(W.parameters(), lr=PROJ_LR)
    flat_l5_norm = flat_l5 / (flat_l5.norm(dim=-1, keepdim=True) + 1e-8)

    for step in range(PROJ_STEPS):
        proj = W(flat_l0)
        proj_norm = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        sims = proj_norm @ flat_l5_norm.T
        same_adapter = flat_adapter.unsqueeze(0) == flat_adapter.unsqueeze(1)
        logits = sims / PROJ_TEMP
        log_probs = F.log_softmax(logits, dim=-1)
        pos_mask = same_adapter.float()
        pos_count = pos_mask.sum(dim=-1)
        pos_log_prob = (log_probs * pos_mask).sum(dim=-1) / (pos_count + 1e-8)
        loss = -pos_log_prob.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    library_keys_l5_cpu = [[k.cpu() if k.is_cuda else k for k in keys] for keys in library_keys_l5]
    W_cpu = W.weight.detach().cpu()

    def route_projected(q_l0):
        q_proj = q_l0 @ W_cpu.T
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for kv in library_keys_l5_cpu[ai]:
                s = cosine(q_proj, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a, best_score

    # ============================================================
    # Run held-out test with full generation logging
    # ============================================================
    print(f"\n{'='*60}")
    print("HELD-OUT INSPECTION (60 trials, full generation logging)")
    print(f"{'='*60}")

    trials = []
    for i, entry in enumerate(library):
        ho_paras = held_out_paraphrase(entry["test"])
        for slot_idx, para in enumerate(ho_paras):
            # Route under base model (no adapter loaded)
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q_l0 = l0_mean(model, ids_t)
            best_a, best_score = route_projected(q_l0)

            # Load the routed adapter and generate
            sd = library[best_a]["sd"]
            sd_gpu = {k: v.to(device) for k, v in sd.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, para, tokenizer, device, GEN_TOKENS)

            # Score
            routed_correct = (best_a == i)
            passkey = entry["test"]["passkey"]
            substring_hit = check_passkey(gen, passkey)
            answer_quality = looks_like_answer(gen, passkey, entry["test"]["type"])

            trials.append({
                "trial":     len(trials),
                "true_idx":  i,
                "routed_idx": best_a,
                "routed_correct": routed_correct,
                "type":      entry["test"]["type"],
                "passkey":   passkey,
                "query":     para,
                "gen":       gen,
                "substring_hit": substring_hit,
                "answer_quality": answer_quality,
                "routing_score": best_score,
            })

    # ============================================================
    # Analysis and printout
    # ============================================================
    n_trials = len(trials)
    n_routed = sum(t["routed_correct"] for t in trials)
    n_substring = sum(t["substring_hit"] for t in trials)
    n_answer = sum(1 for t in trials if t["answer_quality"] == "answer")
    n_present = sum(1 for t in trials if t["answer_quality"] == "present")
    n_absent = sum(1 for t in trials if t["answer_quality"] == "absent")

    print(f"\nTotals over {n_trials} held-out trials:")
    print(f"  Routing correct:                {n_routed}/{n_trials} ({n_routed/n_trials:.0%})")
    print(f"  Substring check_passkey hit:    {n_substring}/{n_trials} ({n_substring/n_trials:.0%})")
    print(f"  Answer-position passkey:        {n_answer}/{n_trials} ({n_answer/n_trials:.0%})")
    print(f"  Passkey present but late:       {n_present}/{n_trials}")
    print(f"  Passkey absent:                 {n_absent}/{n_trials}")

    # Show all trials where the substring hit but the position isn't "answer"
    suspicious = [t for t in trials if t["substring_hit"] and t["answer_quality"] != "answer"]
    print(f"\nSuspicious hits (substring matches but not in answer position): {len(suspicious)}")
    for t in suspicious:
        print(f"\n  trial {t['trial']} ({t['type']}, expected {t['passkey']!r}):")
        print(f"    query: {t['query']}")
        print(f"    gen:   {t['gen']!r}")
        print(f"    label: {t['answer_quality']}")

    # Show all trials where the passkey was absent
    print(f"\n\nFailed retrievals (passkey absent): {n_absent}")
    for t in trials:
        if t["answer_quality"] == "absent":
            print(f"\n  trial {t['trial']} ({t['type']}, expected {t['passkey']!r}):")
            print(f"    query:  {t['query']}")
            print(f"    routed: adapter {t['routed_idx']} (correct={t['routed_correct']})")
            print(f"    gen:    {t['gen']!r}")

    # Show 2 sample successful generations per type
    print(f"\n\nSample successful generations (2 per type):")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        successes = [t for t in trials if t["type"] == ptype and t["answer_quality"] == "answer"]
        for t in successes[:2]:
            print(f"\n  [{ptype}] trial {t['trial']}, expected {t['passkey']!r}")
            print(f"    query: {t['query']}")
            print(f"    gen:   {t['gen']!r}")

    # Save full trial log
    with open(results_dir / "inspection.json", "w") as f:
        json.dump({
            "n_trials":      n_trials,
            "n_routed":      n_routed,
            "n_substring":   n_substring,
            "n_answer":      n_answer,
            "n_present":     n_present,
            "n_absent":      n_absent,
            "trials":        trials,
        }, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
