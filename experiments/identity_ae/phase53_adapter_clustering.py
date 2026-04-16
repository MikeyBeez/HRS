"""Phase 53: Multi-passage adapter clustering.

If a new passage's L0 engram has high cosine similarity to an existing
adapter's key, the new passage is from a related novel domain. Instead of
allocating a fresh adapter, continue training the existing adapter on the
new passage. The adapter already has the right domain structure; the new
passage's gradients should refine rather than fight.

This script tests that hypothesis on the top-5 closest pairs from the
20-passage stratified benchmark, using L0_mean cosine to choose pairs.

Procedure:
  1. Compute the 20×20 L0_mean cosine matrix; pick the top-5 closest pairs.
  2. For each pair (A, B):
       - Train adapter on A alone (baseline A).
       - Continue training on B without resetting (shared A→B).
       - Train adapter on B alone (baseline B).
       - Continue training on A without resetting (shared B→A).
  3. Evaluate same-prompt and training-paraphrase retrieval of BOTH passages
     under each condition. Compare against the standard separate-adapter
     architecture.
  4. Report which pairs survive sharing and whether order matters.

If even one pair survives sharing without forgetting, the storage cost of
the adapter library can be amortized within similarity neighborhoods.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase53_adapter_clustering.py
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
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
ALPHA = 256
N_STEPS = 150
GEN_TOKENS = 50
TOP_K_PAIRS = 5


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-8))


def build_prompts_with_answers(test):
    prompts = [test["prompt"]] + train_paraphrase(test)
    return prompts, [f"{p} {test['passkey']}" for p in prompts]


def evaluate_passage(model, test, tokenizer, device):
    """Returns dict with same-prompt hit and per-paraphrase hits."""
    same_gen = generate_greedy(model, test["prompt"], tokenizer, device, GEN_TOKENS)
    same_hit = check_passkey(same_gen, test["passkey"])

    paras = train_paraphrase(test)
    para_hits = []
    for p in paras:
        gen = generate_greedy(model, p, tokenizer, device, GEN_TOKENS)
        para_hits.append(check_passkey(gen, test["passkey"]))

    return {
        "same_hit": same_hit,
        "same_gen": same_gen[:120],
        "para_hits": para_hits,
        "para_total": len(paras),
        "para_n_hit": sum(para_hits),
    }


def fresh_adapter_on(model, test, tokenizer, device):
    reset_lora_to_zero(model)
    _, prompts_with_answers = build_prompts_with_answers(test)
    train_adapter_multipara(model, test["passage"], prompts_with_answers,
                            tokenizer, device,
                            n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)


def continue_adapter_on(model, test, tokenizer, device):
    """Train current LoRA further on test (NO reset)."""
    _, prompts_with_answers = build_prompts_with_answers(test)
    train_adapter_multipara(model, test["passage"], prompts_with_answers,
                            tokenizer, device,
                            n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase53")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: {len(tests)} passages\n")

    model, _ = load_model(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    # ============================================================
    # STEP 1: Pairwise L0 cosine similarity matrix
    # ============================================================
    print("=" * 60)
    print("STEP 1: pairwise L0 cosine similarities")
    print("=" * 60)

    reset_lora_to_zero(model)
    l0_keys = []
    for test in tests:
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        l0_keys.append(l0_mean(model, ids_t))

    n = len(tests)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = cosine(l0_keys[i], l0_keys[j])
            pairs.append((s, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    top_pairs = pairs[:TOP_K_PAIRS]
    print(f"\nTop {TOP_K_PAIRS} closest pairs by L0 cosine:")
    for rank_idx, (s, i, j) in enumerate(top_pairs):
        print(f"  {rank_idx+1}. cos={s:.4f}  "
              f"[{tests[i]['type']:9s}] {tests[i]['id']}  <->  "
              f"[{tests[j]['type']:9s}] {tests[j]['id']}")

    # ============================================================
    # STEP 2 + 3: For each pair, train conditions and evaluate
    # ============================================================
    print(f"\n{'=' * 60}")
    print("STEP 2/3: train conditions and evaluate")
    print("=" * 60)

    pair_results = []
    t0 = time.time()

    for rank_idx, (sim, i, j) in enumerate(top_pairs):
        test_a = tests[i]
        test_b = tests[j]
        print(f"\nPair {rank_idx+1}: cos={sim:.4f}")
        print(f"  A = [{test_a['type']}] {test_a['id']}  passkey={test_a['passkey']!r}")
        print(f"  B = [{test_b['type']}] {test_b['id']}  passkey={test_b['passkey']!r}")

        # ---- Condition: separate adapter A ----
        fresh_adapter_on(model, test_a, tokenizer, device)
        sep_a_eval_a = evaluate_passage(model, test_a, tokenizer, device)
        sep_a_eval_b = evaluate_passage(model, test_b, tokenizer, device)

        # ---- Condition: separate adapter B ----
        fresh_adapter_on(model, test_b, tokenizer, device)
        sep_b_eval_a = evaluate_passage(model, test_a, tokenizer, device)
        sep_b_eval_b = evaluate_passage(model, test_b, tokenizer, device)

        # ---- Condition: shared A then B ----
        fresh_adapter_on(model, test_a, tokenizer, device)
        continue_adapter_on(model, test_b, tokenizer, device)
        shared_ab_eval_a = evaluate_passage(model, test_a, tokenizer, device)
        shared_ab_eval_b = evaluate_passage(model, test_b, tokenizer, device)

        # ---- Condition: shared B then A ----
        fresh_adapter_on(model, test_b, tokenizer, device)
        continue_adapter_on(model, test_a, tokenizer, device)
        shared_ba_eval_a = evaluate_passage(model, test_a, tokenizer, device)
        shared_ba_eval_b = evaluate_passage(model, test_b, tokenizer, device)

        def fmt(eva):
            same = "Y" if eva["same_hit"] else "N"
            return f"same={same} para={eva['para_n_hit']}/{eva['para_total']}"

        print(f"  separate-A:  A: {fmt(sep_a_eval_a)}   B: {fmt(sep_a_eval_b)}")
        print(f"  separate-B:  A: {fmt(sep_b_eval_a)}   B: {fmt(sep_b_eval_b)}")
        print(f"  shared A→B:  A: {fmt(shared_ab_eval_a)}   B: {fmt(shared_ab_eval_b)}")
        print(f"  shared B→A:  A: {fmt(shared_ba_eval_a)}   B: {fmt(shared_ba_eval_b)}")

        # Survival: BOTH same-prompt hits in a shared condition
        ab_survives = shared_ab_eval_a["same_hit"] and shared_ab_eval_b["same_hit"]
        ba_survives = shared_ba_eval_a["same_hit"] and shared_ba_eval_b["same_hit"]
        survives = ab_survives or ba_survives
        print(f"  -> survives sharing: {'YES' if survives else 'NO'}  "
              f"(A→B: {ab_survives}, B→A: {ba_survives})")

        pair_results.append({
            "rank": rank_idx + 1,
            "cos": sim,
            "a": {"id": test_a["id"], "type": test_a["type"],
                  "passkey": test_a["passkey"]},
            "b": {"id": test_b["id"], "type": test_b["type"],
                  "passkey": test_b["passkey"]},
            "separate_a": {"a": sep_a_eval_a, "b": sep_a_eval_b},
            "separate_b": {"a": sep_b_eval_a, "b": sep_b_eval_b},
            "shared_ab":  {"a": shared_ab_eval_a, "b": shared_ab_eval_b},
            "shared_ba":  {"a": shared_ba_eval_a, "b": shared_ba_eval_b},
            "ab_survives": ab_survives,
            "ba_survives": ba_survives,
            "survives": survives,
        })

    elapsed = time.time() - t0
    print(f"\n[step 2/3 elapsed: {elapsed:.0f}s]")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 72}")
    print("PHASE 53 SUMMARY: shared adapters within L0 similarity neighborhoods")
    print(f"{'=' * 72}")
    print(f"  rank {RANK}, alpha {ALPHA}, {N_STEPS} steps per training round\n")

    print(f"  {'pair':>4} {'cos':>7}  "
          f"{'sepA(a/b)':>12} {'sepB(a/b)':>12} "
          f"{'A→B(a/b)':>12} {'B→A(a/b)':>12}  surv")
    print(f"  {'-'*4} {'-'*7}  {'-'*12} {'-'*12} {'-'*12} {'-'*12}  {'-'*4}")
    for r in pair_results:
        def yn(eva):
            return "Y" if eva["same_hit"] else "N"
        sa = f"{yn(r['separate_a']['a'])}/{yn(r['separate_a']['b'])}"
        sb = f"{yn(r['separate_b']['a'])}/{yn(r['separate_b']['b'])}"
        ab = f"{yn(r['shared_ab']['a'])}/{yn(r['shared_ab']['b'])}"
        ba = f"{yn(r['shared_ba']['a'])}/{yn(r['shared_ba']['b'])}"
        surv = "Y" if r["survives"] else "N"
        print(f"  {r['rank']:>4} {r['cos']:>7.4f}  "
              f"{sa:>12} {sb:>12} {ab:>12} {ba:>12}  {surv:>4}")

    n_survive = sum(1 for r in pair_results if r["survives"])
    n_ab = sum(1 for r in pair_results if r["ab_survives"])
    n_ba = sum(1 for r in pair_results if r["ba_survives"])
    n_either_only = sum(1 for r in pair_results
                        if r["ab_survives"] != r["ba_survives"])

    print(f"\n  pairs surviving sharing (in either order): {n_survive}/{TOP_K_PAIRS}")
    print(f"    A→B order survives: {n_ab}/{TOP_K_PAIRS}")
    print(f"    B→A order survives: {n_ba}/{TOP_K_PAIRS}")
    print(f"    asymmetric (only one order works): {n_either_only}/{TOP_K_PAIRS}")

    if n_survive > 0:
        max_failing = max((r["cos"] for r in pair_results if not r["survives"]),
                          default=None)
        min_surviving = min((r["cos"] for r in pair_results if r["survives"]),
                            default=None)
        if max_failing is not None and min_surviving is not None:
            print(f"\n  cosine threshold band:")
            print(f"    highest failing cos: {max_failing:.4f}")
            print(f"    lowest  surviving cos: {min_surviving:.4f}")
        elif min_surviving is not None:
            print(f"\n  all top-{TOP_K_PAIRS} pairs survived (min cos: {min_surviving:.4f})")
    else:
        print(f"\n  NO pairs survived sharing — one-adapter-per-passage is necessary")
        print(f"  even within the closest L0 similarity neighborhood "
              f"(top cos: {top_pairs[0][0]:.4f}).")

    out = {
        "rank": RANK,
        "alpha": ALPHA,
        "n_steps": N_STEPS,
        "top_k_pairs": TOP_K_PAIRS,
        "all_pairs_sorted": [
            {"cos": s, "i": i, "j": j,
             "a_id": tests[i]["id"], "a_type": tests[i]["type"],
             "b_id": tests[j]["id"], "b_type": tests[j]["type"]}
            for s, i, j in pairs[:20]
        ],
        "pair_results": pair_results,
        "n_survive": n_survive,
        "n_ab_survive": n_ab,
        "n_ba_survive": n_ba,
    }
    with open(results_dir / "clustering.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
