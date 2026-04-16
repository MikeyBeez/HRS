"""Phase 45: Heterogeneous L0+L5 engram pair across all three applications.

Phase 32b found that L0 and L5 mean engrams are duals — each well-aligned at
the layer it was extracted from, weak at the other. Phase 44 showed L0 wins
for routing (90 vs 75 percent held-out). Phase 33b showed they tie at the
standard Application 2 operating point (engram_then_tokens, 91 vs 92 percent
gap closed) but L0 underperforms at extreme single-vector compression
(engram_only, 8 vs 18 percent).

The unified recommendation question: does extracting *both* L0 and L5 — and
using them together — dominate using either alone? Three tests.

  Application 1 (routing): compare three strategies on held-out paraphrases:
    - L0 alone (Phase 44 baseline, 90% expected)
    - L5 alone (Phase 31 baseline, ~75% expected)
    - avg_cos: (cosine(q_L0, k_L0) + cosine(q_L5, k_L5)) / 2

  Application 2 (cache compression): inject the heterogeneous pair as a
  two-position prefix [L0_engram, L5_engram] and measure continuation
  perplexity. Compare to Phase 33 (L5 only) and Phase 33b (L0 only) at
  the engram_only and engram_then_tokens conditions.

  Application 3 (compositional retrieval): apply the same avg_cos
  routing to clause-split top-1 routing on the Phase 29/41/42 pairs.
  Compare to Phase 42 (L5_nonstop_mean) and Phase 42b (L0_mean), both
  of which got 4/5 BOTH at 10/10 routing correctness.

The clean outcomes:
  (1) the pair dominates everywhere → unified recommendation, "use the pair"
  (2) the pair ties at routing and dominates injection → unified, "use the pair"
  (3) the pair adds nothing or hurts → keep L0 default, the duality is real

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase45_l0_l5_pair.py
"""

import json
import math
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
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase33_engram_context import (
    forward_segments, continuation_nll, segments_len,
)
from experiments.identity_ae.phase42b_l0_clause_routing import split_query
from experiments.identity_ae.phase41_activation_composition import (
    stack_two_state_dicts,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
ALPHA = 256
N_STEPS = 150
N_PASSAGES_APP2 = 50
PASSAGE_LEN = 256
CONTEXT_LEN = 200
HALF_LEN = 100


# ----------------------------------------------------------------
# Engram extractors
# ----------------------------------------------------------------
@torch.no_grad()
def l0_mean(model, ids_t):
    """Token embedding mean. Shape: (D,) on device."""
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach()


@torch.no_grad()
def l5_mean(model, ids_t):
    """Layer-5 hidden state mean. Shape: (D,) on device."""
    h = hidden_at_layer(model, ids_t, 5)   # (1, T, D)
    return h.mean(dim=1).squeeze(0).detach()


def avg_cos(q_l0, q_l5, k_l0, k_l5):
    """Average of L0-cosine and L5-cosine."""
    return 0.5 * (cosine(q_l0, k_l0) + cosine(q_l5, k_l5))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase45")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    # ============================================================
    # PHASE A: Build the rank-128 multi-prompt library
    # ============================================================
    model, cfg = load_model(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}\n")

    print("=" * 60)
    print("PHASE A: BUILD LIBRARY")
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
    print(f"Library built in {time.time()-t0:.0f}s")

    # Build paired library keys: each adapter has both L0 and L5 keys for each prompt
    reset_lora_to_zero(model)
    library_keys_l0 = []  # list[adapter] of list[paraphrase] of (D,)
    library_keys_l5 = []
    for entry in library:
        keys_l0 = []
        keys_l5 = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            keys_l0.append(l0_mean(model, ids_t).cpu())
            keys_l5.append(l5_mean(model, ids_t).cpu())
        library_keys_l0.append(keys_l0)
        library_keys_l5.append(keys_l5)

    library_meta = [entry["test"] for entry in library]

    # ============================================================
    # APPLICATION 1: routing strategy comparison on held-out paraphrases
    # ============================================================
    print(f"\n{'='*60}")
    print("APPLICATION 1: HELD-OUT PARAPHRASE ROUTING")
    print(f"{'='*60}")

    def route_l0(q_l0, q_l5):
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for kv in library_keys_l0[ai]:
                s = cosine(q_l0, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    def route_l5(q_l0, q_l5):
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for kv in library_keys_l5[ai]:
                s = cosine(q_l5, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    def route_avg(q_l0, q_l5):
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for k_l0, k_l5 in zip(library_keys_l0[ai], library_keys_l5[ai]):
                s = avg_cos(q_l0, q_l5, k_l0, k_l5)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    routing_strategies = [
        ("L0 only (Phase 44 baseline)", route_l0),
        ("L5 only (Phase 31 baseline)", route_l5),
        ("L0+L5 avg cosine (NEW)",      route_avg),
    ]

    app1_results = {}
    for name, route_fn in routing_strategies:
        n_routed, n_retr = 0, 0
        per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        for i, entry in enumerate(library):
            ho_paras = held_out_paraphrase(entry["test"])
            for slot_idx, para in enumerate(ho_paras):
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                q_l0 = l0_mean(model, ids_t).cpu()
                q_l5 = l5_mean(model, ids_t).cpu()
                best_a = route_fn(q_l0, q_l5)
                if best_a == i:
                    n_routed += 1

                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, 50)
                if check_passkey(gen, entry["test"]["passkey"]):
                    n_retr += 1
                    per_type[entry["test"]["type"]] += 1

        print(f"  {name:34s}  routing {n_routed:2d}/60 ({n_routed/60:.0%})  "
              f"retrieval {n_retr:2d}/60 ({n_retr/60:.0%})")
        print(f"    per type: num {per_type['numeric']:2d}/15  "
              f"ent {per_type['entity']:2d}/15  "
              f"tech {per_type['technical']:2d}/15  "
              f"fact {per_type['fact']:2d}/15")
        app1_results[name] = {
            "routing":   n_routed,
            "retrieval": n_retr,
            "per_type":  per_type,
        }

    # ============================================================
    # APPLICATION 3: clause-split routing on compositional queries
    # ============================================================
    print(f"\n{'='*60}")
    print("APPLICATION 3: CLAUSE-SPLIT ROUTING ON COMPOSITIONAL QUERIES")
    print(f"{'='*60}")

    pairs = []
    numeric_idx = [i for i, m in enumerate(library_meta) if m["type"] == "numeric"]
    entity_idx  = [i for i, m in enumerate(library_meta) if m["type"] == "entity"]
    for k in range(5):
        ai = numeric_idx[k]
        bi = entity_idx[k]
        a = library_meta[ai]
        b = library_meta[bi]
        compo_q = f"{a['prompt']} Also, {b['prompt']}"
        pairs.append({
            "pair_idx":  k,
            "ai":        ai,
            "bi":        bi,
            "a_meta":    a,
            "b_meta":    b,
            "query":     compo_q,
            "true_pair": (ai, bi),
        })

    # Need rank-256 LoRA for activation-level block-stacking
    del model
    torch.cuda.empty_cache()
    model, _ = load_model(device)
    apply_lora(model, rank=2*RANK, alpha=2*ALPHA, target_modules=L45_TARGETS)

    library_sds = [entry["sd"] for entry in library]

    def clause_route(p, route_fn):
        clauses = split_query(p["query"])
        if len(clauses) < 2:
            return p["ai"], p["bi"], 0
        idx_for_clause = []
        for c in clauses[:2]:
            reset_lora_to_zero(model)
            ids = tokenizer.encode(c, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q_l0 = l0_mean(model, ids_t).cpu()
            q_l5 = l5_mean(model, ids_t).cpu()
            best_a = route_fn(q_l0, q_l5)
            idx_for_clause.append(best_a)
        idx1, idx2 = idx_for_clause[0], idx_for_clause[1]
        rc = sum(1 for x in [idx1, idx2] if x in p["true_pair"])
        return idx1, idx2, rc

    app3_results = {}
    for name, route_fn in routing_strategies:
        a_hits, b_hits, both_hits, rc_total = 0, 0, 0, 0
        for p in pairs:
            idx1, idx2, rc = clause_route(p, route_fn)
            rc_total += rc
            sd_combined = stack_two_state_dicts(library_sds[idx1], library_sds[idx2],
                                                  scale1=1.0, scale2=1.0)
            sd_gpu = {k: v.to(device) for k, v in sd_combined.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, p["query"], tokenizer, device, 80)
            ah = check_passkey(gen, p["a_meta"]["passkey"])
            bh = check_passkey(gen, p["b_meta"]["passkey"])
            if ah: a_hits += 1
            if bh: b_hits += 1
            if ah and bh: both_hits += 1
        print(f"  {name:34s}  A {a_hits}/5  B {b_hits}/5  BOTH {both_hits}/5  "
              f"routing {rc_total}/10")
        app3_results[name] = {
            "a_hits": a_hits, "b_hits": b_hits, "both": both_hits,
            "routing_correct": rc_total,
        }

    del model
    torch.cuda.empty_cache()

    # ============================================================
    # APPLICATION 2: heterogeneous prefix injection on WikiText
    # ============================================================
    print(f"\n{'='*60}")
    print("APPLICATION 2: KV CACHE COMPRESSION (heterogeneous L0+L5 prefix)")
    print(f"{'='*60}")

    # Use base model (no LoRA) for the cache compression test
    model, _ = load_model(device)
    reset_lora_to_zero(model)
    model.eval()

    from data import load_wikitext
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    val_ds = splits["validation"]

    torch.manual_seed(0)
    indices = torch.randperm(len(val_ds))[:N_PASSAGES_APP2].tolist()

    conditions = [
        "no_context",
        "full_context",
        "engram_only_L0",
        "engram_only_L5",
        "engram_only_pair",
        "engram_then_tokens_L0",
        "engram_then_tokens_L5",
        "engram_then_tokens_pair",
    ]
    nll_acc = {c: [] for c in conditions}

    for ti, idx in enumerate(indices):
        item = val_ds[idx]
        ids = item[0] if isinstance(item, tuple) else item
        ids = ids[:PASSAGE_LEN]
        if len(ids) < PASSAGE_LEN:
            continue
        context_ids   = ids[:CONTEXT_LEN]
        continuation  = ids[CONTEXT_LEN:]
        first_half    = context_ids[:HALF_LEN]

        # Engrams of full context (for engram_only conditions)
        ctx_t = context_ids.unsqueeze(0).to(device)
        eng_full_l0 = l0_mean(model, ctx_t)
        eng_full_l5 = l5_mean(model, ctx_t)

        # Engrams of first half (for engram_then_tokens conditions)
        h1_t = first_half.unsqueeze(0).to(device)
        eng_h1_l0 = l0_mean(model, h1_t)
        eng_h1_l5 = l5_mean(model, h1_t)

        second_half = context_ids[HALF_LEN:CONTEXT_LEN]

        nll_acc["no_context"].append(
            continuation_nll(model, [], continuation, device))
        nll_acc["full_context"].append(
            continuation_nll(model, [("tokens", context_ids)], continuation, device))

        nll_acc["engram_only_L0"].append(
            continuation_nll(model, [("hidden", eng_full_l0)], continuation, device))
        nll_acc["engram_only_L5"].append(
            continuation_nll(model, [("hidden", eng_full_l5)], continuation, device))
        nll_acc["engram_only_pair"].append(
            continuation_nll(model, [("hidden", eng_full_l0),
                                       ("hidden", eng_full_l5)],
                             continuation, device))

        nll_acc["engram_then_tokens_L0"].append(
            continuation_nll(model, [("hidden", eng_h1_l0),
                                       ("tokens", second_half)],
                             continuation, device))
        nll_acc["engram_then_tokens_L5"].append(
            continuation_nll(model, [("hidden", eng_h1_l5),
                                       ("tokens", second_half)],
                             continuation, device))
        nll_acc["engram_then_tokens_pair"].append(
            continuation_nll(model, [("hidden", eng_h1_l0),
                                       ("hidden", eng_h1_l5),
                                       ("tokens", second_half)],
                             continuation, device))

        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1:2d}/{N_PASSAGES_APP2}] processed")

    # Aggregate
    summary = {}
    for c in conditions:
        nlls = nll_acc[c]
        mean_nll = sum(nlls) / len(nlls)
        ppl = math.exp(mean_nll)
        summary[c] = {"nll": mean_nll, "ppl": ppl, "n": len(nlls)}

    full_ppl = summary["full_context"]["ppl"]
    full_nll = summary["full_context"]["nll"]
    no_nll = summary["no_context"]["nll"]
    nll_gap = no_nll - full_nll

    print(f"\n  {'condition':28s} {'NLL':>8} {'PPL':>10} {'positions':>11} {'gap closed':>12}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*11} {'-'*12}")
    pos_per_cond = {
        "no_context":              0,
        "full_context":          CONTEXT_LEN,
        "engram_only_L0":          1,
        "engram_only_L5":          1,
        "engram_only_pair":        2,
        "engram_then_tokens_L0":   1 + HALF_LEN,
        "engram_then_tokens_L5":   1 + HALF_LEN,
        "engram_then_tokens_pair": 2 + HALF_LEN,
    }
    for c in conditions:
        s = summary[c]
        gap_closed = 1.0 - (s["nll"] - full_nll) / nll_gap if nll_gap > 0 else 0.0
        print(f"  {c:28s} {s['nll']:>8.3f} {s['ppl']:>10.2f} "
              f"{pos_per_cond[c]:>9d}   {gap_closed:>10.0%}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 45 SUMMARY: Heterogeneous L0+L5 pair across all three applications")
    print(f"{'='*72}")
    print(f"\n  Application 1 (held-out paraphrase routing, 60 trials):")
    for name in [n for n, _ in routing_strategies]:
        r = app1_results[name]
        print(f"    {name:34s}  routing {r['routing']:2d}/60 ({r['routing']/60:.0%})  "
              f"retrieval {r['retrieval']:2d}/60 ({r['retrieval']/60:.0%})")
    print(f"\n  Application 3 (clause-split compositional, 5 pairs, K=2):")
    for name in [n for n, _ in routing_strategies]:
        r = app3_results[name]
        print(f"    {name:34s}  BOTH {r['both']}/5  routing {r['routing_correct']}/10")
    print(f"\n  Application 2 (KV cache compression, 50 WikiText passages):")
    print(f"    {'condition':28s} {'gap closed':>12}")
    for c in ["engram_only_L0", "engram_only_L5", "engram_only_pair",
              "engram_then_tokens_L0", "engram_then_tokens_L5", "engram_then_tokens_pair"]:
        s = summary[c]
        gap = 1.0 - (s["nll"] - full_nll) / nll_gap
        print(f"    {c:28s} {gap:>10.0%}")

    out = {
        "rank":     RANK,
        "n_steps":  N_STEPS,
        "app1":     app1_results,
        "app2":     summary,
        "app3":     app3_results,
        "positions_per_condition": pos_per_cond,
    }
    with open(results_dir / "l0_l5_pair.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
