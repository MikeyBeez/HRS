"""Phase 31: Entity-weighted pooling for held-out paraphrase robustness.

Phase 27 measured 63% routing/retrieval on held-out paraphrases with
L5_mean pooling. The diagnosed failure mode is that mean pooling washes
out the discriminating tokens (entity names like "northern") and lets
template tokens (function words, generic nouns like "facility") dominate
the engram. Different paraphrases of the same query collapse to similar
mean-pooled engrams of the question style, not the entity identity.

This script tests several weighting strategies for the pooling step,
keeping everything else from Phase 27 identical (same library, same
keys constructed from training paraphrases, same retrieval pipeline).

Strategies:
  - mean              : uniform (Phase 27 baseline)
  - centered_norm     : weight ∝ ||h_i - mean(h)||  (distinctive tokens)
  - norm              : weight ∝ ||h_i||
  - softmax_centered  : softmax(||h_i - mean(h)||) — sharpens distinctive
  - nonstop_mean      : uniform mean over content (non-stopword) tokens
  - nonstop_centered  : centered_norm over content tokens only

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase31_weighted_pool.py
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
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


N_STEPS = 150
LAYER = 5  # final layer


# ----------------------------------------------------------------
# Stopword list for non-stopword pooling.
# Small, hand-curated list of common English function words and the
# punctuation/template tokens that appear in our prompt distribution.
# ----------------------------------------------------------------
STOPWORDS = set("""
a an the of to in for on at by with from as is are was were be been being
do does did doing have has had having will would shall should may might
must can could and or but if then else when where why how what which who
whom whose this that these those i you he she it we they me him her us them
my your his their our its mine yours hers ours theirs
not no nor so very just only also even
about into onto over under after before during between within without
tell what is on give me look up i need find out at how many code codes
which entry needs all signs are date make made discovery had has have
""".split())
PUNCT = set([",", ".", "?", "!", ":", ";", "—", "-", "'", '"', "(", ")"])


def is_content_token(decoded: str) -> bool:
    """Heuristic: is this token a content word (vs. function/template)?

    `decoded` is the GPT-2 BPE decoded form of a single token, which may
    have a leading space and may be a sub-word piece.
    """
    s = decoded.strip().lower().strip("'\".,;:?!()")
    if not s:
        return False
    if s in STOPWORDS:
        return False
    if s in PUNCT:
        return False
    if s.isdigit():
        return True  # numbers are content
    # Single-letter sub-pieces are usually function-word fragments
    if len(s) <= 1:
        return False
    return True


def make_key_weighted(model, tokenizer, ids_t, strategy: str):
    """Compute a pooled engram from layer-`LAYER` hidden states.

    strategy ∈ {mean, centered_norm, norm, softmax_centered,
                nonstop_mean, nonstop_centered}
    """
    h = hidden_at_layer(model, ids_t, LAYER)  # (1, T, D)
    h = h.squeeze(0)  # (T, D)
    T, D = h.shape

    if strategy == "mean":
        return h.mean(dim=0).detach().cpu()

    if strategy == "norm":
        w = h.norm(dim=-1)  # (T,)
        w = w / (w.sum() + 1e-8)
        return (h * w.unsqueeze(-1)).sum(dim=0).detach().cpu()

    if strategy == "centered_norm":
        m = h.mean(dim=0, keepdim=True)
        w = (h - m).norm(dim=-1)
        w = w / (w.sum() + 1e-8)
        return (h * w.unsqueeze(-1)).sum(dim=0).detach().cpu()

    if strategy == "softmax_centered":
        m = h.mean(dim=0, keepdim=True)
        scores = (h - m).norm(dim=-1)
        w = torch.softmax(scores, dim=0)
        return (h * w.unsqueeze(-1)).sum(dim=0).detach().cpu()

    if strategy in ("nonstop_mean", "nonstop_centered"):
        # Decode each token to identify content vs stopword
        ids = ids_t[0].tolist()
        mask = []
        for t in ids:
            decoded = tokenizer.decode([t])
            mask.append(is_content_token(decoded))
        mask_t = torch.tensor(mask, dtype=torch.bool, device=h.device)
        if mask_t.sum() == 0:
            # Fall back to uniform if no content tokens
            return h.mean(dim=0).detach().cpu()
        h_c = h[mask_t]  # (T_c, D)
        if strategy == "nonstop_mean":
            return h_c.mean(dim=0).detach().cpu()
        # nonstop_centered
        m = h_c.mean(dim=0, keepdim=True)
        w = (h_c - m).norm(dim=-1)
        w = w / (w.sum() + 1e-8)
        return (h_c * w.unsqueeze(-1)).sum(dim=0).detach().cpu()

    raise ValueError(strategy)


def cosine(a, b):
    a_n = a / (a.norm() + 1e-8)
    b_n = b / (b.norm() + 1e-8)
    return float(torch.dot(a_n, b_n))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase31")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Steps per adapter: {N_STEPS}, layer: {LAYER}\n")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL: {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION (same as Phase 26: train + key on training paraphrases)
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
    # KEY EXTRACTION under each strategy
    # ============================================================
    strategies = [
        "mean",            # Phase 27 baseline
        "norm",
        "centered_norm",
        "softmax_centered",
        "nonstop_mean",
        "nonstop_centered",
    ]

    reset_lora_to_zero(model)
    print(f"\n{'='*60}")
    print("KEY EXTRACTION")
    print(f"{'='*60}")

    library_keys = {s: [] for s in strategies}
    for entry in library:
        for s in strategies:
            keys_for_entry = []
            for p in entry["train_prompts"]:
                ids = tokenizer.encode(p, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                keys_for_entry.append(make_key_weighted(model, tokenizer, ids_t, s))
            library_keys[s].append(keys_for_entry)

    # ============================================================
    # HELD-OUT PARAPHRASE EVALUATION FOR EACH STRATEGY
    # ============================================================
    print(f"\n{'='*60}")
    print("HELD-OUT PARAPHRASE ROUTING (60 trials per strategy)")
    print(f"{'='*60}")

    def route_strategy(strategy, query_key):
        best_a, best_score = -1, -2.0
        for ai, keys in enumerate(library_keys[strategy]):
            for k in keys:
                s = cosine(query_key, k)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    all_results = {}
    for strategy in strategies:
        n_routed = 0
        n_retrieved = 0
        per_type_routed = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        per_type_retrieved = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        per_slot_routed = [0, 0, 0]
        per_slot_retrieved = [0, 0, 0]

        for i, test in enumerate(tests):
            ho_paras = held_out_paraphrase(test)
            for slot_idx, para in enumerate(ho_paras):
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                q = make_key_weighted(model, tokenizer, ids_t, strategy)
                best_a = route_strategy(strategy, q)

                correct = (best_a == i)
                if correct:
                    n_routed += 1
                    per_type_routed[test["type"]] += 1
                    per_slot_routed[slot_idx] += 1

                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, 50)
                if check_passkey(gen, test["passkey"]):
                    n_retrieved += 1
                    per_type_retrieved[test["type"]] += 1
                    per_slot_retrieved[slot_idx] += 1

        all_results[strategy] = {
            "routing": n_routed / 60,
            "retrieval": n_retrieved / 60,
            "per_type_routed": {k: v / 5 for k, v in per_type_routed.items()},
            "per_type_retrieved": {k: v / 5 for k, v in per_type_retrieved.items()},
            "per_slot_routed": [x / 20 for x in per_slot_routed],
            "per_slot_retrieved": [x / 20 for x in per_slot_retrieved],
        }
        print(f"  {strategy:20s}  routing={n_routed:2d}/60 ({n_routed/60:.0%})  "
              f"retrieval={n_retrieved:2d}/60 ({n_retrieved/60:.0%})")
        for ptype in ["numeric", "entity", "technical", "fact"]:
            print(f"      {ptype:9s}: routed {per_type_routed[ptype]:2d}/15 "
                  f"retrieved {per_type_retrieved[ptype]:2d}/15")
        print(f"      slots: routed {per_slot_routed} retrieved {per_slot_retrieved}")

    # Drift
    reset_lora_to_zero(model)
    final_ppl = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_ppl - baseline_ppl) / baseline_ppl * 100

    # Summary table
    print(f"\n{'='*60}")
    print("PHASE 31 SUMMARY (held-out paraphrase, pooling ablation)")
    print(f"{'='*60}")
    print(f"  Phase 27 baseline (mean):    63% routing / 63% retrieval")
    print()
    print(f"  {'Strategy':22s} {'Routing':>10} {'Retrieval':>10}")
    print(f"  {'-'*44}")
    for strategy in strategies:
        r = all_results[strategy]
        print(f"  {strategy:22s} {r['routing']:>10.0%} {r['retrieval']:>10.0%}")
    print(f"\n  Val PPL drift: {drift:+.3f}%")

    summary = {
        "n_steps": N_STEPS, "layer": LAYER,
        "drift_pct": drift,
        "strategies": all_results,
    }
    with open(results_dir / "weighted_pool.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
