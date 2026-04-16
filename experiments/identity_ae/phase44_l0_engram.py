"""Phase 44: L0 (token embedding) engram vs L5 hidden-state engram.

The current engram extraction (Phase 31 winner, used everywhere from Phase 38b
onward) does a forward pass through all 6 transformer blocks and then
mean-pools the layer-5 hidden states with stopword filtering. This is
expensive — every routing decision pays a full forward pass.

A cheaper alternative we have never tested: mean-pool the **input
embeddings** directly. `model.drop(model.tok_emb(prompt_ids))` averaged
over the token dimension is one embedding lookup, no block processing.
The question is whether the routing signal needs attention/MLP processing
to be discriminative, or whether the pretrained token embeddings are
already distinctive enough.

We test four conditions on the rank-128 multi-prompt library from
Phase 38b, against the same three query streams used elsewhere:

  L0_mean         token embeddings, uniform mean
  L0_nonstop_mean token embeddings, mean over content tokens only
  L5_mean         layer-5 hidden states, uniform mean (Phase 27 baseline)
  L5_nonstop_mean layer-5 hidden states, content-only (Phase 31 winner)

For each strategy we measure:
  - same-prompt routing/retrieval (sanity check, should be near-perfect)
  - training-distribution paraphrase routing/retrieval (Phase 26 protocol)
  - held-out paraphrase routing/retrieval (Phase 27 protocol)

The headline question: does L0_nonstop_mean match L5_nonstop_mean's 77/77
on held-out paraphrases? If yes, we can drop a forward pass per route.
If no, we have empirical justification for the L5 cost.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase44_l0_engram.py
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
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import (
    is_content_token, cosine,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
ALPHA = 256
N_STEPS = 150
GEN_TOKENS = 50


# ----------------------------------------------------------------
# Engram extractors. The L0 versions never touch a transformer block;
# they just mean-pool the input embeddings.
# ----------------------------------------------------------------
@torch.no_grad()
def l0_token_embeddings(model, ids_t):
    """Output of model.drop(model.tok_emb(ids)) — the input to block 0.
    Shape: (1, T, D)."""
    return model.drop(model.tok_emb(ids_t))


@torch.no_grad()
def l5_hidden_states(model, ids_t):
    """Output of block 5 (the final block). Shape: (1, T, D)."""
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    return h


def make_key(model, tokenizer, ids_t, layer, pooling):
    """Compute an engram key from layer ∈ {L0, L5} with pooling
    ∈ {mean, nonstop_mean}."""
    if layer == "L0":
        h = l0_token_embeddings(model, ids_t)
    elif layer == "L5":
        h = l5_hidden_states(model, ids_t)
    else:
        raise ValueError(layer)
    h = h.squeeze(0)  # (T, D)

    if pooling == "mean":
        return h.mean(dim=0).detach().cpu()

    if pooling == "nonstop_mean":
        ids = ids_t[0].tolist()
        mask = []
        for t in ids:
            decoded = tokenizer.decode([t])
            mask.append(is_content_token(decoded))
        mask_t = torch.tensor(mask, dtype=torch.bool, device=h.device)
        if mask_t.sum() == 0:
            return h.mean(dim=0).detach().cpu()
        h_c = h[mask_t]
        return h_c.mean(dim=0).detach().cpu()

    raise ValueError(pooling)


STRATEGIES = [
    ("L0", "mean"),
    ("L0", "nonstop_mean"),
    ("L5", "mean"),
    ("L5", "nonstop_mean"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase44")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    model, _ = load_model(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")
    print(f"Strategies to test: {STRATEGIES}\n")

    # ============================================================
    # Build the rank-128 multi-prompt library (Phase 38b protocol)
    # ============================================================
    print("=" * 60)
    print("ABSORPTION (Phase 38b protocol)")
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

    reset_lora_to_zero(model)

    # ============================================================
    # For each strategy: extract library keys, run all three tests
    # ============================================================
    all_results = {}

    for layer, pooling in STRATEGIES:
        strat_label = f"{layer}_{pooling}"
        print(f"\n{'='*60}")
        print(f"STRATEGY: {strat_label}")
        print(f"{'='*60}")

        # Build library keys for this strategy
        library_keys = []
        for entry in library:
            keys_for_entry = []
            for p in entry["train_prompts"]:
                ids = tokenizer.encode(p, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                keys_for_entry.append(make_key(model, tokenizer, ids_t, layer, pooling))
            library_keys.append(keys_for_entry)

        def route(query_engram):
            best_a, best_score = -1, -2.0
            for ai, keys in enumerate(library_keys):
                for k in keys:
                    s = cosine(query_engram, k)
                    if s > best_score:
                        best_score = s
                        best_a = ai
            return best_a

        # ----- PART 1: Same-prompt -----
        n_routed_sp, n_retr_sp = 0, 0
        for i, entry in enumerate(library):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(entry["test"]["prompt"], add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q = make_key(model, tokenizer, ids_t, layer, pooling)
            best_a = route(q)
            if best_a == i:
                n_routed_sp += 1

            sd = library[best_a]["sd"]
            sd_gpu = {k: v.to(device) for k, v in sd.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, entry["test"]["prompt"], tokenizer, device, GEN_TOKENS)
            if check_passkey(gen, entry["test"]["passkey"]):
                n_retr_sp += 1

        # ----- PART 2: Training-distribution paraphrase -----
        n_routed_td, n_retr_td = 0, 0
        per_type_td = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        for i, entry in enumerate(library):
            for para in train_paraphrase(entry["test"]):
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                q = make_key(model, tokenizer, ids_t, layer, pooling)
                best_a = route(q)
                if best_a == i:
                    n_routed_td += 1

                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, GEN_TOKENS)
                if check_passkey(gen, entry["test"]["passkey"]):
                    n_retr_td += 1
                    per_type_td[entry["test"]["type"]] += 1

        # ----- PART 3: Held-out paraphrase -----
        n_routed_ho, n_retr_ho = 0, 0
        per_type_ho = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        per_slot_ho = [0, 0, 0]
        for i, entry in enumerate(library):
            ho_paras = held_out_paraphrase(entry["test"])
            for slot_idx, para in enumerate(ho_paras):
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                q = make_key(model, tokenizer, ids_t, layer, pooling)
                best_a = route(q)
                if best_a == i:
                    n_routed_ho += 1

                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, GEN_TOKENS)
                if check_passkey(gen, entry["test"]["passkey"]):
                    n_retr_ho += 1
                    per_type_ho[entry["test"]["type"]] += 1
                    per_slot_ho[slot_idx] += 1

        print(f"  same-prompt:           routing {n_routed_sp:2d}/20 ({n_routed_sp/20:.0%})  "
              f"retrieval {n_retr_sp:2d}/20 ({n_retr_sp/20:.0%})")
        print(f"  training-distribution: routing {n_routed_td:2d}/60 ({n_routed_td/60:.0%})  "
              f"retrieval {n_retr_td:2d}/60 ({n_retr_td/60:.0%})")
        print(f"  held-out:              routing {n_routed_ho:2d}/60 ({n_routed_ho/60:.0%})  "
              f"retrieval {n_retr_ho:2d}/60 ({n_retr_ho/60:.0%})")
        print(f"    per-type held-out: num {per_type_ho['numeric']:2d}/15  "
              f"ent {per_type_ho['entity']:2d}/15  "
              f"tech {per_type_ho['technical']:2d}/15  "
              f"fact {per_type_ho['fact']:2d}/15")
        print(f"    per-slot held-out: {per_slot_ho}")

        all_results[strat_label] = {
            "layer":   layer,
            "pooling": pooling,
            "same_prompt":           {"routing": n_routed_sp, "retrieval": n_retr_sp},
            "training_distribution": {"routing": n_routed_td, "retrieval": n_retr_td},
            "held_out":              {"routing": n_routed_ho, "retrieval": n_retr_ho,
                                       "per_type": per_type_ho, "per_slot": per_slot_ho},
        }

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 44 SUMMARY: L0 (token embedding) vs L5 (hidden state) engram")
    print(f"{'='*72}")
    print(f"  {'Strategy':18s} {'same-prompt':>14}  {'train-distrib':>16}  {'held-out':>14}")
    print(f"  {'-'*18} {'-'*14}  {'-'*16}  {'-'*14}")
    for layer, pooling in STRATEGIES:
        label = f"{layer}_{pooling}"
        r = all_results[label]
        sp = f"{r['same_prompt']['retrieval']}/20 ({r['same_prompt']['retrieval']*100/20:.0f}%)"
        td = f"{r['training_distribution']['retrieval']}/60 ({r['training_distribution']['retrieval']*100/60:.0f}%)"
        ho = f"{r['held_out']['retrieval']}/60 ({r['held_out']['retrieval']*100/60:.0f}%)"
        print(f"  {label:18s} {sp:>14}  {td:>16}  {ho:>14}")

    out = {
        "rank":       RANK,
        "n_steps":    N_STEPS,
        "strategies": [f"{l}_{p}" for l, p in STRATEGIES],
        "results":    all_results,
    }
    with open(results_dir / "l0_vs_l5.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
