"""Phase 22: Per-passage adapter library keyed on the FINAL-LAYER engram.

Phase 21 used mean-pooled layer-3 hidden states as keys, which washed out
the discriminating word in nearly-identical numeric prompts (0/5 routed).

The model's actual engram is the layer-5 (final) hidden state, where
causal attention has propagated the unique tokens through every block.
This script uses the last-token hidden state at layer 5 as the key.

Tests three key sources to ablate:
  - last token, layer 5 (the engram)
  - mean pool, layer 5 (sanity check vs phase 21's layer 3)
  - last token, layer 3 (sanity check on pooling vs depth)

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase22_engram_key.py
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
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict, reset_lora,
)


@torch.no_grad()
def hidden_at_layer(model, ids_t, layer_idx):
    """Run forward through layers 0..layer_idx and return (B, T, D) hidden."""
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == layer_idx:
            return h
    return h


@torch.no_grad()
def make_key(model, ids_t, source):
    """Compute a key from hidden states based on source spec.

    source = "L{layer}_{pool}" where pool ∈ {last, mean}
    """
    layer, pool = source.split("_")
    layer_idx = int(layer[1:])
    h = hidden_at_layer(model, ids_t, layer_idx)  # (1, T, D)
    if pool == "last":
        return h[:, -1, :].squeeze(0).detach().cpu()
    elif pool == "mean":
        return h.mean(dim=1).squeeze(0).detach().cpu()
    else:
        raise ValueError(pool)


def l2_match(q, keys):
    """L2-nearest-neighbor. Returns (best_idx, best_dist)."""
    dists = [float((q - k).norm()) for k in keys]
    best_idx = min(range(len(dists)), key=lambda i: dists[i])
    return best_idx, dists[best_idx]


def cosine_match(q, keys):
    """Cosine-nearest-neighbor. Returns (best_idx, best_sim)."""
    q_n = q / (q.norm() + 1e-8)
    sims = [float(torch.dot(q_n, k / (k.norm() + 1e-8))) for k in keys]
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    return best_idx, sims[best_idx]


def reset_lora_to_zero(model):
    reset_lora(model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.zero_()


def stratified_tests():
    all_tests = generate_passkeys(50)
    by_type = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][:5] + by_type["entity"][:5]
            + by_type["technical"][:5] + by_type["fact"][:5])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase22")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA structure applied: {n_lora:,} params per adapter")

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)

    # Baseline val PPL
    reset_lora_to_zero(model)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL (zero adapter): {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION: train one adapter per passage, save state dicts
    # We compute and store the input token ids alongside; keys are
    # computed later under the BASE model so all configurations
    # use the same key-extraction model.
    # ============================================================
    print(f"{'='*60}")
    print("ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []  # list of (passage_ids_tensor, lora_state_dict_cpu, test_meta)

    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_passage_adapter(model, test["passage"], tokenizer, device,
                               n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)

        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append((ids_t, sd, dict(test)))

        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    print(f"  Library: {len(library)} adapters")

    # ============================================================
    # KEY EXTRACTION: under base model (zero adapter), compute keys
    # from each source spec for both library passages and query prompts.
    # ============================================================
    sources = ["L5_last", "L5_mean", "L3_last", "L3_mean"]
    metrics = ["L2", "cosine"]

    reset_lora_to_zero(model)

    print(f"\n{'='*60}")
    print("KEY EXTRACTION (base model)")
    print(f"{'='*60}")

    library_keys = {src: [] for src in sources}
    for ids_t, _, _ in library:
        for src in sources:
            library_keys[src].append(make_key(model, ids_t, src))

    query_keys = {src: [] for src in sources}
    for test in tests:
        ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        for src in sources:
            query_keys[src].append(make_key(model, ids_t, src))

    # ============================================================
    # ABLATION: try every (source, metric) combination
    # ============================================================
    print(f"\n{'='*60}")
    print("ROUTING ABLATION")
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
                # PASS 1: route with base-model engram (no adapter)
                reset_lora_to_zero(model)
                q1 = query_keys[src][i]
                if metric == "L2":
                    cand_idx, _ = l2_match(q1, library_keys[src])
                else:
                    cand_idx, _ = cosine_match(q1, library_keys[src])

                # Load the candidate adapter
                _, sd, _ = library[cand_idx]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)

                # PASS 2: re-extract query key with the candidate adapter loaded.
                # The adapter shifts the engram geometry; re-rank in that space.
                ids = tokenizer.encode(test["prompt"], add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                q2 = make_key(model, ids_t, src)

                # Recompute library keys under the same loaded adapter, so the
                # comparison is in a consistent geometry.
                lib_keys_loaded = []
                for lib_ids, _, _ in library:
                    lib_keys_loaded.append(make_key(model, lib_ids, src))
                if metric == "L2":
                    best_idx, score = l2_match(q2, lib_keys_loaded)
                else:
                    best_idx, score = cosine_match(q2, lib_keys_loaded)

                # If pass 2 changes its mind, switch to the new candidate.
                if best_idx != cand_idx:
                    _, sd2, _ = library[best_idx]
                    sd_gpu = {k: v.to(device) for k, v in sd2.items()}
                    load_lora_state_dict(model, sd_gpu)

                correct = (best_idx == i)
                if correct:
                    n_correct += 1
                    routed_by_type[test["type"]] += 1

                gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
                found = check_passkey(gen, test["passkey"])
                if found:
                    n_retrieved += 1
                    found_by_type[test["type"]] += 1

                per_test.append({
                    "id": test["id"], "type": test["type"],
                    "pass1_idx": cand_idx, "pass2_idx": best_idx,
                    "switched": cand_idx != best_idx,
                    "score": score, "correct_route": correct, "found": found,
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

    # Forgetting check
    reset_lora_to_zero(model)
    final_baseline = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_baseline - baseline_ppl) / baseline_ppl * 100

    print(f"\n{'='*60}")
    print("PHASE 22 SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':22s} {'Routing':>10} {'Retrieval':>10}")
    print(f"  {'-'*44}")
    for label, r in all_results.items():
        print(f"  {label:22s} {r['routing']:>10.0%} {r['retrieval']:>10.0%}")
    print(f"\n  Val PPL drift (zero adapter): {drift:+.3f}%")

    with open(results_dir / "ablation.json", "w") as f:
        json.dump({"baseline_ppl": baseline_ppl, "drift_pct": drift,
                    "results": all_results}, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
