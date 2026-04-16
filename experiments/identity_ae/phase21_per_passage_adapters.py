"""Phase 21: Per-passage LoRA adapters with content-addressable retrieval.

Each absorbed passage gets its own tiny LoRA adapter, trained from scratch
on that passage alone (phase 14 winning config: rank 512, L4-5, 100 steps,
3e-4 -> 1e-4). At retrieval time, the layer-3 hidden state of the query is
matched against stored keys (mean-pooled hidden states), and the
best-matching adapter is loaded before generation.

Zero interference by construction: each adapter only sees its one passage.
Storage: ~10.5M params per adapter, kept in CPU memory.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase21_per_passage_adapters.py
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    generate_passkeys, check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import (
    val_ppl_ungated, get_hidden_at_layer,
    RANK, N_STEPS, HIGH_LR, BASE_LR,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict, reset_lora,
)


def train_passage_adapter(model, passage, tokenizer, device,
                          n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR):
    """Train LoRA on a single passage. Returns nothing — adapter is in model."""
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    for _ in range(n_steps):
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()


@torch.no_grad()
def passage_key(model, ids_t):
    """Mean-pooled layer-3 hidden state, used as content-addressable key."""
    h = get_hidden_at_layer(model, ids_t)
    return h.mean(dim=1).squeeze(0).detach().cpu()  # (D,)


@torch.no_grad()
def query_key(model, prompt, tokenizer, device):
    """Mean-pooled layer-3 hidden state of a query prompt."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    h = get_hidden_at_layer(model, ids_t)
    return h.mean(dim=1).squeeze(0).detach().cpu()  # (D,)


def best_match(q, keys):
    """Cosine similarity between q and each key. Returns (best_idx, best_sim)."""
    q_n = q / (q.norm() + 1e-8)
    sims = []
    for k in keys:
        k_n = k / (k.norm() + 1e-8)
        sims.append(float(torch.dot(q_n, k_n)))
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    return best_idx, sims[best_idx], sims


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase21")
    results_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA structure applied: {n_lora:,} params per adapter")

    # Stratified 20: 5 of each type
    all_tests = generate_passkeys(50)
    by_type = {"numeric": [], "entity": [], "technical": [], "fact": []}
    for t in all_tests:
        by_type[t["type"]].append(t)
    tests = (by_type["numeric"][:5] + by_type["entity"][:5]
             + by_type["technical"][:5] + by_type["fact"][:5])
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")

    # Baseline val PPL with zero adapter (= pure base model)
    reset_lora(model)
    # zero out lora_B specifically (reset_lora puts A as random small noise)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.zero_()
    baseline_ppl = val_ppl_ungated(model, None, device) if False else None
    # Use the proper val loader
    from data import load_wikitext, build_dataloaders
    splits, _ = load_wikitext(cfg.training.dataset, cfg.model.max_seq_len)
    loaders = build_dataloaders(splits, batch_size=4)
    baseline_ppl = val_ppl_ungated(model, loaders["validation"], device)
    print(f"Baseline val PPL (zero adapter): {baseline_ppl:.3f}\n")

    # ============================================================
    # ABSORPTION: train one adapter per passage, save to CPU
    # ============================================================
    print(f"{'='*60}")
    print("ABSORPTION PHASE")
    print(f"{'='*60}")
    library = []  # list of (key_tensor_cpu, lora_state_dict_cpu, test_meta)

    t0 = time.time()
    for i, test in enumerate(tests):
        # Reset LoRA to zero
        reset_lora(model)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if 'lora_B' in n:
                    p.zero_()

        # Train on this passage
        train_passage_adapter(model, test["passage"], tokenizer, device,
                               n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)

        # Compute and store key from the PASSAGE itself (with adapter active)
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        key = passage_key(model, ids_t)

        # Save LoRA state to CPU
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library.append((key, sd, {"id": test["id"], "type": test["type"],
                                    "passkey": test["passkey"]}))

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({elapsed:.0f}s)")

    print(f"  Library: {len(library)} adapters, ~{n_lora * len(library) / 1e6:.0f}M total params (CPU)")

    # ============================================================
    # RETRIEVAL: for each query, find best-matching adapter, load, generate
    # ============================================================
    print(f"\n{'='*60}")
    print("RETRIEVAL PHASE")
    print(f"{'='*60}")

    # Reset to zero adapter for clean key computation of queries
    # NOTE: query keys are computed with the CURRENT loaded adapter, which
    # may bias matching. Try keys computed with zero adapter (pure base)
    # for consistent retrieval.
    reset_lora(model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.zero_()

    # Precompute query keys for all test prompts using the BASE model (zero adapter)
    # This gives us a fair, content-only retrieval signal.
    base_query_keys = []
    for test in tests:
        qk = query_key(model, test["prompt"], tokenizer, device)
        base_query_keys.append(qk)

    # Also recompute library keys with the BASE model so they're comparable
    print("  Recomputing library keys with base model for consistency...")
    base_library_keys = []
    for _, _, meta in library:
        # Find the original test
        test = next(t for t in tests if t["id"] == meta["id"])
        ids = tokenizer.encode(test["passage"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        bk = passage_key(model, ids_t)
        base_library_keys.append(bk)

    n_correct_route = 0
    n_retrieved = 0
    per_test_results = []
    found_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    routed_by_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}

    for i, test in enumerate(tests):
        q = base_query_keys[i]
        best_idx, best_sim, all_sims = best_match(q, base_library_keys)

        # Did routing pick the right adapter?
        true_idx = i  # library is in same order as tests
        correct_route = (best_idx == true_idx)
        if correct_route:
            n_correct_route += 1
            routed_by_type[test["type"]] += 1

        # Load the routed adapter
        _, sd, _ = library[best_idx]
        # Move to GPU and load
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)

        # Generate
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        found = check_passkey(gen, test["passkey"])
        if found:
            n_retrieved += 1
            found_by_type[test["type"]] += 1

        per_test_results.append({
            "id": test["id"], "type": test["type"], "passkey": test["passkey"],
            "best_match_idx": best_idx, "best_sim": best_sim,
            "correct_route": correct_route, "found": found,
            "gen": gen[:100],
        })

    # ============================================================
    # FORGETTING CHECK: zero adapter for WikiText, library should not affect base
    # ============================================================
    reset_lora(model)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n:
                p.zero_()
    final_baseline = val_ppl_ungated(model, loaders["validation"], device)
    drift = (final_baseline - baseline_ppl) / baseline_ppl * 100

    print()
    print(f"{'='*60}")
    print(f"PHASE 21: PER-PASSAGE ADAPTER LIBRARY")
    print(f"{'='*60}")
    print(f"  Library size:                {len(library)} adapters")
    print(f"  Retrieval (cumulative):      {n_retrieved}/20 ({n_retrieved/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {found_by_type[ptype]}/5")
    print(f"  Routing accuracy:            {n_correct_route}/20 ({n_correct_route/20:.0%})")
    for ptype in ["numeric", "entity", "technical", "fact"]:
        print(f"    {ptype}: {routed_by_type[ptype]}/5")
    print(f"  Val PPL (zero adapter):      {baseline_ppl:.3f} -> {final_baseline:.3f} ({drift:+.3f}%)")

    summary = {
        "library_size": len(library),
        "retrieval": n_retrieved / 20,
        "retrieval_by_type": {k: v / 5 for k, v in found_by_type.items()},
        "routing_accuracy": n_correct_route / 20,
        "routing_by_type": {k: v / 5 for k, v in routed_by_type.items()},
        "baseline_ppl": baseline_ppl,
        "final_ppl_zero_adapter": final_baseline,
        "drift_pct": drift,
        "per_test": per_test_results,
    }
    with open(results_dir / "library.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
