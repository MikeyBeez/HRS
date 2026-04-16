"""Phase 47: Learned L0 → L5 projection for asymmetric routing.

The L0 routing key (Phase 44) gets 90% held-out paraphrase retrieval. Phase 45
showed that mixing L0 and L5 via average cosine drops to 80% — but the per-type
breakdown is interesting: avg_cos gets numeric *perfect* (15/15 vs L0's 11/15)
while breaking entity (15→9) and fact (14→10). L5 carries discriminative signal
exactly where L0 is weakest.

The asymmetric-routing alternative this script tests: store L5 keys per library
entry (capturing the L5-discriminative signal), and at inference time compute
L0 of the query and project it through a learned linear map W: D → D so that
(L0_query @ W) is comparable to L5 keys. The projection is trained once, before
deployment, on the same library's prompts. Inference cost is one embedding
lookup + one matmul instead of a full forward pass — still O(1).

The hope: the projection learns the specific L5 features that distinguish
adapters and applies them to the L0 query, recovering numeric routing
discrimination without losing entity/fact accuracy.

Procedure:
  1. Build the rank-128 multi-prompt library (Phase 38b protocol)
  2. For each adapter, store L5 mean keys for each prompt (under base model)
  3. Generate training pairs: (L0_p, L5_p) for matched p, (L0_p, L5_q) for
     unmatched p, q from different adapters
  4. Train a 1024 × 1024 linear projection W with InfoNCE-style contrastive loss
  5. Test held-out routing: query_engram = L0(query) @ W, compare to L5 keys
  6. Compare to Phase 44 (L0 alone, 90%), Phase 45 (avg_cos, 80%)

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase47_l0_to_l5_projection.py
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
D = 1024


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def l5_mean(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h.mean(dim=1).squeeze(0).detach().cpu()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase47")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}\n")

    # ============================================================
    # PHASE A: Build the rank-128 library (Phase 38b)
    # ============================================================
    model, _ = load_model(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

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

    # ============================================================
    # PHASE B: Extract L0 and L5 keys per library entry
    # ============================================================
    print("\nExtracting L0 and L5 keys for each library entry...")
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

    # ============================================================
    # PHASE C: Build training pairs for the projection W
    # Each (adapter_idx, prompt_idx) pair gives an (L0, L5, label) triple
    # Positives: same adapter; Negatives: different adapter.
    # ============================================================
    print("\nBuilding training pairs for projection learning...")
    flat_l0 = []        # (N, D)
    flat_l5 = []        # (N, D)
    flat_adapter = []   # (N,) — adapter index
    for ai, (keys_l0, keys_l5) in enumerate(zip(library_keys_l0, library_keys_l5)):
        for k_l0, k_l5 in zip(keys_l0, keys_l5):
            flat_l0.append(k_l0)
            flat_l5.append(k_l5)
            flat_adapter.append(ai)
    flat_l0 = torch.stack(flat_l0).to(device)        # (80, 1024)
    flat_l5 = torch.stack(flat_l5).to(device)
    flat_adapter = torch.tensor(flat_adapter, device=device)
    print(f"  Training set: {flat_l0.shape[0]} (L0, L5) pairs, "
          f"{len(library)} adapters")

    # ============================================================
    # PHASE D: Train the projection W
    # InfoNCE-style: for each L0 query, the matching L5 should be the
    # closest among all L5 keys.
    # ============================================================
    print("\nTraining linear projection L0 → L5...")
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)   # initialize as identity
    optimizer = torch.optim.Adam(W.parameters(), lr=PROJ_LR)

    # Pre-normalize the L5 keys for cosine
    flat_l5_norm = flat_l5 / (flat_l5.norm(dim=-1, keepdim=True) + 1e-8)

    for step in range(PROJ_STEPS):
        # Project L0 → L5 space
        proj = W(flat_l0)                                  # (N, D)
        proj_norm = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)

        # Cosine sim matrix: (N, N)
        sims = proj_norm @ flat_l5_norm.T

        # Build labels: positive pairs are entries from the same adapter
        # Build a mask of "same adapter" (positive) entries
        same_adapter = flat_adapter.unsqueeze(0) == flat_adapter.unsqueeze(1)   # (N, N)

        # InfoNCE: for each row, the positives are entries from the same adapter
        # We use a softmax over all entries; the loss pulls positives up and pushes negatives down
        logits = sims / PROJ_TEMP                          # (N, N)
        # log-softmax over the columns (negatives in the row)
        log_probs = F.log_softmax(logits, dim=-1)          # (N, N)
        # Positive log-probs: average over the same-adapter mask per row
        pos_mask = same_adapter.float()
        pos_count = pos_mask.sum(dim=-1)
        pos_log_prob = (log_probs * pos_mask).sum(dim=-1) / (pos_count + 1e-8)
        loss = -pos_log_prob.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            # Diagnostic: top-1 accuracy on the training set
            with torch.no_grad():
                top1 = sims.argmax(dim=-1)
                top1_adapter = flat_adapter[top1]
                accuracy = (top1_adapter == flat_adapter).float().mean().item()
            print(f"  step {step+1}: loss {loss.item():.4f}, train top-1 adapter accuracy {accuracy:.0%}")

    # ============================================================
    # PHASE E: Test held-out paraphrase routing with the projection
    # Compare three strategies:
    #   - L0 alone (Phase 44 baseline, 90% expected)
    #   - L5 alone (Phase 31 baseline, ~75% expected)
    #   - Projected: route(L0 @ W) against L5 keys
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE E: HELD-OUT PARAPHRASE ROUTING")
    print(f"{'='*60}")

    # Move keys to CPU for routing comparison
    library_keys_l0_cpu = [[k.cpu() if k.is_cuda else k for k in keys] for keys in library_keys_l0]
    library_keys_l5_cpu = [[k.cpu() if k.is_cuda else k for k in keys] for keys in library_keys_l5]
    W_cpu = W.weight.detach().cpu()   # (D, D), nn.Linear stores weight as (out, in)

    def route_l0(q_l0):
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for kv in library_keys_l0_cpu[ai]:
                s = cosine(q_l0, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    def route_l5(q_l5):
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for kv in library_keys_l5_cpu[ai]:
                s = cosine(q_l5, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    def route_projected(q_l0):
        # Project L0 query through W (W stored as (out, in), so q @ W.T)
        q_proj = q_l0 @ W_cpu.T
        best_a, best_score = -1, -2.0
        for ai in range(len(library)):
            for kv in library_keys_l5_cpu[ai]:
                s = cosine(q_proj, kv)
                if s > best_score:
                    best_score = s
                    best_a = ai
        return best_a

    routing_strategies = [
        ("L0 only (Phase 44)",                route_l0,        "L0"),
        ("L5 only (Phase 31 baseline)",       route_l5,        "L5"),
        ("L0 @ W vs L5 (projection NEW)",     route_projected, "L0_proj"),
    ]

    app1_results = {}
    for name, route_fn, kind in routing_strategies:
        n_routed, n_retr = 0, 0
        per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
        for i, entry in enumerate(library):
            ho_paras = held_out_paraphrase(entry["test"])
            for slot_idx, para in enumerate(ho_paras):
                reset_lora_to_zero(model)
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                if kind == "L0" or kind == "L0_proj":
                    q = l0_mean(model, ids_t)
                elif kind == "L5":
                    q = l5_mean(model, ids_t)
                best_a = route_fn(q)
                if best_a == i:
                    n_routed += 1

                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, 50)
                if check_passkey(gen, entry["test"]["passkey"]):
                    n_retr += 1
                    per_type[entry["test"]["type"]] += 1

        print(f"  {name:38s}  routing {n_routed:2d}/60 ({n_routed/60:.0%})  "
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
    # SUMMARY
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 47 SUMMARY: L0 → L5 projection for asymmetric routing")
    print(f"{'='*72}")
    print(f"  Projection: linear 1024 → 1024, trained {PROJ_STEPS} steps with InfoNCE")
    print(f"  Phase 44 reference: L0 alone gets 90% held-out retrieval")
    print(f"  Phase 45 reference: L0+L5 average cosine gets 80%")
    print()
    print(f"  Phase 47 results:")
    for name in [n for n, _, _ in routing_strategies]:
        r = app1_results[name]
        print(f"    {name:38s}  routing {r['routing']/60:.0%}  retrieval {r['retrieval']/60:.0%}")

    out = {
        "rank":     RANK,
        "n_steps":  N_STEPS,
        "proj_steps": PROJ_STEPS,
        "proj_lr":  PROJ_LR,
        "results":  app1_results,
    }
    with open(results_dir / "l0_to_l5_projection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
