"""Phase 46: L5-contrastive auxiliary loss during absorption.

The K=2 capacity ceiling Phase 43 measured is set by interference between
simultaneously-loaded adapters in the residual stream. The address/regularizer
framing predicts that adapters trained without any inter-adapter signal will
end up shaping the L5 residual-stream subspace in arbitrary, possibly
overlapping directions — and that the K=4 collapse is driven by that overlap.

This script tests whether adding an explicit L5-contrastive loss during
absorption — pushing each adapter's L5 representation of its own prompt away
from the L5 representations of all other absorbed prompts — produces a
library whose adapters compose better at K > 2.

Procedure:
  1. Build Library A using the standard Phase 38b protocol (no contrastive loss).
  2. Compute L5 anchors: for each adapter i, load it and compute L5_mean of its
     own absorbed prompt under the loaded adapter. Save 20 anchors.
  3. Build Library B from scratch using Phase 38b + an additional L5-contrastive
     loss term: for each adapter i during training, push its current L5_mean of
     its own prompt AWAY from anchors[j] for j != i. Use the smooth-max over
     other anchors so the gradient pushes hardest against the closest competitor.
  4. Run the Phase 43 K-capacity sweep on Library B at K ∈ {2, 4, 8}. Compare
     to Library A's K-capacity from Phase 43.
  5. Also report held-out paraphrase routing (Phase 44 protocol with L0_mean)
     for both libraries — the contrastive training might help routing too if
     it makes adapters more "themselves" in their L5 effect.

Headline question: does the contrastive loss lift the K=4 ceiling? If yes,
Application 3 generalizes to K > 2 without per-adapter scale reduction.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/phase46_l5_contrastive.py
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
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.phase31_weighted_pool import cosine
from experiments.identity_ae.phase43_k_capacity import stack_k_state_dicts
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK_BASE  = 128
ALPHA_BASE = 256
N_STEPS    = 150
GEN_TOKENS = 200
LAYER      = 5

CONTRAST_WEIGHT = 0.1     # λ in nt_loss + λ * contrast_loss
CONTRAST_TEMP   = 0.1     # smooth-max temperature for the contrastive softmax


# ----------------------------------------------------------------
# L5 mean with gradients enabled (for contrastive backward pass)
# ----------------------------------------------------------------
def l5_mean_grad(model, ids_t):
    """Layer-5 hidden state mean. Returns (D,) with grad enabled."""
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == LAYER:
            return h.mean(dim=1).squeeze(0)
    return h.mean(dim=1).squeeze(0)


@torch.no_grad()
def l5_mean_nograd(model, ids_t):
    """Same but no grad — for computing anchors and inference engrams."""
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == LAYER:
            return h.mean(dim=1).squeeze(0)
    return h.mean(dim=1).squeeze(0)


@torch.no_grad()
def l0_mean_nograd(model, ids_t):
    """L0 mean (token embedding mean) — for routing keys."""
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0)


# ----------------------------------------------------------------
# Contrastive training: extends train_adapter_multipara with an L5
# contrastive loss against `other_anchors` (all anchors except mine).
# ----------------------------------------------------------------
def train_adapter_contrastive(model, passage, prompts_with_answers, prompt_only,
                                tokenizer, device, other_anchors,
                                n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR,
                                contrast_weight=CONTRAST_WEIGHT,
                                contrast_temp=CONTRAST_TEMP):
    """Like train_adapter_multipara but adds an L5-contrastive loss against
    other_anchors. The contrastive loss pushes the current L5_mean(prompt_only)
    away from every other anchor, with a smooth-max focus on the closest
    (worst-case) competitor."""

    # Pre-tokenize sources for the standard next-token loss
    sources = []
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    sources.append(torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device))
    for pa in prompts_with_answers:
        ids = tokenizer.encode(pa, add_special_tokens=False)
        sources.append(torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device))

    # Pre-tokenize the prompt-only string for the contrastive loss
    prompt_ids = tokenizer.encode(prompt_only, add_special_tokens=False)
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)

    # other_anchors: (K, D) tensor on device, no grad
    other_anchors = other_anchors.to(device).detach()

    params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()

    for step in range(n_steps):
        # ----- Standard next-token loss -----
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        nt_loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                    ids_t[:, 1:].reshape(-1))

        # ----- L5 contrastive loss against other anchors -----
        # Compute current L5_mean of the prompt under the (trainable) LoRA
        l5_current = l5_mean_grad(model, prompt_t)   # (D,)
        # Cosine similarities to all other anchors
        l5_norm = l5_current / (l5_current.norm() + 1e-8)
        anc_norm = other_anchors / (other_anchors.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (anc_norm @ l5_norm)   # (K,)
        # Smooth max — focus on the closest competitor
        contrast = torch.logsumexp(sims / contrast_temp, dim=0) * contrast_temp
        # We want to MINIMIZE this (push the current vector away from competitors)

        loss = nt_loss + contrast_weight * contrast

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

    model.eval()


# ----------------------------------------------------------------
# K-capacity sweep (Phase 43-style) on a given library
# ----------------------------------------------------------------
def k_capacity_test(library_sds, library_meta, label, device, tokenizer):
    """Run the K-capacity sweep at K ∈ {2, 4, 8} on a precomputed library
    of rank-128 adapters. Returns a dict of results keyed by K."""

    NUM, ENT, TECH, FACT = list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))

    # Build trials per K
    trials_by_k = {}

    trials_by_k[2] = []
    for i in range(5):
        idx = [NUM[i], ENT[i]]
        trials_by_k[2].append({
            "indices": idx,
            "passkeys": [library_meta[j]["passkey"] for j in idx],
            "query": " Also, ".join([library_meta[j]["prompt"] for j in idx]),
        })

    trials_by_k[4] = []
    for i in range(5):
        idx = [NUM[i], ENT[i], TECH[i], FACT[i]]
        trials_by_k[4].append({
            "indices": idx,
            "passkeys": [library_meta[j]["passkey"] for j in idx],
            "query": " Also, ".join([library_meta[j]["prompt"] for j in idx]),
        })

    trials_by_k[8] = []
    for i in range(5):
        j = (i + 1) % 5
        idx = [NUM[i], NUM[j], ENT[i], ENT[j], TECH[i], TECH[j], FACT[i], FACT[j]]
        trials_by_k[8].append({
            "indices": idx,
            "passkeys": [library_meta[k]["passkey"] for k in idx],
            "query": " Also, ".join([library_meta[k]["prompt"] for k in idx]),
        })

    results = {}
    for K in [2, 4, 8]:
        print(f"\n  K={K} (rank {K*RANK_BASE}, alpha {K*ALPHA_BASE})")
        # Reload model fresh and apply LoRA at K-rank
        model, _ = load_model(device)
        apply_lora(model, rank=K*RANK_BASE, alpha=K*ALPHA_BASE,
                   target_modules=L45_TARGETS)

        all_correct, total_pks, found_pks = 0, 0, 0
        per_trial = []
        for ti, trial in enumerate(trials_by_k[K]):
            sds = [library_sds[idx] for idx in trial["indices"]]
            sd_combined = stack_k_state_dicts(sds)
            sd_gpu = {k: v.to(device) for k, v in sd_combined.items()}
            load_lora_state_dict(model, sd_gpu)
            gen = generate_greedy(model, trial["query"], tokenizer, device, GEN_TOKENS)
            hits = [check_passkey(gen, pk) for pk in trial["passkeys"]]
            n_hits = sum(hits)
            total_pks += len(trial["passkeys"])
            found_pks += n_hits
            if n_hits == len(trial["passkeys"]):
                all_correct += 1
            per_trial.append({
                "indices": trial["indices"],
                "n_hits": n_hits,
                "n_total": len(trial["passkeys"]),
            })

        mean_fraction = found_pks / total_pks if total_pks > 0 else 0.0
        print(f"    ALL_K: {all_correct}/5  mean fraction: {mean_fraction:.0%} "
              f"({found_pks}/{total_pks})")
        results[K] = {
            "all_correct": all_correct,
            "mean_fraction": mean_fraction,
            "found_pks": found_pks,
            "total_pks": total_pks,
            "per_trial": per_trial,
        }
        del model
        torch.cuda.empty_cache()
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase46")
    results_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    tests = stratified_tests()
    print(f"Stratified: 5 numeric + 5 entity + 5 technical + 5 fact = {len(tests)}")
    print(f"Contrast weight λ = {CONTRAST_WEIGHT}, smooth-max T = {CONTRAST_TEMP}\n")

    # ============================================================
    # PHASE A: Build Library A (Phase 38b normal) and compute L5 anchors
    # ============================================================
    print("=" * 60)
    print("PHASE A: BUILD LIBRARY A (normal Phase 38b)")
    print("=" * 60)
    model, _ = load_model(device)
    apply_lora(model, rank=RANK_BASE, alpha=ALPHA_BASE, target_modules=L45_TARGETS)

    library_A_sds = []
    library_A_meta = []
    library_A_prompts = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        train_adapter_multipara(model, test["passage"], prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library_A_sds.append(sd)
        library_A_meta.append(dict(test))
        library_A_prompts.append(train_prompts)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s}  ({time.time()-t0:.0f}s)")

    # ============================================================
    # PHASE A.5: Compute L5 anchors from Library A
    # ============================================================
    print("\nComputing L5 anchors from Library A...")
    anchors = []  # 20 vectors, one per adapter
    for i, sd in enumerate(library_A_sds):
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)
        prompt_ids = tokenizer.encode(library_A_meta[i]["prompt"], add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        l5 = l5_mean_nograd(model, prompt_t).cpu()
        anchors.append(l5)
    anchors = torch.stack(anchors)   # (20, D)
    print(f"  Anchor matrix: {anchors.shape}")
    # Sanity check: typical anchor-to-anchor cosine
    a_norm = anchors / (anchors.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = a_norm @ a_norm.T   # (20, 20)
    off_diag = sim_matrix - torch.eye(20)
    mean_off = off_diag.sum() / (20 * 19)
    print(f"  Library A anchors — mean off-diagonal cosine similarity: {mean_off.item():.4f}")
    print(f"  (this is what the contrastive loss will try to reduce)")

    # ============================================================
    # PHASE B: Build Library B with contrastive training
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE B: BUILD LIBRARY B (Phase 38b + L5-contrastive)")
    print(f"{'='*60}")

    library_B_sds = []
    t0 = time.time()
    for i, test in enumerate(tests):
        reset_lora_to_zero(model)
        train_prompts = [test["prompt"]] + train_paraphrase(test)
        prompts_with_answers = [f"{p} {test['passkey']}" for p in train_prompts]
        # Other anchors = all 19 anchors except this index
        other_idx = [j for j in range(20) if j != i]
        other_anchors = anchors[other_idx]   # (19, D)

        train_adapter_contrastive(model, test["passage"], prompts_with_answers,
                                    test["prompt"], tokenizer, device,
                                    other_anchors,
                                    n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        library_B_sds.append(sd)
        if (i + 1) % 5 == 0:
            print(f"  [{i+1:2d}/20] absorbed {test['type']:9s} (contrastive)  "
                  f"({time.time()-t0:.0f}s)")

    # Compute new L5 anchors from Library B for diagnostic
    print("\nComputing L5 anchors from Library B (diagnostic)...")
    anchors_B = []
    for i, sd in enumerate(library_B_sds):
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)
        prompt_ids = tokenizer.encode(library_A_meta[i]["prompt"], add_special_tokens=False)
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
        l5 = l5_mean_nograd(model, prompt_t).cpu()
        anchors_B.append(l5)
    anchors_B = torch.stack(anchors_B)
    a_norm = anchors_B / (anchors_B.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = a_norm @ a_norm.T
    off_diag = sim_matrix - torch.eye(20)
    mean_off_B = off_diag.sum() / (20 * 19)
    print(f"  Library B anchors — mean off-diagonal cosine similarity: {mean_off_B.item():.4f}")
    print(f"  (Library A: {mean_off.item():.4f} — contrastive {'reduced' if mean_off_B < mean_off else 'INCREASED'} the overlap)")

    del model
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE C: Sanity check — same-prompt retrieval on Library B
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE C: SAME-PROMPT RETRIEVAL SANITY (Library B)")
    print(f"{'='*60}")
    model, _ = load_model(device)
    apply_lora(model, rank=RANK_BASE, alpha=ALPHA_BASE, target_modules=L45_TARGETS)
    n_correct = 0
    for i, test in enumerate(tests):
        sd = library_B_sds[i]
        sd_gpu = {k: v.to(device) for k, v in sd.items()}
        load_lora_state_dict(model, sd_gpu)
        gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
        if check_passkey(gen, test["passkey"]):
            n_correct += 1
    print(f"  Library B same-prompt retrieval: {n_correct}/20 ({n_correct/20:.0%})")
    print(f"  (Library A baseline: typically 20/20)")
    del model
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE D: K-capacity test on Library B
    # ============================================================
    print(f"\n{'='*60}")
    print("PHASE D: K-CAPACITY SWEEP ON LIBRARY B")
    print(f"{'='*60}")
    print("  (Phase 43 reference for Library A: K=2 → 4/5 (90%), K=4 → 0/5 (35%), K=8 → 0/5 (5%))")
    results_B = k_capacity_test(library_B_sds, library_A_meta,
                                 "Library B", device, tokenizer)

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*72}")
    print("PHASE 46 SUMMARY: L5-contrastive training and K-capacity")
    print(f"{'='*72}")
    print(f"\n  L5 anchor cosine overlap (lower = more orthogonal adapters):")
    print(f"    Library A (no contrastive): {mean_off.item():.4f}")
    print(f"    Library B (contrastive):    {mean_off_B.item():.4f}")
    print(f"    Reduction: {(mean_off.item() - mean_off_B.item()):.4f}")
    print(f"\n  K-capacity (Library A from Phase 43, Library B this run):")
    print(f"  {'K':>3}  {'A: ALL_K':>10}  {'A: mean':>10}  {'B: ALL_K':>10}  {'B: mean':>10}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    a_reference = {
        2: ("4/5", "90%"),
        4: ("0/5", "35%"),
        8: ("0/5", "5%"),
    }
    for K in [2, 4, 8]:
        a_all, a_mean = a_reference[K]
        r = results_B[K]
        b_all = f"{r['all_correct']}/5"
        b_mean = f"{r['mean_fraction']:.0%}"
        print(f"  {K:>3}  {a_all:>10}  {a_mean:>10}  {b_all:>10}  {b_mean:>10}")

    out = {
        "rank_base": RANK_BASE,
        "n_steps":   N_STEPS,
        "contrast_weight": CONTRAST_WEIGHT,
        "contrast_temp":   CONTRAST_TEMP,
        "anchor_overlap": {
            "library_A": mean_off.item(),
            "library_B": mean_off_B.item(),
        },
        "library_B_same_prompt": n_correct,
        "library_B_k_capacity":  results_B,
    }
    with open(results_dir / "l5_contrastive.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
