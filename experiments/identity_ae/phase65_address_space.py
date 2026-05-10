"""Phase 65: Is the W projection already in the base?

Phase 47 trained a 1024x1024 linear W with InfoNCE so that L0 mean-pooled
queries route correctly against L5 mean-pooled adapter keys. The address-as-
softmax claim is that this W maps L0 engrams into the subspace where the
base model's own selection mechanism operates — i.e. that L5 key-space is
already the base's "address space" and W is finding it, not constructing it.

This script tests the claim by replacing the trained W with W' constructed
from base parameters only (no contrastive training), holding the rest of
the Phase 47 setup fixed, and reporting whether routing accuracy survives.

Conditions (each is a (D, D) matrix W' applied as q_l0 @ W', then cosine
argmax over the library's L5 keys — except C0b which routes against L0 keys):

  C0a  W' = I,                    keys = L5  | cross-space floor (~chance)
  C0b  W' = I,                    keys = L0  | same-space (Phase 44) baseline
  C1   trained Phase 47 W,        keys = L5  | reproduction target
  C2   prod_i out_proj_i.T,       keys = L5  | linear approx to L0->L5 Jacobian
  C3   K(block 0).T,              keys = L5  | K at engram extraction layer
  C4   K(block 5).T,              keys = L5  | K at destination layer
  C5   OLS L5 ~ L0,               keys = L5  | direct test: is the bridge in the base?
  C6   random orthogonal,         keys = L5  | "any d x d projection" control

Stages (cached on disk so re-running condition logic is cheap):
  A. Build LoRA library (20 adapters) + extract base-model L0 + L5 keys
  B. Train Phase 47 W via InfoNCE
  C. For each condition: route 60 held-out paraphrases, generate, score.

Per condition we log: routing top-1, retrieval top-1, mean true-key cosine,
mean best-wrong-key cosine, gap (true - best_wrong) distribution.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_address_space.py
"""

import json
import math
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
GEN_TOKENS = 50
D = 1024


# ============================================================
# Helpers
# ============================================================

@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def l5_mean(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h.mean(dim=1).squeeze(0).detach().cpu()


def cos_against_keys(q, keys_per_adapter):
    """Return (best_adapter, best_score, true_score, best_wrong_score) where:
       - best_score = max cosine across all keys
       - true_score = max cosine across true adapter's keys (caller passes none here;
         we just return all per-adapter best scores).
    Returns (best_adapter, per_adapter_best_score: list[float])."""
    q_n = q / (q.norm() + 1e-8)
    per_adapter = []
    for keys in keys_per_adapter:
        best = -2.0
        for k in keys:
            kn = k / (k.norm() + 1e-8)
            s = float(torch.dot(q_n, kn))
            if s > best:
                best = s
        per_adapter.append(best)
    best_a = max(range(len(per_adapter)), key=lambda i: per_adapter[i])
    return best_a, per_adapter


# ============================================================
# Stage A: build library + base-model L0/L5 keys, cached
# ============================================================

def stage_a_build_library(model, tokenizer, device, cache_path: Path):
    if cache_path.exists():
        print(f"[A] cache hit, loading library from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        return blob["library"], blob["keys_l0"], blob["keys_l5"]

    tests = stratified_tests()
    print(f"[A] building library: {len(tests)} adapters x rank-{RANK} LoRA, "
          f"{N_STEPS} steps each")

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
            print(f"    [{i+1:2d}/{len(tests)}] absorbed {test['type']:9s}  "
                  f"({time.time()-t0:.0f}s elapsed)")

    print(f"[A] extracting base-model L0 + L5 keys (LoRA reset)")
    reset_lora_to_zero(model)
    keys_l0, keys_l5 = [], []
    for entry in library:
        ks_l0, ks_l5 = [], []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            ks_l0.append(l0_mean(model, ids_t))
            ks_l5.append(l5_mean(model, ids_t))
        keys_l0.append(ks_l0)
        keys_l5.append(ks_l5)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"library": library, "keys_l0": keys_l0, "keys_l5": keys_l5},
               cache_path)
    print(f"[A] cached library to {cache_path}")
    return library, keys_l0, keys_l5


# ============================================================
# Stage B: train Phase 47 W via InfoNCE, cached
# ============================================================

def stage_b_train_W(keys_l0, keys_l5, device, cache_path: Path):
    if cache_path.exists():
        print(f"[B] cache hit, loading W from {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    print(f"[B] training Phase 47 W: {PROJ_STEPS} InfoNCE steps, lr={PROJ_LR}")

    flat_l0, flat_l5, flat_a = [], [], []
    for ai, (ks_l0, ks_l5) in enumerate(zip(keys_l0, keys_l5)):
        for k0, k5 in zip(ks_l0, ks_l5):
            flat_l0.append(k0)
            flat_l5.append(k5)
            flat_a.append(ai)
    flat_l0 = torch.stack(flat_l0).to(device)
    flat_l5 = torch.stack(flat_l5).to(device)
    flat_a = torch.tensor(flat_a, device=device)

    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    optim = torch.optim.Adam(W.parameters(), lr=PROJ_LR)
    flat_l5_norm = flat_l5 / (flat_l5.norm(dim=-1, keepdim=True) + 1e-8)

    for step in range(PROJ_STEPS):
        proj = W(flat_l0)
        proj_norm = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)
        sims = proj_norm @ flat_l5_norm.T
        same = flat_a.unsqueeze(0) == flat_a.unsqueeze(1)
        logits = sims / PROJ_TEMP
        log_probs = F.log_softmax(logits, dim=-1)
        pos_mask = same.float()
        pos_count = pos_mask.sum(dim=-1)
        pos_log_prob = (log_probs * pos_mask).sum(dim=-1) / (pos_count + 1e-8)
        loss = -pos_log_prob.mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                top1 = sims.argmax(dim=-1)
                acc = (flat_a[top1] == flat_a).float().mean().item()
            print(f"    step {step+1}: loss {loss.item():.4f}, train top-1 acc {acc:.0%}")

    # Convert to (D, D) matrix in q @ M form: M = W.weight.T
    M_trained = W.weight.detach().cpu().T.contiguous()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(M_trained, cache_path)
    print(f"[B] cached trained W to {cache_path}")
    return M_trained


# ============================================================
# Build base-derived W' for each condition
# ============================================================

def build_M_outproj_composition(model):
    """C2: product of out_proj weights through the stack, in row-vector form.

    Each block applies x -> x @ out_proj.weight.T. Composition over blocks 0..5:
        x -> x @ (out_proj_0.weight.T @ out_proj_1.weight.T @ ... @ out_proj_5.weight.T)
    Returns the product (D, D), no SVD truncation (already D x D).
    Note: ignores residual + MLP, so this is a coarse linear approximation.
    """
    M = torch.eye(D)
    for blk in model.blocks:
        # Walk through to base weight even if wrapped in LoRALayer (.weight returns base)
        Wo = blk.attn.out_proj.weight.detach().cpu()  # (D, D)
        M = M @ Wo.T
    return M.contiguous()


def build_M_K_at_block(model, block_idx):
    """C3/C4: K projection at a given block, applied in row-vector form.
    K is the second third of the fused qkv weight (out=3D, in=D).
    For row vec q, the K-projected query is q @ K.T. So M = K.T.
    """
    qkv_w = model.blocks[block_idx].attn.qkv.weight.detach().cpu()  # (3D, D)
    K = qkv_w[D:2*D, :]  # (D, D)
    return K.T.contiguous()


def build_M_OLS(keys_l0, keys_l5):
    """C5: minimum-norm linear regression solving Y ~ X M on the 80 training pairs.

    With N=80 < D=1024, X^T X is rank-deficient; we use the minimum-norm
    pseudoinverse via X X^T (N, N), full-rank in practice:
        M = X^T (X X^T)^{-1} Y,  shape (D, D).

    For new q (D,), q @ M = (similarity of q to each training X) @ Y, i.e. linear
    kernel regression. This is the "L0 -> L5 mapping evaluated at training points,"
    the most direct base-only analog to what the trained W is doing.
    """
    X_rows, Y_rows = [], []
    for ks_l0, ks_l5 in zip(keys_l0, keys_l5):
        for k0, k5 in zip(ks_l0, ks_l5):
            X_rows.append(k0)
            Y_rows.append(k5)
    X = torch.stack(X_rows).double()  # (N, D)
    Y = torch.stack(Y_rows).double()  # (N, D)
    XXt = X @ X.T  # (N, N)
    # Tiny ridge for numerical safety
    XXt = XXt + 1e-6 * torch.eye(XXt.shape[0], dtype=XXt.dtype)
    M = X.T @ torch.linalg.solve(XXt, Y)  # (D, D)
    return M.float().contiguous()


def build_M_random_orthogonal(seed=0):
    """C6: random orthogonal (D, D) matrix via QR of a Gaussian."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, D, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.contiguous()


# ============================================================
# Stage C: run a single condition
# ============================================================

def run_condition(name, M, key_space, keys_l0_cpu, keys_l5_cpu, library,
                   model, tokenizer, device, generate=True):
    """key_space in {'L0', 'L5'} selects which keys to route against."""
    keys = keys_l0_cpu if key_space == "L0" else keys_l5_cpu

    if M is not None:
        M = M.cpu()

    n_routed, n_retr = 0, 0
    per_type = {"numeric": 0, "entity": 0, "technical": 0, "fact": 0}
    trial_log = []

    for i, entry in enumerate(library):
        ho_paras = held_out_paraphrase(entry["test"])
        for slot_idx, para in enumerate(ho_paras):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            q_l0 = l0_mean(model, ids_t)            # (D,)

            # Project the query through M (None means identity)
            if M is None:
                q_proj = q_l0
            else:
                q_proj = q_l0 @ M

            best_a, per_adapter_best = cos_against_keys(q_proj, keys)
            true_score = per_adapter_best[i]
            best_wrong = max(s for j, s in enumerate(per_adapter_best) if j != i)

            routed_correct = (best_a == i)
            n_routed += int(routed_correct)

            substring_hit = 0
            if generate:
                sd = library[best_a]["sd"]
                sd_gpu = {k: v.to(device) for k, v in sd.items()}
                load_lora_state_dict(model, sd_gpu)
                gen = generate_greedy(model, para, tokenizer, device, GEN_TOKENS)
                if check_passkey(gen, entry["test"]["passkey"]):
                    n_retr += 1
                    per_type[entry["test"]["type"]] += 1
                    substring_hit = 1

            trial_log.append({
                "true_idx": i, "routed_idx": best_a,
                "type": entry["test"]["type"],
                "true_score": true_score,
                "best_score": per_adapter_best[best_a],
                "best_wrong": best_wrong,
                "gap": true_score - best_wrong,
                "routed_correct": routed_correct,
                "substring_hit": substring_hit,
            })

    n_total = len(trial_log)
    true_scores = [t["true_score"] for t in trial_log]
    best_wrongs = [t["best_wrong"] for t in trial_log]
    gaps = [t["gap"] for t in trial_log]

    summary = {
        "name": name,
        "n_total": n_total,
        "n_routed": n_routed,
        "n_retrieval": n_retr,
        "routing_acc": n_routed / n_total,
        "retrieval_acc": n_retr / n_total if generate else None,
        "per_type_retrieval": per_type if generate else None,
        "mean_true_cos": float(sum(true_scores) / n_total),
        "mean_best_wrong_cos": float(sum(best_wrongs) / n_total),
        "mean_gap": float(sum(gaps) / n_total),
        "min_gap": float(min(gaps)),
        "max_gap": float(max(gaps)),
        "n_negative_gap": sum(1 for g in gaps if g < 0),
        "trials": trial_log,
    }
    return summary


def fmt_summary(s):
    line1 = (f"  {s['name']:32s}  routing {s['n_routed']:2d}/{s['n_total']:2d} "
             f"({s['routing_acc']:.0%})")
    if s["retrieval_acc"] is not None:
        line1 += f"  retrieval {s['n_retrieval']:2d}/{s['n_total']:2d} ({s['retrieval_acc']:.0%})"
    line2 = (f"    cos(true)={s['mean_true_cos']:+.3f}  "
             f"cos(best_wrong)={s['mean_best_wrong_cos']:+.3f}  "
             f"gap={s['mean_gap']:+.3f}  "
             f"neg_gap={s['n_negative_gap']}/{s['n_total']}")
    return line1 + "\n" + line2


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase65")
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = results_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    torch.manual_seed(0)

    # ---- Stage A: build library + base-model L0/L5 keys ----
    model, _ = load_model(device)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    library, keys_l0, keys_l5 = stage_a_build_library(
        model, tokenizer, device,
        cache_path=cache_dir / "library.pt",
    )

    # ---- Stage B: train Phase 47 W ----
    M_trained = stage_b_train_W(
        keys_l0, keys_l5, device,
        cache_path=cache_dir / "M_trained.pt",
    )

    # ---- Build base-derived W' for each condition ----
    print("\n[C] constructing base-derived W' matrices...")
    reset_lora_to_zero(model)

    M_outproj = build_M_outproj_composition(model)
    M_Kblock0 = build_M_K_at_block(model, block_idx=0)
    M_Kblock5 = build_M_K_at_block(model, block_idx=5)
    M_OLS     = build_M_OLS(keys_l0, keys_l5)
    M_random  = build_M_random_orthogonal(seed=0)

    # Sanity: matrix Frobenius norms
    def fnorm(M): return float(M.norm())
    print(f"    ||M_trained||_F   = {fnorm(M_trained):.2f}")
    print(f"    ||M_outproj||_F   = {fnorm(M_outproj):.2f}")
    print(f"    ||M_K_block0||_F  = {fnorm(M_Kblock0):.2f}")
    print(f"    ||M_K_block5||_F  = {fnorm(M_Kblock5):.2f}")
    print(f"    ||M_OLS||_F       = {fnorm(M_OLS):.2f}")
    print(f"    ||M_random||_F    = {fnorm(M_random):.2f}")

    # ---- Keys CPU views ----
    keys_l0_cpu = [[k.cpu() if k.is_cuda else k for k in ks] for ks in keys_l0]
    keys_l5_cpu = [[k.cpu() if k.is_cuda else k for k in ks] for ks in keys_l5]

    # ---- Run all conditions ----
    print(f"\n[C] running 8 conditions x 60 held-out trials each")
    print(f"{'='*72}")
    conditions = [
        ("C0a I,         vs L5", None,         "L5"),
        ("C0b I,         vs L0", None,         "L0"),
        ("C1  trained W, vs L5", M_trained,    "L5"),
        ("C2  out_proj_, vs L5", M_outproj,    "L5"),
        ("C3  K block 0, vs L5", M_Kblock0,    "L5"),
        ("C4  K block 5, vs L5", M_Kblock5,    "L5"),
        ("C5  OLS reg ,  vs L5", M_OLS,        "L5"),
        ("C6  rand orth, vs L5", M_random,     "L5"),
    ]

    summaries = []
    for name, M, key_space in conditions:
        t0 = time.time()
        s = run_condition(name, M, key_space, keys_l0_cpu, keys_l5_cpu,
                          library, model, tokenizer, device, generate=True)
        dt = time.time() - t0
        print(fmt_summary(s) + f"   [{dt:.0f}s]")
        summaries.append(s)

    # ---- Save full results ----
    out_path = results_dir / "phase65_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "rank": RANK, "alpha": ALPHA, "n_steps": N_STEPS,
                "proj_steps": PROJ_STEPS, "proj_lr": PROJ_LR, "proj_temp": PROJ_TEMP,
                "gen_tokens": GEN_TOKENS, "d_model": D,
                "n_adapters": len(library),
            },
            "summaries": [{k: v for k, v in s.items() if k != "trials"} for s in summaries],
            "trials": {s["name"]: s["trials"] for s in summaries},
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

    # ---- Verdict against pre-committed decision rules ----
    print(f"\n{'='*72}\nDECISION-RULE VERDICT")
    print(f"{'='*72}")
    by_name = {s["name"]: s for s in summaries}
    c1 = by_name["C1  trained W, vs L5"]["routing_acc"]
    c0a = by_name["C0a I,         vs L5"]["routing_acc"]
    c6 = by_name["C6  rand orth, vs L5"]["routing_acc"]
    base_only = {n: by_name[n]["routing_acc"] for n in
                 ["C2  out_proj_, vs L5", "C3  K block 0, vs L5",
                  "C4  K block 5, vs L5", "C5  OLS reg ,  vs L5"]}
    best_base = max(base_only.values())
    best_base_name = max(base_only, key=base_only.get)

    print(f"  C1 trained W routing acc:    {c1:.0%}")
    print(f"  C0a cross-space floor:       {c0a:.0%}")
    print(f"  C6 random orthogonal:        {c6:.0%}")
    print(f"  Best base-derived W':        {best_base:.0%}  ({best_base_name})")

    if c6 >= 0.90:
        print(f"\n  C6-degeneracy: random projection >=90%; experiment doesn't decide.")
    elif best_base >= c1 - 0.05 and c6 <= c0a + 0.05:
        print(f"\n  STRONG support: base-derived W' within 5pts of trained W,")
        print(f"  random control near floor. Address structure is in the base.")
    elif best_base >= 0.90 and best_base < c1 - 0.05:
        print(f"\n  WEAK support: base-derived W' >=90% but well below trained W.")
        print(f"  Base contains partial address structure; InfoNCE refines it.")
    else:
        print(f"\n  NEGATIVE: all base-derived conditions near control floor.")
        print(f"  W is doing real geometric work, not surfacing pre-existing structure.")


if __name__ == "__main__":
    main()
