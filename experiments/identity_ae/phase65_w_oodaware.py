"""Phase 65 rescue: train W with OOD-aware contrastive loss, then re-eval on
the hard near-neighbor OOD set.

The hard-OOD specificity ablation found C0b's natural separation collapses on
same-template-different-entity queries (FP 0% -> 92%). C1 is symmetrically
broken because Phase 47's InfoNCE never saw OOD negatives. This script trains
a new W with OOD-aware contrastive loss and tests whether C1 acquires real
absolute-distance separation.

Training objective:
    L(W) = InfoNCE(in-library, in-library) + lambda_ood * hinge_OOD(W)

  hinge_OOD(W) = mean over OOD L0 queries of relu( max in-library cos(W q, k) - margin )

Sample 200 OOD passages from WikiText-2 (length-matched to in-library); cache
their (L0, L5) pairs and use only the L0s for the hinge. The original 80
in-library pairs continue to drive the InfoNCE term unchanged.

Sweep lambda_ood in {0.5, 2.0, 8.0} with margin=0.10 to map the trade-off.
For each, evaluate:
  - in-library routing top-1 (via cosine argmax over 80 L5 keys)
  - hard-OOD: FP at 95% in-library recall, separation, 0%-FP recall

Reuses Phase 65 cache; no library rebuild.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_w_oodaware.py
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    load_model, generate_passkeys,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora


SEED = 0
N_OOD_TRAIN = 200          # OOD passages used in training
PROJ_STEPS = 500
PROJ_LR = 1e-3
PROJ_TEMP = 0.05
HINGE_MARGIN = 0.10
LAMBDA_SWEEP = [0.5, 2.0, 8.0]
D = 1024
PHASE65_DIR = Path("results/identity_ae/phase65")
CACHE_DIR = PHASE65_DIR / "cache"


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def l5_mean(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h.mean(dim=1).squeeze(0).detach().cpu()


def cosine(a, b):
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


def best_match(q, keys_per_adapter):
    best_a, best_s = -1, -2.0
    for ai, keys in enumerate(keys_per_adapter):
        for k in keys:
            s = cosine(q, k)
            if s > best_s:
                best_s, best_a = s, ai
    return best_a, best_s


def quantile(xs, q):
    xs = sorted(xs); n = len(xs)
    if n == 0: return float("nan")
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def fraction_above(xs, t):
    return sum(1 for x in xs if x > t) / max(len(xs), 1)


# ============================================================
# OOD training-pair extraction
# ============================================================

def extract_ood_pairs(model, tokenizer, device, n=N_OOD_TRAIN, seed=SEED,
                      cache_path=None):
    if cache_path is not None and cache_path.exists():
        print(f"[A] OOD pair cache hit, loading from {cache_path}")
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        return blob["L0"], blob["L5"]

    from datasets import load_dataset
    print(f"[A] sampling {n} OOD passages from WikiText-2 train")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rng = random.Random(seed)
    # Length-match in-library distribution: 6-16 tokens
    in_lengths = []
    for test in stratified_tests():
        for q in held_out_paraphrase(test):
            in_lengths.append(len(tokenizer.encode(q, add_special_tokens=False)))

    texts = [t for t in ds["text"] if t and len(t.split()) > 20]
    rng.shuffle(texts)

    L0s, L5s = [], []
    cursor = 0
    while len(L0s) < n and cursor < len(texts):
        L = rng.choice(in_lengths)
        text = texts[cursor]; cursor += 1
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < L: continue
        start = rng.randint(0, len(ids) - L)
        chunk_ids = ids[start:start + L]
        ids_t = torch.tensor(chunk_ids, dtype=torch.long).unsqueeze(0).to(device)
        L0s.append(l0_mean(model, ids_t))
        L5s.append(l5_mean(model, ids_t))

    L0 = torch.stack(L0s); L5 = torch.stack(L5s)
    print(f"[A] extracted {L0.shape[0]} OOD (L0, L5) pairs")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"L0": L0, "L5": L5}, cache_path)
    return L0, L5


# ============================================================
# Train W with combined InfoNCE + OOD hinge
# ============================================================

def train_W_ood_aware(in_l0, in_l5, in_a, ood_l0, lam, margin,
                       device, n_steps=PROJ_STEPS, lr=PROJ_LR, temp=PROJ_TEMP):
    """in_l0, in_l5: (N, D); in_a: (N,) adapter idx; ood_l0: (M, D)."""
    in_l0  = in_l0.to(device)
    in_l5  = in_l5.to(device)
    in_a   = in_a.to(device)
    ood_l0 = ood_l0.to(device)

    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    optim = torch.optim.Adam(W.parameters(), lr=lr)
    in_l5_norm = in_l5 / (in_l5.norm(dim=-1, keepdim=True) + 1e-8)

    for step in range(n_steps):
        # InfoNCE on in-library only
        proj_in = W(in_l0)
        proj_in_norm = proj_in / (proj_in.norm(dim=-1, keepdim=True) + 1e-8)
        sims_in = proj_in_norm @ in_l5_norm.T  # (N, N)
        same = in_a.unsqueeze(0) == in_a.unsqueeze(1)
        log_probs = F.log_softmax(sims_in / temp, dim=-1)
        pos_count = same.float().sum(dim=-1)
        info_loss = -((log_probs * same.float()).sum(dim=-1) / (pos_count + 1e-8)).mean()

        # Hinge on OOD: max in-library cosine should be below margin
        proj_ood = W(ood_l0)
        proj_ood_norm = proj_ood / (proj_ood.norm(dim=-1, keepdim=True) + 1e-8)
        sims_ood = proj_ood_norm @ in_l5_norm.T  # (M, N)
        max_ood_cos = sims_ood.max(dim=-1).values  # (M,)
        hinge_loss = F.relu(max_ood_cos - margin).mean()

        loss = info_loss + lam * hinge_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                top1 = sims_in.argmax(dim=-1)
                acc = (in_a[top1] == in_a).float().mean().item()
                mean_max_ood = max_ood_cos.mean().item()
            print(f"    step {step+1}: info {info_loss.item():.3f}  "
                  f"hinge {hinge_loss.item():.3f}  "
                  f"train top-1 {acc:.0%}  "
                  f"mean OOD max-cos {mean_max_ood:+.3f}")

    return W.weight.detach().cpu().T.contiguous()


def build_hard_ood_tests():
    from collections import defaultdict
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][5:10] + by_type["entity"][5:10]
            + by_type["technical"][5:10] + by_type["fact"][5:10])


# ============================================================
# Eval helpers
# ============================================================

def eval_in_library_routing(M, library, keys_l5_cpu, model, tokenizer, device):
    """Routing-only (no generation) over the 60 in-library held-out paraphrases."""
    n_routed, n_total = 0, 0
    scores = []
    for i, entry in enumerate(library):
        for para in held_out_paraphrase(entry["test"]):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            q_l0 = l0_mean(model, ids_t)
            q_proj = q_l0 @ M
            best_a, best_s = best_match(q_proj, keys_l5_cpu)
            n_routed += int(best_a == i)
            n_total += 1
            scores.append(best_s)
    return n_routed, n_total, scores


def eval_hard_ood_routing(M, library, keys_l5_cpu, model, tokenizer, device):
    """Score hard-OOD queries (60 = 20 entities x 3 paraphrases)."""
    hard_ood = build_hard_ood_tests()
    scores, routes = [], []
    for ood_test in hard_ood:
        for para in held_out_paraphrase(ood_test):
            reset_lora_to_zero(model)
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            q_l0 = l0_mean(model, ids_t)
            q_proj = q_l0 @ M
            best_a, best_s = best_match(q_proj, keys_l5_cpu)
            scores.append(best_s)
            routes.append(best_a)
    return scores, routes


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE65_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Cache ----
    print(f"[A] loading Phase 65 cache from {CACHE_DIR}")
    blob = torch.load(CACHE_DIR / "library.pt", map_location="cpu", weights_only=False)
    library, keys_l0, keys_l5 = blob["library"], blob["keys_l0"], blob["keys_l5"]
    M_trained_orig = torch.load(CACHE_DIR / "M_trained.pt",
                                  map_location="cpu", weights_only=False)

    # Build flat in-library tensors
    flat_l0, flat_l5, flat_a = [], [], []
    for ai, (ks_l0, ks_l5) in enumerate(zip(keys_l0, keys_l5)):
        for k0, k5 in zip(ks_l0, ks_l5):
            flat_l0.append(k0); flat_l5.append(k5); flat_a.append(ai)
    flat_l0 = torch.stack(flat_l0)
    flat_l5 = torch.stack(flat_l5)
    flat_a  = torch.tensor(flat_a)

    # Move keys to CPU
    keys_l5_cpu = [[k.cpu() if k.is_cuda else k for k in ks] for ks in keys_l5]

    # ---- Base model + OOD pairs ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    ood_l0, ood_l5 = extract_ood_pairs(
        model, tokenizer, device, n=N_OOD_TRAIN, seed=SEED,
        cache_path=CACHE_DIR / "ood_pairs.pt",
    )

    # ---- Reference: in-library routing under original M_trained ----
    print(f"\n[B] reference (original W from Phase 47) in-library routing:")
    n_in_orig, n_total, in_scores_orig = eval_in_library_routing(
        M_trained_orig, library, keys_l5_cpu, model, tokenizer, device,
    )
    print(f"    in-library top-1: {n_in_orig}/{n_total} ({n_in_orig/n_total:.0%})")
    hard_scores_orig, _ = eval_hard_ood_routing(
        M_trained_orig, library, keys_l5_cpu, model, tokenizer, device,
    )
    print(f"    hard-OOD mean cos: {sum(hard_scores_orig)/len(hard_scores_orig):+.3f}, "
          f"max {max(hard_scores_orig):+.3f}")

    # ---- Sweep lambda_ood ----
    sweep_results = []
    for lam in LAMBDA_SWEEP:
        print(f"\n[C] training W_oodaware with lambda_ood={lam}, margin={HINGE_MARGIN}")
        torch.manual_seed(SEED)  # reset for reproducibility per condition
        M_new = train_W_ood_aware(
            flat_l0, flat_l5, flat_a, ood_l0,
            lam=lam, margin=HINGE_MARGIN, device=device,
        )

        # In-library routing
        n_in, _, in_scores = eval_in_library_routing(
            M_new, library, keys_l5_cpu, model, tokenizer, device,
        )
        # Hard-OOD scoring
        hard_scores, _ = eval_hard_ood_routing(
            M_new, library, keys_l5_cpu, model, tokenizer, device,
        )

        # Calibration
        tau_95 = quantile(in_scores, 0.05)
        fp = fraction_above(hard_scores, tau_95)
        sep = sum(in_scores)/len(in_scores) - sum(hard_scores)/len(hard_scores)
        tau_zero = max(hard_scores) + 1e-9
        rec_zero = fraction_above(in_scores, tau_zero)

        print(f"    in-library top-1: {n_in}/{n_total} ({n_in/n_total:.0%})")
        print(f"    in-library mean {sum(in_scores)/len(in_scores):+.3f}  "
              f"hard-OOD mean {sum(hard_scores)/len(hard_scores):+.3f}  "
              f"sep {sep:+.3f}")
        print(f"    tau@95rec {tau_95:+.3f}  hard-OOD FP {fp:.0%}")
        print(f"    0% hard-OOD FP -> in-library recall {rec_zero:.0%} (tau {tau_zero:+.3f})")

        sweep_results.append({
            "lambda_ood": lam,
            "margin": HINGE_MARGIN,
            "in_library_top1": n_in / n_total,
            "in_library_mean_cos": sum(in_scores) / len(in_scores),
            "hard_ood_mean_cos":   sum(hard_scores) / len(hard_scores),
            "separation":          sep,
            "tau_at_95_recall":    tau_95,
            "hard_ood_fp_at_95":   fp,
            "recall_at_0_fp":      rec_zero,
            "tau_at_0_fp":         tau_zero,
            "in_scores":           in_scores,
            "hard_scores":         hard_scores,
        })

    # ---- Save ----
    out = {
        "config": {
            "n_ood_train": N_OOD_TRAIN, "proj_steps": PROJ_STEPS,
            "proj_lr": PROJ_LR, "proj_temp": PROJ_TEMP,
            "hinge_margin": HINGE_MARGIN, "lambda_sweep": LAMBDA_SWEEP, "seed": SEED,
        },
        "reference_original_W": {
            "in_library_top1": n_in_orig / n_total,
            "in_library_mean_cos": sum(in_scores_orig) / len(in_scores_orig),
            "hard_ood_mean_cos": sum(hard_scores_orig) / len(hard_scores_orig),
            "in_scores": in_scores_orig,
            "hard_scores": hard_scores_orig,
        },
        "sweep": sweep_results,
    }
    out_path = PHASE65_DIR / "w_oodaware.json"
    with open(out_path, "w") as f:
        # The full per-trial scores blow up the JSON; keep them but pretty-print
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Verdict ----
    print(f"\n{'='*72}\nVERDICT")
    print(f"{'='*72}")
    print(f"  original W: in-library {n_in_orig/n_total:.0%}, "
          f"hard-OOD FP @ 95rec ~92% (from prior run)")
    for r in sweep_results:
        print(f"  lambda={r['lambda_ood']:>4}: in-library {r['in_library_top1']:.0%}, "
              f"hard-OOD FP @ 95rec {r['hard_ood_fp_at_95']:.0%}, "
              f"0%-FP recall {r['recall_at_0_fp']:.0%}, "
              f"sep {r['separation']:+.3f}")

    best = min(sweep_results, key=lambda r: r["hard_ood_fp_at_95"])
    if best["hard_ood_fp_at_95"] <= 0.10 and best["in_library_top1"] >= 0.95:
        verdict = (f"OOD-aware W RESCUES C1 (best lambda={best['lambda_ood']}): "
                   f"hard-OOD FP {best['hard_ood_fp_at_95']:.0%} at "
                   f"in-library top-1 {best['in_library_top1']:.0%}.")
    elif best["hard_ood_fp_at_95"] <= 0.10:
        verdict = (f"OOD-aware W trades intra-library accuracy for OOD rejection "
                   f"(best lambda={best['lambda_ood']}): hard-OOD FP "
                   f"{best['hard_ood_fp_at_95']:.0%}, in-library top-1 "
                   f"{best['in_library_top1']:.0%}.")
    else:
        verdict = (f"OOD-aware W does NOT rescue: best hard-OOD FP "
                   f"{best['hard_ood_fp_at_95']:.0%} at lambda={best['lambda_ood']}. "
                   f"Hinge loss + InfoNCE is insufficient; try harder negatives or "
                   f"a richer rejection mechanism.")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
