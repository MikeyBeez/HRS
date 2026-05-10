"""Phase 65 mechanism test: pooling ablation on hard-OOD specificity.

The hard-OOD finding (C0b mean-pool L0 separation collapses from +0.420 on
random WikiText to +0.038 on same-template-different-entity) is consistent
with two stories:

  Mechanistic:    mean-pooling washes out entity tokens — entity information
                  IS in L0 but gets averaged into noise.
  Representational: entity tokens never had distinguishable L0 representations
                  in the first place — pooling can't recover what isn't there.

This script tests four pooling strategies on the same 60 in-library + 60 hard-
OOD queries and the same library state dicts. If ANY pooling separates
entities cleanly (e.g. hard-OOD FP at 95% recall <= 10%), the failure is
mechanistic and a one-line change to engram extraction rescues C0b. If all
four pooling strategies fail, the failure is representational and routing
needs an entity-aware verification stage that doesn't live in L0.

Pooling strategies tested:
  - mean       : baseline (Phase 47 default)
  - last       : last-token L0  (option 1 from the spec)
  - max        : per-dimension max over sequence
  - first      : first-token L0  (sees the question stem, not entity)
  - first3     : mean of first 3 tokens
  - last3      : mean of last 3 tokens (in case entity is near end)

Reuses Phase 65 cache (LoRA state dicts only — keys are re-extracted under
each pooling). No LoRA retraining.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_pooling_ablation.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase10_passkey import load_model, generate_passkeys
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora


SEED = 0
PHASE65_DIR = Path("results/identity_ae/phase65")
CACHE_DIR = PHASE65_DIR / "cache"


# ============================================================
# Pooling strategies
# ============================================================

@torch.no_grad()
def l0_hidden(model, ids_t):
    """Return the (1, T, D) L0 hidden state (post-tok-emb, post-dropout)."""
    return model.drop(model.tok_emb(ids_t))


def pool(h, strategy: str):
    """Apply pooling to a (1, T, D) hidden state and return (D,)."""
    h = h.squeeze(0)  # (T, D)
    T = h.shape[0]
    if strategy == "mean":
        return h.mean(dim=0).cpu()
    if strategy == "last":
        return h[-1].cpu()
    if strategy == "first":
        return h[0].cpu()
    if strategy == "first3":
        return h[:min(3, T)].mean(dim=0).cpu()
    if strategy == "last3":
        return h[-min(3, T):].mean(dim=0).cpu()
    if strategy == "max":
        return h.max(dim=0).values.cpu()
    raise ValueError(strategy)


# ============================================================
# Helpers
# ============================================================

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


def build_hard_ood_tests():
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    return (by_type["numeric"][5:10] + by_type["entity"][5:10]
            + by_type["technical"][5:10] + by_type["fact"][5:10])


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE65_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[A] loading library state dicts from {CACHE_DIR}")
    blob = torch.load(CACHE_DIR / "library.pt", map_location="cpu", weights_only=False)
    library = blob["library"]
    n_adapters = len(library)
    print(f"    {n_adapters} adapters with {sum(len(e['train_prompts']) for e in library)} train prompts")

    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)

    hard_ood = build_hard_ood_tests()

    strategies = ["mean", "last", "first", "first3", "last3", "max"]
    summaries = []

    for strat in strategies:
        print(f"\n[B] pooling strategy: {strat}")

        # ---- Re-extract library L0 keys with this pooling ----
        keys = []
        for entry in library:
            ks = []
            for p in entry["train_prompts"]:
                ids = tokenizer.encode(p, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
                h = l0_hidden(model, ids_t)
                ks.append(pool(h, strat))
            keys.append(ks)

        # ---- Score in-library held-out paraphrases ----
        in_records = []
        for i, entry in enumerate(library):
            for para in held_out_paraphrase(entry["test"]):
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                h = l0_hidden(model, ids_t)
                q = pool(h, strat)
                best_a, best_s = best_match(q, keys)
                in_records.append({
                    "true_idx": i, "routed_idx": best_a,
                    "best_score": best_s,
                    "routed_correct": int(best_a == i),
                })

        # ---- Score hard-OOD ----
        ood_records = []
        for ood_test in hard_ood:
            for para in held_out_paraphrase(ood_test):
                ids = tokenizer.encode(para, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                h = l0_hidden(model, ids_t)
                q = pool(h, strat)
                best_a, best_s = best_match(q, keys)
                ood_records.append({
                    "ood_prompt": ood_test["prompt"],
                    "best_score": best_s,
                    "routed_idx": best_a,
                })

        in_scores = [r["best_score"] for r in in_records]
        ood_scores = [r["best_score"] for r in ood_records]
        n_routed = sum(r["routed_correct"] for r in in_records)
        tau_95 = quantile(in_scores, 0.05)
        fp = fraction_above(ood_scores, tau_95)
        sep = sum(in_scores)/len(in_scores) - sum(ood_scores)/len(ood_scores)
        tau_zero = max(ood_scores) + 1e-9
        rec_zero = fraction_above(in_scores, tau_zero)

        print(f"    in-library top-1: {n_routed}/{len(in_records)} "
              f"({n_routed/len(in_records):.0%})")
        print(f"    in-library mean {sum(in_scores)/len(in_scores):+.3f}, "
              f"hard-OOD mean {sum(ood_scores)/len(ood_scores):+.3f}, "
              f"sep {sep:+.3f}")
        print(f"    tau@95rec {tau_95:+.3f}, hard-OOD FP {fp:.0%}")
        print(f"    0% hard-OOD FP -> in-lib recall {rec_zero:.0%} (tau {tau_zero:+.3f})")

        summaries.append({
            "strategy": strat,
            "in_library_top1": n_routed / len(in_records),
            "in_mean": sum(in_scores) / len(in_scores),
            "ood_mean": sum(ood_scores) / len(ood_scores),
            "separation": sep,
            "tau_95": tau_95,
            "hard_ood_fp_at_95": fp,
            "tau_zero_fp": tau_zero,
            "in_library_recall_at_0_fp": rec_zero,
            "in_scores": in_scores,
            "ood_scores": ood_scores,
            "in_records": in_records,
            "ood_records": ood_records,
        })

    # ---- Save ----
    out = {
        "config": {"strategies": strategies, "seed": SEED},
        "summaries": [{k: v for k, v in s.items()
                        if k not in ("in_records", "ood_records",
                                     "in_scores", "ood_scores")}
                       for s in summaries],
        "details": {s["strategy"]: {k: v for k, v in s.items() if k != "strategy"}
                     for s in summaries},
    }
    out_path = PHASE65_DIR / "pooling_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    bins = [-0.1 + i * 0.04 for i in range(int(2.0 / 0.04) + 1)]
    for ax, s in zip(axes.flat, summaries):
        ax.hist(s["in_scores"],  bins=bins, alpha=0.55, color="C0", label=f"in-lib (n={len(s['in_scores'])})")
        ax.hist(s["ood_scores"], bins=bins, alpha=0.55, color="C3", label=f"hard-OOD (n={len(s['ood_scores'])})")
        ax.axvline(s["tau_95"], color="black", ls="--", lw=1)
        ax.set_title(f"{s['strategy']:6s}  top-1={s['in_library_top1']:.0%}  "
                     f"FP={s['hard_ood_fp_at_95']:.0%}  sep={s['separation']:+.3f}",
                     fontsize=10)
        ax.set_xlabel("max cosine over 80 keys", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
    plt.suptitle("L0 pooling ablation on hard near-neighbor OOD", fontsize=12)
    plt.tight_layout()
    plot_path = PHASE65_DIR / "cosine_distributions_pooling.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict ----
    print(f"\n{'='*72}\nVERDICT")
    print(f"{'='*72}")
    print(f"  {'strategy':<8s}  {'in-lib top1':<12s}  {'sep':<8s}  {'hard-OOD FP':<12s}  {'0%-FP recall'}")
    for s in summaries:
        print(f"  {s['strategy']:<8s}  {s['in_library_top1']:<12.0%}  "
              f"{s['separation']:>+.3f}   {s['hard_ood_fp_at_95']:<12.0%}  "
              f"{s['in_library_recall_at_0_fp']:.0%}")

    best = min(summaries, key=lambda s: s["hard_ood_fp_at_95"])
    if best["hard_ood_fp_at_95"] <= 0.10 and best["in_library_top1"] >= 0.90:
        verdict = (f"MECHANISTIC: '{best['strategy']}' pooling achieves hard-OOD "
                   f"FP {best['hard_ood_fp_at_95']:.0%} at in-library top-1 "
                   f"{best['in_library_top1']:.0%}. Mean-pool was the bottleneck; "
                   f"the entity IS in L0. One-line change to engram extraction "
                   f"rescues C0b.")
    elif best["hard_ood_fp_at_95"] <= 0.30:
        verdict = (f"PARTIAL: best pooling '{best['strategy']}' gives hard-OOD FP "
                   f"{best['hard_ood_fp_at_95']:.0%} (vs mean-pool ~92%). Some "
                   f"entity signal exists in L0 but not enough for clean "
                   f"deployment threshold.")
    else:
        verdict = (f"REPRESENTATIONAL: all pooling strategies fail hard-OOD "
                   f"(best FP {best['hard_ood_fp_at_95']:.0%} at "
                   f"'{best['strategy']}'). The entity isn't in L0 in any "
                   f"recoverable form. Routing must operate at template "
                   f"granularity by design; verification needs a separate stage.")
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
