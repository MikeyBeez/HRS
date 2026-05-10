"""Phase 65 last-token / last-N L0 ablation.

The hard-OOD failure (C0b mean-pool L0 hits 92% FP at 95% in-library recall on
same-template-different-entity OOD) is consistent with two stories:

  Mechanistic:    mean-pool L0 dilutes entity tokens — entity IS in L0 but
                  averaging over template/frame tokens washes the signal out.
  Representational: entity identity isn't in L0 at all; pooling can't recover
                  what isn't there.

This script tests 4 query-side L0 extraction variants and 2 routing paths on
both in-library and hard-OOD query sets.

Variants (all over the L0 hidden state model.drop(model.tok_emb(ids))):
  mean_pool   baseline (Phase 47 default)
  last_1      final token
  last_5      mean over last 5 tokens
  last_10     mean over last 10 tokens

Routings:
  C0b   variant L0 query     vs variant L0 keys (re-extracted per variant)
  C1    variant L0 query @ W vs cached mean-pool L5 keys (W is fixed)

Library state dicts come from Phase 65 cache; M_trained is fixed (Phase 47).
Model is set to .eval() to disable dropout — Phase 47 left dropout active by
oversight (load_model never calls eval), making cached keys mildly stochastic.
A clean baseline under eval mode is what we want for the pooling comparison.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_lasttoken.py
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
    hidden_at_layer, reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora


SEED = 0
PHASE65_DIR = Path("results/identity_ae/phase65")
CACHE_DIR = PHASE65_DIR / "cache"


# ============================================================
# L0 extraction variants
# ============================================================

@torch.no_grad()
def l0_hidden(model, ids_t):
    return model.drop(model.tok_emb(ids_t))  # (1, T, D)


def pool_l0(h, variant: str):
    h = h.squeeze(0)  # (T, D)
    T = h.shape[0]
    if variant == "mean_pool":
        return h.mean(dim=0).cpu()
    if variant == "last_1":
        return h[-1].cpu()
    if variant == "last_5":
        return h[-min(5, T):].mean(dim=0).cpu()
    if variant == "last_10":
        return h[-min(10, T):].mean(dim=0).cpu()
    raise ValueError(variant)


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


def extract_l0_keys(library, model, tokenizer, device, variant):
    """Re-extract per-prompt L0 keys for each library entry under given variant."""
    keys = []
    for entry in library:
        ks = []
        for p in entry["train_prompts"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            h = l0_hidden(model, ids_t)
            ks.append(pool_l0(h, variant))
        keys.append(ks)
    return keys


# ============================================================
# Run a routing condition
# ============================================================

def route_set(query_records, model, tokenizer, device, variant, M, key_set):
    """For each query in query_records (each has 'para' and 'true_idx'),
    compute variant-pooled L0, optionally project through M, score against key_set."""
    out = []
    for r in query_records:
        ids = tokenizer.encode(r["para"], add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        h = l0_hidden(model, ids_t)
        q = pool_l0(h, variant)
        if M is not None:
            q = q @ M
        best_a, best_s = best_match(q, key_set)
        out.append({"true_idx": r["true_idx"], "para": r["para"],
                    "routed_idx": best_a, "best_score": float(best_s)})
    return out


def fmt_summary(in_records, ood_records, label):
    in_scores  = [r["best_score"] for r in in_records]
    ood_scores = [r["best_score"] for r in ood_records]
    n_in_routed = sum(int(r["routed_idx"] == r["true_idx"]) for r in in_records)
    in_top1 = n_in_routed / max(len(in_records), 1)
    in_mean  = sum(in_scores) / max(len(in_scores), 1)
    ood_mean = sum(ood_scores) / max(len(ood_scores), 1)
    sep = in_mean - ood_mean
    tau_95 = quantile(in_scores, 0.05)
    fp = fraction_above(ood_scores, tau_95)
    tau_zero = max(ood_scores) + 1e-9
    rec_zero = fraction_above(in_scores, tau_zero)
    return {
        "label": label, "in_top1": in_top1, "in_mean": in_mean, "ood_mean": ood_mean,
        "separation": sep, "tau_95": tau_95, "hard_ood_fp_at_95": fp,
        "tau_zero_fp": tau_zero, "in_recall_at_0_fp": rec_zero,
        "in_scores": in_scores, "ood_scores": ood_scores,
    }


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
    library = blob["library"]
    cached_keys_l5 = blob["keys_l5"]
    cached_keys_l5 = [[k.cpu() if k.is_cuda else k for k in ks] for ks in cached_keys_l5]
    M_trained = torch.load(CACHE_DIR / "M_trained.pt",
                             map_location="cpu", weights_only=False)
    print(f"    {len(library)} adapters; cached L5 keys + M_trained loaded")

    # ---- Build query sets ----
    in_records = []
    for i, entry in enumerate(library):
        for para in held_out_paraphrase(entry["test"]):
            in_records.append({"true_idx": i, "para": para})

    hard_ood = build_hard_ood_tests()
    ood_records = []
    for ood_test in hard_ood:
        for para in held_out_paraphrase(ood_test):
            # No "true_idx" — these are OOD; mark as -1
            ood_records.append({"true_idx": -1, "para": para})

    print(f"    in-library queries: {len(in_records)}, hard-OOD queries: {len(ood_records)}")

    # ---- Build base model in EVAL mode ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    model.eval()  # disable dropout — clean baseline
    reset_lora_to_zero(model)

    # ---- 4 variants x 2 routing methods x 2 query sets = 16 evaluations ----
    variants = ["mean_pool", "last_1", "last_5", "last_10"]
    all_summaries = []

    for variant in variants:
        print(f"\n[B] variant: {variant}")
        # Re-extract L0 keys for C0b symmetry under this variant
        keys_l0_var = extract_l0_keys(library, model, tokenizer, device, variant)

        # ---- C0b path ----
        in_c0b  = route_set(in_records,  model, tokenizer, device, variant, M=None, key_set=keys_l0_var)
        ood_c0b = route_set(ood_records, model, tokenizer, device, variant, M=None, key_set=keys_l0_var)
        s_c0b = fmt_summary(in_c0b, ood_c0b, label=f"{variant}/C0b")
        print(f"    C0b: in_top1 {s_c0b['in_top1']:.0%}  in_mean {s_c0b['in_mean']:+.3f}  "
              f"ood_mean {s_c0b['ood_mean']:+.3f}  sep {s_c0b['separation']:+.3f}  "
              f"FP@95rec {s_c0b['hard_ood_fp_at_95']:.0%}  0%-FP rec {s_c0b['in_recall_at_0_fp']:.0%}")

        # ---- C1 path: variant L0 query @ M_trained vs cached L5 keys ----
        in_c1  = route_set(in_records,  model, tokenizer, device, variant, M=M_trained, key_set=cached_keys_l5)
        ood_c1 = route_set(ood_records, model, tokenizer, device, variant, M=M_trained, key_set=cached_keys_l5)
        s_c1 = fmt_summary(in_c1, ood_c1, label=f"{variant}/C1")
        print(f"    C1 : in_top1 {s_c1['in_top1']:.0%}  in_mean {s_c1['in_mean']:+.3f}  "
              f"ood_mean {s_c1['ood_mean']:+.3f}  sep {s_c1['separation']:+.3f}  "
              f"FP@95rec {s_c1['hard_ood_fp_at_95']:.0%}  0%-FP rec {s_c1['in_recall_at_0_fp']:.0%}")

        all_summaries.append({
            "variant": variant,
            "c0b": s_c0b, "c1": s_c1,
            "c0b_in_records": in_c0b, "c0b_ood_records": ood_c0b,
            "c1_in_records":  in_c1,  "c1_ood_records":  ood_c1,
        })

    # ---- Save JSON ----
    save_blob = {
        "config": {"variants": variants, "seed": SEED, "model_eval_mode": True},
        "summaries": [
            {"variant": s["variant"],
             "c0b": {k: v for k, v in s["c0b"].items() if k not in ("in_scores", "ood_scores")},
             "c1":  {k: v for k, v in s["c1"].items()  if k not in ("in_scores", "ood_scores")}}
            for s in all_summaries
        ],
        "details": {
            s["variant"]: {
                "c0b": {k: v for k, v in s["c0b"].items()},
                "c1":  {k: v for k, v in s["c1"].items()},
                "c0b_in_records": s["c0b_in_records"],
                "c0b_ood_records": s["c0b_ood_records"],
                "c1_in_records":  s["c1_in_records"],
                "c1_ood_records":  s["c1_ood_records"],
            } for s in all_summaries
        },
    }
    out_path = PHASE65_DIR / "lasttoken.json"
    with open(out_path, "w") as f:
        json.dump(save_blob, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot ----
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    for row, s in enumerate(all_summaries):
        for col, key in enumerate(["c0b", "c1"]):
            ax = axes[row, col]
            ss = s[key]
            bins = ([-0.1 + i * 0.04 for i in range(int(2.0 / 0.04) + 1)] if key == "c0b"
                    else [-0.1 + i * 0.02 for i in range(int(0.8 / 0.02) + 1)])
            ax.hist(ss["in_scores"],  bins=bins, alpha=0.55, color="C0",
                    label=f"in-lib (n={len(ss['in_scores'])})")
            ax.hist(ss["ood_scores"], bins=bins, alpha=0.55, color="C3",
                    label=f"hard-OOD (n={len(ss['ood_scores'])})")
            ax.axvline(ss["tau_95"], color="black", ls="--", lw=1)
            ax.set_title(f"{s['variant']:9s} / {key.upper()}  "
                         f"in_top1={ss['in_top1']:.0%}  FP@95={ss['hard_ood_fp_at_95']:.0%}  "
                         f"sep={ss['separation']:+.3f}", fontsize=9)
            ax.legend(loc="upper left", fontsize=7)
    axes[-1, 0].set_xlabel("max cosine over 80 keys")
    axes[-1, 1].set_xlabel("max cosine over 80 keys")
    plt.suptitle("L0 last-token / last-N ablation, hard near-neighbor OOD\n"
                 "(model.eval() — dropout disabled)", fontsize=11)
    plt.tight_layout()
    plot_path = PHASE65_DIR / "cosine_distributions_lasttoken.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict ----
    print(f"\n{'='*78}\nSUMMARY TABLE")
    print(f"{'='*78}")
    print(f"  {'variant':10s} {'route':>5s} {'in_top1':>7s} {'sep':>7s} "
          f"{'FP@95':>6s} {'0%FP_rec':>9s}")
    for s in all_summaries:
        for key in ["c0b", "c1"]:
            ss = s[key]
            print(f"  {s['variant']:10s} {key.upper():>5s} {ss['in_top1']:>7.0%} "
                  f"{ss['separation']:>+7.3f} {ss['hard_ood_fp_at_95']:>6.0%} "
                  f"{ss['in_recall_at_0_fp']:>9.0%}")

    # Best (variant, route) by hard-OOD FP, conditional on in_top1 >= 0.90
    candidates = [(s["variant"], k, s[k]) for s in all_summaries for k in ["c0b", "c1"]
                  if s[k]["in_top1"] >= 0.90]
    if candidates:
        best_v, best_k, best_s = min(candidates, key=lambda x: x[2]["hard_ood_fp_at_95"])
        print(f"\n  best (in_top1>=90%): {best_v}/{best_k.upper()}  "
              f"FP@95={best_s['hard_ood_fp_at_95']:.0%}  sep={best_s['separation']:+.3f}")
        if best_s["hard_ood_fp_at_95"] <= 0.20:
            verdict = ("OUTCOME A: mechanistic fix found. {v}/{k} achieves hard-OOD "
                       "FP {fp:.0%} at in-library top-1 {top:.0%}. The architecture "
                       "is rescuable with a one-line change to engram extraction."
                       ).format(v=best_v, k=best_k.upper(),
                                fp=best_s["hard_ood_fp_at_95"], top=best_s["in_top1"])
        elif best_s["hard_ood_fp_at_95"] <= 0.50:
            verdict = ("OUTCOME B: partial rescue. Best FP@95 = {fp:.0%} via {v}/{k}. "
                       "Some entity signal exists in late tokens but pooling alone "
                       "isn't a clean fix."
                       ).format(v=best_v, k=best_k.upper(), fp=best_s["hard_ood_fp_at_95"])
        else:
            verdict = ("OUTCOME C: representational. All variants stay >50% FP@95 "
                       "with usable in-library top-1. Entity identity isn't "
                       "recoverable from L0 alone; routing operates at template "
                       "granularity by design.")
    else:
        verdict = ("No variant kept in-library top-1 >= 90%. Pooling itself broke "
                   "intra-library routing. This is itself a mild representational "
                   "result.")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
