"""Phase 65 specificity ablation: hard near-neighbor OOD.

The 128-token and length-matched OOD ablations both gave Outcome D (C0b clean,
C1 poor) but used semantically distant WikiText spans. The deployment-relevant
question is: does C0b's natural L0 separation survive when OOD queries are
*structurally identical* to library queries (same template, same domain) but
reference different entities?

Construction:
  generate_passkeys(50) yields 5+5+5+5 stratified library entries plus 5+5+5+5
  unused near-neighbors per category — same templates, different entities (e.g.
  library asks about "northern facility", OOD asks about "orbital facility").
  Pair each library entry with one near-neighbor entity, generate 3 held-out
  paraphrases per entity, run through C0b and C1 routing. 60 hard-OOD trials,
  apples-to-apples with the 60 in-library trials from Phase 65.

Decision rule (pre-committed):
  - If C0b OOD FP at 95% in-library recall is <= 10%, the two-stage architecture
    (C0b gate -> C1 resolver) is bulletproof; report and stop.
  - If C0b OOD FP > 30%, single-stage C0b breaks under hard OOD; the natural
    next experiment is OOD-aware contrastive retraining of W.
  - If C0b OOD FP between 10-30%, partial; still stronger than C1 (which is at
    70-83% under easy OOD), but not deployment-clean.

Reuses Phase 65 cache; no model retraining.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_specificity_hardood.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase10_passkey import (
    load_model, generate_passkeys,
)
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora


SEED = 0
PHASE65_DIR = Path("results/identity_ae/phase65")
CACHE_DIR = PHASE65_DIR / "cache"


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
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
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return float("nan")
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def fraction_above(xs, threshold):
    return sum(1 for x in xs if x > threshold) / max(len(xs), 1)


def build_hard_ood_tests():
    """Return 20 hard-OOD test dicts: same template per category as the 20 library
    entries, drawn from generate_passkeys()'s next 5 per type. Pair index i in
    library with index i in hard_ood (same category)."""
    all_tests = generate_passkeys(50)
    by_type = defaultdict(list)
    for t in all_tests:
        by_type[t["type"]].append(t)
    # Library uses by_type[t][:5]; hard-OOD uses by_type[t][5:10].
    hard_ood = (by_type["numeric"][5:10] + by_type["entity"][5:10]
                + by_type["technical"][5:10] + by_type["fact"][5:10])
    return hard_ood


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
    M_trained = torch.load(CACHE_DIR / "M_trained.pt", map_location="cpu", weights_only=False)

    # Sanity: confirm library entries match by_type[:5]
    lib_prompts = [e["test"]["prompt"] for e in library]
    expected_prompts = [t["prompt"] for t in stratified_tests()]
    assert lib_prompts == expected_prompts, "library != stratified_tests; pairing assumption wrong"
    print(f"    library uses stratified_tests() (the first 5 per type) — confirmed")

    # ---- Hard-OOD construction ----
    hard_ood = build_hard_ood_tests()
    print(f"[A] hard-OOD: {len(hard_ood)} near-neighbor entities")
    print(f"    library[0] = {library[0]['test']['prompt']!r}")
    print(f"    hard_ood[0] = {hard_ood[0]['prompt']!r}")
    print(f"    library[10] = {library[10]['test']['prompt']!r}")
    print(f"    hard_ood[10] = {hard_ood[10]['prompt']!r}")

    # ---- In-library reference ----
    p65 = json.load(open(PHASE65_DIR / "phase65_results.json"))
    in_c0b = [t["best_score"] for t in p65["trials"]["C0b I,         vs L0"]]
    in_c1  = [t["best_score"] for t in p65["trials"]["C1  trained W, vs L5"]]

    # ---- Base model (LoRA reset for query extraction) ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)

    # ---- Run 60 hard-OOD trials: each entity x 3 held-out paraphrases ----
    print(f"[B] scoring {len(hard_ood) * 3} hard-OOD queries (20 entities x 3 paraphrases)")
    records = []
    for i, ood_test in enumerate(hard_ood):
        paras = held_out_paraphrase(ood_test)
        for p_idx, para in enumerate(paras):
            ids = tokenizer.encode(para, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            q_l0 = l0_mean(model, ids_t)
            a_c0b, s_c0b = best_match(q_l0, keys_l0)
            q_proj = q_l0 @ M_trained
            a_c1, s_c1 = best_match(q_proj, keys_l5)
            records.append({
                "ood_idx": i, "para_idx": p_idx,
                "type": ood_test["type"],
                "ood_prompt": ood_test["prompt"],
                "ood_paraphrase": para,
                "c0b": {"routed_idx": a_c0b, "best_score": float(s_c0b),
                        "routed_lib_prompt": library[a_c0b]["test"]["prompt"]},
                "c1":  {"routed_idx": a_c1,  "best_score": float(s_c1),
                        "routed_lib_prompt": library[a_c1]["test"]["prompt"]},
            })

    ood_c0b = [r["c0b"]["best_score"] for r in records]
    ood_c1  = [r["c1"]["best_score"]  for r in records]

    # ---- Calibration ----
    tau_c0b = quantile(in_c0b, 0.05)
    tau_c1  = quantile(in_c1,  0.05)
    fp_c0b  = fraction_above(ood_c0b, tau_c0b)
    fp_c1   = fraction_above(ood_c1,  tau_c1)
    sep_c0b = sum(in_c0b)/len(in_c0b) - sum(ood_c0b)/len(ood_c0b)
    sep_c1  = sum(in_c1)/len(in_c1)   - sum(ood_c1)/len(ood_c1)

    # ---- Operating points ----
    tau_c0b_zero = max(ood_c0b) + 1e-9
    tau_c1_zero  = max(ood_c1)  + 1e-9
    rec_c0b_zero = fraction_above(in_c0b, tau_c0b_zero)
    rec_c1_zero  = fraction_above(in_c1,  tau_c1_zero)

    print(f"\n[C] HARD-OOD RESULTS (n={len(records)} hard-OOD queries)")
    print(f"    C0b: in_mean {sum(in_c0b)/len(in_c0b):+.3f}  ood_mean {sum(ood_c0b)/len(ood_c0b):+.3f}  "
          f"(min {min(ood_c0b):+.3f}, max {max(ood_c0b):+.3f})  sep {sep_c0b:+.3f}")
    print(f"         tau@95rec {tau_c0b:+.3f}, OOD FP {fp_c0b:.0%}")
    print(f"         0% OOD FP -> recall {rec_c0b_zero:.0%}  (tau {tau_c0b_zero:+.3f})")
    print(f"    C1 : in_mean {sum(in_c1)/len(in_c1):+.3f}   ood_mean {sum(ood_c1)/len(ood_c1):+.3f}   "
          f"(min {min(ood_c1):+.3f}, max {max(ood_c1):+.3f})  sep {sep_c1:+.3f}")
    print(f"         tau@95rec {tau_c1:+.3f}, OOD FP {fp_c1:.0%}")
    print(f"         0% OOD FP -> recall {rec_c1_zero:.0%}  (tau {tau_c1_zero:+.3f})")

    # ---- Compare to easy OOD (128-tok and length-matched) ----
    easy = json.load(open(PHASE65_DIR / "specificity.json"))
    lenmatch = json.load(open(PHASE65_DIR / "specificity_lengthmatched.json"))
    print(f"\n[D] OOD DIFFICULTY LADDER")
    print(f"                                 easy 128-tok | length-matched | hard near-neighbor")
    print(f"    C0b OOD mean:                {easy['ood_summary']['c0b']['mean']:+.3f}      | "
          f"{lenmatch['ood_summary_lengthmatched']['c0b']['mean']:+.3f}         | "
          f"{sum(ood_c0b)/len(ood_c0b):+.3f}")
    print(f"    C0b OOD FP @ 95% recall:     {easy['calibration']['ood_fp_rate_c0b']:.0%}          | "
          f"{lenmatch['calibration_lengthmatched']['ood_fp_rate_c0b']:.0%}             | "
          f"{fp_c0b:.0%}")
    print(f"    C0b 0% FP -> in-library recall: 100%       | 100%           | {rec_c0b_zero:.0%}")
    print(f"    C1  OOD mean:                {easy['ood_summary']['c1']['mean']:+.3f}      | "
          f"{lenmatch['ood_summary_lengthmatched']['c1']['mean']:+.3f}         | "
          f"{sum(ood_c1)/len(ood_c1):+.3f}")
    print(f"    C1  OOD FP @ 95% recall:     {easy['calibration']['ood_fp_rate_c1']:.0%}         | "
          f"{lenmatch['calibration_lengthmatched']['ood_fp_rate_c1']:.0%}            | "
          f"{fp_c1:.0%}")

    # ---- Per-type breakdown ----
    print(f"\n[E] HARD-OOD per-type C0b best_score (sorted):")
    by_type_c0b = defaultdict(list)
    for r in records:
        by_type_c0b[r["type"]].append(r["c0b"]["best_score"])
    for ttype in ["numeric", "entity", "technical", "fact"]:
        scores = sorted(by_type_c0b[ttype])
        print(f"    {ttype:10s}  n={len(scores):2d}  min {min(scores):+.3f}  median {scores[len(scores)//2]:+.3f}  "
              f"max {max(scores):+.3f}  mean {sum(scores)/len(scores):+.3f}")

    # ---- Confusion: which library adapters do hard-OOD spuriously route to? ----
    above_thr_c0b = [r for r in records if r["c0b"]["best_score"] > tau_c0b]
    print(f"\n[F] hard-OOD queries above C0b threshold ({tau_c0b:+.3f}): "
          f"{len(above_thr_c0b)}/{len(records)}")
    if above_thr_c0b:
        for r in above_thr_c0b[:10]:
            print(f"    score {r['c0b']['best_score']:+.3f}: "
                  f"OOD {r['ood_paraphrase']!r} -> "
                  f"lib[{r['c0b']['routed_idx']}] {r['c0b']['routed_lib_prompt']!r}")

    # ---- Save JSON ----
    out = {
        "config": {"n_hard_ood_entities": 20, "n_paraphrases_per": 3,
                   "n_hard_ood_total": len(records), "seed": SEED},
        "in_library_summary": {
            "c0b": {"mean": sum(in_c0b)/len(in_c0b), "p05": quantile(in_c0b, 0.05)},
            "c1":  {"mean": sum(in_c1)/len(in_c1),   "p05": quantile(in_c1, 0.05)},
        },
        "ood_summary_hard": {
            "c0b": {"mean": sum(ood_c0b)/len(ood_c0b),
                    "min": min(ood_c0b), "max": max(ood_c0b),
                    "p50": quantile(ood_c0b, 0.5),
                    "by_type": {k: sum(v)/len(v) for k, v in by_type_c0b.items()}},
            "c1":  {"mean": sum(ood_c1)/len(ood_c1),
                    "min": min(ood_c1),  "max": max(ood_c1),
                    "p50": quantile(ood_c1, 0.5)},
        },
        "calibration_hard": {
            "tau_at_95_recall_c0b": tau_c0b,
            "tau_at_95_recall_c1":  tau_c1,
            "ood_fp_rate_c0b":      fp_c0b,
            "ood_fp_rate_c1":       fp_c1,
            "separation_gap_c0b":   sep_c0b,
            "separation_gap_c1":    sep_c1,
            "recall_at_zero_ood_fp_c0b": rec_c0b_zero,
            "recall_at_zero_ood_fp_c1":  rec_c1_zero,
            "tau_at_zero_ood_fp_c0b": tau_c0b_zero,
            "tau_at_zero_ood_fp_c1":  tau_c1_zero,
        },
        "ood_per_query": records,
    }
    out_path = PHASE65_DIR / "specificity_hardood.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot: C0b on a difficulty ladder ----
    easy_c0b = [r["c0b"]["best_score"] for r in easy["ood_per_query"]]
    lenmatch_c0b = [r["c0b"]["best_score"] for r in lenmatch["ood_per_query"]]
    easy_c1 = [r["c1"]["best_score"] for r in easy["ood_per_query"]]
    lenmatch_c1 = [r["c1"]["best_score"] for r in lenmatch["ood_per_query"]]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    bins_c0b = [-0.1 + i * 0.04 for i in range(int(2.0 / 0.04) + 1)]
    bins_c1  = [-0.1 + i * 0.02 for i in range(int(0.8 / 0.02) + 1)]

    axes[0].hist(in_c0b,        bins=bins_c0b, alpha=0.55, color="C0", label=f"in-library (n={len(in_c0b)})")
    axes[0].hist(easy_c0b,      bins=bins_c0b, alpha=0.45, color="C2", label=f"easy 128-tok OOD (n={len(easy_c0b)})")
    axes[0].hist(lenmatch_c0b,  bins=bins_c0b, alpha=0.45, color="C1", label=f"length-matched OOD (n={len(lenmatch_c0b)})")
    axes[0].hist(ood_c0b,       bins=bins_c0b, alpha=0.55, color="C3", label=f"hard near-neighbor OOD (n={len(ood_c0b)})")
    axes[0].axvline(tau_c0b, color="black", ls="--", lw=1, label=f"tau@95rec={tau_c0b:.3f}")
    axes[0].set_title(f"C0b: best max-cosine across OOD difficulty ladder\n"
                      f"hard-OOD FP @ 95% recall = {fp_c0b:.0%},  "
                      f"0% FP -> in-library recall {rec_c0b_zero:.0%}")
    axes[0].set_xlabel("max cosine over 80 keys")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].hist(in_c1,        bins=bins_c1, alpha=0.55, color="C0", label=f"in-library (n={len(in_c1)})")
    axes[1].hist(easy_c1,      bins=bins_c1, alpha=0.45, color="C2", label=f"easy 128-tok OOD (n={len(easy_c1)})")
    axes[1].hist(lenmatch_c1,  bins=bins_c1, alpha=0.45, color="C1", label=f"length-matched OOD (n={len(lenmatch_c1)})")
    axes[1].hist(ood_c1,       bins=bins_c1, alpha=0.55, color="C3", label=f"hard near-neighbor OOD (n={len(ood_c1)})")
    axes[1].axvline(tau_c1, color="black", ls="--", lw=1, label=f"tau@95rec={tau_c1:.3f}")
    axes[1].set_title(f"C1: best max-cosine across OOD difficulty ladder\n"
                      f"hard-OOD FP @ 95% recall = {fp_c1:.0%},  "
                      f"0% FP -> in-library recall {rec_c1_zero:.0%}")
    axes[1].set_xlabel("max cosine over 80 keys")
    axes[1].set_ylabel("count")
    axes[1].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = PHASE65_DIR / "cosine_distributions_hardood.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict ----
    if fp_c0b <= 0.10:
        verdict = (f"C0b SURVIVES hard-OOD: FP {fp_c0b:.0%} at 95% in-library recall, "
                   f"sep {sep_c0b:+.3f}. Two-stage architecture (C0b gate -> C1 resolver) "
                   f"is bulletproof under near-neighbor attack.")
    elif fp_c0b > 0.30:
        verdict = (f"C0b BREAKS under hard-OOD: FP {fp_c0b:.0%} at 95% recall. "
                   f"Single-stage C0b can't reject near-neighbors. "
                   f"Next step: retrain W with OOD negatives so C1 acquires "
                   f"absolute-distance separation.")
    else:
        verdict = (f"C0b PARTIAL under hard-OOD: FP {fp_c0b:.0%} at 95% recall. "
                   f"Still beats C1 ({fp_c1:.0%}) but not deployment-clean. "
                   f"Two-stage works for plausible OOD; near-neighbor attack is a real failure mode.")
    print(f"\n[G] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
