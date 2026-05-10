"""Phase 65 specificity follow-up: OOD cosine separation under C0b vs C1.

Phase 65 found C0b (L0 keys, no projection) routes 93% of in-library queries
correctly vs C1 (trained W, L5 keys) at 100%. This script tests the hypothesis
that C0b's recall comes at the cost of specificity: because L0 cosines have a
high baseline (mean true cos +0.93, gap +0.029), OOD queries should also score
high under C0b and trigger spurious adapter loads. C1's projection into L5 key
space (mean true cos +0.246, gap +0.148) should leave OOD queries near zero.

Procedure (no retraining):
  1. Load Phase 65 cache: library, L0/L5 keys, trained M.
  2. Sample 30 OOD WikiText-2 passages (~128 tokens), seed=0.
  3. For each: compute L0_q under base model (LoRA reset).
     - C0b path: max cosine over the 80 stored L0 keys; record adapter.
     - C1 path:  max cosine over the 80 stored L5 keys after q @ M; record.
  4. Pull in-library "best_score" per-trial from phase65_results.json.
  5. Calibrate threshold at 95% in-library recall per condition; report FP
     rate on OOD, separation gap, and confusion clusters under C0b.

Outputs:
  results/identity_ae/phase65/specificity.json
  results/identity_ae/phase65/cosine_distributions.png
  results/identity_ae/phase65/specificity_README.md

Note: The library's in-library queries are held-out paraphrased *questions*,
while the OOD queries here are raw WikiText passages — a distribution mismatch
the spec accepted as "slightly less ideal but acceptable." Documented in README.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_specificity.py
"""

import json
import random
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import apply_lora


N_OOD = 30
QUERY_TOKENS = 128
SEED = 0
PHASE65_DIR = Path("results/identity_ae/phase65")
CACHE_DIR = PHASE65_DIR / "cache"


# ============================================================
# Helpers
# ============================================================

@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


def cosine(a, b):
    return float(torch.dot(a / (a.norm() + 1e-8), b / (b.norm() + 1e-8)))


def best_match(q, keys_per_adapter):
    """Return (best_adapter_idx, best_cosine) over a flat search of all keys."""
    best_a, best_s = -1, -2.0
    for ai, keys in enumerate(keys_per_adapter):
        for k in keys:
            s = cosine(q, k)
            if s > best_s:
                best_s = s
                best_a = ai
    return best_a, best_s


def quantile(xs, q):
    """Sample quantile (linear interp), no scipy dependency."""
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    pos = q * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def fraction_above(xs, threshold):
    return sum(1 for x in xs if x > threshold) / max(len(xs), 1)


# ============================================================
# OOD passage sampling
# ============================================================

def sample_ood_passages(tokenizer, n=N_OOD, n_tokens=QUERY_TOKENS, seed=SEED):
    """Pull n random ~n_tokens WikiText-2 passages, seeded."""
    from datasets import load_dataset
    print(f"[A] loading WikiText-2 train for OOD samples...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    rng = random.Random(seed)
    # Build candidate index: lines that tokenize to >= n_tokens
    texts = [t for t in ds["text"] if t and len(t.split()) > n_tokens // 2]
    rng.shuffle(texts)

    passages = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if len(ids) < n_tokens:
            continue
        # Take a random n_tokens-window inside the article
        start = rng.randint(0, len(ids) - n_tokens)
        chunk_ids = ids[start:start + n_tokens]
        passages.append({
            "ids": chunk_ids,
            "text": tokenizer.decode(chunk_ids, skip_special_tokens=True),
        })
        if len(passages) >= n:
            break
    print(f"[A] sampled {len(passages)} OOD passages of {n_tokens} tokens each")
    return passages


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE65_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load cached library + M ----
    print(f"[A] loading Phase 65 cache from {CACHE_DIR}")
    blob = torch.load(CACHE_DIR / "library.pt", map_location="cpu", weights_only=False)
    library = blob["library"]
    keys_l0 = blob["keys_l0"]
    keys_l5 = blob["keys_l5"]
    M_trained = torch.load(CACHE_DIR / "M_trained.pt", map_location="cpu", weights_only=False)
    print(f"    library: {len(library)} adapters, {sum(len(k) for k in keys_l0)} stored keys")
    print(f"    M_trained: {tuple(M_trained.shape)}")

    # ---- Load Phase 65 in-library trials ----
    p65 = json.load(open(PHASE65_DIR / "phase65_results.json"))
    name_c0b = "C0b I,         vs L0"
    name_c1  = "C1  trained W, vs L5"
    in_c0b_scores = [t["best_score"] for t in p65["trials"][name_c0b]]
    in_c0b_correct = [t["routed_correct"] for t in p65["trials"][name_c0b]]
    in_c1_scores  = [t["best_score"] for t in p65["trials"][name_c1]]
    in_c1_correct = [t["routed_correct"] for t in p65["trials"][name_c1]]
    print(f"    in-library C0b trials: n={len(in_c0b_scores)}, mean best_score "
          f"{sum(in_c0b_scores)/len(in_c0b_scores):.3f}")
    print(f"    in-library C1  trials: n={len(in_c1_scores)}, mean best_score "
          f"{sum(in_c1_scores)/len(in_c1_scores):.3f}")

    # ---- Build base model (need it for L0 extraction on OOD passages) ----
    print(f"[A] loading base model for OOD L0 extraction")
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)

    # ---- Sample OOD passages ----
    ood = sample_ood_passages(tokenizer, n=N_OOD, n_tokens=QUERY_TOKENS, seed=SEED)

    # ---- Score each OOD query under both routing paths ----
    print(f"[B] scoring {len(ood)} OOD queries")
    ood_records = []
    for i, p in enumerate(ood):
        ids_t = torch.tensor(p["ids"], dtype=torch.long).unsqueeze(0).to(device)
        q_l0 = l0_mean(model, ids_t)             # (D,)
        # C0b: cosine vs L0 keys
        a_c0b, s_c0b = best_match(q_l0, keys_l0)
        # C1: project then cosine vs L5 keys
        q_proj = q_l0 @ M_trained
        a_c1, s_c1 = best_match(q_proj, keys_l5)
        ood_records.append({
            "ood_idx": i,
            "text_preview": p["text"][:120].replace("\n", " "),
            "c0b": {"routed_idx": a_c0b, "best_score": float(s_c0b)},
            "c1":  {"routed_idx": a_c1,  "best_score": float(s_c1)},
        })

    ood_c0b_scores = [r["c0b"]["best_score"] for r in ood_records]
    ood_c1_scores  = [r["c1"]["best_score"]  for r in ood_records]
    print(f"    OOD C0b mean best_score {sum(ood_c0b_scores)/len(ood_c0b_scores):.3f}")
    print(f"    OOD C1  mean best_score {sum(ood_c1_scores)/len(ood_c1_scores):.3f}")

    # ---- Threshold calibration: at 95% in-library recall ----
    # Take τ s.t. 95% of in-library best_scores exceed it (i.e. 5th percentile).
    tau_c0b = quantile(in_c0b_scores, 0.05)
    tau_c1  = quantile(in_c1_scores,  0.05)
    fp_c0b = fraction_above(ood_c0b_scores, tau_c0b)
    fp_c1  = fraction_above(ood_c1_scores,  tau_c1)
    print(f"\n[C] threshold at 95% in-library recall:")
    print(f"    C0b: tau={tau_c0b:.3f}, OOD FP rate {fp_c0b:.0%}")
    print(f"    C1 : tau={tau_c1 :.3f}, OOD FP rate {fp_c1:.0%}")

    # ---- Separation gap ----
    sep_c0b = (sum(in_c0b_scores) / len(in_c0b_scores)) - (sum(ood_c0b_scores) / len(ood_c0b_scores))
    sep_c1  = (sum(in_c1_scores)  / len(in_c1_scores))  - (sum(ood_c1_scores)  / len(ood_c1_scores))
    print(f"\n[D] separation gap (mean in-library best_score - mean OOD best_score):")
    print(f"    C0b: {sep_c0b:+.3f}")
    print(f"    C1 : {sep_c1:+.3f}")

    # ---- Confusion structure under C0b: which adapters attract OOD? ----
    c0b_routed = Counter(r["c0b"]["routed_idx"] for r in ood_records)
    c1_routed  = Counter(r["c1"]["routed_idx"]  for r in ood_records)
    above_thr_c0b = [r for r in ood_records if r["c0b"]["best_score"] > tau_c0b]
    spurious_c0b = Counter(r["c0b"]["routed_idx"] for r in above_thr_c0b)
    print(f"\n[E] OOD adapter assignments above threshold under C0b ({len(above_thr_c0b)} queries):")
    for ai, cnt in spurious_c0b.most_common():
        ttype = library[ai]["test"]["type"]
        prompt = library[ai]["test"]["prompt"]
        print(f"    adapter {ai:2d} ({ttype:9s}): {cnt} spurious matches  | {prompt[:60]}")

    # ---- Stretch: at what tau does C0b match C1's specificity? ----
    target_fp = max(fp_c1, 1.0 / N_OOD)  # don't go below 1/N for resolution
    sorted_ood_c0b = sorted(ood_c0b_scores)
    # Want the smallest tau such that fraction_above(ood, tau) <= target_fp
    n_allowed = int(target_fp * N_OOD)
    # Threshold strictly above the (N - n_allowed - 1)-th OOD score
    if n_allowed >= N_OOD:
        tau_c0b_match = float("-inf")
    elif n_allowed <= 0:
        tau_c0b_match = sorted_ood_c0b[-1] + 1e-6
    else:
        tau_c0b_match = sorted_ood_c0b[N_OOD - n_allowed - 1] + 1e-9
    recall_c0b_at_match = fraction_above(in_c0b_scores, tau_c0b_match)
    print(f"\n[F] stretch: C0b at C1's specificity ({fp_c1:.0%} OOD FP):")
    print(f"    tau={tau_c0b_match:.3f}, in-library recall {recall_c0b_at_match:.0%}")
    print(f"    (vs C0b's natural recall 95% at tau={tau_c0b:.3f})")

    # ---- Save JSON ----
    out = {
        "config": {
            "n_ood": N_OOD, "query_tokens": QUERY_TOKENS, "seed": SEED,
            "ood_source": "wikitext-2-raw-v1 train",
        },
        "in_library_summary": {
            "c0b": {
                "n": len(in_c0b_scores),
                "mean": sum(in_c0b_scores) / len(in_c0b_scores),
                "p05": quantile(in_c0b_scores, 0.05),
                "p50": quantile(in_c0b_scores, 0.50),
                "p95": quantile(in_c0b_scores, 0.95),
                "min": min(in_c0b_scores), "max": max(in_c0b_scores),
            },
            "c1": {
                "n": len(in_c1_scores),
                "mean": sum(in_c1_scores) / len(in_c1_scores),
                "p05": quantile(in_c1_scores, 0.05),
                "p50": quantile(in_c1_scores, 0.50),
                "p95": quantile(in_c1_scores, 0.95),
                "min": min(in_c1_scores), "max": max(in_c1_scores),
            },
        },
        "ood_summary": {
            "c0b": {
                "n": len(ood_c0b_scores),
                "mean": sum(ood_c0b_scores) / len(ood_c0b_scores),
                "p05": quantile(ood_c0b_scores, 0.05),
                "p50": quantile(ood_c0b_scores, 0.50),
                "p95": quantile(ood_c0b_scores, 0.95),
                "min": min(ood_c0b_scores), "max": max(ood_c0b_scores),
            },
            "c1": {
                "n": len(ood_c1_scores),
                "mean": sum(ood_c1_scores) / len(ood_c1_scores),
                "p05": quantile(ood_c1_scores, 0.05),
                "p50": quantile(ood_c1_scores, 0.50),
                "p95": quantile(ood_c1_scores, 0.95),
                "min": min(ood_c1_scores), "max": max(ood_c1_scores),
            },
        },
        "calibration": {
            "tau_at_95_recall_c0b": tau_c0b,
            "tau_at_95_recall_c1":  tau_c1,
            "ood_fp_rate_c0b":      fp_c0b,
            "ood_fp_rate_c1":       fp_c1,
            "separation_gap_c0b":   sep_c0b,
            "separation_gap_c1":    sep_c1,
        },
        "stretch": {
            "target_ood_fp": target_fp,
            "tau_c0b_at_match": tau_c0b_match,
            "in_library_recall_c0b_at_match": recall_c0b_at_match,
        },
        "ood_assignments": {
            "c0b_top": spurious_c0b.most_common(),
            "c0b_all": dict(c0b_routed),
            "c1_all":  dict(c1_routed),
        },
        "ood_per_query": ood_records,
    }
    out_json = PHASE65_DIR / "specificity.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    bins_c0b = [-0.1 + i * 0.05 for i in range(int(2.0 / 0.05) + 1)]
    bins_c1  = [-0.1 + i * 0.02 for i in range(int(0.8 / 0.02) + 1)]

    axes[0].hist(in_c0b_scores,  bins=bins_c0b, alpha=0.55, label=f"in-library (n={len(in_c0b_scores)})", color="C0")
    axes[0].hist(ood_c0b_scores, bins=bins_c0b, alpha=0.55, label=f"OOD (n={len(ood_c0b_scores)})",        color="C3")
    axes[0].axvline(tau_c0b, color="black", linestyle="--", linewidth=1, label=f"tau_95={tau_c0b:.3f}")
    axes[0].set_title(f"C0b: I, vs L0 keys\nOOD FP@95rec = {fp_c0b:.0%}, sep = {sep_c0b:+.3f}")
    axes[0].set_xlabel("max cosine over 80 keys")
    axes[0].set_ylabel("count")
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].hist(in_c1_scores,   bins=bins_c1, alpha=0.55, label=f"in-library (n={len(in_c1_scores)})", color="C0")
    axes[1].hist(ood_c1_scores,  bins=bins_c1, alpha=0.55, label=f"OOD (n={len(ood_c1_scores)})",        color="C3")
    axes[1].axvline(tau_c1, color="black", linestyle="--", linewidth=1, label=f"tau_95={tau_c1:.3f}")
    axes[1].set_title(f"C1: trained W, vs L5 keys\nOOD FP@95rec = {fp_c1:.0%}, sep = {sep_c1:+.3f}")
    axes[1].set_xlabel("max cosine over 80 keys")
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plot_path = PHASE65_DIR / "cosine_distributions.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict & README ----
    if fp_c0b > 0.30 and fp_c1 <= 0.10:
        verdict = "A"
        verdict_text = ("Outcome A: C0b has poor specificity, C1 is clean. "
                        "C1's value is primarily specificity, not recall.")
    elif fp_c0b > 0.30 and fp_c1 > 0.30:
        verdict = "B"
        verdict_text = ("Outcome B: both methods have specificity issues. "
                        "Neither stack is deployment-ready without an explicit "
                        "rejection mechanism.")
    elif fp_c0b <= 0.10 and fp_c1 <= 0.10:
        verdict = "C"
        verdict_text = ("Outcome C: both methods have clean specificity. "
                        "L5+W only buys the 7-point routing lift, not extra "
                        "specificity.")
    elif fp_c0b <= 0.10 and fp_c1 > fp_c0b:
        verdict = "D"
        verdict_text = ("Outcome D (unexpected): C0b cleaner than C1. "
                        "Investigate W overfitting before publishing.")
    else:
        verdict = "between"
        verdict_text = (f"Between outcomes: C0b FP {fp_c0b:.0%}, C1 FP {fp_c1:.0%}. "
                        f"Re-read decision rules in spec.")

    readme = f"""# Phase 65 specificity follow-up

Tests whether C0b (L0 keys, no projection) keeps its 93% in-library routing
accuracy at deployment-relevant specificity, vs C1 (trained W, L5 keys, 100%
in-library).

## Numbers

|                                    | C0b (I, L0 keys)         | C1 (trained W, L5 keys) |
|------------------------------------|--------------------------|-------------------------|
| In-library mean best_score         | {sum(in_c0b_scores)/len(in_c0b_scores):+.3f}                   | {sum(in_c1_scores)/len(in_c1_scores):+.3f}                  |
| OOD mean best_score                | {sum(ood_c0b_scores)/len(ood_c0b_scores):+.3f}                   | {sum(ood_c1_scores)/len(ood_c1_scores):+.3f}                  |
| Separation gap (in - OOD)          | {sep_c0b:+.3f}                   | {sep_c1:+.3f}                  |
| τ at 95% in-library recall         | {tau_c0b:+.3f}                   | {tau_c1:+.3f}                  |
| OOD FP rate at that τ              | {fp_c0b:.0%}                      | {fp_c1:.0%}                     |

## Stretch

To match C1's OOD FP rate ({fp_c1:.0%}) under C0b, τ_C0b must be raised to
{tau_c0b_match:+.3f}, dropping C0b's in-library recall to {recall_c0b_at_match:.0%}
(from 95%).

## Verdict

{verdict_text}

## Confusion clusters under C0b

OOD queries that exceed τ_C0b ({len(above_thr_c0b)} of {N_OOD}) routed to:
""" + "".join(
    f"  - adapter {ai} ({library[ai]['test']['type']}, prompt: \"{library[ai]['test']['prompt'][:80]}\"): {cnt}\n"
    for ai, cnt in spurious_c0b.most_common()
) + f"""

## Caveat

In-library queries from Phase 65 are held-out *paraphrased questions*; OOD
queries here are raw WikiText-2 passages of {QUERY_TOKENS} tokens. The
distribution mismatch was acknowledged in the spec as "slightly less ideal but
acceptable" — the paraphrase generators in Phase 25/27 are template-bound to
the structured passkey tests and don't apply to arbitrary WikiText.

## Files

- `specificity.json` - per-query records and full summary stats
- `cosine_distributions.png` - in-library vs OOD histograms, both conditions
- `phase65_specificity.py` - the script (under experiments/identity_ae/)

Pre-commitment was Outcome A. Result: {verdict}.
"""
    readme_path = PHASE65_DIR / "specificity_README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"Saved {readme_path}")
    print(f"\nVerdict: {verdict_text}")


if __name__ == "__main__":
    main()
