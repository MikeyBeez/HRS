"""Phase 65 specificity ablation: length-matched OOD.

The 128-token OOD ablation found Outcome D (C0b cleaner than C1). The most
plausible alternative explanation is length asymmetry: in-library queries are
~10-token paraphrased questions, OOD were 128-token WikiText passages. Long
mean-pooled L0 sequences are pulled toward the corpus centroid, which would
artificially lower OOD cosines under C0b.

This ablation rules out length as the driver by sampling OOD spans matching
the empirical in-library length distribution (6-16 tokens, median 11).

Reuses cache from Phase 65; no retraining. Compares against the 128-token
specificity.json so both length conditions sit side-by-side in the writeup.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase65_specificity_lengthmatched.py
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
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.phase27_held_out import held_out_paraphrase
from experiments.identity_ae.lora_wrapper import apply_lora


N_OOD = 30
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


def in_library_length_distribution(tokenizer):
    """Empirical token-length distribution of the 60 held-out paraphrases."""
    lengths = []
    for test in stratified_tests():
        for q in held_out_paraphrase(test):
            lengths.append(len(tokenizer.encode(q, add_special_tokens=False)))
    return lengths


def sample_ood_lengthmatched(tokenizer, length_distribution, n=N_OOD, seed=SEED):
    """Sample n OOD spans from WikiText-2 train, lengths drawn from the empirical
    in-library length distribution."""
    from datasets import load_dataset
    print(f"[A] loading WikiText-2 train")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    rng = random.Random(seed)
    texts = [t for t in ds["text"] if t and len(t.split()) > 20]
    rng.shuffle(texts)

    passages = []
    cursor = 0
    while len(passages) < n and cursor < len(texts):
        L = rng.choice(length_distribution)
        text = texts[cursor]
        cursor += 1
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < L:
            continue
        start = rng.randint(0, len(ids) - L)
        chunk_ids = ids[start:start + L]
        passages.append({
            "ids": chunk_ids,
            "len": L,
            "text": tokenizer.decode(chunk_ids, skip_special_tokens=True),
        })
    print(f"[A] sampled {len(passages)} length-matched OOD spans "
          f"(min {min(p['len'] for p in passages)}, "
          f"max {max(p['len'] for p in passages)}, "
          f"mean {sum(p['len'] for p in passages)/len(passages):.1f})")
    return passages


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    PHASE65_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load cache ----
    print(f"[A] loading Phase 65 cache from {CACHE_DIR}")
    blob = torch.load(CACHE_DIR / "library.pt", map_location="cpu", weights_only=False)
    library, keys_l0, keys_l5 = blob["library"], blob["keys_l0"], blob["keys_l5"]
    M_trained = torch.load(CACHE_DIR / "M_trained.pt", map_location="cpu", weights_only=False)

    # ---- In-library length distribution ----
    in_lengths = in_library_length_distribution(tokenizer)
    print(f"[A] in-library lengths: min {min(in_lengths)}, max {max(in_lengths)}, "
          f"mean {sum(in_lengths)/len(in_lengths):.1f}, n={len(in_lengths)}")

    # ---- In-library reference scores ----
    p65 = json.load(open(PHASE65_DIR / "phase65_results.json"))
    in_c0b = [t["best_score"] for t in p65["trials"]["C0b I,         vs L0"]]
    in_c1  = [t["best_score"] for t in p65["trials"]["C1  trained W, vs L5"]]

    # ---- Build base model ----
    model, _ = load_model(device)
    apply_lora(model, rank=128, alpha=256, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)

    # ---- Length-matched OOD ----
    ood = sample_ood_lengthmatched(tokenizer, in_lengths, n=N_OOD, seed=SEED)

    # ---- Score ----
    print(f"[B] scoring {len(ood)} length-matched OOD queries")
    records = []
    for i, p in enumerate(ood):
        ids_t = torch.tensor(p["ids"], dtype=torch.long).unsqueeze(0).to(device)
        q_l0 = l0_mean(model, ids_t)
        a_c0b, s_c0b = best_match(q_l0, keys_l0)
        q_proj = q_l0 @ M_trained
        a_c1, s_c1 = best_match(q_proj, keys_l5)
        records.append({
            "ood_idx": i, "len": p["len"],
            "text_preview": p["text"][:120].replace("\n", " "),
            "c0b": {"routed_idx": a_c0b, "best_score": float(s_c0b)},
            "c1":  {"routed_idx": a_c1,  "best_score": float(s_c1)},
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
    # 0% OOD FP threshold
    tau_c0b_zero = max(ood_c0b) + 1e-9
    tau_c1_zero  = max(ood_c1)  + 1e-9
    rec_c0b_zero = fraction_above(in_c0b, tau_c0b_zero)
    rec_c1_zero  = fraction_above(in_c1,  tau_c1_zero)

    print(f"\n[C] LENGTH-MATCHED RESULTS")
    print(f"    C0b: in_mean {sum(in_c0b)/len(in_c0b):+.3f}  ood_mean {sum(ood_c0b)/len(ood_c0b):+.3f}  "
          f"sep {sep_c0b:+.3f}")
    print(f"         tau@95rec {tau_c0b:+.3f}, OOD FP {fp_c0b:.0%}; "
          f"0% OOD FP -> recall {rec_c0b_zero:.0%}")
    print(f"    C1 : in_mean {sum(in_c1)/len(in_c1):+.3f}   ood_mean {sum(ood_c1)/len(ood_c1):+.3f}   "
          f"sep {sep_c1:+.3f}")
    print(f"         tau@95rec {tau_c1:+.3f}, OOD FP {fp_c1:.0%}; "
          f"0% OOD FP -> recall {rec_c1_zero:.0%}")

    # ---- Compare to 128-token run ----
    orig = json.load(open(PHASE65_DIR / "specificity.json"))
    orig_c0b = [r["c0b"]["best_score"] for r in orig["ood_per_query"]]
    orig_c1  = [r["c1"]["best_score"]  for r in orig["ood_per_query"]]
    print(f"\n[D] COMPARISON TO 128-TOKEN OOD")
    print(f"    C0b OOD mean: 128tok {sum(orig_c0b)/len(orig_c0b):+.3f}  ->  "
          f"length-matched {sum(ood_c0b)/len(ood_c0b):+.3f}")
    print(f"    C0b OOD FP:   128tok {orig['calibration']['ood_fp_rate_c0b']:.0%}  ->  "
          f"length-matched {fp_c0b:.0%}")
    print(f"    C1  OOD mean: 128tok {sum(orig_c1)/len(orig_c1):+.3f}   ->  "
          f"length-matched {sum(ood_c1)/len(ood_c1):+.3f}")
    print(f"    C1  OOD FP:   128tok {orig['calibration']['ood_fp_rate_c1']:.0%}   ->  "
          f"length-matched {fp_c1:.0%}")

    # ---- Save JSON ----
    out = {
        "config": {
            "n_ood": N_OOD, "seed": SEED,
            "ood_source": "wikitext-2-raw-v1 train",
            "length_sampling": "drawn with replacement from in-library held-out paraphrase lengths",
            "length_distribution_summary": {
                "n": len(in_lengths), "min": min(in_lengths), "max": max(in_lengths),
                "mean": sum(in_lengths)/len(in_lengths),
            },
        },
        "in_library_summary": {
            "c0b": {"mean": sum(in_c0b)/len(in_c0b),
                    "p05": quantile(in_c0b, 0.05), "p95": quantile(in_c0b, 0.95)},
            "c1":  {"mean": sum(in_c1)/len(in_c1),
                    "p05": quantile(in_c1, 0.05), "p95": quantile(in_c1, 0.95)},
        },
        "ood_summary_lengthmatched": {
            "c0b": {"mean": sum(ood_c0b)/len(ood_c0b),
                    "min": min(ood_c0b), "max": max(ood_c0b),
                    "p50": quantile(ood_c0b, 0.5)},
            "c1":  {"mean": sum(ood_c1)/len(ood_c1),
                    "min": min(ood_c1),  "max": max(ood_c1),
                    "p50": quantile(ood_c1, 0.5)},
        },
        "ood_summary_128tok_reference": {
            "c0b": {"mean": sum(orig_c0b)/len(orig_c0b),
                    "min": min(orig_c0b), "max": max(orig_c0b)},
            "c1":  {"mean": sum(orig_c1)/len(orig_c1),
                    "min": min(orig_c1),  "max": max(orig_c1)},
        },
        "calibration_lengthmatched": {
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
    out_path = PHASE65_DIR / "specificity_lengthmatched.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # ---- Plot: 2x2 grid showing length effect on each condition ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")

    bins_c0b = [-0.1 + i * 0.05 for i in range(int(2.0 / 0.05) + 1)]
    bins_c1  = [-0.1 + i * 0.02 for i in range(int(0.8 / 0.02) + 1)]

    # Top row: C0b
    axes[0, 0].hist(in_c0b,   bins=bins_c0b, alpha=0.55, color="C0", label=f"in-library (n={len(in_c0b)})")
    axes[0, 0].hist(orig_c0b, bins=bins_c0b, alpha=0.55, color="C3", label=f"OOD 128tok (n={len(orig_c0b)})")
    axes[0, 0].axvline(tau_c0b, color="black", ls="--", lw=1)
    axes[0, 0].set_title(f"C0b: 128-tok OOD\nFP@95rec={orig['calibration']['ood_fp_rate_c0b']:.0%}, "
                         f"sep={sum(in_c0b)/len(in_c0b) - sum(orig_c0b)/len(orig_c0b):+.3f}")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(in_c0b,  bins=bins_c0b, alpha=0.55, color="C0", label=f"in-library (n={len(in_c0b)})")
    axes[0, 1].hist(ood_c0b, bins=bins_c0b, alpha=0.55, color="C3", label=f"OOD len-matched (n={len(ood_c0b)})")
    axes[0, 1].axvline(tau_c0b, color="black", ls="--", lw=1, label=f"tau_95={tau_c0b:.3f}")
    axes[0, 1].set_title(f"C0b: length-matched OOD\nFP@95rec={fp_c0b:.0%}, sep={sep_c0b:+.3f}")
    axes[0, 1].legend(loc="upper left", fontsize=8)

    # Bottom row: C1
    axes[1, 0].hist(in_c1,   bins=bins_c1, alpha=0.55, color="C0", label=f"in-library (n={len(in_c1)})")
    axes[1, 0].hist(orig_c1, bins=bins_c1, alpha=0.55, color="C3", label=f"OOD 128tok (n={len(orig_c1)})")
    axes[1, 0].axvline(tau_c1, color="black", ls="--", lw=1)
    axes[1, 0].set_title(f"C1: 128-tok OOD\nFP@95rec={orig['calibration']['ood_fp_rate_c1']:.0%}, "
                         f"sep={sum(in_c1)/len(in_c1) - sum(orig_c1)/len(orig_c1):+.3f}")
    axes[1, 0].set_xlabel("max cosine over 80 keys")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend(loc="upper left", fontsize=8)

    axes[1, 1].hist(in_c1,  bins=bins_c1, alpha=0.55, color="C0", label=f"in-library (n={len(in_c1)})")
    axes[1, 1].hist(ood_c1, bins=bins_c1, alpha=0.55, color="C3", label=f"OOD len-matched (n={len(ood_c1)})")
    axes[1, 1].axvline(tau_c1, color="black", ls="--", lw=1, label=f"tau_95={tau_c1:.3f}")
    axes[1, 1].set_title(f"C1: length-matched OOD\nFP@95rec={fp_c1:.0%}, sep={sep_c1:+.3f}")
    axes[1, 1].set_xlabel("max cosine over 80 keys")
    axes[1, 1].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = PHASE65_DIR / "cosine_distributions_lengthmatched.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved {plot_path}")

    # ---- Verdict on the length confound ----
    survives_c0b = fp_c0b <= 0.10
    survives_c1  = fp_c1  >  0.30
    if survives_c0b and survives_c1:
        verdict = ("Length confound RULED OUT. Outcome D survives length matching: "
                   "C0b retains clean OOD rejection and C1 retains heavy overlap. "
                   "The architectural claim (C0b > C1 for open-world rejection) is robust.")
    elif not survives_c0b and survives_c1:
        verdict = ("Length confound PARTIAL: C0b's separation shrunk under length matching "
                   "while C1's overlap remained. The C0b advantage is partly length-driven; "
                   "the C1 overlap is intrinsic to the trained band.")
    elif survives_c0b and not survives_c1:
        verdict = ("Surprising: C1's specificity recovered under length matching. "
                   "Investigate — this suggests C1 is sensitive to query length in a way "
                   "that the 128-tok run obscured.")
    else:
        verdict = ("Both conditions degraded under length matching. The 128-tok separation "
                   "for both was likely length-driven; need a third experiment to disentangle.")
    print(f"\n[E] {verdict}")


if __name__ == "__main__":
    main()
