"""Score answer quality and compute coverage stats.

Answer quality: similarity between each condition's generation and the
full-context (ceiling) generation. We use a simple-but-meaningful
metric: token-level F1 (proportion of overlapping content tokens between
two answers). Standard in QA evaluation; doesn't require external models.

Coverage: for each condition, count how many of the expected-relevant
turns' answer-key tokens appear in the generation. The "answer key"
for a turn is its title + a few keywords. This catches the case where
the model hallucinates plausible Civil War content from pretraining
without using the engrams.

(BERTScore would be better but requires loading another model.
Token-F1 is a defensible substitute.)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.topics import ALL_TOPICS

STOPWORDS = set("""a an the and or of in on at to for from with by is was are were be
been being it its this that these those i you he she they we us them
him her his its my your their our as if but not no yes also so very
just only than then now thus into upon between within without through
during about across over under after before above below toward up down
out off again further while because where when how what which who whom
whose why all any most some many few both each every other another
do does did done can could should would may might must shall will
have has had having more less much fewer same different""".split())


def tokens(text):
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", text)
            if w.lower() not in STOPWORDS and len(w) > 2]


def f1_overlap(pred, ref):
    """Standard token-overlap F1 between two strings (case-insensitive,
    whitespace tokenized after stripping common stopwords)."""
    p = tokens(pred); r = tokens(ref)
    if not p or not r:
        return 0.0
    p_set = set(p); r_set = set(r)
    overlap = p_set & r_set
    if not overlap: return 0.0
    precision = len(overlap) / len(p_set)
    recall    = len(overlap) / len(r_set)
    return 2 * precision * recall / (precision + recall)


def title_keywords(title):
    """Extract content keywords from a turn title (drop stopwords)."""
    return set(tokens(title))


def coverage(generation, relevant_turn_ids, all_topics=ALL_TOPICS):
    """Fraction of relevant turn titles whose keywords appear in
    `generation`. A turn "covered" iff at least one of its title
    keywords (post-stopword filtering) appears in the generation."""
    if not relevant_turn_ids:
        return 0.0
    gen_tokens = set(tokens(generation))
    n_covered = 0
    for tid in relevant_turn_ids:
        title = all_topics[tid][0]
        kw = title_keywords(title)
        if kw & gen_tokens:
            n_covered += 1
    return n_covered / len(relevant_turn_ids)


def main():
    data = json.loads((EXP / "results/conditions.json").read_text())
    results = data["results"]

    # For each probe + condition, compute F1 vs ceiling and coverage
    summary = {
        "f1": {"recent_only": [], "random_engrams": [],
                "uniform_pool": [], "engram_routing": []},
        "coverage": {"full_context": [], "recent_only": [],
                      "random_engrams": [], "uniform_pool": [],
                      "engram_routing": []},
        "routing_p_at_k": [],
        "routing_r_at_k": [],
        "routing_r_at_20": [],
    }

    for r in results:
        ceiling = r["gen_full_context"]
        for cond in ("recent_only", "random_engrams", "uniform_pool", "engram_routing"):
            cand = r[f"gen_{cond}"]
            summary["f1"][cond].append(f1_overlap(cand, ceiling))
        for cond in ("full_context", "recent_only", "random_engrams",
                     "uniform_pool", "engram_routing"):
            gen = r[f"gen_{cond}"]
            summary["coverage"][cond].append(
                coverage(gen, r["relevant_ids"]))
        summary["routing_p_at_k"].append(r["routing_p_at_k"])
        summary["routing_r_at_k"].append(r["routing_r_at_k"])
        summary["routing_r_at_20"].append(r["routing_r_at_20"])

    # Aggregate
    print(f"\n=== Summary across {len(results)} test probes ===\n")
    print("Answer quality (token-F1 vs full-context ceiling):")
    for cond in ("recent_only", "random_engrams", "uniform_pool", "engram_routing"):
        v = summary["f1"][cond]
        print(f"  {cond:18s}  mean={np.mean(v):.3f}  median={np.median(v):.3f}  "
              f"p25={np.quantile(v, 0.25):.3f}  p75={np.quantile(v, 0.75):.3f}")

    print("\nCoverage of relevant turns (frac. of relevant titles whose "
          "keywords appear in the generation):")
    for cond in ("full_context", "recent_only", "random_engrams",
                  "uniform_pool", "engram_routing"):
        v = summary["coverage"][cond]
        print(f"  {cond:18s}  mean={np.mean(v):.3f}  median={np.median(v):.3f}")

    print(f"\nRouting precision/recall (k={data['config']['top_k']}):")
    print(f"  P@k = {np.mean(summary['routing_p_at_k']):.3f}")
    print(f"  R@k = {np.mean(summary['routing_r_at_k']):.3f}")
    print(f"  R@20 = {np.mean(summary['routing_r_at_20']):.3f}")

    # Save summary
    out = {
        "summary": {
            "f1_means":       {c: float(np.mean(v))
                                for c, v in summary["f1"].items()},
            "coverage_means": {c: float(np.mean(v))
                                for c, v in summary["coverage"].items()},
            "routing_p_at_k_mean": float(np.mean(summary["routing_p_at_k"])),
            "routing_r_at_k_mean": float(np.mean(summary["routing_r_at_k"])),
            "routing_r_at_20_mean": float(np.mean(summary["routing_r_at_20"])),
        },
        "per_probe": [
            {
                "probe_idx": r["probe_idx"], "prompt": r["prompt"],
                "n_relevant": r["n_relevant"],
                "routing_p_at_k": r["routing_p_at_k"],
                "routing_r_at_k": r["routing_r_at_k"],
                "f1_recent":         f1_overlap(r["gen_recent_only"], r["gen_full_context"]),
                "f1_random":         f1_overlap(r["gen_random_engrams"], r["gen_full_context"]),
                "f1_uniform":        f1_overlap(r["gen_uniform_pool"], r["gen_full_context"]),
                "f1_engram_routing": f1_overlap(r["gen_engram_routing"], r["gen_full_context"]),
                "cov_recent":         coverage(r["gen_recent_only"], r["relevant_ids"]),
                "cov_random":         coverage(r["gen_random_engrams"], r["relevant_ids"]),
                "cov_uniform":        coverage(r["gen_uniform_pool"], r["relevant_ids"]),
                "cov_engram_routing": coverage(r["gen_engram_routing"], r["relevant_ids"]),
                "cov_full":           coverage(r["gen_full_context"], r["relevant_ids"]),
            }
            for r in results
        ],
    }
    out_path = EXP / "results/scores.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
