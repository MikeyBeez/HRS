"""Dump v3 generations from all 11 conditions × 20 probes (and a
5-probe subset) into structured markdown for external judging.

For each probe: probe text, list of relevant turn IDs, EXCERPT of the
relevant turns from the original dataset, and all 11 condition
generations with their per-probe F1 scores.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"
PRIOR = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))


# Token-F1 (same definition as the rest of the experiment)
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
    p = tokens(pred); r = tokens(ref)
    if not p or not r: return 0.0
    p_set = set(p); r_set = set(r)
    overlap = p_set & r_set
    if not overlap: return 0.0
    precision = len(overlap) / len(p_set)
    recall    = len(overlap) / len(r_set)
    return 2 * precision * recall / (precision + recall)


CONDITIONS = [
    ("full_context",       "gen_full_context"),
    ("recent_only",        "gen_recent_only"),
    ("prompts_as_context", "gen_prompts_as_context"),
    ("engram_8_p",         "gen_engram_8_p"),
    ("engram_8_pr",        "gen_engram_8_pr"),
    ("engram_16_p",        "gen_engram_16_p"),
    ("engram_16_pr",       "gen_engram_16_pr"),
    ("engram_24_p",        "gen_engram_24_p"),
    ("engram_24_pr",       "gen_engram_24_pr"),
    ("uniform_pool",       "gen_uniform_pool"),
    ("random_engrams",     "gen_random_engrams"),
]


def render_probe(r, turns):
    """r: a v3 result record. turns: list of all 100 turn dicts."""
    lines = []
    lines.append(f"=== PROBE {r['probe_idx']} ===\n")
    lines.append(f"PROBE TEXT: {r['prompt']}\n")
    lines.append(
        f"RELEVANT TURNS (manually annotated, {r['n_relevant']} total): "
        f"{r['relevant_ids']}\n"
    )

    lines.append("CONVERSATION HISTORY EXCERPT (relevant turns only):\n")
    lines.append("```")
    for tid in r["relevant_ids"]:
        t = turns[tid]
        lines.append(f"[Turn {tid}: {t['title']}]")
        lines.append(f"USER: {t['prompt']}")
        lines.append(f"ASSISTANT: {t['response'].strip()}")
        lines.append("")
    lines.append("```\n")

    ceiling = r["gen_full_context"]
    for cond_label, gen_key in CONDITIONS:
        gen = r[gen_key]
        if cond_label == "full_context":
            f1_str = "F1=1.000"
        else:
            f1_str = f"F1={f1_overlap(gen, ceiling):.3f}"
        lines.append(f"--- {cond_label} ({f1_str}) ---")
        lines.append("```")
        lines.append(gen.strip())
        lines.append("```\n")

    return "\n".join(lines)


def main():
    data = json.loads((EXP / "results/conditions_v3.json").read_text())
    results = data["results"]
    turns = json.loads((PRIOR / "data/turns.json").read_text())["turns"]

    # 5-probe subset (same IDs as the inspection task)
    subset_idx = [0, 13, 11, 4, 12]

    # Build full dump
    full_lines = []
    full_lines.append("# v3 generation dump for external judging\n")
    full_lines.append(
        "All 20 v3 probes × 11 conditions = 220 generated answers.\n"
        "F1 values are per-probe token-F1 vs full_context (ceiling).\n"
        "Conversation excerpts include only the manually-annotated "
        "relevant turns (full 100-turn corpus is at "
        "`experiments/multi_engram/data/turns.json`).\n"
        "Conditions ordered: full_context (ceiling) → text baselines → "
        "engrams (sweep) → uniform_pool → random_engrams (floor).\n"
    )
    full_lines.append("---\n")

    subset_lines = []
    subset_lines.append("# v3 generation dump — 5 probes for initial review\n")
    subset_lines.append(
        "Same 5 probes used in the earlier `answer_inspection.md` "
        "(probes 0, 4, 11, 12, 13).\n"
        "Each probe: relevant turn excerpts + 11 condition generations "
        "+ per-condition F1 vs the full_context ceiling.\n"
    )
    subset_lines.append("---\n")

    for r in sorted(results, key=lambda x: x["probe_idx"]):
        block = render_probe(r, turns)
        full_lines.append(block)
        full_lines.append("---\n")
        if r["probe_idx"] in subset_idx:
            subset_lines.append(block)
            subset_lines.append("---\n")

    out_full = EXP / "results/v3_generations_for_judging.md"
    out_full.write_text("\n".join(full_lines))
    out_subset = EXP / "results/v3_generations_5probes.md"
    out_subset.write_text("\n".join(subset_lines))
    print(f"saved {out_full}  ({out_full.stat().st_size:,} bytes, "
          f"{len(full_lines)} sections)")
    print(f"saved {out_subset}  ({out_subset.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
