"""Pick 5 probes spanning topics with wide F1 spread, dump answers."""
from __future__ import annotations

import json
import sys
import re
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"
sys.path.insert(0, str(REPO))

STOPWORDS = set("""a an the and or of in on at to for from with by is was are were be
been being it its this that these those i you he she they we us them
him her his its my your their our as if but not no yes also so very
just only than then now thus into upon between within without through
during about across over under after before above below toward up down
out off again further while because where when how what which who whom
whose why all any most some many few both each every other another
do does did done can could should should would may might must shall will
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


def main():
    data = json.loads((EXP / "results/conditions.json").read_text())
    results = data["results"]

    # Per-probe F1s
    rows = []
    for r in results:
        ceiling = r["gen_full_context"]
        rows.append({
            "idx": r["probe_idx"],
            "prompt": r["prompt"],
            "ans_ceiling": r["gen_full_context"],
            "ans_recent": r["gen_recent_only"],
            "ans_prompts": r["gen_prompts_as_context"],
            "ans_engram": r["gen_engram_8_pr"],
            "ans_random": r["gen_random_engrams"],
            "f1_recent": f1_overlap(r["gen_recent_only"], ceiling),
            "f1_prompts": f1_overlap(r["gen_prompts_as_context"], ceiling),
            "f1_engram": f1_overlap(r["gen_engram_8_pr"], ceiling),
            "f1_random": f1_overlap(r["gen_random_engrams"], ceiling),
        })

    # Spread = max - min across the 4 non-ceiling conditions
    for r in rows:
        f1s = [r["f1_recent"], r["f1_prompts"], r["f1_engram"], r["f1_random"]]
        r["spread"] = max(f1s) - min(f1s)

    # Print all probes with F1s + spread, sorted by spread desc
    rows_sorted = sorted(rows, key=lambda r: r["spread"], reverse=True)
    print("All 20 probes by spread (recent / prompts / engram_8_pr / random / spread):")
    for r in rows_sorted:
        print(f"  [{r['idx']:2d}] spread={r['spread']:.3f}  "
              f"R={r['f1_recent']:.3f} P={r['f1_prompts']:.3f} "
              f"E={r['f1_engram']:.3f} X={r['f1_random']:.3f}  "
              f"| {r['prompt'][:60]}")

    # Pick 5 probes spanning topics, preferring wider F1 spread:
    # 0  Grant (general/strategy)               spread 0.70
    # 13 eastern theater (battles/campaigns)    spread 0.63
    # 11 political path Fort Sumter to Appomattox (political) spread 0.46
    # 4  African American troops (social)       spread 0.36
    # 12 mobilization/recruitment (logistics/social) spread 0.25
    selected_idx = [0, 13, 11, 4, 12]
    selected = [r for r in rows if r["idx"] in selected_idx]
    selected.sort(key=lambda r: selected_idx.index(r["idx"]))

    out = []
    out.append("# Answer inspection: 5 probes × 5 conditions\n")
    out.append("Manual dump of generated answers from "
                "`results/conditions.json`. F1 values are per-probe "
                "token-F1 vs `gen_full_context` (the ceiling). "
                "5 probes selected to span topic areas with the widest "
                "available F1 spread: generals/strategy (Grant), "
                "battles/campaigns (eastern theater), political "
                "(political path Fort Sumter to Appomattox), social "
                "(African American troops), logistics/recruitment "
                "(manpower mobilization).\n")
    out.append("Generations are dumped verbatim, no editing or "
                "summarization.\n")
    out.append("---\n")

    for i, r in enumerate(selected):
        out.append(f"## Probe {i+1} (test_idx={r['idx']})\n")
        out.append(f"**PROBE:** {r['prompt']}\n")
        out.append(f"### CEILING (full_context, F1=1.000):")
        out.append("```")
        out.append(r["ans_ceiling"].strip())
        out.append("```\n")
        out.append(f"### RECENT_ONLY (F1={r['f1_recent']:.3f}):")
        out.append("```")
        out.append(r["ans_recent"].strip())
        out.append("```\n")
        out.append(f"### PROMPTS_AS_CONTEXT (F1={r['f1_prompts']:.3f}):")
        out.append("```")
        out.append(r["ans_prompts"].strip())
        out.append("```\n")
        out.append(f"### ENGRAM_8_PR (F1={r['f1_engram']:.3f}):")
        out.append("```")
        out.append(r["ans_engram"].strip())
        out.append("```\n")
        out.append(f"### RANDOM_ENGRAMS (F1={r['f1_random']:.3f}):")
        out.append("```")
        out.append(r["ans_random"].strip())
        out.append("```\n")
        out.append("---\n")

    out_path = EXP / "results/answer_inspection.md"
    out_path.write_text("\n".join(out))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
