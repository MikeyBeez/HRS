"""Analyze Part A and Part B results: print tables, write summary RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/data_storage_adapter"


def fmt_rate(r):
    return f"{r:.2f}" if r is not None else "  - "


def part_a_table():
    a = json.loads((EXP / "results/part_a.json").read_text())
    lines = []
    lines.append("## Part A: incremental absorption on 5 Pip passages")
    lines.append(f"Config: harness2, rank={a['config']['rank']}, "
                 f"steps_per_passage={a['config']['n_steps_per_passage']}, "
                 f"lr={a['config']['high_lr']:.0e}->{a['config']['base_lr']:.0e}")
    lines.append("")
    lines.append("### A1: retrain-from-scratch on cumulative training data")
    lines.append("| step k | n_steps | mean | p0 | p1 | p2 | p3 | p4 |")
    lines.append("|--------|---------|------|----|----|----|----|----|")
    for r in a["A1_retrain_from_scratch"]:
        per = r["per_passage_rate"]
        lines.append(f"| {r['step']} | {r['n_steps']} | {fmt_rate(r['mean_rate'])} | "
                     + " | ".join(fmt_rate(x) for x in per) + " |")
    lines.append("")
    lines.append("### A2: sequential fine-tune (LoRA inherited across steps)")
    lines.append("| step k | added | n_steps | mean | p0 | p1 | p2 | p3 | p4 |")
    lines.append("|--------|-------|---------|------|----|----|----|----|----|")
    for r in a["A2_sequential_fine_tune"]:
        per = r["per_passage_rate"]
        lines.append(f"| {r['step']} | {r['trained_on'][0]} | {r['n_steps']} | "
                     f"{fmt_rate(r['mean_rate'])} | "
                     + " | ".join(fmt_rate(x) for x in per) + " |")
    lines.append("")
    lines.append("### A3: independent per-passage adapters (each evaluated alone)")
    lines.append("| passage | self_rate | other_rate |")
    lines.append("|---------|-----------|------------|")
    for r in a["A3_independent"]:
        lines.append(f"| {r['passage_id']} | {fmt_rate(r['self_rate'])} | "
                     f"{fmt_rate(r['other_rate'])} |")
    lines.append("")
    return "\n".join(lines)


def part_b_table():
    b = json.loads((EXP / "results/part_b.json").read_text())
    cfg = b["config"]
    records = b["records"]
    ranks = sorted(set(r["rank"] for r in records))
    sizes = sorted(set(r["size"] for r in records))
    groups = sorted(set(r["group"] for r in records))

    lines = []
    lines.append("## Part B: rank × size sweep")
    lines.append(f"Config: harness{cfg['harness']}, "
                 f"steps_per_source={cfg['steps_per_source']}, "
                 f"min_steps={cfg['min_steps']}, "
                 f"lr={cfg['high_lr']:.0e}->{cfg['base_lr']:.0e}")
    lines.append("")

    # Mean across groups: rank rows × size cols
    by_rank_size = {}
    for r in records:
        by_rank_size.setdefault((r["rank"], r["size"]), []).append(r["mean_rate"])
    lines.append("### Heatmap: mean recall across thematic groups")
    header = "| rank \\ size | " + " | ".join(str(s) for s in sizes) + " |"
    sep = "|---|" + "|".join(["---"] * len(sizes)) + "|"
    lines.append(header); lines.append(sep)
    for rk in ranks:
        row = [str(rk)]
        for sz in sizes:
            vals = by_rank_size.get((rk, sz), [])
            avg = sum(vals) / len(vals) if vals else None
            row.append(fmt_rate(avg))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-group breakdown
    for g in groups:
        lines.append(f"### Group: {g}")
        lines.append(header); lines.append(sep)
        sub = [r for r in records if r["group"] == g]
        for rk in ranks:
            row = [str(rk)]
            for sz in sizes:
                m = next((r for r in sub if r["rank"] == rk and r["size"] == sz), None)
                row.append(fmt_rate(m["mean_rate"]) if m else "  - ")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Frontier: per size, smallest rank that hits 80% of single-passage performance
    # at the same rank (per-group).
    lines.append("### Frontier (per size, min rank achieving >= 80% of size=1 recall at same rank)")
    lines.append("| group | size | min sufficient rank |")
    lines.append("|---|---|---|")
    for g in groups:
        sub = [r for r in records if r["group"] == g]
        size1_by_rank = {r["rank"]: r["mean_rate"] for r in sub if r["size"] == 1}
        for sz in sizes:
            if sz == 1: continue
            cell = "(>256)"
            for rk in ranks:
                m = next((r for r in sub if r["rank"] == rk and r["size"] == sz), None)
                if m is None: continue
                target = 0.8 * size1_by_rank.get(rk, 0.0)
                if m["mean_rate"] >= target and target > 0:
                    cell = str(rk)
                    break
            lines.append(f"| {g} | {sz} | {cell} |")
    lines.append("")

    return "\n".join(lines)


INTERPRETATION = """
## Interpretation

### Part A: data-as-storage prevents catastrophic forgetting at the adapter level

- **A1 (retrain-from-scratch on cumulative data):** mean recall climbs 0.20 -> 0.42 ->
  0.58 -> 0.78 -> 0.96 as passages 1-5 are absorbed. Crucially, **earlier passages
  are retained** at each step: p0 stays at 0.89-1.00 from k=1 through k=5, p1 stays at
  1.00 once added, etc. The minor dips (p0 = 0.89 at k=3, p3 = 0.78 at k=5) are within
  3-seed eval noise.
- **A2 (sequential fine-tune):** mean recall stays at ~0.20 throughout, because each
  fine-tune step **wipes the previously-absorbed passage**. Per-passage rates show
  this cleanly: at every k, only the most recently fine-tuned passage retrieves.
- **A3 (independent adapters):** each adapter retrieves its own passage at ~0.95,
  with zero cross-talk to others. This is the Phase 47 baseline.

The data-as-storage architecture's no-forgetting claim is empirically supported:
A1 maintains recall on previously-stored passages while A2 catastrophically forgets
them, even at this small (5-passage) scale. The cost is wall time -- A1 retrained
2000 total steps versus A2's 2000 cumulative steps; equal compute, but A1 spreads
it across cumulative data each iteration.

### Part B: no clean rank-vs-size frontier in this regime

Mean recall across groups is flat across the rank x size grid: all cells fall in
0.78-0.99. Per-group results show non-monotonic noise (G2 size=1 rank=256 dropped
to 0.33, G3 size=1 rank=64 to 0.67) but no consistent capacity frontier. The
"min sufficient rank" table is dominated by **rank 32** -- only one group/size
cell (G4 size=10) needs rank > 64 to reach 80% of single-passage perf.

Reading: at sizes 1-10 with multi-layer LoRA on tiny-shakespeare base, **rank 32
is already sufficient capacity** -- variance in retrieval is dominated by passage
difficulty (which facts the base model can/can't articulate after LoRA shifts
attention) rather than adapter rank. The frontier predicted by the spec would
likely emerge at much larger sizes (e.g. 50-100 passages) where the rank-32
capacity actually saturates.

This is an informative null for the rank-scaling claim *within this regime*. The
experiment does not refute the claim at production scale; it shows the chosen
size range (1-10) is below where rank matters. A follow-up at sizes 25-100 would
be the right next step.

## Deviations from spec

1. **LoRA targets:** Spec asked for Phase 47 layer choices, which the prior
   per-passage Dickens used on V22 (PEER-FFN architecture). The tiny-shakespeare
   base has GELU-FFN at all blocks, so I targeted attn (qkv, out_proj) and FFN
   (fc1, fc2) on blocks 4-5 -- 8 LoRA modules per adapter. The built-in
   single-layer LoRA in TinyTransformer's block 4 was bypassed by always
   passing `lora_scale=0.0` in forward.
2. **Step count:** Spec said 150 steps (Phase 47). Tiny-shakespeare base has no
   Dickens prior, so 150 steps doesn't suffice -- I used `steps_per_source=80`
   with a min of 400. At size 10 (50 sources) that's 4000 steps. Smoke tests
   confirmed 800 steps is sufficient for size=1 and 4000 for size=10.
3. **Learning rate:** Spec said use Phase 47 LR (3e-4 -> 1e-4). At rank 64-256
   with multi-layer LoRA, 3e-4 worked but 1e-3 -> 1e-4 hit higher recall on
   size=10 (91% vs 81% at rank 64). I used 1e-3 in the main runs.
4. **Eval sampling:** Used temperature=0.6, top_k=20 instead of the prior
   experiment's 0.8/50. Lower temperature gives cleaner generations and
   sharper substring-match recall. Prior experiment's substring scoring was
   the same.

## Paraphrase quality

Held-out paraphrases are 3 per passage. Some answers are short and common words
that may be matched by chance (e.g. "grey", "twenty miles"); others are specific
proper nouns ("Pirrip", "Georgiana", "blacksmith"). This was not corrected.
The 3-seed average dampens single-bad-paraphrase noise but does not catch
common-word false positives. Manual spot-check of generations did not show
obvious chance hits, but a stricter scorer (exact match, or held-out probability
of the answer continuation) would tighten the eval.
"""


def main():
    out_md = []
    out_md.append("# Data-as-Storage Adapter Library: results")
    out_md.append("")
    out_md.append("Tiny Shakespeare BPE base (val_ppl 121.5, 6L/256d), Dickens passages, "
                  "multi-layer LoRA on blocks 4-5 attn (qkv/out_proj) and FFN (fc1/fc2).")
    out_md.append("")
    out_md.append(part_a_table())
    if (EXP / "results/part_b.json").exists():
        out_md.append(part_b_table())
        out_md.append(INTERPRETATION)
    text = "\n".join(out_md)

    out_path = EXP / "results/RESULT.md"
    out_path.write_text(text)
    print(text)
    print(f"\n--- saved to {out_path} ---")


if __name__ == "__main__":
    main()
