"""Aggregate all ablation results into a single RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/mnt/data/Code/HRS")
RES = REPO / "experiments/hrs_ablations/results"

# Phase 47 baseline numbers (from per_passage_dickens/results/RESULT.md)
BASELINE_ROUTING = 1.000
BASELINE_RETRIEVAL = 0.929


def fmt_delta(val, baseline):
    delta = val - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:+.3f}"


def section_baseline():
    return [
        "## Phase 47 baseline (reference)",
        "",
        "From `experiments/per_passage_dickens/results/RESULT.md` "
        "(50 Dickens adapters, V22-Dickens base, rank-128 LoRA on L45):",
        "",
        f"- Routing accuracy: **{BASELINE_ROUTING:.3f}** (150/150 held-out queries)",
        f"- Retrieval accuracy: **{BASELINE_RETRIEVAL:.3f}** (substring match, "
        "3 stochastic seeds)",
        "",
        "Reproduced in `baseline_check.py` at 1.000 routing in 0.3s.",
        "",
    ]


def section_ablation1():
    p = RES / "ablation1_pooling.json"
    if not p.exists():
        return ["## Ablation 1: Engram pooling (NOT RUN)", ""]
    data = json.loads(p.read_text())
    lines = [
        "## Ablation 1: Engram pooling operation",
        "",
        "Reuses adapter library; recomputes stored library keys per (layer, "
        "pool); retrains projection W (500 InfoNCE steps); measures "
        "routing on 150 held-out queries.",
        "",
        "| variant | q_layer | s_layer | pool | proj_train | routing | max_sim | retrieval | wall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in data["results"]:
        retrieval = f"{r.get('retrieval_acc', float('nan')):.3f}" if "retrieval_acc" in r else "—"
        lines.append(
            f"| {r['variant']} | {r['q_layer']} | {r['s_layer']} | "
            f"{r['pool_op']} | {r['proj_train_acc']:.3f} | "
            f"**{r['routing_acc']:.3f}** | {r['max_sim_mean']:.3f} | "
            f"{retrieval} | {r['wall_s']:.0f}s |"
        )
    lines.append("")
    lines.append(f"*Total wall: {data['wall_total_s']:.0f}s*")
    lines.append("")
    return lines


def section_ablation2():
    p = RES / "ablation2_qproj.json"
    if not p.exists():
        return ["## Ablation 2: Q projection (NOT RUN)", ""]
    data = json.loads(p.read_text())
    lines = [
        "## Ablation 2: Engram-as-key vs separate Q projection",
        "",
        "All variants use query=L0_mean. Stored side and projection vary.",
        "",
        "| variant | routing | max_sim | delta vs baseline |",
        "|---|---:|---:|---:|",
    ]
    for r in data["results"]:
        d = fmt_delta(r["routing_acc"], BASELINE_ROUTING)
        lines.append(
            f"| {r['variant']} | **{r['routing_acc']:.3f}** | "
            f"{r['max_sim_mean']:.3f} | {d} |"
        )
    lines.append("")
    lines.append(f"*Total wall: {data['wall_s']:.0f}s*")
    lines.append("")
    return lines


def section_ablation3():
    p = RES / "ablation3_routing.json"
    if not p.exists():
        return ["## Ablation 3: Routing mechanism (NOT RUN)", ""]
    data = json.loads(p.read_text())
    lines = [
        "## Ablation 3: Routing mechanism",
        "",
        "Same engrams (L0_mean → W → L5_aggregate); routing function varies.",
        "",
        "| variant | routing | delta vs baseline |",
        "|---|---:|---:|",
    ]
    for r in data["results"]:
        d = fmt_delta(r["routing_acc"], BASELINE_ROUTING)
        lines.append(
            f"| {r['variant']} | **{r['routing_acc']:.3f}** | {d} |"
        )
    lines.append("")
    lines.append(f"*Total wall: {data['wall_s']:.0f}s*")
    lines.append("")
    return lines


def section_ablation4():
    p = RES / "ablation4_lora.json"
    if not p.exists():
        return ["## Ablation 4: LoRA configuration (NOT RUN OR INCOMPLETE)", ""]
    data = json.loads(p.read_text())
    lines = [
        "## Ablation 4: LoRA configuration",
        "",
        "Re-train 50 adapters per variant (150 steps each, V22-Dickens base, "
        "150 paraphrase-mixed sources). Routing uses canonical W; only "
        "adapter weights vary.",
        "",
        "| variant | rank | targets | params/ad | routing | retrieval | retr Δ | wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in data["results"]:
        d = fmt_delta(r["retrieval_acc"], BASELINE_RETRIEVAL)
        lines.append(
            f"| {r['variant']} | {r['rank']} | {r['n_targets']} | "
            f"{r['n_lora_params_per_adapter']:,} | "
            f"{r['routing_acc']:.3f} | **{r['retrieval_acc']:.3f}** | "
            f"{d} | {r['train_wall_s']+r['eval_wall_s']:.0f}s |"
        )
    lines.append("")
    lines.append(f"*Total wall: {data['wall_total_s']:.0f}s*")
    lines.append("")
    return lines


def section_ablation5():
    p = RES / "ablation5_unfreeze.json"
    if not p.exists():
        return ["## Ablation 5: Partial unfreezing (NOT RUN)", ""]
    data = json.loads(p.read_text())
    lines = [
        "## Ablation 5: Partial base unfreezing",
        "",
        "Each adapter is trained with both LoRA params AND block 5 of the "
        "base unfrozen. Per-adapter snapshots of (lora, block5). Routing on "
        f"canonical frozen base; generation with snapshot swap. "
        f"last_block_lr={data.get('last_block_lr', 1e-5):.0e}.",
        "",
        "| variant | rank | routing | retrieval | retr Δ vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    d = fmt_delta(data["retrieval_acc"], BASELINE_RETRIEVAL)
    lines.append(
        f"| unfreeze_block5 | {data['rank']} | "
        f"{data['routing_acc']:.3f} | **{data['retrieval_acc']:.3f}** | {d} |"
    )
    lines.append("")
    fp = data.get("forgetting_probe", [])
    if fp:
        lines.append("**Forgetting probe** — (lora=A, block5=B mismatched). "
                     "Tests whether the LoRA dominates the per-adapter "
                     "block5 snapshot:")
        lines.append("")
        lines.append("| a | b | matched (lora=A,block5=A) | mismatched (lora=A,block5=B) |")
        lines.append("|---|---|---|---|")
        for r in fp:
            lines.append(f"| {r['a']} | {r['b']} | {r['match_match']} | "
                          f"{r['mismatch_match']} |")
        lines.append("")
    return lines


def section_summary():
    return [
        "## Summary table — load-bearing vs incidental decisions",
        "",
        "| Architectural decision | Verdict | Evidence |",
        "|---|---|---|",
        "| **Frozen base model** | **Load-bearing (the simpler version is correct)** | Unfreezing block 5 *hurts* retrieval (0.900 vs 0.929 baseline) and adds 78M per-adapter params. Forgetting probe: 4/5 mismatched (lora=A, block5=B) fail to retrieve. |",
        "| Per-passage LoRA adapters | (Architectural premise, not ablated independently) | Phase 47 baseline already establishes this |",
        "| **Engram = first-layer mean** | **Not strictly required (incidental)** | Same-layer L0_mean→W→L0_aggregate also hits 100% routing / 0.929 retrieval. Mid-stack (L2) hits 99.3%. Cross-layer is canonical but not load-bearing. |",
        "| **Pool operation (mean)** | **Not load-bearing for argmax routing (incidental)** | Max and attention-pool both hit 100% routing. Caveat: max-pool drops max_sim to 0.20 (matters for any confidence-gating threshold). |",
        "| **Engram-as-key (no separate Q projection)** | **The framing is wrong — Phase 47 DOES use a projection W.** Removing it is load-bearing. | Without W: 79.3% same-layer / 6.7% cross-layer. With W (500-step InfoNCE): 100%. 5000-step W gives no improvement. |",
        "| **Argmax cosine similarity routing** | **Load-bearing — cosine specifically** | Cosine 100%, dot product 99.3% (close), euclidean 2.0% (broken, magnitude-dominated), learned MLP 79.3% (overfits 200 training paras). Top-K offers no headroom (top-1 already perfect). |",
        "| **LoRA rank** | **Not load-bearing in 32-256 range at 50 adapters** | Rank 32/64/128/256 all retrieve 0.929-0.936. Compute scales linearly with rank but accuracy doesn't. |",
        "| **LoRA targets (attn+FFN on blocks 4-5)** | **Either subset alone is sufficient at 50 adapters** | attn-only (4 modules): 0.924. FFN-only (4 modules): 0.936. Combined (8 modules, baseline): 0.929. |",
        "",
        "## Implications for the recruitment ask",
        "",
        "**Specific claims that are empirically validated and load-bearing:**",
        "1. Cosine-similarity routing in a learned-projection space.",
        "2. Frozen base model — unfreezing strictly hurts at this scale.",
        "3. Linear projection W trained with InfoNCE is the right inductive bias for the routing classifier (beats a 2-layer MLP).",
        "",
        "**Claims that are not load-bearing at the 50-adapter scale and could be relaxed:**",
        "1. Cross-layer (L0→L5) engram structure — same-layer routing also works.",
        "2. Mean-pooling specifically — max and attention-pool work too (modulo confidence-gating considerations).",
        "3. Rank 128 specifically — rank 32 is sufficient with no quality loss.",
        "4. Both attn AND FFN LoRA — either alone is sufficient.",
        "",
        "**The recruitment ask should be sharpened:** validate cosine + InfoNCE-projection routing + frozen base at scale (200, 500, 1000+ adapters). The other choices (rank, targets, pool op, layer choice) are tunable parameters, not load-bearing architectural commitments.",
        "",
        "## Caveats",
        "",
        "1. All ablations are at **50 adapters**. Several null results (rank doesn't matter, targets don't matter) may not survive at 200-1000+ adapters where the rank/target capacity floor binds.",
        "2. Retrieval differences of ±0.01 are within stochastic-decoding noise (3 seeds × 150 queries ≈ ±2pp std).",
        "3. Substring match on natural-prose answers can hit on chance for short/common answers — same scorer as Phase 47 baseline, so comparisons within this study are valid.",
        "4. The learned MLP router was a 2-layer 1024→512→50 with light regularization. A more carefully tuned discriminator might close the 79.3% → 100% gap with cosine; the headline finding is that cosine is *not* obviously inferior to a simple learned classifier at this scale.",
        "",
        "## Wall-clock totals",
        "",
        "| Stage | Wall |",
        "|---|---:|",
        "| Baseline reproduction | 0.3s |",
        "| Ablation 1 (pooling, 6 variants) | 235s |",
        "| Ablation 2 (Q projection) | 7s |",
        "| Ablation 3 (routing mechanism) | 3s |",
        "| Ablation 4 (LoRA config, 5 variants × 50 adapters) | 1413s (~24 min) |",
        "| Ablation 5 (block-5 unfreeze) | ~6 min |",
        "| **Total** | **~32 min** |",
        "",
    ]


def main():
    out = []
    out.append("# HRS Architecture Ablation Study\n")
    out.append("Phase 47 architectural decisions tested in isolation. "
                "Substrate: 50 Dickens adapters on V22-Dickens base, "
                "rank-128 LoRA on layers 4-5, L0_mean→W→L5_aggregate "
                "cosine routing.\n")
    out.extend(section_baseline())
    out.extend(section_ablation1())
    out.extend(section_ablation2())
    out.extend(section_ablation3())
    out.extend(section_ablation4())
    out.extend(section_ablation5())
    out.extend(section_summary())
    out_path = RES / "RESULT.md"
    out_path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n--- saved to {out_path} ---")


if __name__ == "__main__":
    main()
