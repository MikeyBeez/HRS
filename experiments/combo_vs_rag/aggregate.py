"""Combine combo-adapter results (existing) + new RAG results into a
single side-by-side RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path("/mnt/data/Code/HRS")
COMBO = REPO / "experiments/combo_adapter"
CVR = REPO / "experiments/combo_vs_rag"


def main():
    combo_tests = json.loads((COMBO / "results/tests.json").read_text())
    rag = json.loads((CVR / "results/rag.json").read_text())
    divs = json.loads((CVR / "results/divergences.json").read_text())

    # Maps from combination name to combo adapter results
    combo_t1 = {r["name"]: r for r in combo_tests["test1"]}
    combo_t2 = {r["name"]: r for r in combo_tests["test2"]}
    rag_t1 = {r["name"]: r for r in rag["test1"]}
    rag_t2 = {r["name"]: r for r in rag["test2"]}

    out = []
    out.append("# Combination Adapter vs Oracle RAG\n")
    out.append("**Question:** at this scale, does training content into a "
                "combination adapter (parameter-space absorption) provide "
                "any advantage over having the same content in context "
                "(oracle RAG)? Same passages, same queries, same scoring, "
                "same base model.")
    out.append("")
    out.append("**Verdict: Combination adapter wins decisively on this "
                "substrate. The architecture's substitution claim is "
                "supported here.** Combo retrieval is 0.65-0.74 across "
                "K=2/3/4; RAG retrieval is 0.03-0.06 across the same K. "
                "Combo wins by 60-71 percentage points on per-constituent "
                "retrieval and by 22-38pp on cross-passage queries. "
                "**Important caveat:** the V22-Dickens base used here is "
                "a small (~510M-param) Dickens-pretrained model, not "
                "instruction-tuned. Its in-context-learning ability for "
                "Q/A-style queries is essentially zero — it just continues "
                "with novel Dickensian prose. A larger instruction-tuned "
                "base would likely close most or all of this gap.")
    out.append("")

    out.append("## Setup\n")
    out.append("- Same 10 combinations from the combo adapter experiment "
                "(reused without retraining): 4 K=2, 4 K=3, 2 K=4.")
    out.append("- Same query sets: 3 held-out paraphrases per constituent "
                "for Test 1; 3 hand-crafted cross-passage queries per "
                "combination for Test 2.")
    out.append("- Procedure A (combo adapter): K=1 inference with combo "
                "adapter loaded.")
    out.append("- Procedure B (oracle RAG): build prompt as "
                "`{passage_1}\\n\\n{passage_2}\\n\\n...\\n\\n{probe}`, run "
                "inference with NO adapter (LoRA zeroed), score same way.")
    out.append("- All RAG prompts fit comfortably under the 512-token "
                "context limit (max prompt = 292 tokens at K=4 with full "
                "passages — no truncation needed).")
    out.append("")

    out.append("## Test 1: per-constituent retrieval\n")
    out.append("| combination | K | combo | RAG | gap (combo-RAG) |")
    out.append("|---|---:|---:|---:|---:|")
    avgs = {2: [], 3: [], 4: []}
    for name in combo_t1:
        c = combo_t1[name]
        K = c["k"]
        c_avg = np.mean([p["combo_rate"] for p in c["per_constituent"]])
        r_avg = rag_t1[name]["avg_rag_rate"]
        avgs[K].append((c_avg, r_avg))
        out.append(f"| {name} | {K} | {c_avg:.3f} | {r_avg:.3f} | "
                    f"{c_avg-r_avg:+.3f} |")
    out.append("")
    out.append("### Aggregated by K\n")
    out.append("| K | combo (avg) | RAG (avg) | gap |")
    out.append("|---:|---:|---:|---:|")
    for k in (2, 3, 4):
        rows = avgs[k]
        c = np.mean([r[0] for r in rows]); r = np.mean([r[1] for r in rows])
        out.append(f"| {k} | {c:.3f} | {r:.3f} | {c-r:+.3f} |")
    out.append("")

    out.append("## Test 2: cross-passage queries (fragment coverage)\n")
    out.append("| combination | K | combo frac | RAG frac | gap | "
                "combo full | RAG full |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    avgs2 = {2: [], 3: [], 4: []}
    for name in combo_t2:
        c = combo_t2[name]
        K = c["k"]
        c_frac = np.mean([x["frac_hits_mean"] for x in c["combo_results"]])
        c_full = np.mean([x["full_hit_rate"] for x in c["combo_results"]])
        r_frac = rag_t2[name]["avg_frac_hits"]
        r_full = rag_t2[name]["avg_full_hit"]
        avgs2[K].append((c_frac, r_frac, c_full, r_full))
        out.append(f"| {name} | {K} | {c_frac:.3f} | {r_frac:.3f} | "
                    f"{c_frac-r_frac:+.3f} | {c_full:.3f} | {r_full:.3f} |")
    out.append("")
    out.append("### Aggregated by K\n")
    out.append("| K | combo frac | RAG frac | gap | combo full | RAG full |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for k in (2, 3, 4):
        rows = avgs2[k]
        cf = np.mean([r[0] for r in rows]); rf = np.mean([r[1] for r in rows])
        cu = np.mean([r[2] for r in rows]); ru = np.mean([r[3] for r in rows])
        out.append(f"| {k} | {cf:.3f} | {rf:.3f} | {cf-rf:+.3f} | "
                    f"{cu:.3f} | {ru:.3f} |")
    out.append("")

    out.append("## Failure analysis (Test 3)\n")
    out.append("Inspecting actual generations on representative queries.")
    out.append("")
    for ex in divs:
        out.append(f"### {ex['case']}\n")
        out.append(f"**Expected answer:** `{ex['expected']}`\n")
        out.append(f"**Probe:** `{ex['probe']}`\n")
        prompt_key = "rag_prompt_first_180" if "rag_prompt_first_180" in ex else "rag_prompt_first_120"
        out.append(f"**RAG prompt prefix:** `{ex[prompt_key]}`\n")
        out.append(f"**Combo generation:** `{ex['combo_gen']}`\n")
        out.append(f"**RAG generation:** `{ex['rag_gen']}`\n")
        out.append("")

    out.append("## What the divergences show\n")
    out.append("**Combo's success mode:** the adapter regurgitates the "
                "training passages near-verbatim when probed. E.g., asked "
                "for Joe's profession, it produces \"Joe Gargery, who "
                "married the blacksmith. Joe's forge adjoined our house...\" "
                "— the literal training text. The substring matcher counts "
                "this as a hit because \"blacksmith\" appears.")
    out.append("")
    out.append("**RAG's failure mode:** the V22-Dickens base, even with "
                "the relevant passage right there in context, doesn't "
                "extract the answer. It produces novel Dickens-style "
                "continuation that doesn't reference the answer. E.g., "
                "asked for Joe's profession with the passage in context, "
                "it produces \"Joe Gargery, with a kind of assurance that "
                "he would soon ask you to live...\" — fluent Dickens style "
                "but no extraction.")
    out.append("")
    out.append("**This is consistent with the V22-Dickens base lacking "
                "in-context-learning skill.** It's a 510M-param "
                "language-model-only base, no instruction tuning, no "
                "Q/A-format training. Asking it to extract from "
                "context-provided passages is asking for a capability "
                "it doesn't have.")
    out.append("")

    out.append("## What this experiment establishes — and what it doesn't\n")
    out.append("**Establishes (at this scale):**")
    out.append("- Training content into a parameter-space adapter "
                "produces a model that, on probes designed to elicit "
                "that content, retrieves it. The combo adapter "
                "internalizes facts and reproduces them on demand.")
    out.append("- The same content placed in context for an "
                "instruction-untrained base does NOT yield extraction. "
                "The base just continues with style-matched novel text.")
    out.append("- The architecture's substitution claim — \"absorbing "
                "content into a model is meaningfully different from "
                "having it in context\" — is **empirically supported** "
                "for this base.")
    out.append("")
    out.append("**Does NOT establish:**")
    out.append("- Whether the same advantage holds with a larger "
                "instruction-tuned base. A modern 7B+ Q/A-capable base "
                "would likely extract answers from in-context passages "
                "at much higher rate. The 60-71pp gap here might shrink "
                "to 10pp, 0pp, or invert.")
    out.append("- Whether combo's \"retrieval\" is doing anything beyond "
                "memorizing training passages and regurgitating them. "
                "The example divergences suggest much of the combo "
                "advantage is verbatim reproduction. Combo has not "
                "demonstrably learned to *integrate* facts (cross-passage "
                "queries are at 30-50% fragment coverage, far from the "
                "100% an integrating model would hit).")
    out.append("- How combo compares to RAG on novel queries that "
                "weren't in training. The held-out paraphrases share "
                "the same `{subject}` string structure with training "
                "paraphrases — they're surface variants, not "
                "fundamentally new queries.")
    out.append("")

    out.append("## Implications for the architecture's claims\n")
    out.append("The combination adapter approach has a real, measurable "
                "advantage over oracle RAG when the base model can't do "
                "in-context Q/A. This is the regime the HRS architecture "
                "is currently operating in (V22-Dickens, 510M params). "
                "In this regime, parameter-space absorption is the only "
                "way to make the model produce the answer.")
    out.append("")
    out.append("Whether this advantage transfers to bases that *can* do "
                "in-context Q/A is the next experiment that matters most "
                "for the recruitment ask. The ideal substrate test:")
    out.append("- Same combination adapters approach on a larger base "
                "(e.g., the prior PEER 2B-target, or a Llama-3-8B-Instruct).")
    out.append("- Same RAG comparison on the same larger base.")
    out.append("- Measure whether combo retains its 60-71pp lead, falls "
                "to a 10pp lead, ties, or loses.")
    out.append("")
    out.append("If combo retains a lead at scale, the architecture's "
                "value proposition is real and durable: parameter-space "
                "adapters do something context can't replicate.")
    out.append("")
    out.append("If combo ties or loses at scale, the architecture's "
                "value proposition shifts to cost structure: combo "
                "adapters are smaller-context, possibly faster at "
                "inference, but provide no fundamental capability "
                "advantage. This is still useful but it's a "
                "different story than \"the architecture provides "
                "capabilities other approaches don't.\"")
    out.append("")

    out.append("## Wall-clock totals\n")
    rag_wall = rag["wall_total_s"]
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| RAG eval (Tests 1+2) | {rag_wall:.0f}s |")
    out.append(f"| Divergence inspection | <30s |")
    out.append(f"| Combo eval (already done in combo_adapter experiment) | "
                f"reused |")
    out.append(f"| **Total new compute** | **~"
                f"{(rag_wall+30)/60:.0f} min** |")
    out.append("")

    out.append("## Summary\n")
    out.append("On the V22-Dickens base, combination adapters dominate "
                "oracle RAG: 0.65-0.74 retrieval vs 0.03-0.06 across "
                "K=2/3/4. The architecture's claim that "
                "parameter-space absorption is meaningfully different "
                "from in-context access is **empirically supported at "
                "this scale**.")
    out.append("")
    out.append("The headline caveat: this comparison is between a "
                "small base that can't do in-context Q/A and an adapter "
                "trained to regurgitate Dickens passages. The combo "
                "adapter's main mechanism is verbatim reproduction of "
                "training data, which the substring scorer counts as "
                "retrieval. RAG's failure is the base's lack of "
                "extractive Q/A capability, not a problem with content "
                "being in context per se.")
    out.append("")
    out.append("**The experiment that would change my read:** run the "
                "same head-to-head on a base that *can* do in-context "
                "Q/A. If combo still wins by a large margin there, the "
                "architecture's case is robust. If combo's lead shrinks "
                "or inverts, the architecture's value proposition needs "
                "honest reframing toward cost structure rather than "
                "capability advantage.")

    out_path = CVR / "results/RESULT.md"
    out_path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n--- saved to {out_path} ---")


if __name__ == "__main__":
    main()
