"""Engram-Routed RAG on Dickens-50.

Tests whether prepending the routed passage's training text to the
inference prompt closes the 7-percentage-point gap (93% retrieval) of
the published per_passage_dickens architecture.

Five conditions, 3 stochastic-decoding seeds each, 150 held-out probes:

  C1_baseline_adapter_only          correct adapter + prompt
                                    [= published per_passage_dickens]
  C2_correct_adapter_correct_rag    correct adapter + correct passage
                                    + prompt [the new variant]
  C3_no_adapter_correct_rag         base model + correct passage + prompt
                                    [tests RAG-alone contribution]
  C4_correct_adapter_wrong_rag      correct adapter + a DIFFERENT passage
                                    + prompt [tests whether model uses
                                    retrieved text or ignores it]
  C5_floor                          base model + prompt only

Adapter selection: ORACLE for all conditions that use an adapter (C1,
C2, C4). Original Dickens-50 routing was 100%, so this removes routing
as a confound and isolates the inference-time question. C4 uses
randomly chosen wrong adapter id for the RAG text only (the LOADED
adapter remains correct).

RAG injection method: passage text prepended as plain tokens with a
clear separator, then the probe. Format:

    {passage}
    {probe}

Passages are short (max 109 tokens; mean 64) — no truncation needed.
Plus probe (~10-15 tokens) + 30 generation tokens, well under MAX_CTX_POS=512.

Reuses the trained adapters and probes from experiments/per_passage_dickens.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/engram_routed_rag"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)


RANK = 128
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
MAX_CTX_POS = 512
RAG_TEMPLATE = "{passage}\n{probe}"


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed,
              temperature=TEMPERATURE, top_k=TOP_K):
    """Same generate as evaluate.py — token-only autoregressive sampling."""
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -MAX_CTX_POS:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :].float() / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


CONDITIONS = [
    "C1_baseline_adapter_only",
    "C2_correct_adapter_correct_rag",
    "C3_no_adapter_correct_rag",
    "C4_correct_adapter_wrong_rag",
    "C5_floor",
]


def build_input_text(condition, probe, correct_passage, wrong_passage):
    """Return the full input text fed to the tokenizer."""
    if condition == "C1_baseline_adapter_only":
        return probe
    if condition == "C2_correct_adapter_correct_rag":
        return RAG_TEMPLATE.format(passage=correct_passage, probe=probe)
    if condition == "C3_no_adapter_correct_rag":
        return RAG_TEMPLATE.format(passage=correct_passage, probe=probe)
    if condition == "C4_correct_adapter_wrong_rag":
        return RAG_TEMPLATE.format(passage=wrong_passage, probe=probe)
    if condition == "C5_floor":
        return probe
    raise ValueError(condition)


def use_adapter(condition):
    return condition in ("C1_baseline_adapter_only",
                          "C2_correct_adapter_correct_rag",
                          "C4_correct_adapter_wrong_rag")


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())
    print(f"Library: {len(library)} entries.")

    # Adapter index -> passage text (the training text used during adapter training)
    aid_to_passage = {e["id"]: e["passage"] for e in library}

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()
    print(f"Model loaded. tok_emb dtype: {model.tok_emb.weight.dtype}")

    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location="cpu", weights_only=False)
        adapter_sds[e["id"]] = {k: v.to(device) for k, v in sd.items()}
    print(f"Cached {len(adapter_sds)} adapters on GPU.")

    held_out = []
    for entry in library:
        for q in entry["paraphrases_held_out"]:
            held_out.append({
                "adapter_id": entry["id"],
                "fact_type": entry["fact_type"],
                "probe": q,
                "answer": entry["answer"],
            })
    print(f"Held-out queries: {len(held_out)}\n")

    all_results = []
    per_probe = {c: {} for c in CONDITIONS}

    t0 = time.time()
    for condition in CONDITIONS:
        for seed in [0, 1, 2]:
            t_seed = time.time()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

            n_retrieved = 0
            per_type = {}
            details = []

            for qi, q in enumerate(held_out):
                aid = q["adapter_id"]
                # Adapter (always correct when used)
                if use_adapter(condition):
                    load_lora_state_dict(model, adapter_sds[aid])
                else:
                    reset_lora_to_zero(model)

                # Passage selection for RAG
                correct_passage = aid_to_passage[aid]
                wrong_aid = None
                wrong_passage = None
                if condition == "C4_correct_adapter_wrong_rag":
                    candidates = [i for i in aid_to_passage if i != aid]
                    wrong_aid = random.choice(candidates)
                    wrong_passage = aid_to_passage[wrong_aid]

                input_text = build_input_text(condition, q["probe"],
                                                correct_passage, wrong_passage)
                ids = tokenizer.encode(input_text, add_special_tokens=False)
                # Safety: cap input at MAX_CTX_POS - GEN_TOKENS
                max_input = MAX_CTX_POS - GEN_TOKENS
                if len(ids) > max_input:
                    ids = ids[:max_input]
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

                gen_seed = seed * 10000 + qi
                gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
                full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                cont = full[len(input_text):]
                hit = check_match(q["answer"], cont)
                if hit: n_retrieved += 1
                ft = q["fact_type"]
                per_type.setdefault(ft, {"n": 0, "retrieved": 0})
                per_type[ft]["n"] += 1
                per_type[ft]["retrieved"] += int(hit)

                details.append({
                    "qi": qi, "adapter_id_true": aid,
                    "adapter_loaded": (aid if use_adapter(condition) else None),
                    "rag_passage_source_aid": (aid if condition in (
                        "C2_correct_adapter_correct_rag",
                        "C3_no_adapter_correct_rag") else
                        (wrong_aid if condition == "C4_correct_adapter_wrong_rag" else None)),
                    "fact_type": ft,
                    "probe": q["probe"], "answer": q["answer"],
                    "input_n_tokens": len(ids),
                    "continuation": cont, "retrieved": hit,
                })
                per_probe[condition].setdefault(qi, []).append(hit)

            n = len(held_out)
            ret_acc = n_retrieved / n
            print(f"  {condition:>34s}  seed={seed}  retrieval={ret_acc:.3f}  "
                  f"wall={time.time()-t_seed:.0f}s")
            rec = {
                "condition": condition, "seed": seed,
                "retrieval_accuracy": ret_acc,
                "per_fact_type": {ft: {**v, "retrieval": v["retrieved"]/v["n"]}
                                    for ft, v in per_type.items()},
            }
            all_results.append(rec)
            (EXP / f"results/eval_{condition}_seed{seed}.json").write_text(
                json.dumps({"summary": rec, "details": details}, indent=2))

    # ---- Aggregate ----
    print(f"\n{'='*82}\nSUMMARY (mean ± std over 3 seeds)\n{'='*82}")
    summary = {}
    for c in CONDITIONS:
        rs = [x for x in all_results if x["condition"] == c]
        m = float(np.mean([x["retrieval_accuracy"] for x in rs]))
        s = float(np.std([x["retrieval_accuracy"] for x in rs]))
        summary[c] = {"retrieval_mean": m, "retrieval_std": s,
                       "per_seed": [x["retrieval_accuracy"] for x in rs]}
        print(f"  {c:>34s}  retrieval = {m:.3f} ± {s:.3f}  "
              f"(seeds: {[round(x,3) for x in summary[c]['per_seed']]})")

    # ---- Per-probe deltas ----
    def categorize_deltas(cond_a, cond_b):
        gain = matched = lost = 0
        per_probe_summary = []
        for qi in sorted(per_probe[cond_a].keys()):
            a = sum(per_probe[cond_a][qi])
            b = sum(per_probe[cond_b][qi])
            d = a - b
            if d > 0:    gain += 1
            elif d < 0:  lost += 1
            else:        matched += 1
            per_probe_summary.append({"qi": qi, f"{cond_a}_hits": a,
                                       f"{cond_b}_hits": b, "delta": d})
        return {"n_gain": gain, "n_match": matched, "n_lost": lost,
                "per_probe": per_probe_summary}

    deltas = {
        "C2_vs_C1": categorize_deltas("C2_correct_adapter_correct_rag",
                                        "C1_baseline_adapter_only"),
        "C3_vs_C1": categorize_deltas("C3_no_adapter_correct_rag",
                                        "C1_baseline_adapter_only"),
        "C4_vs_C1": categorize_deltas("C4_correct_adapter_wrong_rag",
                                        "C1_baseline_adapter_only"),
        "C2_vs_C3": categorize_deltas("C2_correct_adapter_correct_rag",
                                        "C3_no_adapter_correct_rag"),
    }
    print(f"\nPer-probe deltas (out of {len(per_probe[CONDITIONS[0]])} probes, "
          f"summing 3 seeds each):")
    for k, v in deltas.items():
        print(f"  {k:>10s}:  gain={v['n_gain']:3d}  match={v['n_match']:3d}  "
              f"lost={v['n_lost']:3d}")

    # Concentration analysis: of probes where C1 failed (sum < 3), how
    # many did C2 fix (sum > C1's sum)?
    c1_failures = [qi for qi in per_probe["C1_baseline_adapter_only"]
                   if sum(per_probe["C1_baseline_adapter_only"][qi]) < 3]
    c2_fixes = sum(1 for qi in c1_failures
                   if sum(per_probe["C2_correct_adapter_correct_rag"][qi])
                       > sum(per_probe["C1_baseline_adapter_only"][qi]))
    c3_fixes = sum(1 for qi in c1_failures
                   if sum(per_probe["C3_no_adapter_correct_rag"][qi])
                       > sum(per_probe["C1_baseline_adapter_only"][qi]))
    print(f"\nC1 had partial/full failure on {len(c1_failures)} probes "
          f"(out of {len(per_probe[CONDITIONS[0]])}).")
    print(f"  C2 improved over C1 on {c2_fixes} of those.")
    print(f"  C3 improved over C1 on {c3_fixes} of those.")

    (EXP / "results/summary.json").write_text(json.dumps({
        "all_results": all_results, "summary": summary, "deltas": deltas,
        "concentration_analysis": {
            "c1_failures_n": len(c1_failures),
            "c2_fixed_of_c1_failures": c2_fixes,
            "c3_fixed_of_c1_failures": c3_fixes,
        },
        "config": {
            "rank": RANK, "gen_tokens": GEN_TOKENS,
            "temperature": TEMPERATURE, "top_k": TOP_K,
            "rag_template": RAG_TEMPLATE,
            "max_input_tokens": MAX_CTX_POS - GEN_TOKENS,
            "adapter_selection": "ORACLE for C1/C2/C4 (original routing was 100%); C3/C5 use base model only",
            "wrong_passage_selection_C4": "uniform random among adapters != true adapter",
        },
        "wall_total_s": time.time() - t0,
    }, indent=2))
    print(f"\nSaved {EXP/'results/summary.json'}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
