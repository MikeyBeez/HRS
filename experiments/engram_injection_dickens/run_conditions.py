"""Does adding engram injection at inference beat the Dickens-50 93% baseline?

Three conditions on the same held-out probes (50 adapters x 3 paraphrases
= 150 probes), 3 stochastic-decoding seeds each.

  C1_baseline_no_engram:      correct adapter + prompt only  (= original
                               published architecture, target ~0.93)
  C2_correct_engram_injected: correct adapter + correct adapter's engram
                               injected as inputs_embeds prefix + prompt
  C3_wrong_engram_injected:   correct adapter + a DIFFERENT adapter's
                               engram injected + prompt  (control: tests
                               whether the engram's specific signal matters)

Adapter selection: ORACLE for all three (the Dickens-50 routing was
already 100%, so using the oracle adapter removes routing as a confound
and isolates the inference-time question).

Engram injection method: single L5 mean-pool hidden state prepended at
position 0, processed via raw-block forward loop. Mirrors
experiments/identity_ae/phase33_engram_context.py exactly. Engram source
is `library_l5_aggregate[adapter_id]` from
experiments/per_passage_dickens/results/library_keys.json.

For C3, each probe's "wrong engram" is a uniformly random adapter
distinct from the probe's true adapter (same RNG seed across seeds for
reproducibility within a seed).

Pre-run background:
- Adapters were trained on raw passage text + (paraphrase + answer)
  strings. NO ENGRAM in training input. So C2/C3 test how the adapter
  responds to a prefix it never saw during training.
- Original Dickens-50 evaluate.py never injected engrams at inference.
  C1 here reproduces that exact path.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/engram_injection_dickens"
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
D = 1024
MAX_CTX_POS = 512


# ---------------- Forward path with engram-as-position-0 ----------------
# Mirrors phase33_engram_context.py: raw-block loop with one hidden-state
# position spliced before the token embeddings.

@torch.no_grad()
def forward_with_engram_prefix(model, engram_h, ids_t):
    eng_part = engram_h.view(1, 1, -1).to(ids_t.device).to(model.tok_emb.weight.dtype)
    tok_part = model.drop(model.tok_emb(ids_t))
    h = torch.cat([eng_part, tok_part], dim=1)
    if h.shape[1] > MAX_CTX_POS:
        h = h[:, -MAX_CTX_POS:]
    for block in model.blocks:
        eb = (model.engram_buffer if getattr(model, "_engram_buffer_initialized", False)
              else None)
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
    h = model.ln_f(h)
    return model.lm_head(h)


@torch.no_grad()
def generate_with_engram(model, engram_h, ids_t, n_tokens, gen_seed,
                          temperature=TEMPERATURE, top_k=TOP_K):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -(MAX_CTX_POS - 1):]
        logits_full = forward_with_engram_prefix(model, engram_h, idx)
        logits = logits_full[:, -1, :].float() / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


@torch.no_grad()
def generate_no_engram(model, ids_t, n_tokens, gen_seed,
                        temperature=TEMPERATURE, top_k=TOP_K):
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
    "C1_baseline_no_engram",
    "C2_correct_engram_injected",
    "C3_wrong_engram_injected",
]


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())

    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    print(f"Library L5 keys: {library_l5.shape}")

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
    # Per-probe outcomes keyed by (condition, qi) for delta analysis
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
                # Always load the CORRECT adapter (oracle, since original
                # routing was 100%)
                load_lora_state_dict(model, adapter_sds[aid])

                ids = tokenizer.encode(q["probe"], add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

                gen_seed = seed * 10000 + qi
                wrong_aid = None
                if condition == "C1_baseline_no_engram":
                    gen_ids = generate_no_engram(model, ids_t, GEN_TOKENS,
                                                   gen_seed=gen_seed)
                elif condition == "C2_correct_engram_injected":
                    eng = library_l5[aid]
                    gen_ids = generate_with_engram(model, eng, ids_t, GEN_TOKENS,
                                                     gen_seed=gen_seed)
                elif condition == "C3_wrong_engram_injected":
                    candidates = [i for i in adapter_sds if i != aid]
                    wrong_aid = random.choice(candidates)
                    eng = library_l5[wrong_aid]
                    gen_ids = generate_with_engram(model, eng, ids_t, GEN_TOKENS,
                                                     gen_seed=gen_seed)

                full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                cont = full[len(q["probe"]):]
                hit = check_match(q["answer"], cont)
                if hit: n_retrieved += 1
                ft = q["fact_type"]
                per_type.setdefault(ft, {"n": 0, "retrieved": 0})
                per_type[ft]["n"] += 1
                per_type[ft]["retrieved"] += int(hit)

                details.append({
                    "qi": qi, "adapter_id_true": aid,
                    "engram_source_aid": (aid if condition == "C2_correct_engram_injected"
                                           else (wrong_aid if condition == "C3_wrong_engram_injected"
                                                 else None)),
                    "fact_type": ft,
                    "probe": q["probe"], "answer": q["answer"],
                    "continuation": cont, "retrieved": hit,
                })
                # Track per-probe outcomes for delta analysis (use seed-0 only
                # for the first reduction; we'll compute multi-seed deltas later)
                per_probe[condition].setdefault(qi, []).append(hit)

            n = len(held_out)
            ret_acc = n_retrieved / n
            print(f"  {condition:>32s}  seed={seed}  retrieval={ret_acc:.3f}  "
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
    print(f"\n{'='*80}\nSUMMARY (mean ± std over 3 seeds)\n{'='*80}")
    summary = {}
    for c in CONDITIONS:
        rs = [x for x in all_results if x["condition"] == c]
        m = float(np.mean([x["retrieval_accuracy"] for x in rs]))
        s = float(np.std([x["retrieval_accuracy"] for x in rs]))
        summary[c] = {"retrieval_mean": m, "retrieval_std": s,
                       "per_seed": [x["retrieval_accuracy"] for x in rs]}
        print(f"  {c:>32s}  retrieval = {m:.3f} ± {s:.3f}  "
              f"(seeds: {[round(x,3) for x in summary[c]['per_seed']]})")

    # ---- Per-probe deltas ----
    # For each probe, sum its hits over 3 seeds per condition (in [0,3]),
    # then categorize C2 vs C1 and C3 vs C1.
    def categorize_deltas(cond_a, cond_b):
        """Per-probe delta (cond_a - cond_b) summed over seeds."""
        gain = matched = lost = 0
        per_probe_summary = []
        for qi in sorted(per_probe[cond_a].keys()):
            a = sum(per_probe[cond_a][qi])  # 0..3
            b = sum(per_probe[cond_b][qi])  # 0..3
            d = a - b
            if d > 0:    gain += 1
            elif d < 0:  lost += 1
            else:        matched += 1
            per_probe_summary.append({"qi": qi, f"{cond_a}_hits": a,
                                       f"{cond_b}_hits": b, "delta": d})
        return {"n_gain": gain, "n_match": matched, "n_lost": lost,
                "per_probe": per_probe_summary}

    deltas = {
        "C2_vs_C1": categorize_deltas("C2_correct_engram_injected",
                                        "C1_baseline_no_engram"),
        "C3_vs_C1": categorize_deltas("C3_wrong_engram_injected",
                                        "C1_baseline_no_engram"),
    }
    print(f"\nPer-probe deltas (out of {len(per_probe[CONDITIONS[0]])} probes, "
          f"summing 3 seeds each):")
    for k, v in deltas.items():
        print(f"  {k:>10s}:  gain={v['n_gain']:3d}  match={v['n_match']:3d}  "
              f"lost={v['n_lost']:3d}")

    (EXP / "results/summary.json").write_text(json.dumps({
        "all_results": all_results, "summary": summary, "deltas": deltas,
        "background": {
            "adapter_training_case": "a (raw passage + (paraphrase + answer); no engram in training input)",
            "original_inference_engram_injection": False,
            "original_published_retrieval": "≈0.93 (eval_A_full mean across 3 seeds)",
            "this_experiment": "C1 reproduces original; C2 adds engram injection; C3 controls with wrong-adapter engram",
        },
        "config": {
            "rank": RANK, "gen_tokens": GEN_TOKENS,
            "temperature": TEMPERATURE, "top_k": TOP_K,
            "engram_source": "library_l5_aggregate (mean of L5 mean-pool over training paraphrases)",
            "engram_injection_method": "single hidden-state position 0 via raw-block forward (mirrors phase33_engram_context.py)",
            "adapter_selection": "ORACLE for all three conditions; original routing was 100% so this removes routing as a confound",
        },
        "wall_total_s": time.time() - t0,
    }, indent=2))
    print(f"\nSaved {EXP/'results/summary.json'}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
