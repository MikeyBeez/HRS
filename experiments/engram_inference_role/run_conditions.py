"""Does the engram do inference-time work, or only routing?

Tests on the existing Dickens-50 architecture (per-passage rank-128 LoRA
adapters on layers 4-5 of GPT-2 V22 Dickens-pretrained, plus L0->L5
projection W trained via InfoNCE).

PRE-RUN BACKGROUND CHECK
========================
Adapter training (experiments/per_passage_dickens/train_adapters.py:90-100):
  Each adapter trained on
    (a) raw passage text, next-token LM
    (b) each (training paraphrase + answer) string, next-token LM
  NO ENGRAM is in the training input. This is case (a) of the spec.

Existing eval (experiments/per_passage_dickens/evaluate.py): generates
from `tokenizer.encode(probe)` only. NEVER injects the engram at
inference. The L5 mean-pool key is used solely for routing (cosine
similarity for adapter selection).

Therefore: the published 93% retrieval is the spec's "Condition 2"
(adapter-only-at-inference), not the spec's "Condition 1." The original
architecture has no inference-time engram injection. The spec's
"Condition 1" (engram + adapter at inference) is a NEW variant.

Conditions (relabeled for clarity, but covering all five spec cases):
  C1_engram_plus_correct_adapter   — NEW: oracle correct adapter
                                     + engram-as-inputs_embeds prefix
                                     + prompt
  C2_correct_adapter_only          — ORIGINAL ARCHITECTURE: oracle correct
                                     adapter + prompt only
  C3_engram_only_no_adapter        — engram prefix + base model
                                     (no adapter) + prompt
  C4_engram_plus_wrong_adapter     — engram prefix + RANDOM WRONG adapter
                                     + prompt
  C5_floor_prompt_only             — base model + prompt only

Engram source: library_l5_aggregate[adapter_id] — exactly the same vector
used for routing (mean of L5 mean-pool over training paraphrases).
Injected as a single hidden-state position 0 via raw-block forward,
mirroring phase33_engram_context.py.

Adapter selection: ORACLE for C1/C2 (since the original routing was
already 100% — using oracle removes routing as a confound). C4 picks a
uniformly random wrong adapter per query. C3/C5 use no adapter.

Probes: same held-out set as evaluate.py (50 adapters x 3 paraphrases =
150 probes). Same metric: case-insensitive substring match in
continuation under stochastic decoding (3 seeds).
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
EXP = REPO / "experiments/engram_inference_role"
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

@torch.no_grad()
def forward_with_engram_prefix(model, engram_h, ids_t):
    """[engram_h_as_pos_0] + tok_emb(ids_t) -> logits.

    Mirrors phase33_engram_context.py forward_segments. Uses the raw
    block loop instead of model.forward() so we can splice a hidden
    state in at position 0.
    """
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
        idx = ids_t[:, -(MAX_CTX_POS - 1):]   # +1 for engram pos
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
    """Same as evaluate.py generate."""
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


# ---------------- Match metric (matches evaluate.py) ----------------

def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


CONDITIONS = [
    "C1_engram_plus_correct_adapter",
    "C2_correct_adapter_only",
    "C3_engram_only_no_adapter",
    "C4_engram_plus_wrong_adapter",
    "C5_floor_prompt_only",
]


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ---- Library / keys / W (W loaded for completeness; we use oracle
    # adapter IDs to remove routing as a confound, since original was 100%) ----
    library = json.loads((DICKENS / "data/library.json").read_text())
    keys = json.loads((DICKENS / "results/library_keys.json").read_text())

    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    print(f"Library L5 keys: {library_l5.shape}")

    # ---- Model + LoRA structure (LoRA values loaded per query) ----
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

    # ---- Pre-cache adapter state_dicts on GPU ----
    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location="cpu", weights_only=False)
        adapter_sds[e["id"]] = {k: v.to(device) for k, v in sd.items()}
    print(f"Cached {len(adapter_sds)} adapters on GPU.")

    # ---- Held-out queries ----
    held_out = []
    for entry in library:
        for q in entry["paraphrases_held_out"]:
            held_out.append({
                "adapter_id": entry["id"],
                "fact_type": entry["fact_type"],
                "probe": q,
                "answer": entry["answer"],
            })
    print(f"Held-out queries: {len(held_out)} "
          f"(= {len(library)} adapters × 3 paraphrases)\n")

    # ---- Run conditions ----
    all_results = []
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
                # Pick adapter
                if condition in ("C1_engram_plus_correct_adapter",
                                  "C2_correct_adapter_only"):
                    routed = aid
                elif condition == "C4_engram_plus_wrong_adapter":
                    # uniformly random wrong adapter
                    candidates = [i for i in adapter_sds if i != aid]
                    routed = random.choice(candidates)
                else:  # C3, C5: no adapter
                    routed = -1

                if routed == -1:
                    reset_lora_to_zero(model)
                else:
                    load_lora_state_dict(model, adapter_sds[routed])

                # Build inputs
                ids = tokenizer.encode(q["probe"], add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

                # Engram is the ORACLE engram for this probe (the L5
                # aggregate of the adapter that owns the fact)
                eng = library_l5[aid]

                # Choose generator
                gen_seed = seed * 10000 + qi
                if condition in ("C1_engram_plus_correct_adapter",
                                  "C3_engram_only_no_adapter",
                                  "C4_engram_plus_wrong_adapter"):
                    gen_ids = generate_with_engram(model, eng, ids_t, GEN_TOKENS,
                                                     gen_seed=gen_seed)
                else:  # C2, C5: prompt only
                    gen_ids = generate_no_engram(model, ids_t, GEN_TOKENS,
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
                    "adapter_id_loaded": routed, "fact_type": ft,
                    "probe": q["probe"], "answer": q["answer"],
                    "continuation": cont, "retrieved": hit,
                })

            n = len(held_out)
            ret_acc = n_retrieved / n
            print(f"  {condition:>40s}  seed={seed}  retrieval={ret_acc:.3f}  "
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

    # Aggregate
    print(f"\n{'='*78}\nSUMMARY (mean ± std over 3 seeds)\n{'='*78}")
    summary = {}
    for c in CONDITIONS:
        rs = [x for x in all_results if x["condition"] == c]
        m = float(np.mean([x["retrieval_accuracy"] for x in rs]))
        s = float(np.std([x["retrieval_accuracy"] for x in rs]))
        summary[c] = {"retrieval_mean": m, "retrieval_std": s,
                       "per_seed": [x["retrieval_accuracy"] for x in rs]}
        print(f"  {c:>40s}  retrieval = {m:.3f} ± {s:.3f}  "
              f"(seeds: {[round(x,3) for x in summary[c]['per_seed']]})")

    (EXP / "results/summary.json").write_text(json.dumps({
        "all_results": all_results, "summary": summary,
        "background_check": {
            "adapter_training_case": "a (raw passage + (paraphrase + answer) strings; no engram in training input)",
            "original_inference_engram_injection": False,
            "original_published_retrieval_eq_C2": True,
        },
        "config": {
            "rank": RANK, "gen_tokens": GEN_TOKENS,
            "temperature": TEMPERATURE, "top_k": TOP_K,
            "engram_source": "library_l5_aggregate (mean of L5 mean-pool over training paraphrases)",
            "engram_injection": "single hidden-state position 0 via raw-block forward",
            "adapter_selection": "ORACLE for C1/C2 (original routing was 100%); random wrong for C4",
        },
        "wall_total_s": time.time() - t0,
    }, indent=2))
    print(f"\nSaved {EXP/'results/summary.json'}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
