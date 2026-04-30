"""Pull a few example queries where combo and RAG diverge, with actual
generations.

Reuses the eval infrastructure but only on a small set of queries for
inspection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import load_lora_state_dict

from experiments.combo_adapter.combos import COMBINATIONS
from experiments.combo_adapter.evaluate import (
    encode, generate, build_model, GEN_TOKENS, RANK,
)

PPD = REPO / "experiments/per_passage_dickens"
COMBO = REPO / "experiments/combo_adapter"


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((PPD / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    # Build one model — we'll toggle adapter on/off
    model = build_model(device, rank=RANK)

    # Pick a few illustrative queries
    examples = []

    # Case 1: K=2 single-content where RAG = 0% but combo = high
    # P1_Pip_Joe, retrieve Joe's profession
    e2 = by_id[2]
    probe = e2["paraphrases_held_out"][0]  # e.g. "Recall: Joe Gargery's profession = "
    rag_prompt = "\n\n".join([by_id[0]["passage"].strip(),
                                by_id[2]["passage"].strip(), probe])

    # --- combo gen ---
    combo_sd = torch.load(COMBO / "adapters/P1_Pip_Joe.pt",
                           map_location=device, weights_only=False)
    reset_lora_to_zero(model)
    load_lora_state_dict(model, combo_sd)
    ids_t = encode(tokenizer, probe, device)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    combo_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                    skip_special_tokens=True)

    # --- RAG gen ---
    reset_lora_to_zero(model)
    ids_t = encode(tokenizer, rag_prompt, device, ctx=512)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    rag_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                  skip_special_tokens=True)

    examples.append({
        "case": "Test 1 K=2: Joe's profession (P1_Pip_Joe)",
        "expected": e2["answer"],
        "probe": probe,
        "rag_prompt_first_120": rag_prompt[:120].replace("\n", " ↵ ") + "...",
        "combo_gen": combo_cont,
        "rag_gen": rag_cont,
    })

    # Case 2: Test 2 cross-passage K=3 (T2 Joe+Estella+Drummle)
    combo_t2 = next(c for c in COMBINATIONS if c["name"] == "T2_Joe_Estella_Drummle")
    q = combo_t2["cross_queries"][0]  # 3-fact chained
    probe = q["probe"]
    constituents = [by_id[i] for i in combo_t2["constituents"]]
    rag_prompt = "\n\n".join([e["passage"].strip() for e in constituents] + [probe])

    combo_sd = torch.load(COMBO / "adapters/T2_Joe_Estella_Drummle.pt",
                           map_location=device, weights_only=False)
    reset_lora_to_zero(model)
    load_lora_state_dict(model, combo_sd)
    ids_t = encode(tokenizer, probe, device)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    combo_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                    skip_special_tokens=True)

    reset_lora_to_zero(model)
    ids_t = encode(tokenizer, rag_prompt, device, ctx=512)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    rag_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                  skip_special_tokens=True)

    examples.append({
        "case": "Test 2 K=3: 3-fact cross-passage (T2_Joe_Estella_Drummle)",
        "expected": q["fragments"],
        "probe": probe,
        "rag_prompt_first_180": rag_prompt[:180].replace("\n", " ↵ ") + "...",
        "combo_gen": combo_cont,
        "rag_gen": rag_cont,
    })

    # Case 3: K=4 single-content (Q1 retrieve Estella)
    e16 = by_id[16]
    probe = e16["paraphrases_held_out"][0]
    combo_q1 = next(c for c in COMBINATIONS
                     if c["name"] == "Q1_Pip_Estella_Magwitch_Provis")
    constituents = [by_id[i] for i in combo_q1["constituents"]]
    rag_prompt = "\n\n".join([e["passage"].strip() for e in constituents] + [probe])

    combo_sd = torch.load(COMBO / "adapters/Q1_Pip_Estella_Magwitch_Provis.pt",
                           map_location=device, weights_only=False)
    reset_lora_to_zero(model)
    load_lora_state_dict(model, combo_sd)
    ids_t = encode(tokenizer, probe, device)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    combo_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                    skip_special_tokens=True)

    reset_lora_to_zero(model)
    ids_t = encode(tokenizer, rag_prompt, device, ctx=512)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    rag_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                  skip_special_tokens=True)

    examples.append({
        "case": "Test 1 K=4: Estella retrieval (Q1)",
        "expected": e16["answer"],
        "probe": probe,
        "rag_prompt_first_180": rag_prompt[:180].replace("\n", " ↵ ") + "...",
        "combo_gen": combo_cont,
        "rag_gen": rag_cont,
    })

    # Case 4: where RAG happens to hit (P3_Magwitch_Provis was best at 0.167)
    combo_p3 = next(c for c in COMBINATIONS if c["name"] == "P3_Magwitch_Provis")
    e22 = by_id[22]
    probe = e22["paraphrases_held_out"][0]
    constituents = [by_id[i] for i in combo_p3["constituents"]]
    rag_prompt = "\n\n".join([e["passage"].strip() for e in constituents] + [probe])

    combo_sd = torch.load(COMBO / "adapters/P3_Magwitch_Provis.pt",
                           map_location=device, weights_only=False)
    reset_lora_to_zero(model)
    load_lora_state_dict(model, combo_sd)
    ids_t = encode(tokenizer, probe, device)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    combo_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                    skip_special_tokens=True)

    reset_lora_to_zero(model)
    ids_t = encode(tokenizer, rag_prompt, device, ctx=512)
    gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
    rag_cont = tokenizer.decode(gen[0, ids_t.shape[1]:],
                                  skip_special_tokens=True)

    examples.append({
        "case": "Test 1 K=2: Provis retrieval (P3, RAG's best cell)",
        "expected": e22["answer"],
        "probe": probe,
        "rag_prompt_first_180": rag_prompt[:180].replace("\n", " ↵ ") + "...",
        "combo_gen": combo_cont,
        "rag_gen": rag_cont,
    })

    out_path = REPO / "experiments/combo_vs_rag/results/divergences.json"
    out_path.write_text(json.dumps(examples, indent=2))
    print(json.dumps(examples, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
