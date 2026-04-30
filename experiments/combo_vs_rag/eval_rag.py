"""Procedure B: oracle RAG with the same source content as combo adapter.

For each query:
  prompt = passage_1 + "\n\n" + passage_2 + ... + "\n\n" + probe
  Run inference with NO adapter (LoRA at zero).
  Score same way as combo eval (substring match for Test 1, fragment
  coverage + full-hit for Test 2).

Reuses the 10 combinations and the same query sets from
experiments/combo_adapter.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import apply_lora

from experiments.combo_adapter.combos import COMBINATIONS
from experiments.combo_adapter.evaluate import (
    encode, generate, check_match, build_model,
    GEN_TOKENS, SEEDS, RANK,
)

PPD = REPO / "experiments/per_passage_dickens"


def build_rag_prompt(constituents_data, probe):
    """Concatenate passages then probe. Passages separated by double newline."""
    parts = []
    for entry in constituents_data:
        parts.append(entry["passage"].strip())
    parts.append(probe)
    return "\n\n".join(parts)


def measure(model, prompt, expected_fragments, tokenizer, device,
            n_seeds=3, gen_tokens=GEN_TOKENS):
    """Generate from prompt N seeds. Return frac_hits, full_hit_rate, gens."""
    fracs = []; full_hits = 0
    gens = []
    for seed in range(n_seeds):
        ids_t = encode(tokenizer, prompt, device, ctx=512)
        gen = generate(model, ids_t, gen_tokens,
                        gen_seed=seed * 10000 + hash(prompt) % 1000)
        full = tokenizer.decode(gen[0], skip_special_tokens=True)
        # Decode just the generated continuation (after the original prompt
        # tokens). We use prompt token count to find the boundary in the
        # decoded text — easier to just strip the prompt prefix from `full`
        # since text-level decode preserves it.
        # The prompt may or may not be a clean prefix of `full` due to
        # tokenizer round-trip; use string strip when possible.
        if full.startswith(prompt):
            cont = full[len(prompt):]
        else:
            # Fallback: token-level slicing
            cont_ids = gen[0, ids_t.shape[1]:]
            cont = tokenizer.decode(cont_ids, skip_special_tokens=True)
        gens.append(cont[:200])
        hits = sum(1 for f in expected_fragments if check_match(f, cont))
        fracs.append(hits / max(1, len(expected_fragments)))
        if hits == len(expected_fragments):
            full_hits += 1
    return {
        "frac_hits_mean": float(np.mean(fracs)),
        "full_hit_rate": full_hits / n_seeds,
        "gens": gens,
    }


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((PPD / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    # Build base model with LoRA structure but zeroed (= base behavior).
    print("Loading V22-Dickens base ...")
    model = build_model(device, rank=RANK)
    reset_lora_to_zero(model)

    # ----- Test 1: per-constituent retrieval, RAG style -----
    print("\n=== Test 1: RAG on per-constituent held-out queries ===")
    t0 = time.time()
    test1_results = []
    for combo in COMBINATIONS:
        cname = combo["name"]; K = combo["k"]
        constituents = combo["constituents"]
        constituent_entries = [by_id[i] for i in constituents]

        per_constituent = []
        for entry in constituent_entries:
            n = 0; n_hit = 0
            for q_text in entry["paraphrases_held_out"]:
                rag_prompt = build_rag_prompt(constituent_entries, q_text)
                # Quick token count check
                n_tokens = len(tokenizer.encode(rag_prompt,
                                                 add_special_tokens=False))
                for seed in SEEDS:
                    ids_t = encode(tokenizer, rag_prompt, device, ctx=512)
                    gen = generate(model, ids_t, GEN_TOKENS,
                                    gen_seed=seed * 10000 + hash(q_text) % 1000)
                    cont_ids = gen[0, ids_t.shape[1]:]
                    cont = tokenizer.decode(cont_ids, skip_special_tokens=True)
                    n += 1
                    if check_match(entry["answer"], cont): n_hit += 1
            per_constituent.append({
                "library_id": entry["id"], "answer": entry["answer"],
                "rag_rate": n_hit / n, "n": n, "rag_prompt_tokens": n_tokens,
            })
        avg = float(np.mean([p["rag_rate"] for p in per_constituent]))
        test1_results.append({
            "name": cname, "k": K,
            "per_constituent": per_constituent,
            "avg_rag_rate": avg,
        })
        print(f"  [{cname}] K={K}  avg_rag={avg:.3f}  "
              f"prompt_tokens={per_constituent[0]['rag_prompt_tokens']}")
    print(f"  Test 1 wall: {time.time()-t0:.0f}s")

    # ----- Test 2: cross-passage queries, RAG style -----
    print("\n=== Test 2: RAG on cross-passage queries ===")
    t0 = time.time()
    test2_results = []
    for combo in COMBINATIONS:
        cname = combo["name"]; K = combo["k"]
        constituent_entries = [by_id[i] for i in combo["constituents"]]
        per_query = []
        for q in combo["cross_queries"]:
            rag_prompt = build_rag_prompt(constituent_entries, q["probe"])
            r = measure(model, rag_prompt, q["fragments"], tokenizer, device)
            per_query.append({**q, **r})
        avg_full = float(np.mean([r["full_hit_rate"] for r in per_query]))
        avg_frac = float(np.mean([r["frac_hits_mean"] for r in per_query]))
        test2_results.append({
            "name": cname, "k": K,
            "per_query": per_query,
            "avg_full_hit": avg_full, "avg_frac_hits": avg_frac,
        })
        print(f"  [{cname}] K={K}  rag: full={avg_full:.3f} "
              f"frac={avg_frac:.3f}")
    print(f"  Test 2 wall: {time.time()-t0:.0f}s")

    out = {
        "test1": test1_results,
        "test2": test2_results,
        "wall_total_s": time.time() - t0,
    }
    out_path = REPO / "experiments/combo_vs_rag/results/rag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
