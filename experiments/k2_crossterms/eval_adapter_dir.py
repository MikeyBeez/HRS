"""Evaluate a directory of 50 adapters with the same N=1 + N=2 protocol
as Phase 1. Used for orthogonal-lambda{...} and crosstermaware variants.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python eval_adapter_dir.py \\
        --adapter-dir experiments/k2_crossterms/adapters_orthogonal_lambda0.1 \\
        --label orthogonal_lambda_0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/k2_crossterms"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.k2_crossterms.multi_lora import (
    apply_multi_lora, set_active_adapters, reset_multi_lora,
)


RANK = 128
ALPHA = 256
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
MAX_CTX_POS = 512
N_PAIRS = 50
PAIR_SEED = 42


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-pairs", type=int, default=N_PAIRS)
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    label = args.label
    out_path = EXP / "results" / f"eval_{label}.json"

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_multi_lora(model, rank=RANK, alpha=ALPHA,
                      target_modules=L45_TARGETS, max_k=2)
    reset_multi_lora(model)
    model.eval()

    # Load adapters from the specified directory
    adapter_sds = {}
    for entry in library:
        sd_path = adapter_dir / f"adapter_{entry['id']:03d}.pt"
        if not sd_path.exists():
            print(f"WARN: missing {sd_path}")
            continue
        adapter_sds[entry["id"]] = torch.load(
            sd_path, map_location=device, weights_only=False)
    print(f"[{label}] Loaded {len(adapter_sds)} adapters from {adapter_dir}")

    held_out_per_adapter = {}
    for entry in library:
        held_out_per_adapter[entry["id"]] = [
            {"probe": q, "answer": entry["answer"],
             "fact_type": entry["fact_type"]}
            for q in entry["paraphrases_held_out"]
        ]

    # ----- N=1 -----
    n1_results = []
    t0 = time.time()
    for aid in sorted(adapter_sds.keys()):
        set_active_adapters(model, [adapter_sds[aid]])
        for probe_rec in held_out_per_adapter[aid]:
            probe = probe_rec["probe"]
            ids = tokenizer.encode(probe, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            for seed in [0, 1, 2]:
                gen_seed = seed * 10000 + aid * 100
                gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
                full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                cont = full[len(probe):]
                hit = check_match(probe_rec["answer"], cont)
                n1_results.append({
                    "adapter_id": aid, "fact_type": probe_rec["fact_type"],
                    "seed": seed, "retrieved": hit,
                })
    n1_acc = sum(r["retrieved"] for r in n1_results) / len(n1_results)
    n1_per_type = defaultdict(list)
    for r in n1_results:
        n1_per_type[r["fact_type"]].append(r["retrieved"])
    print(f"[{label}] N=1 = {n1_acc:.3f}  "
          f"(elapsed={time.time()-t0:.0f}s)")
    for ft, hits in sorted(n1_per_type.items()):
        print(f"    {ft:>8s}: {sum(hits)/len(hits):.3f} (n={len(hits)})")

    # ----- N=2 -----
    rng = random.Random(PAIR_SEED)
    aids = sorted(adapter_sds.keys())
    pairs = []
    while len(pairs) < args.max_pairs:
        i, j = rng.sample(aids, 2)
        pairs.append((i, j))

    n2_results = []
    t0 = time.time()
    for pi, (i, j) in enumerate(pairs):
        set_active_adapters(model, [adapter_sds[i], adapter_sds[j]])
        for target_aid, companion_aid in [(i, j), (j, i)]:
            for probe_rec in held_out_per_adapter[target_aid]:
                probe = probe_rec["probe"]
                ids = tokenizer.encode(probe, add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                for seed in [0, 1, 2]:
                    gen_seed = seed * 100000 + pi * 1000 + target_aid
                    gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
                    full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    cont = full[len(probe):]
                    hit = check_match(probe_rec["answer"], cont)
                    n2_results.append({
                        "pair_id": pi, "target_adapter": target_aid,
                        "companion_adapter": companion_aid,
                        "fact_type": probe_rec["fact_type"], "seed": seed,
                        "retrieved": hit,
                    })
    n2_acc = sum(r["retrieved"] for r in n2_results) / len(n2_results)
    n2_per_type = defaultdict(list)
    for r in n2_results:
        n2_per_type[r["fact_type"]].append(r["retrieved"])
    print(f"[{label}] N=2 = {n2_acc:.3f}  "
          f"(elapsed={time.time()-t0:.0f}s)")
    for ft, hits in sorted(n2_per_type.items()):
        print(f"    {ft:>8s}: {sum(hits)/len(hits):.3f} (n={len(hits)})")

    # Save
    summary = {
        "label": label,
        "adapter_dir": str(adapter_dir),
        "n1_overall": n1_acc,
        "n2_overall": n2_acc,
        "gap_n1_n2": n1_acc - n2_acc,
        "n1_per_fact_type": {ft: sum(h)/len(h) for ft, h in n1_per_type.items()},
        "n2_per_fact_type": {ft: sum(h)/len(h) for ft, h in n2_per_type.items()},
        "n_pairs": len(pairs),
        "pair_seed": PAIR_SEED,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[{label}] Saved {out_path}")


if __name__ == "__main__":
    main()
