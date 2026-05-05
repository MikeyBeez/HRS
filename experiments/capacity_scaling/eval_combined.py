"""Evaluate a combined adapter on its constituent passages' held-out probes.

For each constituent passage P, run all 3 held-out paraphrases × 3 seeds
through the model with the combined adapter loaded. Score retrieval via
substring match (same as Dickens-50 evaluate.py).

Output: JSON with overall + per-passage + per-fact-type retrieval.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python eval_combined.py \\
        --adapter-path .../adapters/size_05.pt \\
        --label size_05
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/capacity_scaling"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)


DEFAULT_RANK = 128
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
MAX_CTX_POS = 512


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
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    ck = torch.load(args.adapter_path, map_location=device, weights_only=False)
    sd = ck["lora_state_dict"]
    passage_ids = ck["passage_ids"]
    rank = ck.get("rank", DEFAULT_RANK)
    alpha = ck.get("alpha", rank * 2)
    print(f"[{args.label}] Loaded adapter trained on {len(passage_ids)} "
          f"passages (rank={rank}, alpha={alpha}): {passage_ids}")

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=rank, alpha=alpha, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    load_lora_state_dict(model, sd)
    model.eval()

    per_probe = []
    t0 = time.time()
    for aid in passage_ids:
        entry = by_id[aid]
        for probe in entry["paraphrases_held_out"]:
            ids = tokenizer.encode(probe, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            for seed in [0, 1, 2]:
                gen_seed = seed * 100000 + aid * 100 + len(per_probe)
                gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
                full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                cont = full[len(probe):]
                hit = check_match(entry["answer"], cont)
                per_probe.append({
                    "adapter_size": len(passage_ids),
                    "passage_id": aid, "fact_type": entry["fact_type"],
                    "seed": seed, "probe": probe,
                    "answer": entry["answer"], "continuation": cont,
                    "retrieved": hit,
                })

    n = len(per_probe)
    overall = sum(r["retrieved"] for r in per_probe) / n
    per_passage = defaultdict(list)
    per_type = defaultdict(list)
    for r in per_probe:
        per_passage[r["passage_id"]].append(r["retrieved"])
        per_type[r["fact_type"]].append(r["retrieved"])

    per_passage_acc = {pid: sum(h) / len(h) for pid, h in per_passage.items()}
    per_passage_min = min(per_passage_acc.values())
    per_passage_max = max(per_passage_acc.values())
    per_passage_mean = sum(per_passage_acc.values()) / len(per_passage_acc)

    print(f"[{args.label}] OVERALL retrieval = {overall:.3f}  "
          f"(per-passage mean {per_passage_mean:.3f}, "
          f"min {per_passage_min:.3f}, max {per_passage_max:.3f})  "
          f"(elapsed={time.time()-t0:.0f}s)")
    for ft in ["entity", "numeric", "place", "relation"]:
        if ft in per_type:
            v = per_type[ft]
            print(f"    {ft:>10s}: {sum(v)/len(v):.3f}  (n={len(v)})")

    # Save
    out_path = EXP / f"results/eval_{args.label}.json"
    summary = {
        "label": args.label,
        "adapter_path": args.adapter_path,
        "size": len(passage_ids),
        "passage_ids": passage_ids,
        "rank": rank,
        "alpha": alpha,
        "n_lora_params": ck.get("n_lora_params"),
        "training_time_s": ck.get("training_time_s"),
        "n_steps": ck.get("n_steps"),
        "final_loss_mean50": ck.get("final_loss_mean50"),
        "n_probes_total": n,
        "overall_retrieval": overall,
        "per_passage_retrieval": per_passage_acc,
        "per_passage_mean": per_passage_mean,
        "per_passage_min": per_passage_min,
        "per_passage_max": per_passage_max,
        "per_fact_type": {ft: sum(v)/len(v) for ft, v in per_type.items()},
        "per_fact_type_n": {ft: len(v) for ft, v in per_type.items()},
        "wall_eval_s": time.time() - t0,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[{args.label}] Saved {out_path}")

    # CSV per-probe
    csv_path = EXP / f"results/eval_{args.label}_per_probe.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adapter_size", "passage_id", "fact_type", "seed",
                     "probe", "answer", "continuation", "retrieved"])
        for r in per_probe:
            w.writerow([r["adapter_size"], r["passage_id"], r["fact_type"],
                         r["seed"], r["probe"], r["answer"],
                         r["continuation"][:200], int(r["retrieved"])])


if __name__ == "__main__":
    main()
