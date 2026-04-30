"""Tests 1, 2, 3 for combination adapters.

Test 1 (per-constituent retrieval): for each combination adapter, score
substring match on each constituent's held-out paraphrases. Compare to:
  - Single-passage K=1 baseline (the canonical per_passage_dickens adapter)
  - Multi-adapter K=N composition (constituents block-stacked at rank K*128)

Test 2 (cross-passage queries): for each combination adapter, run the
hand-crafted cross-passage queries and score the fraction of expected
answer fragments hit. Compare to multi-adapter K=N composition.

Test 3 (capacity sweep): aggregate Test 1 & 2 results across K=2,3,4.
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
from experiments.identity_ae.phase43_k_capacity import stack_k_state_dicts
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)

from experiments.combo_adapter.combos import COMBINATIONS

PPD = REPO / "experiments/per_passage_dickens"
COMBO = REPO / "experiments/combo_adapter"
RANK = 128
GEN_TOKENS = 30
TEMPERATURE = 0.6
TOP_K = 20
SEEDS = (0, 1, 2)


def encode(tokenizer, text, device, ctx=512):
    ids = tokenizer.encode(text, add_special_tokens=False)[:ctx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / TEMPERATURE
        v, _ = torch.topk(logits, TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, gen):
    if answer.lower() in gen.lower():
        return True
    a = answer.replace(",", "").replace(" ", "").lower()
    g = gen.replace(",", "").replace(" ", "").lower()
    return bool(a) and a in g


def build_model(device, rank=RANK):
    model, _ = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()
    return model


def measure_one(model, probe, expected_fragments, tokenizer, device,
                 n_seeds=3, gen_tokens=GEN_TOKENS):
    """Generate from probe N seeds. Return (any_seed_full_hit, frac_fragments
    hit per seed averaged over seeds, raw generations)."""
    n = 0; full_hits = 0; frac_hits = []
    gens = []
    for seed in range(n_seeds):
        ids_t = encode(tokenizer, probe, device)
        gen = generate(model, ids_t, gen_tokens,
                        gen_seed=seed * 10000 + hash(probe) % 1000)
        full = tokenizer.decode(gen[0], skip_special_tokens=True)
        cont = full[len(probe):]
        gens.append(cont[:120])
        n += 1
        # Per-fragment hit
        hits = sum(1 for f in expected_fragments if check_match(f, cont))
        frac = hits / max(1, len(expected_fragments))
        frac_hits.append(frac)
        if hits == len(expected_fragments):
            full_hits += 1
    return {
        "n_seeds": n,
        "full_hit_rate": full_hits / n,
        "frac_hits_mean": float(np.mean(frac_hits)),
        "gens": gens,
    }


# ----- Test 1 -----
def test1(combo_paths_by_name, library, tokenizer, device):
    """Per-constituent retrieval. For each combination, eval each
    constituent's held-out probes under three conditions:
    - combo: combo adapter loaded alone
    - single: single-passage adapter (canonical) loaded alone
    - multi: constituent single-passage adapters block-stacked at rank K*128
    """
    by_id = {e["id"]: e for e in library}
    out = []
    for combo in COMBINATIONS:
        cname = combo["name"]
        K = combo["k"]
        constituents = combo["constituents"]
        single_paths = [PPD / f"adapters/adapter_{cid:03d}.pt"
                        for cid in constituents]

        # Build the multi-adapter stacked state once
        sds = [torch.load(p, map_location="cpu", weights_only=False)
               for p in single_paths]
        stacked_multi = stack_k_state_dicts(sds) if len(sds) > 1 else sds[0]

        per_constituent = []
        for cid in constituents:
            entry = by_id[cid]
            held = entry["paraphrases_held_out"]
            answer = entry["answer"]

            # ----- combo adapter (rank 128) -----
            model_combo = build_model(device, rank=RANK)
            combo_sd = torch.load(combo_paths_by_name[cname], map_location=device,
                                   weights_only=False)
            load_lora_state_dict(model_combo, combo_sd)
            n_h, n_t = 0, 0
            for q in held:
                for seed in SEEDS:
                    ids_t = encode(tokenizer, q, device)
                    gen = generate(model_combo, ids_t, GEN_TOKENS,
                                    gen_seed=seed * 10000 + hash(q) % 1000)
                    full = tokenizer.decode(gen[0], skip_special_tokens=True)
                    cont = full[len(q):]
                    n_t += 1
                    if check_match(answer, cont): n_h += 1
            r_combo = n_h / n_t
            del model_combo; torch.cuda.empty_cache()

            # ----- single-passage K=1 -----
            model_s = build_model(device, rank=RANK)
            sd_single = torch.load(single_paths[constituents.index(cid)],
                                    map_location=device, weights_only=False)
            load_lora_state_dict(model_s, sd_single)
            n_h, n_t = 0, 0
            for q in held:
                for seed in SEEDS:
                    ids_t = encode(tokenizer, q, device)
                    gen = generate(model_s, ids_t, GEN_TOKENS,
                                    gen_seed=seed * 10000 + hash(q) % 1000)
                    full = tokenizer.decode(gen[0], skip_special_tokens=True)
                    cont = full[len(q):]
                    n_t += 1
                    if check_match(answer, cont): n_h += 1
            r_single = n_h / n_t
            del model_s; torch.cuda.empty_cache()

            # ----- multi-adapter K=N composition -----
            model_m = build_model(device, rank=K * RANK)
            load_lora_state_dict(model_m, {k: v.to(device) for k, v in stacked_multi.items()})
            n_h, n_t = 0, 0
            for q in held:
                for seed in SEEDS:
                    ids_t = encode(tokenizer, q, device)
                    gen = generate(model_m, ids_t, GEN_TOKENS,
                                    gen_seed=seed * 10000 + hash(q) % 1000)
                    full = tokenizer.decode(gen[0], skip_special_tokens=True)
                    cont = full[len(q):]
                    n_t += 1
                    if check_match(answer, cont): n_h += 1
            r_multi = n_h / n_t
            del model_m; torch.cuda.empty_cache()

            per_constituent.append({
                "library_id": cid,
                "answer": answer,
                "combo_rate": r_combo,
                "single_rate": r_single,
                "multi_rate": r_multi,
            })

        out.append({"name": cname, "k": K, "per_constituent": per_constituent})
        avg_combo = np.mean([p["combo_rate"] for p in per_constituent])
        avg_single = np.mean([p["single_rate"] for p in per_constituent])
        avg_multi = np.mean([p["multi_rate"] for p in per_constituent])
        print(f"  [{cname}] K={K}  avg_combo={avg_combo:.3f}  "
              f"avg_single={avg_single:.3f}  avg_multi={avg_multi:.3f}")
    return out


# ----- Test 2 -----
def test2(combo_paths_by_name, tokenizer, device):
    """Cross-passage queries. Combo adapter vs multi-adapter K=N composition."""
    out = []
    for combo in COMBINATIONS:
        cname = combo["name"]
        K = combo["k"]
        constituents = combo["constituents"]
        single_paths = [PPD / f"adapters/adapter_{cid:03d}.pt"
                        for cid in constituents]
        sds = [torch.load(p, map_location="cpu", weights_only=False)
               for p in single_paths]
        stacked_multi = stack_k_state_dicts(sds)

        # Combo adapter
        model_combo = build_model(device, rank=RANK)
        combo_sd = torch.load(combo_paths_by_name[cname], map_location=device,
                               weights_only=False)
        load_lora_state_dict(model_combo, combo_sd)
        combo_results = []
        for q in combo["cross_queries"]:
            r = measure_one(model_combo, q["probe"], q["fragments"],
                             tokenizer, device)
            combo_results.append({**q, **r})
        del model_combo; torch.cuda.empty_cache()

        # Multi-adapter K=N
        model_m = build_model(device, rank=K * RANK)
        load_lora_state_dict(model_m, {k: v.to(device) for k, v in stacked_multi.items()})
        multi_results = []
        for q in combo["cross_queries"]:
            r = measure_one(model_m, q["probe"], q["fragments"],
                             tokenizer, device)
            multi_results.append({**q, **r})
        del model_m; torch.cuda.empty_cache()

        out.append({
            "name": cname, "k": K,
            "combo_results": combo_results,
            "multi_results": multi_results,
        })
        avg_combo_full = np.mean([r["full_hit_rate"] for r in combo_results])
        avg_combo_frac = np.mean([r["frac_hits_mean"] for r in combo_results])
        avg_multi_full = np.mean([r["full_hit_rate"] for r in multi_results])
        avg_multi_frac = np.mean([r["frac_hits_mean"] for r in multi_results])
        print(f"  [{cname}] K={K}  combo: full={avg_combo_full:.3f} "
              f"frac={avg_combo_frac:.3f}  multi: full={avg_multi_full:.3f} "
              f"frac={avg_multi_frac:.3f}")
    return out


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((PPD / "data/library.json").read_text())

    combo_paths_by_name = {c["name"]: COMBO / f"adapters/{c['name']}.pt"
                            for c in COMBINATIONS}

    out = {}
    t0 = time.time()

    print("\n=== Test 1: per-constituent retrieval ===")
    out["test1"] = test1(combo_paths_by_name, library, tokenizer, device)

    print("\n=== Test 2: cross-passage queries ===")
    out["test2"] = test2(combo_paths_by_name, tokenizer, device)

    out["wall_total_s"] = time.time() - t0
    out_path = COMBO / "results/tests.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTotal eval wall: {out['wall_total_s']:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
