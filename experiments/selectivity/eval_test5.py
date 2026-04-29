"""Test 5: K-sweep composition with all 6 domain adapters.

For K in {1, 2, 4, 6}:
  - Build a stacked LoRA at rank K*128 from a deterministic K-subset.
  - For each domain in the K-subset, evaluate retrieval on its held-out
    positives. Average across the K domains.
  - Measure passivity on bread (Domain C, never seen).

Compare selectivity (λ=0.1) and baseline procedures across the K sweep.
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
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, hidden_at_layer,
)
from experiments.identity_ae.phase43_k_capacity import stack_k_state_dicts
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)
from experiments.selectivity.data import domain_data
from experiments.selectivity.evaluate import (
    encode, generate, check_match, build_model,
    precompute_base_artifacts, passivity_with_h5,
    GEN_TOKENS, SEEDS, RANK,
)

PPD = REPO / "experiments/per_passage_dickens"
SEL = REPO / "experiments/selectivity"

DOMAINS = ("A", "B", "D", "E", "F", "G")
K_VALUES = [1, 2, 4, 6]


def adapter_path(procedure, domain):
    if procedure == "selectivity":
        return SEL / f"adapters/sel_{domain}_lam0.1.pt"
    else:  # baseline
        return SEL / f"adapters/baseline_{domain}.pt"


def k_subset(K):
    """First K domains in DOMAINS order."""
    return list(DOMAINS[:K])


@torch.no_grad()
def measure_retrieval(model, examples, tokenizer, device):
    """Substring match on examples × 3 seeds."""
    n = 0; n_hit = 0
    for ex in examples:
        for seed in SEEDS:
            ids_t = encode(tokenizer, ex["probe"], device)
            gen = generate(model, ids_t, GEN_TOKENS,
                            gen_seed=seed * 10000 + hash(ex["probe"]) % 1000)
            full = tokenizer.decode(gen[0], skip_special_tokens=True)
            cont = full[len(ex["probe"]):]
            n += 1
            if check_match(ex["answer"], cont):
                n_hit += 1
    return {"n": n, "n_hit": n_hit, "rate": n_hit / n}


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = domain_data()

    out = {"K_values": K_VALUES, "domains_loaded": {}, "results": {}}
    t_total = time.time()

    for procedure in ("selectivity", "baseline"):
        out["results"][procedure] = {}
        print(f"\n=== Procedure: {procedure} ===")
        for K in K_VALUES:
            t_k = time.time()
            subset = k_subset(K)
            out["domains_loaded"][f"K{K}"] = subset

            # Build stacked LoRA from this K-subset
            sds = [torch.load(adapter_path(procedure, d), map_location="cpu",
                              weights_only=False) for d in subset]
            stacked = stack_k_state_dicts(sds) if len(sds) > 1 else sds[0]
            model = build_model(device, rank=K * RANK)
            load_lora_state_dict(model, {k: v.to(device) for k, v in stacked.items()})

            # Retrieval on each loaded domain's held-out positives
            per_domain = {}
            n_total_loaded = 0; n_hit_loaded = 0
            for d in subset:
                r = measure_retrieval(model, data[d]["held_out"][:10],
                                       tokenizer, device)
                per_domain[d] = r
                n_total_loaded += r["n"]; n_hit_loaded += r["n_hit"]
            avg_loaded = n_hit_loaded / max(1, n_total_loaded)

            # Passivity on bread (Domain C, held-out)
            base_logits, base_h5 = precompute_base_artifacts(
                model, data["C"]["held_out"][:20], tokenizer, device,
            )
            load_lora_state_dict(model, {k: v.to(device) for k, v in stacked.items()})
            passivity = passivity_with_h5(model, base_logits, base_h5,
                                           data["C"]["held_out"][:20],
                                           tokenizer, device)

            wall = time.time() - t_k
            out["results"][procedure][f"K{K}"] = {
                "K": K, "subset": subset,
                "per_domain_retrieval": per_domain,
                "avg_retrieval_loaded": avg_loaded,
                "passivity_on_C": passivity,
                "wall_s": wall,
            }
            print(f"  K={K}: avg_retrieval={avg_loaded:.3f}  "
                  f"C_passivity_KL={passivity['kl_mean']:.2f}  "
                  f"C_h5_cos={passivity['h5_cos_mean']:.3f}  "
                  f"wall={wall:.0f}s")
            del model
            torch.cuda.empty_cache()

    out["wall_total_s"] = time.time() - t_total
    out_path = SEL / "results/test5.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTest 5 wall: {time.time()-t_total:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
