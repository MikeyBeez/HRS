"""Test 4: multi-pass processing baseline.

For each K=3 and K=4 cross-passage query that requires content from K
passages:
  Pass 1..K: load each constituent adapter alone (K=1), generate
             continuation from the original probe.
  Pass K+1: with NO adapter (LoRA zeroed), feed
             "{probe} {cont_1} {cont_2} ... {cont_K}\nFinal answer: "
             and generate the final continuation.

Score the final continuation against the expected answer fragments.

Compare to combination adapter (K=1, the combo loaded alone) on the same
queries.
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
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)

from experiments.combo_adapter.combos import COMBINATIONS
from experiments.combo_adapter.evaluate import (
    encode, generate, check_match, build_model,
    GEN_TOKENS, SEEDS, RANK,
)

PPD = REPO / "experiments/per_passage_dickens"
COMBO = REPO / "experiments/combo_adapter"


def measure_combo_only(probe, fragments, model_combo, tokenizer, device):
    """Generate from probe with the combo adapter active, fraction-of-hits
    fragment scoring."""
    fracs = []; full_hits = 0
    for seed in SEEDS:
        ids_t = encode(tokenizer, probe, device)
        gen = generate(model_combo, ids_t, GEN_TOKENS,
                        gen_seed=seed * 10000 + hash(probe) % 1000)
        full = tokenizer.decode(gen[0], skip_special_tokens=True)
        cont = full[len(probe):]
        hits = sum(1 for f in fragments if check_match(f, cont))
        fracs.append(hits / max(1, len(fragments)))
        if hits == len(fragments):
            full_hits += 1
    return {"frac_hits_mean": float(np.mean(fracs)),
            "full_hit_rate": full_hits / len(SEEDS)}


def measure_multipass(probe, fragments, constituents, adapter_sds, model,
                       tokenizer, device):
    """For each constituent, run K=1 inference, capture continuation. Then
    run a final pass with no adapter (LoRA zeroed) on the concatenated
    intermediate outputs.
    """
    fracs = []; full_hits = 0
    for seed in SEEDS:
        # K constituent passes
        intermediates = []
        for cid in constituents:
            reset_lora_to_zero(model)
            load_lora_state_dict(model, adapter_sds[cid])
            ids_t = encode(tokenizer, probe, device)
            gen = generate(model, ids_t, GEN_TOKENS,
                            gen_seed=seed * 10000 + hash(probe) % 1000 + cid)
            full = tokenizer.decode(gen[0], skip_special_tokens=True)
            cont = full[len(probe):]
            intermediates.append(cont.strip())
        # Final pass with LoRA off
        reset_lora_to_zero(model)
        composite = (probe + "\n" +
                     "".join(f"Note {i+1}: {it}\n"
                             for i, it in enumerate(intermediates))
                     + "Final answer: ")
        ids_t = encode(tokenizer, composite, device, ctx=512)
        gen = generate(model, ids_t, GEN_TOKENS,
                        gen_seed=seed * 10000 + hash(probe) % 1000 + 999)
        full = tokenizer.decode(gen[0], skip_special_tokens=True)
        cont = full[len(composite):]
        hits = sum(1 for f in fragments if check_match(f, cont))
        fracs.append(hits / max(1, len(fragments)))
        if hits == len(fragments):
            full_hits += 1
    return {"frac_hits_mean": float(np.mean(fracs)),
            "full_hit_rate": full_hits / len(SEEDS)}


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((PPD / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    # Restrict to K=3 and K=4 combinations
    target_combos = [c for c in COMBINATIONS if c["k"] in (3, 4)]
    print(f"Test 4: {len(target_combos)} K>=3 combinations × 3 cross-queries "
          f"× 3 seeds × (multipass + combo)")

    # Pre-load all constituent adapter state dicts onto GPU
    needed_ids = set()
    for c in target_combos:
        needed_ids.update(c["constituents"])
    adapter_sds = {}
    for cid in needed_ids:
        sd = torch.load(PPD / f"adapters/adapter_{cid:03d}.pt",
                        map_location="cpu", weights_only=False)
        adapter_sds[cid] = {k: v.to(device) for k, v in sd.items()}
    print(f"Loaded {len(adapter_sds)} constituent adapters")

    # Build a single rank-128 model for both multipass and combo paths
    model = build_model(device, rank=RANK)

    out = []
    t0 = time.time()
    for combo in target_combos:
        cname = combo["name"]
        K = combo["k"]
        constituents = combo["constituents"]
        # Combo adapter
        combo_sd = torch.load(COMBO / f"adapters/{cname}.pt",
                               map_location=device, weights_only=False)

        per_query = []
        for q in combo["cross_queries"]:
            # Combo path
            reset_lora_to_zero(model)
            load_lora_state_dict(model, combo_sd)
            r_combo = measure_combo_only(q["probe"], q["fragments"],
                                          model, tokenizer, device)
            # Multipass path
            r_mp = measure_multipass(q["probe"], q["fragments"],
                                      constituents, adapter_sds, model,
                                      tokenizer, device)
            per_query.append({
                "probe": q["probe"][:80] + "...",
                "fragments": q["fragments"],
                "combo": r_combo,
                "multipass": r_mp,
            })

        avg_combo_frac = np.mean([p["combo"]["frac_hits_mean"] for p in per_query])
        avg_mp_frac = np.mean([p["multipass"]["frac_hits_mean"] for p in per_query])
        avg_combo_full = np.mean([p["combo"]["full_hit_rate"] for p in per_query])
        avg_mp_full = np.mean([p["multipass"]["full_hit_rate"] for p in per_query])
        print(f"  [{cname}] K={K}  combo: full={avg_combo_full:.3f} "
              f"frac={avg_combo_frac:.3f}  multipass: full={avg_mp_full:.3f} "
              f"frac={avg_mp_frac:.3f}")
        out.append({"name": cname, "k": K, "per_query": per_query,
                    "avg_combo_frac": avg_combo_frac,
                    "avg_multipass_frac": avg_mp_frac,
                    "avg_combo_full": avg_combo_full,
                    "avg_multipass_full": avg_mp_full})

    out_path = COMBO / "results/test4_multipass.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "results": out, "wall_total_s": time.time() - t0,
    }, indent=2))
    print(f"\nTest 4 wall: {time.time()-t0:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
