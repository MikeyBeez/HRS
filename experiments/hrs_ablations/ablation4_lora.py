"""Ablation 4: LoRA configuration variants.

For each (rank, target set), retrain all 50 adapters and run a routing+
retrieval eval. The base model and stored engrams are unchanged (engrams
come from the base model with LoRA at zero), so we reuse the canonical
projection_W and library_keys.

Variants:
  baseline_r128_L45         : Phase 47 (rank 128, L45 = attn+FFN on blocks 4-5)
  rank32_L45                : rank 32, same targets
  rank64_L45                : rank 64
  rank256_L45               : rank 256
  rank128_attn_only         : rank 128, attn (qkv, out_proj) on blocks 4-5
  rank128_ffn_only          : rank 128, FFN (input_proj, output_proj) on blocks 4-5
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.hrs_ablations.util import (
    PPD, D, get_tokenizer, get_hidden, pool, held_out_queries,
    generate, check_match, GEN_TOKENS,
)
from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)

ATTN_ONLY = ['blocks.4.attn.qkv', 'blocks.4.attn.out_proj',
              'blocks.5.attn.qkv', 'blocks.5.attn.out_proj']
FFN_ONLY  = ['blocks.4.peer_ffn.input_proj', 'blocks.4.peer_ffn.output_proj',
              'blocks.5.peer_ffn.input_proj', 'blocks.5.peer_ffn.output_proj']

VARIANTS = [
    {"name": "rank32_L45",        "rank": 32,  "targets": L45_TARGETS},
    {"name": "rank64_L45",        "rank": 64,  "targets": L45_TARGETS},
    {"name": "rank256_L45",       "rank": 256, "targets": L45_TARGETS},
    {"name": "rank128_attn_only", "rank": 128, "targets": ATTN_ONLY},
    {"name": "rank128_ffn_only",  "rank": 128, "targets": FFN_ONLY},
]


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    library = json.loads((PPD / "data/library.json").read_text())
    keys = json.loads((PPD / "results/library_keys.json").read_text())
    queries = held_out_queries(library)

    # Library L5 stored keys + projection W (canonical, base-model-derived,
    # so unchanged across LoRA configs).
    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    library_l5_n = F.normalize(library_l5, dim=-1)
    proj_ck = torch.load(PPD / "results/projection_W.pt",
                         map_location=device, weights_only=False)
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(proj_ck["W_state"])
    W.eval()

    # Load V22-Dickens base
    print("Loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                            map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)

    out_dir = REPO / "experiments/hrs_ablations/results"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []
    t_total = time.time()

    for v in VARIANTS:
        t0 = time.time()
        print(f"\n=== {v['name']}: rank={v['rank']} targets={len(v['targets'])} ===")

        # Reset model to base and apply this variant's LoRA structure.
        model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
        n_lora = apply_lora(model, rank=v["rank"], alpha=v["rank"] * 2,
                             target_modules=v["targets"])
        print(f"  LoRA params per adapter: {n_lora:,}")

        # Train all 50 adapters
        t_train = time.time()
        adapter_sds = {}
        for entry in library:
            i = entry["id"]
            passage = entry["passage"]
            train_paras = entry["paraphrases_train"]
            answer = entry["answer"]
            prompts_with_answers = [f"{p}{answer}" for p in train_paras]
            reset_lora_to_zero(model)
            train_adapter_multipara(model, passage, prompts_with_answers,
                                     tokenizer, device, n_steps=150)
            sd = get_lora_state_dict(model)
            adapter_sds[i] = {k: vv.detach().cpu().clone() for k, vv in sd.items()}
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/50] adapter trained ({time.time()-t_train:.0f}s)")
        train_wall = time.time() - t_train
        print(f"  All 50 adapters trained: {train_wall:.0f}s")

        # Move adapters to device for eval
        adapter_sds = {i: {k: vv.to(device) for k, vv in sd.items()}
                       for i, sd in adapter_sds.items()}

        # Evaluate: routing+retrieval, A_full only, 3 seeds
        t_eval = time.time()
        n = 0; n_routing = 0; n_retrieval = 0
        for seed in (0, 1, 2):
            for qi, q in enumerate(queries):
                ids = tokenizer.encode(q["probe"], add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                with torch.no_grad():
                    reset_lora_to_zero(model)
                    l0 = pool(get_hidden(model, ids_t, "L0"), "mean")
                    proj = W(l0.unsqueeze(0))
                    proj_n = F.normalize(proj, dim=-1)
                    sim = (proj_n @ library_l5_n.T).squeeze(0)
                    routed = sim.argmax().item()
                load_lora_state_dict(model, adapter_sds[routed])
                gen = generate(model, ids_t, GEN_TOKENS,
                                gen_seed=seed * 10000 + qi)
                full = tokenizer.decode(gen[0], skip_special_tokens=True)
                cont = full[len(q["probe"]):]
                n += 1
                if routed == q["adapter_id"]: n_routing += 1
                if check_match(q["answer"], cont): n_retrieval += 1
        eval_wall = time.time() - t_eval

        rec = {
            "variant": v["name"],
            "rank": v["rank"],
            "n_targets": len(v["targets"]),
            "n_lora_params_per_adapter": n_lora,
            "train_wall_s": train_wall,
            "eval_wall_s": eval_wall,
            "routing_acc": n_routing / n,
            "retrieval_acc": n_retrieval / n,
            "n": n,
        }
        results_summary.append(rec)
        print(f"  routing_acc={rec['routing_acc']:.3f}  "
              f"retrieval_acc={rec['retrieval_acc']:.3f}  "
              f"eval_wall={eval_wall:.0f}s  total={time.time()-t0:.0f}s")

        out_path = out_dir / "ablation4_lora.json"
        out_path.write_text(json.dumps({
            "results": results_summary,
            "wall_total_s": time.time() - t_total,
        }, indent=2))

    print(f"\nAblation 4 wall: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
