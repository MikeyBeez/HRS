"""Phase A: train one rank-128 LoRA adapter per library entry.

For each entry, the adapter is trained on:
  - the passage text (next-token LM)
  - each (training paraphrase + answer) string (next-token LM, with the
    paraphrase + " " + answer joined as one sequence)

Phase 47 protocol via experiments.identity_ae.phase26_multikey.train_adapter_multipara.
Saves adapter state_dict + L5 mean key (under base model, computed from the
training paraphrases) for each library entry.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero,
)
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
import os
N_STEPS = int(os.environ.get("N_STEPS", "150"))


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0).detach().cpu()


@torch.no_grad()
def l5_mean(model, ids_t):
    h = hidden_at_layer(model, ids_t, 5)
    return h.mean(dim=1).squeeze(0).detach().cpu()


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 1. Load V22 + Dickens-pretrained checkpoint
    model, cfg = load_model(device)
    dickens_ck = torch.load(
        REPO / "experiments/per_passage_dickens/results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    missing, unexpected = model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    print(f"Loaded Dickens-pretrained base. missing={len(missing)} unexpected={len(unexpected)}")
    print(f"  GE val_ppl in ckpt: {dickens_ck.get('ge_val_ppl', 'N/A')}")

    # 2. Apply LoRA structure
    n_lora = apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"Steps per adapter: {N_STEPS}")

    # 3. Load library
    library = json.loads((REPO / "experiments/per_passage_dickens/data/library.json").read_text())
    print(f"Library: {len(library)} entries")

    suffix = f"_steps{N_STEPS}" if N_STEPS != 150 else ""
    out_dir = REPO / f"experiments/per_passage_dickens/adapters{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys_dir = REPO / "experiments/per_passage_dickens/results"
    keys_dir.mkdir(parents=True, exist_ok=True)

    library_keys = []  # one record per entry: {id, l5_mean (averaged over training paras), l0_means_per_para, l5_means_per_para, sd_path}
    t0 = time.time()
    for entry in library:
        i = entry["id"]
        passage = entry["passage"]
        train_paras = entry["paraphrases_train"]
        answer = entry["answer"]

        # Each prompt joined with answer as the (prompt + " " + answer) string
        prompts_with_answers = [f"{p}{answer}" for p in train_paras]

        reset_lora_to_zero(model)
        train_adapter_multipara(
            model, passage, prompts_with_answers, tokenizer, device,
            n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR,
        )

        # Save adapter
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        sd_path = out_dir / f"adapter_{i:03d}.pt"
        torch.save(sd, sd_path)

        # Compute L5 mean keys under BASE model (LoRA reset to zero) per paraphrase.
        # The library's L5 key for this adapter is the mean of L5 over training paras.
        reset_lora_to_zero(model)
        l5_per_para = []
        l0_per_para = []
        for p in train_paras:
            ids = tokenizer.encode(p, add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
            l5_per_para.append(l5_mean(model, ids_t))
            l0_per_para.append(l0_mean(model, ids_t))
        # Aggregated L5 key
        l5_aggregate = torch.stack(l5_per_para).mean(dim=0)

        library_keys.append({
            "id": i,
            "fact_type": entry["fact_type"],
            "answer": answer,
            "sd_path": str(sd_path.relative_to(REPO)),
            "l5_aggregate": l5_aggregate.tolist(),
            "l5_per_para": [v.tolist() for v in l5_per_para],
            "l0_per_para": [v.tolist() for v in l0_per_para],
        })

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(library)}] adapter trained ({time.time()-t0:.0f}s)")

    # Save library_keys (the routing/library index)
    keys_path = keys_dir / f"library_keys{suffix}.json"
    keys_path.write_text(json.dumps(library_keys, indent=2))
    print(f"\n[Phase A] DONE  wall={time.time()-t0:.0f}s  saved {keys_path}")


if __name__ == "__main__":
    main()
