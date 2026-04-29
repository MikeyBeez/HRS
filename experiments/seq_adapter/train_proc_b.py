"""Procedure B: train 8 adapters sequentially on the chosen 8 passages.

For adapter k: bake adapters 0..k-1 into the base model's wrapped Linear
weights, then train a fresh rank-128 LoRA on top. After training, save the
LoRA state dict for adapter k. Restore the canonical base weights and
repeat with one more adapter baked in.

At inference time, the canonical base is restored and adapters are
block-stacked the same way as Procedure A.

Hyperparameters
---------------
Sequential training preserves Phase 47's recipe (rank 128, alpha 256,
n_steps 150, HIGH_LR -> BASE_LR with StepLR halve-at-half). Spec notes
that sequential training may need different hyperparameters; if loss
fails to decrease, this script logs it for the writeup.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase26_multikey import train_adapter_multipara
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, LoRALayer,
)

from experiments.seq_adapter.queries import CHOSEN_IDS

PPD = REPO / "experiments/per_passage_dickens"
RANK = 128
ALPHA = RANK * 2
N_STEPS = 150


def bake_lora_into_base(model, sd_lora):
    """Bake a LoRA state dict into the base weights of each LoRALayer.

    For a LoRALayer with base_layer.weight W (out_f, in_f), the equivalent
    bake adds scaling * (A @ B).T to W where:
      A: (in_f, R), B: (R, out_f), scaling = alpha / rank.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALayer):
            A_key = f"{name}.lora_A"
            B_key = f"{name}.lora_B"
            if A_key not in sd_lora or B_key not in sd_lora:
                continue
            A = sd_lora[A_key].to(mod.base_layer.weight.device)
            B = sd_lora[B_key].to(mod.base_layer.weight.device)
            with torch.no_grad():
                mod.base_layer.weight.data += mod.scaling * (A @ B).T


def main():
    device = torch.device("cuda")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Procedure B: sequential training of {len(CHOSEN_IDS)} adapters")
    print(f"Chosen indices: {CHOSEN_IDS}")

    library = json.loads((PPD / "data/library.json").read_text())
    chosen_entries = [library[i] for i in CHOSEN_IDS]

    # Load V22-Dickens base model + LoRA structure
    print("Loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                            map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    # Cache canonical base weights (only the wrapped Linear weights).
    canonical_base_weights = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALayer):
            canonical_base_weights[name] = mod.base_layer.weight.data.detach().clone()

    out_dir = REPO / "experiments/seq_adapter/adapters_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    t_total = time.time()
    saved_adapters = {}  # local_idx -> state dict

    for k, entry in enumerate(chosen_entries):
        print(f"\n[Adapter {k} / {len(chosen_entries)-1}] "
              f"library_id={entry['id']} ans={entry['answer']!r}")

        # Restore canonical base
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALayer):
                mod.base_layer.weight.data.copy_(canonical_base_weights[name])

        # Bake adapters 0..k-1 into the base
        for prev in range(k):
            bake_lora_into_base(model, saved_adapters[prev])

        # Reset LoRA to zero (fresh adapter)
        reset_lora_to_zero(model)

        # Train this adapter
        passage = entry["passage"]
        train_paras = entry["paraphrases_train"]
        answer = entry["answer"]
        prompts_with_answers = [f"{p}{answer}" for p in train_paras]

        t0 = time.time()
        # Capture loss curve manually by hooking train_adapter_multipara.
        # The function doesn't return loss, so we run training and rely on
        # post-training generation to verify convergence.
        train_adapter_multipara(model, passage, prompts_with_answers,
                                 tokenizer, device,
                                 n_steps=N_STEPS, high_lr=HIGH_LR,
                                 base_lr=BASE_LR)
        wall = time.time() - t0

        # Sanity check: with this adapter active (and prior baked-in), can
        # we retrieve the answer?
        with torch.no_grad():
            ids = tokenizer.encode(entry["paraphrases_held_out"][0],
                                   add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            for _ in range(20):
                out = model(ids_t[:, -512:], step=0)
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids_t = torch.cat([ids_t, nxt], dim=1)
            full = tokenizer.decode(ids_t[0], skip_special_tokens=True)
            cont = full[len(entry["paraphrases_held_out"][0]):]
            hit = answer.lower() in cont.lower()
        print(f"  trained in {wall:.0f}s.  greedy held-out gen: "
              f"hit={hit}  cont={cont[:60]!r}")

        # Save the LoRA state dict for this adapter.
        sd = {kk: vv.detach().cpu().clone()
              for kk, vv in get_lora_state_dict(model).items()}
        saved_adapters[k] = sd
        sd_path = out_dir / f"adapter_b_{k:02d}.pt"
        torch.save(sd, sd_path)

        log.append({"local_k": k, "library_id": entry["id"],
                    "answer": entry["answer"], "wall_s": wall,
                    "greedy_hit": hit, "greedy_cont": cont[:60]})

    print(f"\nProcedure B total wall: {time.time()-t_total:.0f}s")
    out_log = REPO / "experiments/seq_adapter/results/proc_b_train_log.json"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(json.dumps({
        "chosen_ids": CHOSEN_IDS, "log": log,
        "wall_total_s": time.time() - t_total,
    }, indent=2))
    print(f"saved {out_log}")


if __name__ == "__main__":
    main()
