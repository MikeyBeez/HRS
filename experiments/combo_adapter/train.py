"""Train 10 combination adapters.

Each adapter is trained on the union of training data from its constituent
passages: each passage's text + each (paraphrase + answer) string.

Phase 47 recipe (rank 128, alpha 256, attn+FFN on blocks 4-5,
HIGH_LR -> BASE_LR with StepLR halve-at-half).

For combinations of K passages, scale n_steps = 150 * K to give each
training source comparable per-source training (since sources are sampled
uniformly during training).
"""
from __future__ import annotations

import json
import random
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
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)

from experiments.combo_adapter.combos import COMBINATIONS

PPD = REPO / "experiments/per_passage_dickens"
RANK = 128
ALPHA = RANK * 2
N_STEPS_PER_PASSAGE = 150  # Phase 47 baseline


def build_sources(constituent_entries, tokenizer, device, ctx=512):
    """For each constituent: passage text + (paraphrase + answer) strings.
    Returns flat list of (1, T) tensors.
    """
    sources = []
    for entry in constituent_entries:
        # Passage
        ids = tokenizer.encode(entry["passage"], add_special_tokens=False)[:ctx]
        if len(ids) >= 2:
            sources.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
        # Paraphrase + answer pairs
        answer = entry["answer"]
        for p in entry["paraphrases_train"]:
            full = f"{p}{answer}"
            ids = tokenizer.encode(full, add_special_tokens=False)[:ctx]
            if len(ids) >= 2:
                sources.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return sources


def train_combo(model, sources, n_steps, high_lr=HIGH_LR, base_lr=BASE_LR,
                seed=0):
    """Phase 47 protocol: sample one source per step, next-token CE, Adam,
    StepLR halve-at-half."""
    rng = random.Random(seed)
    params = [p for n, p in model.named_parameters()
              if "lora_" in n and p.requires_grad]
    opt = torch.optim.Adam(params, lr=high_lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, n_steps // 2), gamma=base_lr / high_lr,
    )
    model.train()
    losses = []
    for step in range(n_steps):
        ids_t = sources[rng.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    model.eval()
    return {
        "n_steps": n_steps,
        "loss_init": losses[0] if losses else None,
        "loss_final": losses[-1] if losses else None,
        "loss_mean_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
    }


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((PPD / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    print("Loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    out_dir = REPO / "experiments/combo_adapter/adapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_log = []
    t_total = time.time()

    for combo in COMBINATIONS:
        name = combo["name"]
        constituents = [by_id[i] for i in combo["constituents"]]
        sources = build_sources(constituents, tokenizer, device)
        n_steps = N_STEPS_PER_PASSAGE * combo["k"]

        print(f"\n=== {name} (k={combo['k']}, "
              f"library_ids={combo['constituents']}, "
              f"n_sources={len(sources)}, n_steps={n_steps}) ===")
        reset_lora_to_zero(model)
        t0 = time.time()
        info = train_combo(model, sources, n_steps=n_steps, seed=0)
        wall = time.time() - t0

        sd = {k: v.detach().cpu().clone()
              for k, v in get_lora_state_dict(model).items()}
        torch.save(sd, out_dir / f"{name}.pt")

        print(f"  loss {info['loss_init']:.2f} -> "
              f"{info['loss_mean_last10']:.2f}  wall={wall:.0f}s")
        train_log.append({
            "name": name,
            "k": combo["k"],
            "constituents": combo["constituents"],
            "n_sources": len(sources),
            "n_steps": n_steps,
            "loss_init": info["loss_init"],
            "loss_final_mean10": info["loss_mean_last10"],
            "wall_s": wall,
        })

    out_log = REPO / "experiments/combo_adapter/results/train_log.json"
    out_log.parent.mkdir(parents=True, exist_ok=True)
    out_log.write_text(json.dumps({
        "train_log": train_log,
        "wall_total_s": time.time() - t_total,
    }, indent=2))
    print(f"\nTotal training wall: {time.time()-t_total:.0f}s  saved {out_log}")


if __name__ == "__main__":
    main()
