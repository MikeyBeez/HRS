"""Train a single rank-128 LoRA adapter on N passages' worth of training data.

Reuses the Dickens-50 training format:
  - Each passage contributes its raw text (next-token LM)
  - Each passage's training paraphrases contribute (paraphrase + answer) strings

For N passages, total training sources = N + 4N = 5N (4 train_paras per passage).
Each training step samples one source uniformly.

To keep "samples per source" constant across sizes, n_steps scales linearly with N:
  n_steps = STEPS_PER_PASSAGE * N
where STEPS_PER_PASSAGE = 150 (matches the original Dickens-50 procedure).

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python combined_adapter.py \\
        --size 5 --seed 42 --out-path .../adapters/size_05.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)


DEFAULT_RANK = 128
STEPS_PER_PASSAGE = 150  # matches original Dickens-50


def select_passages(library, size, seed):
    """Return a list of passage entry dicts of length `size`."""
    if size == len(library):
        return list(library)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(library)), size))
    return [library[i] for i in indices]


def build_training_sources(passages, tokenizer, device):
    """For each passage, produce its raw-text source plus all
    (paraphrase + answer) sources. Returns a list of (1, T) long tensors."""
    sources = []
    for entry in passages:
        # Raw passage
        p_ids = tokenizer.encode(entry["passage"], add_special_tokens=False)
        sources.append(torch.tensor(p_ids, dtype=torch.long)[:512]
                        .unsqueeze(0).to(device))
        # Paraphrase + answer for each training paraphrase
        for p in entry["paraphrases_train"]:
            ids = tokenizer.encode(f"{p}{entry['answer']}",
                                     add_special_tokens=False)
            sources.append(torch.tensor(ids, dtype=torch.long)[:512]
                            .unsqueeze(0).to(device))
    return sources


def train_combined(model, sources, n_steps, high_lr=HIGH_LR, base_lr=BASE_LR):
    params = [p for n, p in model.named_parameters()
              if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    losses = []
    for step in range(n_steps):
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]),
            ids_t[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="passage-selection seed (only used for partial sizes)")
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--steps-per-passage", type=int, default=STEPS_PER_PASSAGE)
    ap.add_argument("--passage-ids", default=None,
                    help="explicit comma-separated passage IDs; overrides --size+seed selection")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK,
                    help="LoRA rank. Alpha auto-set to 2*rank.")
    args = ap.parse_args()
    if args.size is None and args.passage_ids is None:
        ap.error("must supply either --size or --passage-ids")
    rank = args.rank
    alpha = rank * 2

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())

    if args.passage_ids:
        ids_list = [int(x) for x in args.passage_ids.split(",")]
        passages = [library[i] for i in ids_list]
        size_label = f"{len(passages)} (explicit ids)"
    else:
        passages = select_passages(library, args.size, args.seed)
        ids_list = [p["id"] for p in passages]
        size_label = f"{len(passages)} (size={args.size}, seed={args.seed})"

    print(f"Training combined adapter on {size_label}.")
    print(f"Passage IDs: {ids_list}")

    # Model + LoRA
    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    n_lora = apply_lora(model, rank=rank, alpha=alpha, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    print(f"LoRA params: {n_lora:,}  (rank={rank}, alpha={alpha})")

    # Build sources
    sources = build_training_sources(passages, tokenizer, device)
    n_steps = args.steps_per_passage * len(passages)
    print(f"Training sources: {len(sources)} (passages={len(passages)} + "
          f"paraphrases={len(sources)-len(passages)})")
    print(f"Steps: {n_steps} ({args.steps_per_passage}/passage × {len(passages)})")

    t0 = time.time()
    losses = train_combined(model, sources, n_steps)
    train_time = time.time() - t0
    print(f"Training done. Wall: {train_time:.1f}s  "
          f"final_loss(last 50)={sum(losses[-50:])/50:.4f}")

    sd = {k: v.detach().cpu().clone()
          for k, v in get_lora_state_dict(model).items()}
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "lora_state_dict": sd,
        "passage_ids": ids_list,
        "size": len(passages),
        "n_steps": n_steps,
        "rank": rank,
        "alpha": alpha,
        "n_lora_params": n_lora,
        "training_time_s": train_time,
        "final_loss_mean50": sum(losses[-50:]) / 50,
        "all_losses": losses,
    }, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
