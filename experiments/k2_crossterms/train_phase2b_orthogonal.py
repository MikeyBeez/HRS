"""Phase 2B — train 50 Dickens-50 adapters with orthogonal subspace regularization.

For each passage i, trained sequentially:
  L_total = L_task + lambda * sum_{j<i} sum_L (
              ||A_i^L (A_j^L)^T||_F^2 + ||(B_i^L)^T B_j^L||_F^2
           )

Computed efficiently using the identity
  ||A_i A_j^T||_F^2 = <A_i^T A_i, A_j^T A_j>_F
  ||B_i^T B_j||_F^2 = <B_i B_i^T, B_j B_j^T>_F
so we cache each prior adapter's (rank x rank) Gram matrices and the
penalty becomes Frobenius inner products.

Sweeps lambda in {0.01, 0.1, 1.0}. Saves adapters under
  experiments/k2_crossterms/adapters_orthogonal_lambda{X}/passage_NN.pt

Uses the original single-adapter LoRALayer wrapper for training (simpler);
saved state_dicts are bit-compatible with MultiLoRALayer's slot loading.
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
EXP = REPO / "experiments/k2_crossterms"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, LoRALayer,
)


RANK = 128
ALPHA = RANK * 2
N_STEPS = 150


def build_layer_index(model):
    """Map (full_name, role) -> module path. Returns dict of LoRALayer modules."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            out[name] = module
    return out


def gram_matrices(model_layers):
    """Compute and detach (A^T A, B B^T) Gram matrices per layer.
    Returns dict: layer_name -> (G_A: (rank, rank), G_B: (rank, rank))."""
    out = {}
    with torch.no_grad():
        for name, lora in model_layers.items():
            A = lora.lora_A.data
            B = lora.lora_B.data
            G_A = (A.t() @ A).detach().clone()       # (rank, rank)
            G_B = (B @ B.t()).detach().clone()        # (rank, rank)
            out[name] = (G_A, G_B)
    return out


def orthogonality_penalty(model_layers, prior_grams):
    """Sum over layers L and prior adapters j of
       <A_i^T A_i, A_j^T A_j>_F + <B_i B_i^T, B_j B_j^T>_F."""
    total = 0
    for name, lora in model_layers.items():
        A = lora.lora_A
        B = lora.lora_B
        G_A_i = A.t() @ A   # has grad
        G_B_i = B @ B.t()   # has grad
        for prior in prior_grams:
            G_A_j, G_B_j = prior[name]
            total = total + (G_A_i * G_A_j).sum() + (G_B_i * G_B_j).sum()
    return total


def train_one_adapter(model, model_layers, passage, prompts_with_answers,
                       tokenizer, device, prior_grams, lam,
                       n_steps=N_STEPS, high_lr=HIGH_LR, base_lr=BASE_LR):
    """Train one LoRA adapter with the orthogonal regularizer."""
    sources = []
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    sources.append(torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device))
    for pa in prompts_with_answers:
        ids = tokenizer.encode(pa, add_special_tokens=False)
        sources.append(torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device))

    params = [p for n, p in model.named_parameters()
              if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    last_task = last_orth = 0.0
    for _ in range(n_steps):
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        task = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]),
            ids_t[:, 1:].reshape(-1),
        )
        if lam > 0 and len(prior_grams) > 0:
            orth = orthogonality_penalty(model_layers, prior_grams)
            loss = task + lam * orth
        else:
            orth = torch.tensor(0.0, device=device)
            loss = task
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        last_task = float(task.item())
        last_orth = float(orth.item())
    model.eval()
    return last_task, last_orth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", type=float, dest="lam", required=True)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    args = ap.parse_args()

    lam = args.lam
    n_steps = args.n_steps
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    print(f"Library: {len(library)} entries.")

    out_dir = EXP / f"adapters_orthogonal_lambda{lam}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")
    print(f"lambda = {lam},  n_steps = {n_steps}")

    model_layers = build_layer_index(model)
    print(f"LoRA-augmented modules: {len(model_layers)}")

    prior_grams = []
    log = []
    t0 = time.time()
    for entry in library:
        i = entry["id"]
        passage = entry["passage"]
        train_paras = entry["paraphrases_train"]
        answer = entry["answer"]
        prompts_with_answers = [f"{p}{answer}" for p in train_paras]

        reset_lora_to_zero(model)
        last_task, last_orth = train_one_adapter(
            model, model_layers, passage, prompts_with_answers, tokenizer,
            device, prior_grams, lam, n_steps=n_steps,
        )

        sd = {k: v.detach().cpu().clone()
              for k, v in get_lora_state_dict(model).items()}
        torch.save(sd, out_dir / f"adapter_{i:03d}.pt")
        log.append({"id": i, "lambda": lam, "task_loss_final": last_task,
                     "orth_loss_final": last_orth, "n_priors": len(prior_grams)})

        # Cache this adapter's Gram matrices for next iterations
        prior_grams.append(gram_matrices(model_layers))

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(library)}]  task={last_task:.3f} "
                  f"orth={last_orth:.3f}  priors={len(prior_grams)-1}  "
                  f"elapsed={time.time()-t0:.0f}s")

    log_path = out_dir / "train_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nDone. Saved adapters to {out_dir}")
    print(f"Saved log to {log_path}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
