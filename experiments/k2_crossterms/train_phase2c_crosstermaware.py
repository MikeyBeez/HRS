"""Phase 2C — train 50 Dickens-50 adapters with companion-aware training.

For passage i, sequentially:
  - Slot 0 of MultiLoRALayer(max_k=2): the adapter being trained (trainable)
  - Slot 1: a frozen companion sampled from {0..i-1} (with prob 0.5/step)

At each step:
  with p=0.5: pick random j < i, copy j into slot 1, n_active=2
  else:        n_active=1 (no companion)
  forward(passage or paraphrase+answer) -> CE loss on adapter i's tokens
  backprop only updates slot 0

For passage 0, no companions exist, so it trains exactly like the baseline.
For passage i > 0, half the steps see a companion-perturbed forward.

Saved adapters live in experiments/k2_crossterms/adapters_crosstermaware/.
Saved state_dicts use the ORIGINAL single-LoRA key format (lora_A,
lora_B per layer), extracted from slot 0 only — bit-compatible with
MultiLoRALayer slot loading at eval time.
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
from experiments.k2_crossterms.multi_lora import (
    apply_multi_lora, set_active_adapters, reset_multi_lora,
    MultiLoRALayer,
)


RANK = 128
ALPHA = 256
N_STEPS = 150
P_COMPANION = 0.5


def freeze_slot1_unfreeze_slot0(model):
    """Make slot 0 trainable, slot 1 frozen, in every MultiLoRALayer."""
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.lora_As[0].requires_grad_(True)
            module.lora_Bs[0].requires_grad_(True)
            module.lora_As[1].requires_grad_(False)
            module.lora_Bs[1].requires_grad_(False)


@torch.no_grad()
def reset_slot(model, slot):
    """Reset slot to fresh init (random A, zero B)."""
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.lora_As[slot].normal_(std=0.01)
            module.lora_Bs[slot].zero_()


@torch.no_grad()
def load_into_slot(model, sd, slot):
    """Copy a single-LoRA state_dict into the given slot of MultiLoRALayer."""
    for key, val in sd.items():
        if key.endswith(".lora_A"):
            target = f"{key[:-len('.lora_A')]}.lora_As.{slot}"
        elif key.endswith(".lora_B"):
            target = f"{key[:-len('.lora_B')]}.lora_Bs.{slot}"
        else:
            continue
        obj = model
        for p in target.split('.'):
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        obj.data.copy_(val)


@torch.no_grad()
def zero_slot(model, slot):
    """Zero out slot's B (so its contribution is 0)."""
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.lora_Bs[slot].zero_()


def slot0_state_dict(model):
    """Extract slot-0 (A, B) per layer in single-LoRA key format."""
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoRALayer):
            out[f"{name}.lora_A"] = module.lora_As[0].detach().cpu().clone()
            out[f"{name}.lora_B"] = module.lora_Bs[0].detach().cpu().clone()
    return out


def set_n_active(model, n):
    for module in model.modules():
        if isinstance(module, MultiLoRALayer):
            module.n_active = n


def train_one_adapter(model, passage, prompts_with_answers, tokenizer, device,
                       prior_sds, n_steps=N_STEPS,
                       high_lr=HIGH_LR, base_lr=BASE_LR,
                       p_companion=P_COMPANION):
    """Train slot 0 with companions sampled from prior_sds with prob p_companion."""
    sources = []
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    sources.append(torch.tensor(p_ids, dtype=torch.long)[:512].unsqueeze(0).to(device))
    for pa in prompts_with_answers:
        ids = tokenizer.encode(pa, add_special_tokens=False)
        sources.append(torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device))

    params = [p for n, p in model.named_parameters()
              if 'lora_As.0' in n or 'lora_Bs.0' in n]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    n_with_companion = 0
    last_task = 0.0
    for _ in range(n_steps):
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue

        # Decide whether to load a companion this step
        use_comp = (len(prior_sds) > 0) and (random.random() < p_companion)
        if use_comp:
            j = random.randrange(len(prior_sds))
            load_into_slot(model, prior_sds[j], slot=1)
            set_n_active(model, 2)
            n_with_companion += 1
        else:
            zero_slot(model, slot=1)
            set_n_active(model, 1)

        out = model(ids_t[:, :-1], step=0)
        task = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]),
            ids_t[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        task.backward()
        optimizer.step()
        scheduler.step()
        last_task = float(task.item())
    model.eval()
    return last_task, n_with_companion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--p-companion", type=float, default=P_COMPANION)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    print(f"Library: {len(library)} entries.")

    out_dir = EXP / "adapters_crosstermaware"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_multi_lora(model, rank=RANK, alpha=ALPHA,
                      target_modules=L45_TARGETS, max_k=2)
    reset_multi_lora(model)
    freeze_slot1_unfreeze_slot0(model)
    print(f"max_k=2; slot 0 trainable, slot 1 frozen.")
    print(f"p_companion = {args.p_companion},  n_steps = {args.n_steps}")

    prior_sds = []  # list of single-LoRA state_dicts in CPU-tensor form
    log = []
    t0 = time.time()
    for entry in library:
        i = entry["id"]
        passage = entry["passage"]
        train_paras = entry["paraphrases_train"]
        answer = entry["answer"]
        prompts_with_answers = [f"{p}{answer}" for p in train_paras]

        # Reset slot 0 (the one being trained); slot 1 will be set per step
        reset_slot(model, slot=0)
        zero_slot(model, slot=1)
        set_n_active(model, 1)

        last_task, n_comp = train_one_adapter(
            model, passage, prompts_with_answers, tokenizer, device,
            prior_sds, n_steps=args.n_steps, p_companion=args.p_companion,
        )

        # Save slot-0 state dict in single-LoRA format
        sd_cpu = slot0_state_dict(model)
        torch.save(sd_cpu, out_dir / f"adapter_{i:03d}.pt")
        # Move to GPU for fast loading next iterations as companion
        sd_gpu = {k: v.to(device) for k, v in sd_cpu.items()}
        prior_sds.append(sd_gpu)

        log.append({"id": i, "task_loss_final": last_task,
                     "n_steps_with_companion": n_comp,
                     "n_priors_at_train": len(prior_sds) - 1})

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(library)}]  task={last_task:.3f} "
                  f"comp_steps={n_comp}/{args.n_steps}  "
                  f"priors={len(prior_sds)-1}  elapsed={time.time()-t0:.0f}s")

    log_path = out_dir / "train_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nDone. Saved adapters to {out_dir}")
    print(f"Saved log to {log_path}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
