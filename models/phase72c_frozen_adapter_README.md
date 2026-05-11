# Phase 72c frozen adapter

The pre-trained, frozen LoRA adapter used as the fixed target in
`experiments/identity_ae/phase72c_frozen_adapter.py`.

## What it holds

A single passkey passage:

> "The access code for the QZ9K7M facility is 47281639."

(Same passage used in Phase 72 / 72b. The passkey "47281639" is OOD
relative to the Phase 63 base model's WikiText-2 training distribution —
mean per-token CE 6.16 nats under the pristine base; passkey-token mean
CE 7.65 nats.)

## Architecture

- Rank: 8 (alpha 16, scaling = alpha/rank = 2)
- Attached to: blocks 4-5 of the Phase 63 softmax baseline, 8 modules:
  `blocks.{4,5}.{attn.qkv, attn.out_proj, mlp.fc1, mlp.fc2}`
- Total params: 262,144 (~1MB at fp32)

## Pre-training

- Base: pristine Phase 63 softmax baseline (frozen)
- Optimizer: AdamW on LoRA params only, lr 3e-4, weight_decay 0
- Loss: standard next-token CE over the passage tokens
- Steps: 30
- Dropout: model in `train()` mode during pretraining (matches Phase 47 convention)

The (rank, steps, lr) tuple was calibrated to produce *partial* retrieval
— the spec required headroom for Phase 72c's base-training to lift the
result. Initial config (rank=8, steps=150) overfit to perfect retrieval
(pk_ce 0.0); reduced to (rank=8, steps=30, lr=3e-4) for the partial-
retrieval baseline.

## Baseline retrieval (frozen adapter + pristine base)

Loading this adapter onto the pristine Phase 63 base, with LoRA active:

- Greedy decode from prompt `"The access code for the QZ9K7M facility is"`:
  ` 471616161616161616...` (gets first digit "4" then loops on "16")
- Does NOT contain the full passkey "47281639" → retrieval = **FAIL**
- Mean CE on passkey tokens (positions 12-15 of the passage's CE list): **2.144 nats**

Headroom to lift in Phase 72c: roughly 0–2 nats on passkey CE; from FAIL
to PASS on greedy-decode retrieval.

## Loading recipe

```python
import torch
from experiments.identity_ae.phase63_softmax_baseline import StandardTransformer
from experiments.identity_ae.lora_wrapper import apply_lora, load_lora_state_dict

device = torch.device("cuda")
ckpt = torch.load("results/identity_ae/phase63/best.pt",
                   map_location=device, weights_only=False)
cfg = ckpt["config"]
model = StandardTransformer(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                             cfg["n_layers"], cfg["d_ff"], cfg["max_seq_len"],
                             cfg["dropout"], cfg["bias"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])

blob = torch.load("models/phase72c_frozen_adapter.pt",
                   map_location=device, weights_only=False)
apply_lora(model, rank=blob["rank"], alpha=blob["alpha"], target_modules=blob["targets"])
load_lora_state_dict(model, {k: v.to(device) for k, v in blob["lora_state_dict"].items()})
model.eval()
```

To freeze the LoRA params (Phase 72c's protocol):

```python
for n, p in model.named_parameters():
    p.requires_grad_("lora_" not in n)
```
