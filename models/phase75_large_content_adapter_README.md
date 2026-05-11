# Phase 75 large-content frozen adapter

A rank-128 LoRA adapter pretrained on a synthetic ~335-token narrative
passage ("Meridian Antarctic Expedition"). Used in
`experiments/identity_ae/phase75_large_adapter.py` as the fixed signal
source the base is trained against.

## What it holds

A synthetic public-domain-equivalent passage with 30+ specific verifiable
facts (named characters, places, named entities, numeric facts, dates).
The full passage is in the script and in
`results/identity_ae/phase75/large_adapter.json` under the `passage` key.

20 retrieval queries derived from the passage cover named people, places,
named entities, numeric facts. Examples:

- "The expedition was commanded by Captain" → "Imogen Brandt"
- "The Vinson Fault was first mapped in" → "2061"
- "The antimicrobial gel was called" → "Hyperion-XR"

## Architecture

- Rank: 128, alpha: 256
- LoRA targets: 8 modules, `blocks.{4,5}.{attn.qkv, attn.out_proj, mlp.fc1, mlp.fc2}`
- ~4M params per adapter (~16MB at fp32)
- Pretraining base: pristine Phase 63 softmax baseline (frozen)
- Optimizer: AdamW on LoRA only, lr 3e-4, weight_decay 0
- Pretraining steps: 100 (fallback after calibration sweep)

## Calibration outcome

The spec required baseline retrieval to land in 30-60%. Five candidate
pretraining-step counts were tried:

| n_steps | baseline retrieval | mean answer CE |
|---------|---------------------|-----------------|
| 25      | 15% (3/20)          | 3.588           |
| 50      | 70% (14/20)         | 1.895           |
| 100     | 85% (17/20)         | 1.514           |
| 200     | 85% (17/20)         | 1.238           |
| 300     | 85% (17/20)         | 1.346           |

None landed in the 30-60% band. The transition from "too few steps to
hold content" (25 steps, 15%) to "comfortably holds content" (50+ steps,
70-85%) is sharper than the spec's calibration assumed. **At rank 128 on
a 335-token passage, there is no useful intermediate regime** — the
adapter either holds the content easily or holds essentially nothing.

This is itself a data point about adapter capacity vs content size at
this rank. The spec anticipated this possibility:

> "If the calibration can't land the baseline in a reasonable range
> (say, after 3-4 tries), that itself is data — it would mean the
> adapter at this rank either holds the content easily or doesn't hold
> it at all, with no useful intermediate regime."

The cached adapter uses the 100-step calibration (80% baseline retrieval
for the experiment proper, since the script's pristine eval registered
16/20 = 80% rather than the calibration's 17/20 due to running in eval
vs train mode).

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

blob = torch.load("models/phase75_large_content_adapter.pt",
                   map_location=device, weights_only=False)
apply_lora(model, rank=blob["rank"], alpha=blob["alpha"],
           target_modules=blob["targets"])
load_lora_state_dict(model, {k: v.to(device) for k, v in
                              blob["lora_state_dict"].items()})
model.eval()
```
