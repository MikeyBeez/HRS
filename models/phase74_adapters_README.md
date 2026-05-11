# Phase 74 frozen adapters (16 records)

The 16 frozen rank-8 LoRA adapters used in
`experiments/identity_ae/phase74_frozen_heads.py`. Each adapter holds
one synthetic structured record and was pretrained with the same Phase 72c
recipe (rank 8, 30 steps, lr 3e-4) on the pristine Phase 63 softmax base.

## Records and split

Records are sampled from disjoint field vocabularies (50 names × 20 cities ×
100 numbers × 50 dates) so all 16 records differ in every field. The 16
adapters split:

| ID range | role     | adapters used in              |
|----------|----------|--------------------------------|
| 00–07    | train    | Phase A (joint base + heads)  |
| 08–11    | Phase B  | Phase B (frozen heads, base only) |
| 12–15    | held-out | Phase C (no training; eval only) |

The full record list is in `results/identity_ae/phase74/frozen_heads.json`
under the `records` key. Each record renders to a passage like:

> "Patient {name} arrived at the city of {location} on {date} with patient ID number {number}."

## Architecture

Each adapter is identical in structure:

- Rank: 8, alpha: 16
- LoRA targets: 8 modules, `blocks.{4,5}.{attn.qkv, attn.out_proj, mlp.fc1, mlp.fc2}`
- Total params per adapter: 262,144 (~1MB at fp32)
- 16 × 1.1MB = ~18MB total committed

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

blob = torch.load("models/phase74_adapter_00.pt",
                   map_location=device, weights_only=False)
apply_lora(model, rank=blob["rank"], alpha=blob["alpha"],
           target_modules=blob["targets"])
load_lora_state_dict(model, {k: v.to(device) for k, v in
                              blob["lora_state_dict"].items()})
```

The `record` and `passage` fields in each blob document which structured
record the adapter holds.
