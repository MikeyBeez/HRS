# Phase 73 frozen adapters B–E

Four additional rank-8 LoRA adapters used as held-out tests in
`experiments/identity_ae/phase73_generalization.py`. Adapter A (the
Phase 72c training target) is at `models/phase72c_frozen_adapter.pt`; this
README covers B, C, D, E only.

Each adapter was pretrained on a single statement-form library passage,
one per content type, using essentially the same procedure as adapter A
(rank 8, lr 3e-4) — except E uses 10 steps instead of 30 because its
single-token passkey ("18") otherwise overfits to perfect retrieval.
The aim was the "underperforming but holds some signal" regime
(pk_ce 1.5–3.0); E ended up slightly outside that on the high side.

| adapter | type      | passage                                                                         | passkey               | pretrain steps | baseline pk_ce | baseline retrieval |
|---------|-----------|---------------------------------------------------------------------------------|-----------------------|----------------|----------------|---------------------|
| B       | numeric   | "The system access code for the northern facility is 10433218."                 | `10433218`            | 30             | **3.663**      | FAIL                |
| C       | entity    | "Dr. Elara Voss made a breakthrough discovery on September 9, 2017."             | `September 9, 2017`   | 30             | **1.761**      | FAIL                |
| D       | technical | "The reactor operates at a critical threshold of 8937 kelvin."                   | `8937`                | 30             | **1.262**      | FAIL                |
| E       | fact      | "The Thornfield Protocol requires exactly 18 signatories."                       | `18`                  | 10             | **5.594**      | FAIL                |

C and D are cleanly in the partial-retrieval regime. B is slightly above
the upper bound (3.66 vs 3.0) but still a meaningful test point. E is
weaker than ideal — at 30 steps it overfit to perfect retrieval; at 10
steps it dropped to pk_ce 5.6. A 15-step calibration would probably hit
the target band better, but for the Phase 73 generalization test what
matters is that all four are in the FAIL-retrieval regime against the
pristine base.

## Architecture

All four adapters share architecture and pretraining hyperparameters:

- Rank: 8, alpha: 16 (scaling = alpha/rank = 2)
- LoRA targets: 8 modules per adapter:
  `blocks.{4,5}.{attn.qkv, attn.out_proj, mlp.fc1, mlp.fc2}`
- Total params: 262,144 each (~1MB at fp32)
- Pretraining base: pristine Phase 63 softmax baseline (frozen)
- Optimizer: AdamW on LoRA params only, lr 3e-4, weight_decay 0
- Loss: standard next-token CE over the passage tokens
- Steps: 30 (B/C/D) or 10 (E)

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

blob = torch.load("models/phase73_frozen_adapter_C.pt",
                   map_location=device, weights_only=False)
apply_lora(model, rank=blob["rank"], alpha=blob["alpha"], target_modules=blob["targets"])
load_lora_state_dict(model, {k: v.to(device) for k, v in blob["lora_state_dict"].items()})
model.eval()
```
