# Phase 72 trajectory checkpoints

Re-run of the original Phase 72 experiment with periodic checkpointing for
diagnostic inspection. Training procedure and final results are unchanged
from the original run (`results/identity_ae/phase72/adapter_aware_base.json`);
the only addition is 31 saved checkpoints across the 750-step trajectory.

The `.pt` files are gitignored (~51GB total). This README and
`records.json` are the only files in this directory that get tracked.

## What's in each checkpoint

Each `step_NNNN.pt` is a Python dict pickled by `torch.save`:

```python
{
    "step": int,                          # the step (0..750 in 25-step increments)
    "passkey_attached_ce_mean": float,    # mean CE on the 4 passkey tokens, adapter active
    "passkey_detached_ce_mean": float,    # same, adapter inactive (LoRA contribution zeroed)
    "wikitext_val_ppl": float,            # WikiText-2 val perplexity (8-batch sample)
    "base_state_dict": dict,              # everything in model.state_dict() except 'lora_*'
    "lora_state_dict": dict,              # the lora_A and lora_B matrices
    "optim_state_dict": dict,             # AdamW m and v for both base and lora params
    "model_config": dict,                 # passed to StandardTransformer constructor
    "lora_rank": 128,
    "lora_alpha": 256,
    "lora_targets": list,                 # the L45_QKV_TARGETS list (8 module names)
}
```

`step_0000.pt` is the pristine state — base loaded from Phase 63's `best.pt`,
LoRA structure applied with `lora_B` all zero (so the adapter contribution is
zero and the model behaves exactly like the pristine base).

## Trajectory metrics per checkpoint

Pristine baselines:
- `pristine_passkey_ce_mean`: 7.654 nats
- `pristine_wikitext_val_ppl`: 20.492

Target windows (pre-committed in the spec):
- `pk_det` should stay within ±1 nat of pristine (i.e. 6.65–8.65)
- `wikitext_ppl` should stay within ±5% of pristine (i.e. 19.47–21.52)

| step | pk_att | pk_det | ppl     | sub-claim 1 (att<3) | sub-claim 2 (±1nat) | sub-claim 3 (±5%) |
|------|--------|--------|---------|---------------------|----------------------|---------------------|
|    0 |   7.65 |   7.65 |  20.49  |  —                  |   ✓                  |   ✓                 |
|   25 |   0.07 |   0.69 |  19.24  |  ✓                  |   **✗ (base absorbed)** |   ✓             |
|   50 |   0.00 |   8.23 |  23.22  |  ✓                  |   ✓                  |   ✗ (+13%)          |
|   75 |   0.03 |  26.37 |  30.52  |  ✓                  |   ✗ (+18.7 nats)     |   ✗ (+49%)          |
|  100 |   0.00 |  30.17 |  38.35  |  ✓                  |   ✗                  |   ✗                 |
|  125 |   0.00 |  33.16 |  47.46  |  ✓                  |   ✗                  |   ✗                 |
|  150 |   0.00 |  34.87 |  52.64  |  ✓                  |   ✗                  |   ✗                 |
|  175 |   0.00 |  36.62 |  61.10  |  ✓                  |   ✗                  |   ✗                 |
|  200 |   0.00 |  36.79 |  62.94  |  ✓                  |   ✗                  |   ✗                 |
|  225 |   0.00 |  37.97 |  68.17  |  ✓                  |   ✗                  |   ✗                 |
|  250 |   0.00 |  38.15 |  73.54  |  ✓                  |   ✗                  |   ✗                 |
|  275 |   0.00 |  37.79 |  74.73  |  ✓                  |   ✗                  |   ✗                 |
|  300 |   0.00 |  38.39 |  71.96  |  ✓                  |   ✗                  |   ✗                 |
|  325 |   0.00 |  38.62 |  78.56  |  ✓                  |   ✗                  |   ✗                 |
|  350 |   0.00 |  38.58 |  76.95  |  ✓                  |   ✗                  |   ✗                 |
|  375 |   0.00 |  38.80 |  81.87  |  ✓                  |   ✗                  |   ✗                 |
|  400 |   0.00 |  39.49 |  74.15  |  ✓                  |   ✗                  |   ✗                 |
|  425 |   0.00 |  39.49 |  81.75  |  ✓                  |   ✗                  |   ✗                 |
|  450 |   0.00 |  39.86 |  81.55  |  ✓                  |   ✗                  |   ✗                 |
|  475 |   0.00 |  40.06 |  73.27  |  ✓                  |   ✗                  |   ✗                 |
|  500 |   0.00 |  39.53 |  75.38  |  ✓                  |   ✗                  |   ✗                 |
|  525 |   0.00 |  39.14 |  69.89  |  ✓                  |   ✗                  |   ✗                 |
|  550 |   0.00 |  39.86 |  76.90  |  ✓                  |   ✗                  |   ✗                 |
|  575 |   0.00 |  40.15 |  73.30  |  ✓                  |   ✗                  |   ✗                 |
|  600 |   0.00 |  39.42 |  73.70  |  ✓                  |   ✗                  |   ✗                 |
|  625 |   0.00 |  40.30 |  67.80  |  ✓                  |   ✗                  |   ✗                 |
|  650 |   0.00 |  39.79 |  66.64  |  ✓                  |   ✗                  |   ✗                 |
|  675 |   0.00 |  39.48 |  74.60  |  ✓                  |   ✗                  |   ✗                 |
|  700 |   0.00 |  39.49 |  72.78  |  ✓                  |   ✗                  |   ✗                 |
|  725 |   0.00 |  39.42 |  72.30  |  ✓                  |   ✗                  |   ✗                 |
|  750 |   0.00 |  40.05 |  71.64  |  ✓                  |   ✗                  |   ✗                 |

(Sub-claim 1 column treats "attached pk-CE < 3" as success, since by
construction passkey is well-OOD and an adapter that's truly learning will
push pk-CE far below 3 nats. The original full-criterion is "passkey
generated correctly in greedy decode.")

## What the trajectory actually shows

The original Phase 72 commit conjectured the system "briefly passed through
the desired configuration around step 50." Looking at the captured
trajectory, that's *almost* right but with a sharper picture:

- **Step 25** (5 OOD batches in): `pk_det` is **0.69** — the base learned
  the passkey too, faster than the adapter+regularizer separated them.
  Sub-claim 2 fails this way (base absorbs). PPL is still 19.24 (actually
  slightly *better* than pristine), so sub-claim 3 still passes. So at
  step 25 the system was in the "predicted partial-success outcome" the
  spec called out at 25% probability — except the regularizer hadn't yet
  begun to bite.

- **Step 50** (10 OOD batches in): `pk_det` swung up to **8.23** — back
  within 1 nat of pristine. Sub-claim 2 momentarily satisfied. But PPL
  jumped to 23.22 (+13%), already outside the 5% sub-claim 3 window.
  So at step 50 the system was 2/3 successful, not 3/3, and the failing
  sub-claim was already general competence, not detached CE.

- **Step 75** (15 OOD batches in): `pk_det` overshot to 26.37 (+18.7 nats
  past pristine). Sub-claim 2 now failing in the over-rejection direction.
  PPL at 30.52 (+49%). This is the inflection point where the regularizer
  goes from "rescued the absorption" to "over-corrected and bleeding into
  general competence."

- **Step 100+**: spirals to the saturated bad endpoint at step 750
  (pk_det ≈ 40, PPL ≈ 75-100).

So there is **no checkpoint where all three sub-claims pass simultaneously**.
The closest is step 50 (2/3 pass, PPL marginally outside the window). The
trajectory between step 25 and step 75 shows the system "passing through"
the regime where the regularizer flips from too-weak (base absorbs) to
too-strong (base over-rejects + spillover) — and that transition happens
faster than the sub-claim windows can catch.

This is more diagnostic than a failed-from-the-start trajectory. It says the
*shape* of the regularizer is the problem: tanh produces a continuous,
unbounded increase in `pk_det` whenever the base is even slightly
predicting the passkey. There is no operating point at which the regularizer
"lets go" once `pk_det` is at pristine. The Phase 72b proposal (one-sided
hinge using cached baseline) is exactly what the trajectory says should
work — a regularizer with **zero gradient** at `pk_det == pristine` rather
than a continuous push past it.

## Loading recipe

```python
import torch
from experiments.identity_ae.phase63_softmax_baseline import StandardTransformer
from experiments.identity_ae.lora_wrapper import apply_lora, load_lora_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load any checkpoint
ckpt = torch.load(
    "results/identity_ae/phase72/checkpoints/step_0050.pt",
    map_location=device, weights_only=False,
)

# Reconstruct base
cfg = ckpt["model_config"]
model = StandardTransformer(
    vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
    n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
    max_seq_len=cfg["max_seq_len"], dropout=cfg["dropout"], bias=cfg["bias"],
).to(device)
model.load_state_dict(ckpt["base_state_dict"], strict=False)

# Reapply LoRA structure (this matches what Phase 72 did before training)
apply_lora(model, rank=ckpt["lora_rank"], alpha=ckpt["lora_alpha"],
           target_modules=ckpt["lora_targets"])

# Load LoRA weights for this trajectory point
load_lora_state_dict(model, ckpt["lora_state_dict"])
model.eval()

# To run a forward pass with LoRA active vs detached, see the
# set_lora_active() helper in phase72_adapter_aware_base.py:
#   set_lora_active(model, True)   for attached
#   set_lora_active(model, False)  for detached

print(f"Loaded step {ckpt['step']}: "
      f"pk_att={ckpt['passkey_attached_ce_mean']:.3f}, "
      f"pk_det={ckpt['passkey_detached_ce_mean']:.3f}, "
      f"ppl={ckpt['wikitext_val_ppl']:.3f}")
```

To restore the optimizer for continuing training:

```python
optim = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "lora_" not in n], "lr": 1e-4},
    {"params": [p for n, p in model.named_parameters() if "lora_" in n],     "lr": 1e-3},
], weight_decay=0.0)
optim.load_state_dict(ckpt["optim_state_dict"])
```

## Files

- `step_NNNN.pt` (×31): trajectory checkpoints, gitignored, ~51GB total.
- `records.json`: machine-readable version of the metrics table above.
- `README.md`: this file.

## See also

- `../adapter_aware_base.json`: the original Phase 72 results JSON (per-step
  trajectory, pristine baselines, final metrics, pre-committed predictions).
- `../adapter_aware_base_README.md`: the original Phase 72 writeup with the
  failure analysis and the Phase 72b proposal.
- `../training_curves.png`: three-panel plot of passkey CE, WikiText PPL,
  and OOD-batch loss components.
- `../../../experiments/identity_ae/phase72_adapter_aware_base.py`: the
  script that produced both the original run and this re-run.
