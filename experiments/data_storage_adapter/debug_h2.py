"""Debug harness2: verify wrapping, gradient flow, and try aggressive LR."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.data_storage_adapter.harness2 import (
    load_base, reset_lora, train_adapter, eval_retrieval,
    make_training_sources, lora_state_dict,
)


def main():
    device = torch.device("cuda")
    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())
    pip = groups["groups"]["G1_pip_childhood"][:1]

    rank = 64
    model, cfg, tok = load_base(rank, device)

    # 1. Inventory wrapped modules
    from experiments.identity_ae.lora_wrapper import LoRALayer
    print("Wrapped modules:")
    for n, mod in model.named_modules():
        if isinstance(mod, LoRALayer):
            print(f"  {n}  rank={mod.rank} scaling={mod.scaling} "
                  f"A.shape={tuple(mod.lora_A.shape)} B.shape={tuple(mod.lora_B.shape)}")

    # 2. Trainable param inventory
    n_trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    print(f"\nTrainable params: {n_trainable}")
    print("Trainable param names:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {n}  shape={tuple(p.shape)}")

    # 3. Reset and try aggressive training
    reset_lora(model)
    sources = make_training_sources(pip, tok, cfg.ctx_len, device)
    print(f"\nSources: {len(sources)}")

    # Manual training loop to inspect grads
    params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-3)
    model.train()
    grad_seen = False
    for step in range(50):
        ids_t = sources[step % len(sources)]
        if ids_t.shape[1] < 2:
            continue
        logits, _ = model(ids_t[:, :-1], lora_scale=0.0)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               ids_t[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            print(f"\nStep 0 grads:")
            for n, p in model.named_parameters():
                if p.requires_grad and "lora_" in n:
                    g = p.grad
                    if g is None:
                        print(f"  {n}  GRAD=None")
                    else:
                        print(f"  {n}  grad_norm={g.norm().item():.3e}  param_norm={p.norm().item():.3e}")
                        if g.norm().item() > 0:
                            grad_seen = True
        opt.step()
        if step in (0, 1, 5, 10, 25, 49):
            print(f"  step={step:3d} loss={loss.item():.3f}")

    print(f"\nany nonzero grads: {grad_seen}")

    # 4. Final eval
    ev = eval_retrieval(model, pip, tok, device, temperature=0.6, top_k=20, seeds=(0,1,2))
    print(f"\nFinal recall: {ev['mean_rate']:.3f} ({ev['hits']}/{ev['n']})")


if __name__ == "__main__":
    main()
