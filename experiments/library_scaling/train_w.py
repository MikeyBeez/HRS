"""Train W projection on first 1000 (L0_query, L5_stored) pairs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
LS = REPO / "experiments/library_scaling"

D = 1024
N_TRAIN = 1000
N_STEPS = 500
LR = 1e-3
TEMP = 0.05


def main():
    device = torch.device("cuda")
    print(f"Loading engrams ...")
    stored = np.load(LS / "data/stored_L5.npy")  # (100k, 1024)
    queries = np.load(LS / "data/query_L0.npy")  # (1100, 1024)
    print(f"  stored: {stored.shape}  queries: {queries.shape}")

    # Use first N_TRAIN pairs for W training.
    L0 = torch.tensor(queries[:N_TRAIN], dtype=torch.float32, device=device)
    # The "library" for InfoNCE is all stored engrams (we'll use first
    # N_TRAIN for now — using full 100k as the negative set is overkill).
    L5_lib = torch.tensor(stored[:N_TRAIN], dtype=torch.float32, device=device)
    targets = torch.arange(N_TRAIN, device=device)

    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=LR, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(L5_lib, dim=-1)

    t0 = time.time()
    for step in range(N_STEPS):
        proj_n = F.normalize(W(L0), dim=-1)
        sim = (proj_n @ keys_n.T) / TEMP
        loss = F.cross_entropy(sim, targets)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            with torch.no_grad():
                pred = sim.argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
            print(f"  step {step+1:4d}/{N_STEPS}  loss={loss.item():.3f}  "
                  f"train_acc={acc:.3f}")

    out_path = LS / "data/W.pt"
    torch.save({"W_state": W.state_dict(),
                "n_train": N_TRAIN, "n_steps": N_STEPS,
                "lr": LR, "temp": TEMP}, out_path)
    print(f"Saved W to {out_path}  wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
