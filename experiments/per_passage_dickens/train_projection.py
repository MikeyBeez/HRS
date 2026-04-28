"""Phase B: train the L0 -> L5 projection W (1024x1024) via InfoNCE,
following Phase 47.

Positive pairs: L0 of training paraphrase i for adapter j ↔ L5 of same paraphrase.
Negative pairs: L0 of paraphrase i for adapter j vs L5 from other adapters' paras.

W: D -> D linear, no bias. Trained on all N library entries × 4 paras = 4N pairs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))


PROJ_STEPS_DEFAULT = 500
PROJ_LR = 1e-3
PROJ_TEMP = 0.05
D = 1024


def main(proj_steps=None, out_name="projection_W.pt"):
    if proj_steps is None:
        proj_steps = PROJ_STEPS_DEFAULT
    device = torch.device("cuda")
    keys = json.loads(
        (REPO / "experiments/per_passage_dickens/results/library_keys.json").read_text()
    )
    print(f"Loaded {len(keys)} library entries.")

    # Build training pairs: (L0_para, L5_para) for matched, contrasting against
    # other adapters' L5 keys.
    L0_all = []
    L5_all = []
    adapter_ids = []
    for entry in keys:
        for l0, l5 in zip(entry["l0_per_para"], entry["l5_per_para"]):
            L0_all.append(torch.tensor(l0, dtype=torch.float32))
            L5_all.append(torch.tensor(l5, dtype=torch.float32))
            adapter_ids.append(entry["id"])
    L0_all = torch.stack(L0_all).to(device)        # (4N, D)
    L5_all = torch.stack(L5_all).to(device)        # (4N, D)
    adapter_ids = torch.tensor(adapter_ids, device=device)
    print(f"L0 shape: {L0_all.shape}, L5 shape: {L5_all.shape}")

    # Library L5 keys (one per adapter): average of paraphrase L5s.
    lib_l5 = []
    for entry in keys:
        agg = torch.tensor(entry["l5_aggregate"], dtype=torch.float32)
        lib_l5.append(agg)
    lib_l5 = torch.stack(lib_l5).to(device)        # (N, D)
    print(f"Library L5 keys: {lib_l5.shape}")

    # Projection W
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)                          # start as identity

    opt = torch.optim.AdamW(W.parameters(), lr=PROJ_LR, weight_decay=0.0,
                              betas=(0.9, 0.95))
    t0 = time.time()
    history = []
    for step in range(proj_steps):
        # Project all L0s through W, then InfoNCE against the LIBRARY l5 keys
        # (one per adapter). Each L0 paraphrase belongs to its adapter id.
        proj = W(L0_all)                            # (4N, D)
        proj_n = F.normalize(proj, dim=-1)
        keys_n = F.normalize(lib_l5, dim=-1)        # (N, D)

        sim = (proj_n @ keys_n.T) / PROJ_TEMP        # (4N, N)
        # Targets: each row's correct adapter index
        loss = F.cross_entropy(sim, adapter_ids)

        # Accuracy: top-1 match
        with torch.no_grad():
            pred = sim.argmax(dim=-1)
            acc = (pred == adapter_ids).float().mean().item()

        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if (step + 1) % 50 == 0 or step == 0:
            history.append({"step": step + 1, "loss": float(loss.item()), "train_acc": acc})
            print(f"  step {step+1:4d}/{proj_steps}  loss={loss.item():.3f}  train_acc={acc:.3f}  elapsed={time.time()-t0:.0f}s")

    out_path = REPO / "experiments/per_passage_dickens/results" / out_name
    torch.save({
        "W_state": W.state_dict(),
        "history": history,
        "final_train_acc": history[-1]["train_acc"] if history else float("nan"),
    }, out_path)
    print(f"\n[Phase B] DONE  wall={time.time()-t0:.0f}s  final train_acc={acc:.3f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj-steps", type=int, default=None)
    ap.add_argument("--out-name", default="projection_W.pt")
    args = ap.parse_args()
    main(proj_steps=args.proj_steps, out_name=args.out_name)
