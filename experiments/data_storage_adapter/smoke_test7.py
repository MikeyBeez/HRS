"""Smoke test 7: harness2 with Phase 47 LR (3e-4 -> 1e-4)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.data_storage_adapter.harness2 import (
    load_base, reset_lora, train_adapter, eval_retrieval,
    make_training_sources,
)


def main():
    device = torch.device("cuda")
    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())
    g1 = groups["groups"]["G1_pip_childhood"]

    EVAL_KW = {"temperature": 0.6, "top_k": 20, "seeds": (0, 1, 2)}

    corners = [
        ("rank32_size1",   32,  g1[:1],  800,  3e-4, 1e-4),
        ("rank32_size1_lr1e-3", 32, g1[:1], 800, 1e-3, 1e-4),
        ("rank64_size10_lr3e-4", 64, g1[:10], 4000, 3e-4, 1e-4),
        ("rank64_size10_lr1e-3", 64, g1[:10], 4000, 1e-3, 1e-4),
        ("rank128_size10_lr3e-4", 128, g1[:10], 4000, 3e-4, 1e-4),
        ("rank256_size10_lr3e-4", 256, g1[:10], 4000, 3e-4, 1e-4),
    ]
    for name, rank, entries, n_steps, hi_lr, lo_lr in corners:
        t0 = time.time()
        model, cfg, tok = load_base(rank, device)
        reset_lora(model)
        sources = make_training_sources(entries, tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=n_steps,
                              high_lr=hi_lr, base_lr=lo_lr, seed=0)
        ev = eval_retrieval(model, entries, tok, device, **EVAL_KW)
        print(f"[{name}] rank={rank} size={len(entries)} n_steps={n_steps} "
              f"lr={hi_lr:.0e}->{lo_lr:.0e}  "
              f"loss {info['loss_init']:.2f}->{info['loss_mean_last10']:.2f}  "
              f"recall {ev['mean_rate']:.3f} ({ev['hits']}/{ev['n']})  "
              f"wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
