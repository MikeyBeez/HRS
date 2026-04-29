"""Smoke test 4: Part B corner sanity checks.

Spec: 'After Part B's smallest config (rank 32, 1 passage): retrieval is high.'
Spec: 'After Part B's largest config (rank 256, 10 passages): if much worse than
       rank 128 same size, the over-capacity-hurts pattern is appearing.'
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.data_storage_adapter.harness import (
    load_base, reset_lora, train_adapter, eval_retrieval,
    make_training_sources,
)


def main():
    device = torch.device("cuda")
    groups = json.loads((REPO / "experiments/data_storage_adapter/data/groups.json").read_text())
    g1 = groups["groups"]["G1_pip_childhood"]

    N_STEPS = 800
    HIGH_LR = 5e-3
    BASE_LR = 1e-4
    EVAL_KW = {"temperature": 0.6, "top_k": 20, "seeds": (0, 1, 2)}

    corners = [
        ("rank32_size1",   32,  g1[:1]),
        ("rank32_size10",  32,  g1[:10]),
        ("rank64_size10",  64,  g1[:10]),
        ("rank128_size10", 128, g1[:10]),
        ("rank256_size10", 256, g1[:10]),
    ]
    for name, rank, entries in corners:
        t0 = time.time()
        model, cfg, tok = load_base(rank, device)
        reset_lora(model)
        sources = make_training_sources(entries, tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=N_STEPS,
                              high_lr=HIGH_LR, base_lr=BASE_LR, seed=0)
        ev = eval_retrieval(model, entries, tok, device, lora_scale=1.0,
                            **EVAL_KW)
        print(f"[{name}] rank={rank} size={len(entries)} sources={len(sources)}  "
              f"loss {info['loss_init']:.2f}->{info['loss_mean_last10']:.2f}  "
              f"recall {ev['mean_rate']:.3f} ({ev['hits']}/{ev['n']})  "
              f"wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
