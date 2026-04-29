"""Smoke test 6: validate harness2 (multi-layer LoRA) at the corner cases.

Compare with harness1's single-layer LoRA results:
  rank32_size1   single: 100% (9/9)   multi: ?
  rank32_size10  single:   8% (7/90)  multi: ?
  rank256_size10 single: 16% (8000st) multi: ?
"""
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

    HIGH_LR = 5e-3
    BASE_LR = 1e-4
    EVAL_KW = {"temperature": 0.6, "top_k": 20, "seeds": (0, 1, 2)}

    corners = [
        ("rank32_size1",   32,  g1[:1],  800),
        ("rank64_size1",   64,  g1[:1],  800),
        ("rank32_size10",  32,  g1[:10], 4000),
        ("rank64_size10",  64,  g1[:10], 4000),
        ("rank128_size10", 128, g1[:10], 4000),
        ("rank256_size10", 256, g1[:10], 4000),
    ]
    for name, rank, entries, n_steps in corners:
        t0 = time.time()
        model, cfg, tok = load_base(rank, device)
        reset_lora(model)
        sources = make_training_sources(entries, tok, cfg.ctx_len, device)
        info = train_adapter(model, sources, n_steps=n_steps,
                              high_lr=HIGH_LR, base_lr=BASE_LR, seed=0)
        ev = eval_retrieval(model, entries, tok, device, **EVAL_KW)
        print(f"[{name}] rank={rank} size={len(entries)} n_steps={n_steps} "
              f"sources={len(sources)}  "
              f"loss {info['loss_init']:.2f}->{info['loss_mean_last10']:.2f}  "
              f"recall {ev['mean_rate']:.3f} ({ev['hits']}/{ev['n']})  "
              f"wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
