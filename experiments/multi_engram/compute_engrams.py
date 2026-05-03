"""Compute 100 engrams via Mistral-7B-v0.1 mid-layer mean pooling.

Layer choice: layer 16 (middle of 32). The position-erosion experiment
showed late layers leave the token regime; mid-layer is a reasonable
compromise that should preserve some semantic content while being
distinct across topics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/multi_engram"

BASE = "mistralai/Mistral-7B-v0.1"
LAYER = 16    # of 32; middle


@torch.no_grad()
def main():
    device = torch.device("cuda")
    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    turns = json.loads((EXP / "data/turns.json").read_text())["turns"]
    print(f"  {len(turns)} turns loaded")

    d_model = model.config.hidden_size
    engrams = np.zeros((len(turns), d_model), dtype=np.float32)
    t0 = time.time()
    for i, turn in enumerate(turns):
        full_text = turn["full_text"]
        ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
        # Forward, request all hidden states
        out = model(ids, output_hidden_states=True, return_dict=True)
        # hidden_states is a tuple of (n_layers + 1) tensors of shape (1, T, D)
        h = out.hidden_states[LAYER]  # (1, T, D)
        eng = h.mean(dim=1).squeeze(0).float().cpu().numpy()
        engrams[i] = eng
        del out, h
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(turns)}]  elapsed={time.time()-t0:.0f}s")
    np.save(EXP / "data/engrams.npy", engrams)
    print(f"\nSaved engrams shape={engrams.shape} to data/engrams.npy")
    print(f"Total wall: {time.time()-t0:.0f}s")

    # Quick sanity: pairwise mean cos
    e_t = torch.tensor(engrams, dtype=torch.float32, device=device)
    e_n = torch.nn.functional.normalize(e_t, dim=-1)
    sim = e_n @ e_n.T
    iu = torch.triu_indices(len(turns), len(turns), offset=1)
    pairs = sim[iu[0], iu[1]]
    print(f"\nPairwise cosine: mean={pairs.mean().item():.3f}  "
          f"max={pairs.max().item():.3f}  "
          f"min={pairs.min().item():.3f}")


if __name__ == "__main__":
    main()
