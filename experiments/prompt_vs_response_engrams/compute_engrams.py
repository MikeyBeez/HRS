"""Compute 6 sets of engrams for the 100 reused Civil War turns:

  layers ∈ {8, 16, 24}
  content ∈ {prompt_only, prompt_plus_response}

Each engram is the LAST-TOKEN hidden state at the given layer (no
mean-pool — we're testing whether the prior experiment's anisotropy
came from the mean-pooling).

Reports anisotropy (mean & max pairwise cosine) for each (layer, content)
combination.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"
PRIOR = REPO / "experiments/multi_engram"

BASE = "mistralai/Mistral-7B-v0.1"
LAYERS = [8, 16, 24]


@torch.no_grad()
def main():
    device = torch.device("cuda")
    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    d_model = model.config.hidden_size

    turns = json.loads((PRIOR / "data/turns.json").read_text())["turns"]
    print(f"  {len(turns)} turns reused from multi_engram")

    # We'll compute hidden_states for each turn ONCE per content variant
    # (prompt_only and prompt_plus_response), then extract last-token at
    # each of 3 layers.
    engrams = {f"{layer}_{kind}": np.zeros((len(turns), d_model),
                                              dtype=np.float32)
               for layer in LAYERS for kind in ("p", "pr")}

    t0 = time.time()
    for i, turn in enumerate(turns):
        # Build the two text variants
        prompt_only = turn["prompt"]                      # "Tell me about X."
        prompt_plus_response = turn["full_text"]          # "Q: ... \nA: ..."

        for kind, text in (("p", prompt_only),
                            ("pr", prompt_plus_response)):
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            out = model(ids, output_hidden_states=True, return_dict=True)
            for layer in LAYERS:
                # hidden_states[layer]: (1, T, D), output of layer `layer`
                # (layer=0 is post-embedding).
                last_tok = out.hidden_states[layer][0, -1, :].float().cpu().numpy()
                engrams[f"{layer}_{kind}"][i] = last_tok
            del out
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(turns)}]  elapsed={time.time()-t0:.0f}s")

    EXP.mkdir(exist_ok=True)
    (EXP / "data").mkdir(exist_ok=True)
    np.savez(
        EXP / "data/engrams_v2.npz",
        **engrams,
    )
    print(f"\nSaved engrams to {EXP / 'data/engrams_v2.npz'}")
    print(f"Total wall: {time.time()-t0:.0f}s")

    # Anisotropy report
    print("\n=== Anisotropy (pairwise cosine) ===")
    print(f"  {'set':>20s}  {'mean':>6s}  {'max':>6s}  {'p90':>6s}  {'min':>6s}")
    aniso = {}
    for key, e in engrams.items():
        et = torch.tensor(e, dtype=torch.float32, device=device)
        en = F.normalize(et, dim=-1)
        sim = en @ en.T
        iu = torch.triu_indices(len(turns), len(turns), offset=1)
        pairs = sim[iu[0], iu[1]]
        aniso[key] = {
            "mean": float(pairs.mean().item()),
            "max":  float(pairs.max().item()),
            "p90":  float(pairs.quantile(0.9).item()),
            "min":  float(pairs.min().item()),
        }
        print(f"  {key:>20s}  {aniso[key]['mean']:6.3f}  {aniso[key]['max']:6.3f}  "
              f"{aniso[key]['p90']:6.3f}  {aniso[key]['min']:6.3f}")

    (EXP / "data/anisotropy.json").write_text(json.dumps(aniso, indent=2))


if __name__ == "__main__":
    main()
