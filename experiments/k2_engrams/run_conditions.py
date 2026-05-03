"""Run engram_2_separated and engram_2_random_split with same metadata
+ sampling protocol as v3. For each probe, route by cosine similarity
to the two engrams (no W projection — there's nothing meaningful to
train at K=2). Top-1 selection: inject the chosen single engram as
inputs_embeds prefix.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/k2_engrams"
PRIOR_ME = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.probes import (
    VAL_PROBES, TEST_PROBES, build_probe_records,
)

BASE = "mistralai/Mistral-7B-v0.1"
LAYER = 16
TEMP_GEN = 0.7
REP_PENALTY = 1.15
GEN_TOKENS = 200


def engram_prefix():
    return ("[The model has access to compressed memories of the prior "
            "conversation, retrieved by relevance to the current question. "
            "These memories appear as the initial context below.]\n")


def engram_query_suffix(probe_text):
    return (f"\n\n[Current question — please answer using the compressed "
            f"memories above]\n"
            f"USER: {probe_text}\n"
            f"ASSISTANT:")


def get_layer_pool(model, ids, layer):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
        return out.hidden_states[layer][0, -1, :].float()


def generate_text(model, tokenizer, prompt, inputs_embeds, max_new_tokens,
                   device="cuda", seed=0):
    """inputs_embeds: 1D (D,) for a single prefix vector, or 2D (P, D)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        emb = model.model.embed_tokens(ids)
        # Ensure prefix is 2D (P, D) then add batch dim → (1, P, D)
        if inputs_embeds.dim() == 1:
            inputs_embeds = inputs_embeds.unsqueeze(0)  # (1, D)
        prefix = inputs_embeds.unsqueeze(0).to(emb.dtype)  # (1, P, D)
        full = torch.cat([prefix, emb], dim=1)
        n_input = full.shape[1]
        out = model.generate(
            inputs_embeds=full, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=TEMP_GEN,
            repetition_penalty=REP_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True), n_input


def main():
    device = torch.device("cuda")
    print(f"Loading {BASE} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    # Load K=2 engrams
    eng = np.load(EXP / "data/k2_engrams.npz")
    eng_A = torch.tensor(eng["topic_A"], dtype=torch.float32, device=device)
    eng_B = torch.tensor(eng["topic_B"], dtype=torch.float32, device=device)
    eng_R1 = torch.tensor(eng["random_R1"], dtype=torch.float32, device=device)
    eng_R2 = torch.tensor(eng["random_R2"], dtype=torch.float32, device=device)
    setup = json.loads((EXP / "data/setup.json").read_text())
    print(f"  cos(A, B)   = {setup['cos_topic_AB']:.3f}")
    print(f"  cos(R1, R2) = {setup['cos_random_R1R2']:.3f}")

    test = build_probe_records(TEST_PROBES)

    # Compute probe-side L16 last-token for routing
    print("\nComputing probe L16 last-token embeddings ...")
    probe_emb = []
    for p in test:
        ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
        probe_emb.append(get_layer_pool(model, ids, LAYER))
    probe_emb = torch.stack(probe_emb).to(device)
    print(f"  shape: {probe_emb.shape}")

    # ---- Routing decisions ----
    def route2(probe_e, e1, e2):
        """Cosine routing between two engrams. Return top-1 index (0 or 1)
        and (cos1, cos2)."""
        e1n = F.normalize(e1.unsqueeze(0), dim=-1)
        e2n = F.normalize(e2.unsqueeze(0), dim=-1)
        pen = F.normalize(probe_e.unsqueeze(0), dim=-1)
        c1 = (pen @ e1n.T).item()
        c2 = (pen @ e2n.T).item()
        return (0 if c1 >= c2 else 1, c1, c2)

    # Check predicted vs actual routing for both splits
    print("\nRouting decisions (engram_2_separated):")
    print(f"  {'idx':>3s}  {'predicted':>10s}  {'cos_A':>6s}  {'cos_B':>6s}  "
          f"{'actual':>6s}  {'match':>6s}")
    sep_routes = []
    n_correct = 0; n_eval = 0
    for i, p in enumerate(test):
        actual_idx, c_A, c_B = route2(probe_emb[i], eng_A, eng_B)
        actual = "A" if actual_idx == 0 else "B"
        predicted = setup["predictions"][i]["predicted_target"]
        if predicted == "AMBIGUOUS":
            match = "amb"
        else:
            match = "yes" if actual == predicted else "no"
            n_eval += 1
            if actual == predicted: n_correct += 1
        sep_routes.append({"probe_idx": i, "predicted": predicted,
                            "cos_A": c_A, "cos_B": c_B, "actual": actual,
                            "match": match})
        print(f"  {i:3d}  {predicted:>10s}  {c_A:6.3f}  {c_B:6.3f}  "
              f"{actual:>6s}  {match:>6s}")
    print(f"\nRouting accuracy: {n_correct}/{n_eval} = "
          f"{n_correct/max(1,n_eval):.3f} (excludes ambiguous probes)")

    # Same for random split (no predicted target — just record)
    print("\nRouting decisions (engram_2_random_split):")
    rand_routes = []
    for i, p in enumerate(test):
        actual_idx, c_R1, c_R2 = route2(probe_emb[i], eng_R1, eng_R2)
        actual = "R1" if actual_idx == 0 else "R2"
        rand_routes.append({"probe_idx": i, "cos_R1": c_R1, "cos_R2": c_R2,
                             "actual": actual})

    # ---- Generate ----
    print(f"\n=== Generating: 20 probes × 2 conditions "
          f"(temp={TEMP_GEN}, rep_pen={REP_PENALTY}) ===")
    results = []
    t0 = time.time()
    for pi, p in enumerate(test):
        probe_text = p["prompt"]
        rec = {"probe_idx": pi, "prompt": probe_text,
               "n_relevant": p["n_relevant"],
               "relevant_ids": p["relevant_ids"]}

        prompt = engram_prefix() + engram_query_suffix(probe_text)

        # engram_2_separated
        sr = sep_routes[pi]
        chosen_eng = eng_A if sr["actual"] == "A" else eng_B
        gen, n_in = generate_text(model, tokenizer, prompt,
                                    chosen_eng, GEN_TOKENS,
                                    device=device, seed=pi*7)
        rec["gen_engram_2_separated"] = gen
        rec["tokens_engram_2_separated"] = n_in
        rec["sep_route"] = sr

        # engram_2_random_split
        rr = rand_routes[pi]
        chosen_eng = eng_R1 if rr["actual"] == "R1" else eng_R2
        gen, n_in = generate_text(model, tokenizer, prompt,
                                    chosen_eng, GEN_TOKENS,
                                    device=device, seed=pi*7+1)
        rec["gen_engram_2_random_split"] = gen
        rec["tokens_engram_2_random_split"] = n_in
        rec["rand_route"] = rr

        results.append(rec)
        if (pi + 1) % 5 == 0 or pi == 0:
            print(f"  [{pi+1}/{len(test)}]  elapsed={time.time()-t0:.0f}s")

    out = {
        "results": results,
        "config": {"temp": TEMP_GEN, "repetition_penalty": REP_PENALTY,
                    "gen_tokens": GEN_TOKENS, "layer": LAYER,
                    "version": "k2"},
        "setup": setup,
        "routing_accuracy_separated": n_correct / max(1, n_eval),
        "n_correct": n_correct, "n_eval": n_eval,
        "wall_total_s": time.time() - t0,
    }
    (EXP / "results/conditions.json").parent.mkdir(parents=True, exist_ok=True)
    (EXP / "results/conditions.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {EXP/'results/conditions.json'}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
