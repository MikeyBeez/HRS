"""V3 rerun with metadata wrapping + repetition_penalty + temperature
0.7. Reuses v2 engrams and W projections.

Key changes vs v2:
  - Each text condition now wraps content in role-labeled blocks.
  - Engram conditions get a text prefix describing the engrams as
    "compressed memories of the prior conversation."
  - Generation: temperature=0.7, repetition_penalty=1.15 (was greedy
    in v2, which produced repetitive output).
  - Outputs go to results/conditions_v3.json (does not overwrite v2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"
PRIOR = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.probes import (
    VAL_PROBES, TEST_PROBES, build_probe_records,
)

BASE = "mistralai/Mistral-7B-v0.1"
LAYERS = [8, 16, 24]
TOP_K = 10
TEMP = 0.7
REP_PENALTY = 1.15
GEN_TOKENS = 200
RECENT_N = 8


# ----------------- METADATA WRAPPERS -----------------

def wrap_full_context(turns):
    """All turns wrapped with [Conversation turn N] / USER / ASSISTANT."""
    parts = []
    for n, t in enumerate(turns, start=1):
        parts.append(
            f"[Conversation turn {n}]\n"
            f"USER: {t['prompt']}\n"
            f"ASSISTANT: {t['response']}\n"
            f"[end of turn {n}]"
        )
    return "\n\n".join(parts)


def wrap_recent_only(turns, n_recent):
    """Last N turns + prefix note."""
    note = ("[Note: earlier conversation turns omitted to save space; "
            "the most recent 8 turns follow]\n\n")
    body = []
    start = len(turns) - n_recent
    for offset, t in enumerate(turns[-n_recent:]):
        n = start + offset + 1
        body.append(
            f"[Conversation turn {n}]\n"
            f"USER: {t['prompt']}\n"
            f"ASSISTANT: {t['response']}\n"
            f"[end of turn {n}]"
        )
    return note + "\n\n".join(body)


def wrap_prompts_as_context(turns):
    """Each prompt wrapped with [Earlier conversation turn N — prompt only ...]."""
    parts = []
    for n, t in enumerate(turns, start=1):
        parts.append(
            f"[Earlier conversation turn {n} — prompt only, response removed]\n"
            f"USER: {t['prompt']}\n"
            f"[end of turn {n}]"
        )
    return "\n\n".join(parts)


def wrap_probe(probe_text):
    """Append the synthesis probe with current-question framing."""
    return (f"\n\n[Current question — please answer using the conversation "
            f"history above]\n"
            f"USER: {probe_text}\n"
            f"ASSISTANT:")


def engram_prefix():
    return ("[The model has access to compressed memories of the prior "
            "conversation, retrieved by relevance to the current question. "
            "These memories appear as the initial context below.]\n")


def engram_query_suffix(probe_text):
    return (f"\n\n[Current question — please answer using the compressed "
            f"memories above]\n"
            f"USER: {probe_text}\n"
            f"ASSISTANT:")


# ----------------- TRAINING / ROUTING (reused from v2) -----------------

def get_layer_pool(model, ids, layer):
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
        return out.hidden_states[layer][0, -1, :].float()


def train_W(probe_engrams, probe_relevant_ids, library_engrams,
            n_steps=500, lr=1e-3, temp=0.05, device="cuda"):
    D = library_engrams.shape[-1]
    W = nn.Linear(D, D, bias=False).to(device)
    nn.init.eye_(W.weight)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=0.0,
                              betas=(0.9, 0.95))
    keys_n = F.normalize(library_engrams, dim=-1)
    pairs = []
    for pi, rel in enumerate(probe_relevant_ids):
        for tid in rel:
            pairs.append((pi, tid))
    if not pairs: return W, 0.0
    pairs_p = torch.tensor([p[0] for p in pairs], device=device)
    pairs_t = torch.tensor([p[1] for p in pairs], device=device)
    for _ in range(n_steps):
        proj_n = F.normalize(W(probe_engrams), dim=-1)
        sim = (proj_n[pairs_p] @ keys_n.T) / temp
        loss = F.cross_entropy(sim, pairs_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        proj_n = F.normalize(W(probe_engrams), dim=-1)
        sim = (proj_n[pairs_p] @ keys_n.T) / temp
        train_acc = (sim.argmax(dim=-1) == pairs_t).float().mean().item()
    return W, train_acc


def routing_pr(top_k_ids, gt_ids, k):
    if k == 0 or not gt_ids: return 0.0, 0.0
    pred = set(top_k_ids[:k]); gt = set(gt_ids)
    overlap = pred & gt
    return len(overlap) / k, len(overlap) / len(gt)


def generate_text(model, tokenizer, prompt, inputs_embeds=None,
                   max_new_tokens=GEN_TOKENS, device="cuda",
                   max_input_tokens=2000, temperature=TEMP,
                   repetition_penalty=REP_PENALTY, seed=0):
    """Sampling with temperature + repetition_penalty."""
    torch.manual_seed(seed)
    with torch.no_grad():
        if inputs_embeds is None:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if ids.shape[1] > max_input_tokens:
                ids = ids[:, -max_input_tokens:]
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
            return tokenizer.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=True), ids.shape[1]
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        emb = model.model.embed_tokens(ids)
        prefix = inputs_embeds.unsqueeze(0).to(emb.dtype)
        full = torch.cat([prefix, emb], dim=1)
        n_input_tokens = full.shape[1]
        out = model.generate(
            inputs_embeds=full, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True), n_input_tokens


def main():
    device = torch.device("cuda")
    rng = np.random.default_rng(42)

    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    turns = json.loads((PRIOR / "data/turns.json").read_text())["turns"]

    eng_npz = np.load(EXP / "data/engrams_v2.npz")
    eng_sets = {key: torch.tensor(eng_npz[key], dtype=torch.float32, device=device)
                for key in eng_npz.files}

    aniso = json.loads((EXP / "data/anisotropy.json").read_text())

    val = build_probe_records(VAL_PROBES)
    test = build_probe_records(TEST_PROBES)

    # Probe-side embeddings at 3 layers
    print("Computing probe-side L8/L16/L24 last-token ...")
    probe_emb = {}
    for layer in LAYERS:
        ve = []; te = []
        for p in val:
            ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
            ve.append(get_layer_pool(model, ids, layer))
        for p in test:
            ids = tokenizer.encode(p["prompt"], return_tensors="pt").to(device)
            te.append(get_layer_pool(model, ids, layer))
        probe_emb[layer] = {"val": torch.stack(ve), "test": torch.stack(te)}

    # Train W's
    print("Training 6 W projections ...")
    val_relevant = [p["relevant_ids"] for p in val]
    Ws = {}; train_accs = {}
    for layer in LAYERS:
        for kind in ("p", "pr"):
            key = f"{layer}_{kind}"
            W, ta = train_W(probe_emb[layer]["val"], val_relevant,
                              eng_sets[key], device=device)
            Ws[key] = W.half()
            train_accs[key] = ta
    torch.cuda.empty_cache()

    # Pre-build wrapped texts
    full_text = wrap_full_context(turns)
    recent_text = wrap_recent_only(turns, RECENT_N)
    prompts_text = wrap_prompts_as_context(turns)

    print(f"\nToken budgets (with metadata wrapping):")
    print(f"  full_context:      {len(tokenizer.encode(full_text)):5d}")
    print(f"  recent_only:       {len(tokenizer.encode(recent_text)):5d}")
    print(f"  prompts_as_context:{len(tokenizer.encode(prompts_text)):5d}")
    print(f"  engram prefix:     {len(tokenizer.encode(engram_prefix())):5d}")

    print(f"\n=== Running v3 conditions × {len(test)} test probes ===")
    print(f"  temp={TEMP}, repetition_penalty={REP_PENALTY}, "
          f"max_new_tokens={GEN_TOKENS}")
    results = []
    t_total = time.time()

    for pi, p in enumerate(test):
        probe_text = p["prompt"]
        gt_ids = p["relevant_ids"]
        rec = {"probe_idx": pi, "prompt": probe_text,
               "n_relevant": len(gt_ids), "relevant_ids": gt_ids}
        print(f"\n[{pi+1}/{len(test)}] {probe_text[:60]}...  "
              f"({len(gt_ids)} relevant)")

        probe_suffix_text = wrap_probe(probe_text)
        probe_suffix_engram = engram_query_suffix(probe_text)

        # Cond 1: full_context (wrapped)
        full_prompt = full_text + probe_suffix_text
        gen, n_in = generate_text(model, tokenizer, full_prompt, seed=pi*7)
        rec["gen_full_context"] = gen; rec["tokens_full_context"] = n_in

        # Cond 2: random_engrams
        rand_ids = rng.choice(100, size=TOP_K, replace=False).tolist()
        rand_engrams = eng_sets["16_pr"][rand_ids]
        full_engram_prompt = engram_prefix() + probe_suffix_engram
        gen, n_in = generate_text(model, tokenizer, full_engram_prompt,
                                    inputs_embeds=rand_engrams, seed=pi*7+1)
        rec["gen_random_engrams"] = gen; rec["tokens_random_engrams"] = n_in
        rec["random_ids"] = rand_ids

        # Cond 3: prompts_as_context (wrapped) — KEY CONDITION
        prompts_prompt = prompts_text + probe_suffix_text
        gen, n_in = generate_text(model, tokenizer, prompts_prompt, seed=pi*7+2)
        rec["gen_prompts_as_context"] = gen
        rec["tokens_prompts_as_context"] = n_in

        # Cond 4: recent_only (wrapped)
        recent_prompt = recent_text + probe_suffix_text
        gen, n_in = generate_text(model, tokenizer, recent_prompt, seed=pi*7+3)
        rec["gen_recent_only"] = gen; rec["tokens_recent_only"] = n_in

        # Conds 5-10: 6 engram routings
        for layer in LAYERS:
            for kind in ("p", "pr"):
                key = f"{layer}_{kind}"
                W = Ws[key]; eng_bank = eng_sets[key]
                keys_n = F.normalize(eng_bank, dim=-1)
                pe = probe_emb[layer]["test"][pi].half()
                with torch.no_grad():
                    proj = W(pe.unsqueeze(0))
                    proj_n = F.normalize(proj.float(), dim=-1)
                    sim = (proj_n @ keys_n.T).squeeze(0)
                topk = sim.argsort(descending=True)[:TOP_K].cpu().tolist()
                p_at_k, r_at_k = routing_pr(topk, gt_ids, TOP_K)
                rec[f"routing_p_at_k_{key}"] = p_at_k
                rec[f"routing_r_at_k_{key}"] = r_at_k
                rec[f"topk_{key}"] = topk
                topk_e = eng_bank[topk]
                gen, n_in = generate_text(
                    model, tokenizer, full_engram_prompt,
                    inputs_embeds=topk_e, seed=pi*7+4+layer,
                )
                rec[f"gen_engram_{key}"] = gen
                rec[f"tokens_engram_{key}"] = n_in

        # Cond 11: uniform_pool (16_pr mean)
        uniform = eng_sets["16_pr"].mean(dim=0, keepdim=True)
        gen, n_in = generate_text(
            model, tokenizer, full_engram_prompt,
            inputs_embeds=uniform, seed=pi*7+50,
        )
        rec["gen_uniform_pool"] = gen; rec["tokens_uniform_pool"] = n_in

        results.append(rec)

    out = {
        "results": results,
        "anisotropy": aniso,
        "train_accs": train_accs,
        "config": {"top_k": TOP_K, "temp": TEMP,
                    "repetition_penalty": REP_PENALTY,
                    "recent_n": RECENT_N, "gen_tokens": GEN_TOKENS,
                    "layers": LAYERS, "version": "v3"},
        "wall_total_s": time.time() - t_total,
    }
    out_path = EXP / "results/conditions_v3.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Total wall: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
