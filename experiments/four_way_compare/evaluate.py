"""Four-way comparison on Mistral-7B with a single rank-512 PEFT model
that block-stacks adapters into the rank-512 LoRA slot.

Key trick (Phase 43-style): wrap Mistral with rank-512 LoRA, alpha=1024
(scaling = 2.0). Adapters trained at rank 128 (alpha=256, scaling=2.0)
produce equivalent contributions when block-stacked into the rank-512
slot. Single adapters get block-padded with zeros up to rank 512.

Procedures:
  A: RAG       — disable adapter, prepend constituent passages.
  B: Combo     — load combo (rank-128) padded into rank-512 slot.
  C: Multi-stack — block-stack K constituent adapters at rank K*128,
                  pad to 512 with zeros.
  D: Multi-pass — K passes with each constituent loaded as Combo,
                  then synthesis pass with adapter disabled.

Same query sets as combo_adapter; greedy decoding, 1 seed for time.
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
from peft import LoraConfig, get_peft_model

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.combo_adapter.combos import COMBINATIONS, CHOSEN_IDS

PPD = REPO / "experiments/per_passage_dickens"
FWC = REPO / "experiments/four_way_compare"
ADAPTERS = FWC / "adapters_mistral"

BASE = "mistralai/Mistral-7B-v0.1"
RANK_TRAIN = 128                 # Adapters were trained at rank 128
ALPHA_TRAIN = RANK_TRAIN * 2     # alpha=256 → scaling=2.0
RANK_INFER = 512                 # 4 × 128, max K
ALPHA_INFER = RANK_INFER * 2     # alpha=1024 → scaling=2.0
LAYERS = [30, 31]
TARGET_MODULES = ["q_proj", "v_proj", "gate_proj", "down_proj"]
GEN_TOKENS = 30

FEWSHOT = """Read the passage and answer the question with a short extracted answer.

Passage: The cat was named Whiskers and lived in a small blue house on Elm Street.
Question: What was the cat's name?
Answer: Whiskers

Passage: The lighthouse on the cliff was 90 feet tall and painted red and white.
Question: How tall was the lighthouse?
Answer: 90 feet

"""


def lora_config_inference():
    return LoraConfig(
        r=RANK_INFER, lora_alpha=ALPHA_INFER,
        target_modules=TARGET_MODULES,
        layers_to_transform=LAYERS,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )


def load_inference_model(device):
    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    cfg = lora_config_inference()
    peft_model = get_peft_model(base, cfg)
    peft_model.eval()
    print(f"  loaded; LoRA rank={RANK_INFER} alpha={ALPHA_INFER}")
    return peft_model, tokenizer


def stack_and_pad_state_dicts(sds, target_rank=RANK_INFER, train_rank=RANK_TRAIN):
    """Block-stack K rank-128 state dicts into one rank-K*128 stacked sd,
    then zero-pad to target_rank. Returns a dict mapping rank-512 PEFT keys.

    Train state dict keys: '...lora_A.default.weight' shape (rank, in)
                           '...lora_B.default.weight' shape (out, rank)
    PEFT lora_A has shape (rank, in_features) — input projection.
    PEFT lora_B has shape (out_features, rank) — output projection.

    Block-stacking:
      A_stacked: concat along dim=0  (stacked_rank, in)
      B_stacked: concat along dim=1  (out, stacked_rank)

    Then pad to target_rank with zero rows/cols.
    """
    K = len(sds)
    stacked_rank = K * train_rank
    assert stacked_rank <= target_rank, f"K={K} too large for target_rank={target_rank}"

    # Find shared keys, group by base linear name (stripping .lora_A.default.weight etc.)
    sample_keys = list(sds[0].keys())
    out = {}
    for k in sample_keys:
        if "lora_A" in k:
            # Stack along dim 0: (rank, in) → (K*rank, in)
            stacked = torch.cat([sd[k] for sd in sds], dim=0)
            in_f = stacked.shape[1]
            padded = torch.zeros((target_rank, in_f),
                                  dtype=stacked.dtype, device=stacked.device)
            padded[:stacked_rank] = stacked
            out[k] = padded
        elif "lora_B" in k:
            # Stack along dim 1: (out, rank) → (out, K*rank)
            stacked = torch.cat([sd[k] for sd in sds], dim=1)
            out_f = stacked.shape[0]
            padded = torch.zeros((out_f, target_rank),
                                  dtype=stacked.dtype, device=stacked.device)
            padded[:, :stacked_rank] = stacked
            out[k] = padded
        else:
            out[k] = sds[0][k]
    return out


def load_zeros_into_slot(peft_model):
    """Zero out the LoRA — equivalent to disabling. Easier than disable_adapter
    in some PEFT versions."""
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_" in n:
                p.zero_()


def load_state_into_default(peft_model, sd_dict, target_rank=RANK_INFER):
    """Load a (possibly stacked) state dict into PEFT's default slot.
    Handles both rank-128 (single, padded to 512) and rank-512 (already
    stacked) inputs."""
    # First zero-out so we don't carry over previous state.
    load_zeros_into_slot(peft_model)

    # Check expected shape from current params
    sample_param = None
    for n, p in peft_model.named_parameters():
        if "lora_A" in n:
            sample_param = p; break
    expected_rank, expected_in = sample_param.shape

    # If incoming SD is at rank < target, pad. If at target rank, load directly.
    remapped = {}
    for k, v in sd_dict.items():
        if "lora_A" in k:
            r, in_f = v.shape
            if r < target_rank:
                padded = torch.zeros((target_rank, in_f),
                                      dtype=v.dtype, device="cuda")
                padded[:r] = v.to("cuda")
                remapped[k] = padded
            else:
                remapped[k] = v.to("cuda")
        elif "lora_B" in k:
            out_f, r = v.shape
            if r < target_rank:
                padded = torch.zeros((out_f, target_rank),
                                      dtype=v.dtype, device="cuda")
                padded[:, :r] = v.to("cuda")
                remapped[k] = padded
            else:
                remapped[k] = v.to("cuda")
        else:
            remapped[k] = v.to("cuda") if torch.is_tensor(v) else v
    missing, unexpected = peft_model.load_state_dict(remapped, strict=False)
    if unexpected:
        # Unexpected may be empty if our keys match; sanity print
        print(f"WARN unexpected keys (first 3): {unexpected[:3]}")


@torch.no_grad()
def greedy_generate(peft_model, tokenizer, prompt, max_new_tokens=GEN_TOKENS,
                     device="cuda"):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = peft_model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    cont_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True)


def check_match(answer, gen):
    if answer.lower() in gen.lower():
        return True
    a = answer.replace(",", "").replace(" ", "").lower()
    g = gen.replace(",", "").replace(" ", "").lower()
    return bool(a) and a in g


QMAP = {
    0:  ("What was Pip's father's family name?", "Pirrip"),
    2:  ("What was Joe Gargery's profession?", "blacksmith"),
    16: ("What was the name of Miss Havisham's adopted daughter?", "Estella"),
    17: ("What was the name of Mr. Jaggers's clerk?", "Wemmick"),
    22: ("What was the name of Estella's father?", "Provis"),
    30: ("Who was Pip's roommate at Barnard's Inn?", "Herbert"),
    31: ("Who was Pip's secret benefactor?", "Magwitch"),
    36: ("Who did Estella marry?", "Drummle"),
}


def make_rag_prompt(constituent_entries, question):
    passage_block = "\n\n".join(e["passage"].strip() for e in constituent_entries)
    return (FEWSHOT + f"Passage: {passage_block}\nQuestion: {question}\nAnswer:")


def make_combo_prompt(question):
    return (FEWSHOT + f"Question: {question}\nAnswer:")


def make_multipass_synthesis_prompt(question, intermediate_notes):
    notes_block = "\n".join(f"Note {i+1}: {n}"
                              for i, n in enumerate(intermediate_notes))
    return (FEWSHOT
            + f"Notes:\n{notes_block}\n\nQuestion: {question}\nAnswer:")


def first_line(s): return s.split("\n")[0].strip()


def main():
    device = torch.device("cuda")
    library = json.loads((PPD / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    peft_model, tokenizer = load_inference_model(device)

    # Pre-load all 18 state dicts to CPU (will move per-load)
    print("Loading 18 saved state dicts ...")
    sds = {}
    for cid in CHOSEN_IDS:
        sds[f"single_{cid}"] = torch.load(
            ADAPTERS / f"single_{cid}.pt", map_location="cpu", weights_only=False,
        )
    for c in COMBINATIONS:
        sds[f"combo_{c['name']}"] = torch.load(
            ADAPTERS / f"combo_{c['name']}.pt", map_location="cpu", weights_only=False,
        )
    print(f"  loaded {len(sds)} state dicts")

    # ----- TEST 1 -----
    print("\n=== Test 1: per-constituent retrieval ===")
    test1 = []
    t0 = time.time()
    for combo in COMBINATIONS:
        cname = combo["name"]; K = combo["k"]
        constituents = [by_id[i] for i in combo["constituents"]]

        per_constituent = []
        for cid in combo["constituents"]:
            entry = by_id[cid]
            question, answer = QMAP[cid]

            # ---- A: RAG ----
            load_zeros_into_slot(peft_model)
            rag_prompt = make_rag_prompt(constituents, question)
            gen_rag = greedy_generate(peft_model, tokenizer, rag_prompt)

            # ---- B: Combo ----
            load_state_into_default(peft_model, sds[f"combo_{cname}"])
            gen_combo = greedy_generate(peft_model, tokenizer,
                                          make_combo_prompt(question))

            # ---- C: Multi-stack ----
            constituent_sds = [sds[f"single_{c}"] for c in combo["constituents"]]
            stacked = stack_and_pad_state_dicts(constituent_sds)
            load_state_into_default(peft_model, stacked)
            gen_ms = greedy_generate(peft_model, tokenizer,
                                       make_combo_prompt(question))

            # ---- D: Multi-pass ----
            intermediates = []
            for c in combo["constituents"]:
                load_state_into_default(peft_model, sds[f"single_{c}"])
                gen = greedy_generate(peft_model, tokenizer,
                                       make_combo_prompt(question))
                intermediates.append(first_line(gen))
            load_zeros_into_slot(peft_model)
            synth_prompt = make_multipass_synthesis_prompt(question, intermediates)
            gen_mp = greedy_generate(peft_model, tokenizer, synth_prompt)

            per_constituent.append({
                "cid": cid, "question": question, "answer": answer,
                "gen_rag":   first_line(gen_rag),   "hit_rag":   check_match(answer, first_line(gen_rag)),
                "gen_combo": first_line(gen_combo), "hit_combo": check_match(answer, first_line(gen_combo)),
                "gen_ms":    first_line(gen_ms),    "hit_ms":    check_match(answer, first_line(gen_ms)),
                "gen_mp":    first_line(gen_mp),    "hit_mp":    check_match(answer, first_line(gen_mp)),
            })

        rates = {k: float(np.mean([1.0 if p[f"hit_{k}"] else 0.0
                                      for p in per_constituent]))
                 for k in ("rag", "combo", "ms", "mp")}
        test1.append({"name": cname, "k": K, "per_constituent": per_constituent,
                      "rates": rates})
        print(f"  [{cname}] K={K}  rag={rates['rag']:.3f} combo={rates['combo']:.3f} "
              f"ms={rates['ms']:.3f} mp={rates['mp']:.3f}")
    print(f"  Test 1 wall: {time.time()-t0:.0f}s")

    # ----- TEST 2 -----
    print("\n=== Test 2: cross-passage queries ===")
    test2 = []
    t1 = time.time()
    for combo in COMBINATIONS:
        cname = combo["name"]; K = combo["k"]
        constituents = [by_id[i] for i in combo["constituents"]]

        per_query = []
        for q in combo["cross_queries"]:
            question = q["probe"]

            # A: RAG
            load_zeros_into_slot(peft_model)
            g_rag = greedy_generate(peft_model, tokenizer,
                                      make_rag_prompt(constituents, question),
                                      max_new_tokens=80)
            # B: Combo
            load_state_into_default(peft_model, sds[f"combo_{cname}"])
            g_combo = greedy_generate(peft_model, tokenizer,
                                        make_combo_prompt(question),
                                        max_new_tokens=80)
            # C: Multi-stack
            constituent_sds = [sds[f"single_{c}"] for c in combo["constituents"]]
            stacked = stack_and_pad_state_dicts(constituent_sds)
            load_state_into_default(peft_model, stacked)
            g_ms = greedy_generate(peft_model, tokenizer,
                                     make_combo_prompt(question),
                                     max_new_tokens=80)
            # D: Multi-pass
            intermediates = []
            for c in combo["constituents"]:
                load_state_into_default(peft_model, sds[f"single_{c}"])
                gen = greedy_generate(peft_model, tokenizer,
                                       make_combo_prompt(question),
                                       max_new_tokens=GEN_TOKENS)
                intermediates.append(first_line(gen))
            load_zeros_into_slot(peft_model)
            g_mp = greedy_generate(peft_model, tokenizer,
                                     make_multipass_synthesis_prompt(question, intermediates),
                                     max_new_tokens=80)

            frags = q["fragments"]
            def frac(g): return sum(1 for f in frags if check_match(f, g)) / max(1, len(frags))
            per_query.append({
                "probe": q["probe"][:80] + "...", "fragments": frags,
                "gen_rag": g_rag[:120],   "frac_rag": frac(g_rag),
                "gen_combo": g_combo[:120], "frac_combo": frac(g_combo),
                "gen_ms": g_ms[:120],      "frac_ms": frac(g_ms),
                "gen_mp": g_mp[:120],      "frac_mp": frac(g_mp),
            })

        rates = {k: float(np.mean([p[f"frac_{k}"] for p in per_query]))
                 for k in ("rag", "combo", "ms", "mp")}
        test2.append({"name": cname, "k": K, "per_query": per_query,
                      "rates": rates})
        print(f"  [{cname}] K={K}  rag={rates['rag']:.3f} combo={rates['combo']:.3f} "
              f"ms={rates['ms']:.3f} mp={rates['mp']:.3f}")
    print(f"  Test 2 wall: {time.time()-t1:.0f}s")

    out = {"test1": test1, "test2": test2,
           "wall_total_s": time.time() - t0}
    out_path = FWC / "results/tests.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTotal eval wall: {out['wall_total_s']:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
