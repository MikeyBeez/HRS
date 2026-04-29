"""Case 1: single-adapter retrieval under composition.

For each of the 8 chosen passages, take its 3 held-out paraphrases (from
per_passage_dickens). Evaluate retrieval at K in {1, 2, 4, 8}, where the
non-relevant adapters are added with a deterministic offset scheme:
  K=2: relevant + (i+1) mod 8
  K=4: relevant + {(i+1), (i+2), (i+3)} mod 8
  K=8: all 8 adapters

For each (K, i, paraphrase, seed in {0,1,2}), block-stack the K adapters
into one rank-K*128 LoRA, generate 30 tokens, substring-match the answer.

Run for both Procedure A (independent) and Procedure B (sequential).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase43_k_capacity import stack_k_state_dicts
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)

from experiments.seq_adapter.queries import CHOSEN_IDS

PPD = REPO / "experiments/per_passage_dickens"
SEQ = REPO / "experiments/seq_adapter"
RANK_BASE = 128
ALPHA_BASE = RANK_BASE * 2
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
SEEDS = (0, 1, 2)


def encode(tokenizer, text, device, ctx=512):
    ids = tokenizer.encode(text, add_special_tokens=False)[:ctx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / TEMPERATURE
        v, _ = torch.topk(logits, TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, gen):
    if answer.lower() in gen.lower():
        return True
    a = answer.replace(",", "").replace(" ", "").lower()
    g = gen.replace(",", "").replace(" ", "").lower()
    return bool(a) and a in g


def adapter_subset(i, K, n=8):
    """Deterministic K-adapter subset around relevant adapter i.
    K=1: just i
    K>=2: i, (i+1)%n, (i+2)%n, ..., (i+K-1)%n
    """
    return [(i + j) % n for j in range(K)]


def build_model_at_rank(device, rank):
    """Fresh V22-Dickens base + apply LoRA at the given rank."""
    model, _ = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()
    return model


def load_adapter_set(procedure, indices):
    """Load 8 single-adapter state dicts from disk for the requested indices."""
    base = SEQ / ("adapters_a" if procedure == "A" else "adapters_b")
    pref = "adapter_a_" if procedure == "A" else "adapter_b_"
    return [torch.load(base / f"{pref}{i:02d}.pt", map_location="cpu",
                       weights_only=False)
            for i in indices]


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((PPD / "data/library.json").read_text())
    chosen_entries = [library[i] for i in CHOSEN_IDS]

    # Build Case 1 questions: 8 adapters × 3 held-out paraphrases.
    questions = []
    for local_i, entry in enumerate(chosen_entries):
        for q in entry["paraphrases_held_out"]:
            questions.append({
                "local_i": local_i,
                "library_id": entry["id"],
                "answer": entry["answer"],
                "probe": q,
            })
    print(f"Case 1 questions: {len(questions)} "
          f"(8 adapters × 3 paraphrases)")

    K_VALUES = [1, 2, 4, 8]

    all_results = {}
    t_total = time.time()
    for procedure in ["A", "B"]:
        print(f"\n=== Procedure {procedure} ===")
        results = []
        for K in K_VALUES:
            t_k = time.time()
            # Build a fresh model wrapped at rank K * 128
            model = build_model_at_rank(device, K * RANK_BASE)

            # For each question, build the adapter set and the stacked sd
            n_correct = 0; n_total = 0
            details = []
            for qi, q in enumerate(questions):
                idxs = adapter_subset(q["local_i"], K)
                sds = load_adapter_set(procedure, idxs)
                if len(sds) > 1:
                    stacked = stack_k_state_dicts(sds)
                else:
                    stacked = sds[0]
                stacked_gpu = {k: v.to(device) for k, v in stacked.items()}
                load_lora_state_dict(model, stacked_gpu)
                for seed in SEEDS:
                    ids_t = encode(tokenizer, q["probe"], device)
                    gen = generate(model, ids_t, GEN_TOKENS,
                                    gen_seed=seed * 10000 + qi)
                    full = tokenizer.decode(gen[0], skip_special_tokens=True)
                    cont = full[len(q["probe"]):]
                    hit = check_match(q["answer"], cont)
                    n_total += 1
                    if hit: n_correct += 1
            wall = time.time() - t_k
            results.append({
                "K": K,
                "n_total": n_total,
                "n_correct": n_correct,
                "retrieval": n_correct / n_total,
                "wall_s": wall,
            })
            print(f"  K={K}  retrieval={n_correct/n_total:.3f} "
                  f"({n_correct}/{n_total})  wall={wall:.0f}s")
            del model
            torch.cuda.empty_cache()
        all_results[procedure] = results

    out_path = REPO / "experiments/seq_adapter/results/case1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "results": all_results,
        "wall_total_s": time.time() - t_total,
        "K_values": K_VALUES,
    }, indent=2))
    print(f"\nCase 1 wall: {time.time()-t_total:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
