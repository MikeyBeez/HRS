"""Case 2: 12 hand-crafted composition queries × K in {2, 4, 8} × {A, B}.

Each query has two target answer fragments. Score: ans_a hit, ans_b hit,
both hit. The "both hit" rate is the headline metric.

For K=2 the adapters are exactly (a, b) of the query.
For K=4 we add 2 distractors: indices (a+1)%8, (b+1)%8 (skipping if equal
to a or b; replace with next available index).
For K=8 all 8 adapters are loaded.
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

from experiments.seq_adapter.queries import CHOSEN_IDS, COMPOSITION_QUERIES

PPD = REPO / "experiments/per_passage_dickens"
SEQ = REPO / "experiments/seq_adapter"
RANK_BASE = 128
ALPHA_BASE = RANK_BASE * 2
GEN_TOKENS = 50  # longer to accommodate two answers
TEMPERATURE = 0.8
TOP_K = 50
SEEDS = (0, 1, 2)
N_LOCAL = 8


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


def build_model_at_rank(device, rank):
    model, _ = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()
    return model


def adapter_subset_for_pair(a, b, K):
    """K=2 -> [a, b].  K=4 -> [a, b, distractor1, distractor2] picked round-robin
    avoiding a/b.  K=8 -> all 8 in order [a, b, ...]."""
    base = [a, b]
    if K == 2:
        return base
    avail = [i for i in range(N_LOCAL) if i not in base]
    if K == 4:
        return base + avail[:2]
    if K == 8:
        return base + avail
    raise ValueError(K)


def lookup_local_id(library_id):
    """Map a library id (0, 2, 16, ...) to its local position 0..7."""
    return CHOSEN_IDS.index(library_id)


def load_adapter_local(procedure, local_idx):
    base = SEQ / ("adapters_a" if procedure == "A" else "adapters_b")
    pref = "adapter_a_" if procedure == "A" else "adapter_b_"
    return torch.load(base / f"{pref}{local_idx:02d}.pt",
                       map_location="cpu", weights_only=False)


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Case 2: {len(COMPOSITION_QUERIES)} composition queries × "
          f"K in {{2,4,8}} × procedures {{A,B}}")

    # Pre-resolve adapter local indices for each query
    queries = []
    for q in COMPOSITION_QUERIES:
        a_local = lookup_local_id(q["a"])
        b_local = lookup_local_id(q["b"])
        queries.append({**q, "a_local": a_local, "b_local": b_local})

    K_VALUES = [2, 4, 8]
    all_results = {}
    t_total = time.time()

    for procedure in ["A", "B"]:
        print(f"\n=== Procedure {procedure} ===")
        results = []
        for K in K_VALUES:
            t_k = time.time()
            model = build_model_at_rank(device, K * RANK_BASE)

            n_a = 0; n_b = 0; n_both = 0; n_total = 0
            details = []
            for qi, q in enumerate(queries):
                idxs = adapter_subset_for_pair(q["a_local"], q["b_local"], K)
                sds = [load_adapter_local(procedure, i) for i in idxs]
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
                    hit_a = check_match(q["ans_a"], cont)
                    hit_b = check_match(q["ans_b"], cont)
                    n_total += 1
                    if hit_a: n_a += 1
                    if hit_b: n_b += 1
                    if hit_a and hit_b: n_both += 1
                    details.append({
                        "query_i": qi, "K": K, "seed": seed,
                        "ans_a": q["ans_a"], "ans_b": q["ans_b"],
                        "hit_a": hit_a, "hit_b": hit_b,
                        "both": hit_a and hit_b,
                        "cont": cont[:120],
                    })
            wall = time.time() - t_k
            results.append({
                "K": K,
                "n_total": n_total,
                "n_a": n_a, "n_b": n_b, "n_both": n_both,
                "rate_a": n_a / n_total,
                "rate_b": n_b / n_total,
                "rate_both": n_both / n_total,
                "wall_s": wall,
                "details": details,
            })
            print(f"  K={K}  rate_a={n_a/n_total:.3f}  rate_b={n_b/n_total:.3f}  "
                  f"rate_both={n_both/n_total:.3f}  wall={wall:.0f}s")
            del model
            torch.cuda.empty_cache()
        all_results[procedure] = results

    out_path = REPO / "experiments/seq_adapter/results/case2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "results": all_results,
        "queries": COMPOSITION_QUERIES,
        "wall_total_s": time.time() - t_total,
        "K_values": K_VALUES,
    }, indent=2))
    print(f"\nCase 2 wall: {time.time()-t_total:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
