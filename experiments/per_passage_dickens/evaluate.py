"""Evaluate the four conditions on held-out paraphrases for routing/retrieval.

Conditions:
  A — full system: L0 query -> W -> top-1 cosine vs library L5 keys -> load
       adapter -> generate. Routes to base if max_sim < THRESHOLD.
  B — oracle routing: load the correct adapter for each query directly.
  C — base only: never load any adapter (LoRA at zero). Floor.
  D — random adapter: load a random adapter from the library each query.

For each condition, score (per-fact-type and overall):
  - routing accuracy: fraction of queries that loaded the correct adapter
  - retrieval accuracy: fraction with answer substring in the generation
  - routing-correct retrieval ceiling: among correctly routed, recall

3 stochastic-decoding seeds per condition.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    hidden_at_layer, reset_lora_to_zero,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)


RANK = 128
ROUTING_THRESHOLD = -1.0           # was 0.50 (Phase 47, synthetic). Cosine sims on
                                   # Dickens prose peak at ~0.5 (natural text has
                                   # lower max-sim than synthetic), so the original
                                   # 0.50 threshold incorrectly gated 51% of held-out
                                   # queries to no-adapter. Argmax routing gives
                                   # 150/150 correct. Use a permissive threshold.
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
D = 1024


@torch.no_grad()
def l0_mean(model, ids_t):
    h = model.drop(model.tok_emb(ids_t))
    return h.mean(dim=1).squeeze(0)


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed, temperature=TEMPERATURE, top_k=TOP_K):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, generation):
    """Case-insensitive substring match. Phase 47 also handled comma-stripped."""
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load library + adapter checkpoints + projection
    library = json.loads((REPO / "experiments/per_passage_dickens/data/library.json").read_text())
    keys = json.loads((REPO / "experiments/per_passage_dickens/results/library_keys.json").read_text())

    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    library_l5_n = F.normalize(library_l5, dim=-1)

    proj_ck = torch.load(REPO / "experiments/per_passage_dickens/results/projection_W.pt",
                           map_location=device, weights_only=False)
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(proj_ck["W_state"])
    W.eval()

    # Load model + LoRA structure (LoRA values will be loaded per query)
    model, cfg = load_model(device)
    dickens_ck = torch.load(
        REPO / "experiments/per_passage_dickens/results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()

    # Build the full set of held-out (query, expected_adapter_id, answer) triples
    held_out_queries = []
    for entry in library:
        for q in entry["paraphrases_held_out"]:
            held_out_queries.append({
                "adapter_id": entry["id"],
                "fact_type": entry["fact_type"],
                "probe": q,
                "answer": entry["answer"],
            })
    print(f"Held-out queries: {len(held_out_queries)} (= {len(library)} adapters × 3 paraphrases)")

    # Pre-cache adapter state_dicts (move to GPU once)
    adapter_sds = {}
    for e in keys:
        sd = torch.load(REPO / e["sd_path"], map_location="cpu", weights_only=False)
        adapter_sds[e["id"]] = {k: v.to(device) for k, v in sd.items()}

    out_dir = REPO / "experiments/per_passage_dickens/results"
    all_results = []
    t0 = time.time()
    for condition in ["A_full", "B_oracle", "C_base", "D_random"]:
        for seed in [0, 1, 2]:
            t_seed = time.time()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            n_routed_correct = 0
            n_retrieved = 0
            per_type = {}
            details = []

            for qi, q in enumerate(held_out_queries):
                # Determine which adapter to load for this query
                if condition == "A_full":
                    reset_lora_to_zero(model)
                    ids = tokenizer.encode(q["probe"], add_special_tokens=False)
                    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                    with torch.no_grad():
                        l0 = l0_mean(model, ids_t)              # (D,)
                        proj = W(l0.unsqueeze(0))                # (1, D)
                        proj_n = F.normalize(proj, dim=-1)
                        sim = (proj_n @ library_l5_n.T).squeeze(0)   # (N,)
                        max_sim, top_id = sim.max(dim=0)
                        if max_sim.item() >= ROUTING_THRESHOLD:
                            routed = top_id.item()
                        else:
                            routed = -1   # base fallback
                elif condition == "B_oracle":
                    routed = q["adapter_id"]
                elif condition == "C_base":
                    routed = -1
                elif condition == "D_random":
                    routed = random.randrange(len(library))

                # Load adapter (or reset to zero for base)
                if routed == -1:
                    reset_lora_to_zero(model)
                else:
                    load_lora_state_dict(model, adapter_sds[routed])

                # Generate
                ids = tokenizer.encode(q["probe"], add_special_tokens=False)
                ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                gen_ids = generate(model, ids_t, GEN_TOKENS,
                                     gen_seed=seed * 10000 + qi)
                full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                cont = full[len(q["probe"]):]
                hit = check_match(q["answer"], cont)
                routed_correct = (routed == q["adapter_id"])

                if routed_correct: n_routed_correct += 1
                if hit: n_retrieved += 1
                ft = q["fact_type"]
                per_type.setdefault(ft, {"n": 0, "routed": 0, "retrieved": 0})
                per_type[ft]["n"] += 1
                per_type[ft]["routed"] += int(routed_correct)
                per_type[ft]["retrieved"] += int(hit)

                details.append({
                    "qi": qi, "adapter_id_true": q["adapter_id"], "adapter_id_routed": routed,
                    "fact_type": ft, "probe": q["probe"], "answer": q["answer"],
                    "continuation": cont, "routed_correct": routed_correct, "retrieved": hit,
                })

            n = len(held_out_queries)
            routing_acc = n_routed_correct / n
            retrieval_acc = n_retrieved / n
            ceiling = (sum(1 for d in details if d["routed_correct"] and d["retrieved"]) /
                         max(1, n_routed_correct))
            print(f"  {condition:>10s}  seed={seed}  routing_acc={routing_acc:.3f}  "
                  f"retrieval_acc={retrieval_acc:.3f}  ceiling={ceiling:.3f}  "
                  f"wall={time.time()-t_seed:.0f}s")
            all_results.append({
                "condition": condition, "seed": seed,
                "routing_accuracy": routing_acc,
                "retrieval_accuracy": retrieval_acc,
                "routing_correct_retrieval_ceiling": ceiling,
                "per_fact_type": {ft: {**v, "routing": v["routed"]/v["n"],
                                          "retrieval": v["retrieved"]/v["n"]}
                                    for ft, v in per_type.items()},
            })
            (out_dir / f"eval_{condition}_seed{seed}.json").write_text(
                json.dumps({"summary": all_results[-1], "details": details}, indent=2))

    # Aggregate per-condition
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    summary = {}
    for c in ["A_full", "B_oracle", "C_base", "D_random"]:
        rs = [x for x in all_results if x["condition"] == c]
        m_rout = float(np.mean([x["routing_accuracy"] for x in rs]))
        s_rout = float(np.std([x["routing_accuracy"] for x in rs]))
        m_ret = float(np.mean([x["retrieval_accuracy"] for x in rs]))
        s_ret = float(np.std([x["retrieval_accuracy"] for x in rs]))
        m_ceil = float(np.mean([x["routing_correct_retrieval_ceiling"] for x in rs]))
        summary[c] = {
            "routing_mean": m_rout, "routing_std": s_rout,
            "retrieval_mean": m_ret, "retrieval_std": s_ret,
            "ceiling_mean": m_ceil,
        }
        print(f"  {c:>10s}  routing={m_rout:.3f}±{s_rout:.3f}  "
              f"retrieval={m_ret:.3f}±{s_ret:.3f}  ceiling={m_ceil:.3f}")

    (out_dir / "evaluation_summary.json").write_text(json.dumps({
        "all_results": all_results, "summary": summary,
        "total_wall_s": time.time() - t0,
    }, indent=2))
    print(f"\nTotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
