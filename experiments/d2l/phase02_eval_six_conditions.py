"""Phase 02 (D2L) — Six-condition cloze evaluation.

Loads the trained Perceiver checkpoint and runs every Phase 01 cloze item
through six conditions:

  C1 baseline               -- no document, no adapter, no engram
                               (cached from Phase 01 baseline_results.json)
  C2 RAG                    -- oracle in-context passage + cloze prefix
  C3 D2L adapter only       -- Perceiver(passage) -> LoRA, run cloze prefix
  C4 adapter + engram       -- LoRA + mean-pooled layer-16 engram as soft prefix
  C5 adapter + context      -- LoRA + passage in context
  C6 anti-suppression prompt -- prepend instruction not to emit <NAME>

For each item × condition, record top-1 token, top-5 tokens, target rank,
target log-prob, and whether top-1 == the literal token "<NAME>".

The "source passage" for each cloze item is the cloze item's
full_sentence_for_context, surrounded by a small window of Bleak House text
(±150 chars). This is what RAG and adapter conditions ingest.

Usage
-----
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/d2l/phase02_eval_six_conditions.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.d2l.phase02_perceiver_train import (
    Perceiver, install_lora_wrappers, set_lora, clear_lora,
    MODEL_ID, LORA_RANK, LATENT_N, LATENT_D, N_CROSS, N_SELF,
    extract_passage_hidden,
)


CHECKPOINT_PATH = Path("results/d2l/phase02/perceiver_checkpoint.pt")
BLEAK_PATH = Path("results/d2l/phase02/bleak_house.txt") if Path(
    "results/d2l/phase02/bleak_house.txt"
).exists() else Path("results/d2l/phase01/bleak_house.txt")
CLOZE_PATH = Path("results/d2l/phase01/cloze_items.json")
PHASE01_RESULTS_PATH = Path("results/d2l/phase01/baseline_results.json")
RESULTS_DIR = Path("results/d2l/phase02")

ANTI_SUPPRESSION_PREFIX = (
    "In the following passage, fill in the missing word with the actual name "
    "from the book, not the literal token <NAME>. The passage continues:\n\n"
)

ENGRAM_LAYER = 16
PASSAGE_WINDOW = 150          # chars around the source sentence for context conditions
MAX_PREFIX_TOKENS = 1500


def find_source_passage(bleak_text, sentence):
    """Locate the source sentence in the book and return a window around it."""
    if not sentence:
        return ""
    idx = bleak_text.find(sentence[:80])
    if idx < 0:
        return sentence
    start = max(0, idx - PASSAGE_WINDOW)
    end = min(len(bleak_text), idx + len(sentence) + PASSAGE_WINDOW)
    return bleak_text[start:end]


# ============================================================
# Scoring
# ============================================================

NAME_PLACEHOLDER = "<NAME>"


@torch.no_grad()
def score_one(base, tokenizer, prefix_ids, target_id, device,
              prefix_embeds=None):
    """Run base forward on prefix_ids (or prefix_embeds), return
    (top1_id, top5_ids, rank_of_target, target_logprob)."""
    if prefix_embeds is not None:
        out = base(inputs_embeds=prefix_embeds, use_cache=False)
    else:
        ids = torch.tensor([prefix_ids[-MAX_PREFIX_TOKENS:]], dtype=torch.long, device=device)
        out = base(ids, use_cache=False)
    logits = out.logits[0, -1].float()
    lp = F.log_softmax(logits, dim=-1)
    sorted_ids = torch.argsort(logits, descending=True)
    top1 = int(sorted_ids[0].item())
    top5 = [int(x) for x in sorted_ids[:5].tolist()]
    rank = int((sorted_ids == target_id).nonzero(as_tuple=True)[0].item())
    target_lp = float(lp[target_id].item())
    return top1, top5, rank, target_lp


def name_placeholder_id(tokenizer):
    # Single-token id for "<NAME>" if available.
    ids = tokenizer.encode(NAME_PLACEHOLDER, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


# ============================================================
# Condition runners
# ============================================================

def run_C1_baseline_from_phase01():
    """Reuse Phase 01 results."""
    if not PHASE01_RESULTS_PATH.exists():
        return None
    d = json.load(open(PHASE01_RESULTS_PATH))
    return d["items"]


def run_C2_rag(base, tokenizer, items, bleak_text, device):
    """Place source passage before the cloze prefix, score."""
    results = []
    for it in items:
        passage = find_source_passage(bleak_text, it.get("full_sentence_for_context", ""))
        full_prefix = passage + "\n\n" + it["prefix"]
        ids = tokenizer.encode(full_prefix, add_special_tokens=False)
        top1, top5, rank, lp = score_one(base, tokenizer, ids, it["target_token_id"], device)
        results.append({**it, "rank": rank, "top1_hit": rank == 0,
                        "top5_hit": rank < 5, "target_log_prob": lp,
                        "model_top1_id": top1, "model_top1_str": tokenizer.decode([top1]),
                        "model_top5_strs": [tokenizer.decode([t]) for t in top5]})
    return results


def run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak_text,
                      device, mode):
    """mode in {'adapter', 'adapter_engram', 'adapter_context'}."""
    results = []
    for it in items:
        passage = find_source_passage(bleak_text, it.get("full_sentence_for_context", ""))
        passage_ids = tokenizer.encode(passage, add_special_tokens=False)[:512]

        # Perceiver forward to produce LoRA
        passage_hidden = extract_passage_hidden(base, tokenizer, passage_ids, device)
        passage_hidden = passage_hidden.float()
        A_mats, B_mats = perceiver(passage_hidden)
        A_list = [A_mats[i].to(torch.float16) for i in range(A_mats.size(0))]
        B_list = [B_mats[i].to(torch.float16) for i in range(B_mats.size(0))]
        set_lora(wrappers, A_list, B_list)

        if mode == "adapter":
            ids = tokenizer.encode(it["prefix"], add_special_tokens=False)
            top1, top5, rank, lp = score_one(base, tokenizer, ids,
                                              it["target_token_id"], device)
        elif mode == "adapter_engram":
            # Mean-pooled engram from layer ENGRAM_LAYER, prepended at embedding level.
            with torch.no_grad():
                # Recompute hidden states with output_hidden_states for the chosen layer
                pids = torch.tensor([passage_ids], dtype=torch.long, device=device)
                base_out = base(pids, output_hidden_states=True, use_cache=False)
                h_layer = base_out.hidden_states[ENGRAM_LAYER]
                engram = h_layer.mean(dim=1, keepdim=True)            # (1, 1, H)
                # Prefix prompt embedding
                pre_ids = tokenizer.encode(it["prefix"], add_special_tokens=False)
                pre_ids_t = torch.tensor([pre_ids], dtype=torch.long, device=device)
                pre_embs = base.get_input_embeddings()(pre_ids_t)     # (1, T, H)
                full = torch.cat([engram.to(pre_embs.dtype), pre_embs], dim=1)
                top1, top5, rank, lp = score_one(base, tokenizer, None,
                                                  it["target_token_id"], device,
                                                  prefix_embeds=full)
        elif mode == "adapter_context":
            full_prefix = passage + "\n\n" + it["prefix"]
            ids = tokenizer.encode(full_prefix, add_special_tokens=False)
            top1, top5, rank, lp = score_one(base, tokenizer, ids,
                                              it["target_token_id"], device)
        else:
            raise ValueError(mode)

        clear_lora(wrappers)
        results.append({**it, "rank": rank, "top1_hit": rank == 0,
                        "top5_hit": rank < 5, "target_log_prob": lp,
                        "model_top1_id": top1, "model_top1_str": tokenizer.decode([top1]),
                        "model_top5_strs": [tokenizer.decode([t]) for t in top5]})
    return results


def run_C6_anti_suppression(base, tokenizer, items, device):
    """Prepend an anti-suppression prompt before the cloze prefix; no adapter."""
    results = []
    for it in items:
        full_prefix = ANTI_SUPPRESSION_PREFIX + it["prefix"]
        ids = tokenizer.encode(full_prefix, add_special_tokens=False)
        top1, top5, rank, lp = score_one(base, tokenizer, ids,
                                          it["target_token_id"], device)
        results.append({**it, "rank": rank, "top1_hit": rank == 0,
                        "top5_hit": rank < 5, "target_log_prob": lp,
                        "model_top1_id": top1, "model_top1_str": tokenizer.decode([top1]),
                        "model_top5_strs": [tokenizer.decode([t]) for t in top5]})
    return results


# ============================================================
# Aggregation
# ============================================================

def aggregate(results, name_id):
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    out = {}
    for cat, rows in by_cat.items():
        n = len(rows)
        top1 = sum(int(r["top1_hit"]) for r in rows) / n
        top5 = sum(int(r["top5_hit"]) for r in rows) / n
        mean_lp = sum(r["target_log_prob"] for r in rows) / n
        mean_rank = sum(r["rank"] for r in rows) / n
        if name_id is not None:
            name_rate = sum(1 for r in rows if r["model_top1_id"] == name_id) / n
        else:
            name_rate = None
        out[cat] = {"n": n, "top1_acc": top1, "top5_acc": top5,
                    "mean_target_log_prob": mean_lp, "mean_target_rank": mean_rank,
                    "name_placeholder_rate": name_rate}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 02 — Six-condition evaluation")
    print(f"  base: {MODEL_ID}")
    print(f"  checkpoint: {CHECKPOINT_PATH}")
    print()

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16,
    ).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False
    print(f"  base loaded in {time.time() - t_load:.1f}s")

    wrappers = install_lora_wrappers(base)

    # Load Perceiver checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    perceiver = Perceiver(
        base_hidden=cfg["base_hidden"], latent_n=cfg["latent_n"], latent_d=cfg["latent_d"],
        n_cross=cfg["n_cross"], n_self=cfg["n_self"],
        n_layers=cfg["n_layers"], rank=cfg["rank"],
        dim_in=cfg["ffn_in"], dim_out=cfg["ffn_out"],
    ).to(device).float().eval()
    perceiver.load_state_dict(ckpt["perceiver_state_dict"])
    print(f"  Perceiver loaded (step {ckpt['step']})")

    bleak = BLEAK_PATH.read_text(encoding="utf-8")
    items = json.load(open(CLOZE_PATH))
    print(f"  cloze items: {len(items)}")

    name_id = name_placeholder_id(tokenizer)
    print(f"  <NAME> token id: {name_id}")

    all_aggregates = {}

    # C1 baseline from Phase 01
    c1 = run_C1_baseline_from_phase01()
    if c1 is not None:
        agg = aggregate(c1, name_id)
        (RESULTS_DIR / "results_C1.json").write_text(json.dumps(c1, indent=2))
        all_aggregates["C1_baseline"] = agg
        print("\nC1 baseline (from Phase 01):")
        for cat, m in agg.items():
            print(f"  {cat:15s} top1={m['top1_acc']*100:5.1f}%  "
                  f"top5={m['top5_acc']*100:5.1f}%  "
                  f"name_rate={m['name_placeholder_rate']*100:5.1f}%  "
                  f"mean_lp={m['mean_target_log_prob']:+.2f}")

    with torch.no_grad():
        print("\nC2 RAG ...")
        t = time.time()
        c2 = run_C2_rag(base, tokenizer, items, bleak, device)
        (RESULTS_DIR / "results_C2.json").write_text(json.dumps(c2, indent=2))
        agg = aggregate(c2, name_id)
        all_aggregates["C2_rag"] = agg
        print(f"  ({time.time() - t:.0f}s)")
        for cat, m in agg.items():
            print(f"  {cat:15s} top1={m['top1_acc']*100:5.1f}%  "
                  f"top5={m['top5_acc']*100:5.1f}%  "
                  f"name_rate={m['name_placeholder_rate']*100:5.1f}%  "
                  f"mean_lp={m['mean_target_log_prob']:+.2f}")

        print("\nC3 D2L adapter only ...")
        t = time.time()
        c3 = run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak,
                               device, mode="adapter")
        (RESULTS_DIR / "results_C3.json").write_text(json.dumps(c3, indent=2))
        agg = aggregate(c3, name_id)
        all_aggregates["C3_adapter"] = agg
        print(f"  ({time.time() - t:.0f}s)")
        for cat, m in agg.items():
            print(f"  {cat:15s} top1={m['top1_acc']*100:5.1f}%  "
                  f"top5={m['top5_acc']*100:5.1f}%  "
                  f"name_rate={m['name_placeholder_rate']*100:5.1f}%  "
                  f"mean_lp={m['mean_target_log_prob']:+.2f}")

        print("\nC4 adapter + engram ...")
        t = time.time()
        c4 = run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak,
                               device, mode="adapter_engram")
        (RESULTS_DIR / "results_C4.json").write_text(json.dumps(c4, indent=2))
        agg = aggregate(c4, name_id)
        all_aggregates["C4_adapter_engram"] = agg
        print(f"  ({time.time() - t:.0f}s)")
        for cat, m in agg.items():
            print(f"  {cat:15s} top1={m['top1_acc']*100:5.1f}%  "
                  f"top5={m['top5_acc']*100:5.1f}%  "
                  f"name_rate={m['name_placeholder_rate']*100:5.1f}%  "
                  f"mean_lp={m['mean_target_log_prob']:+.2f}")

        print("\nC5 adapter + context ...")
        t = time.time()
        c5 = run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak,
                               device, mode="adapter_context")
        (RESULTS_DIR / "results_C5.json").write_text(json.dumps(c5, indent=2))
        agg = aggregate(c5, name_id)
        all_aggregates["C5_adapter_context"] = agg
        print(f"  ({time.time() - t:.0f}s)")
        for cat, m in agg.items():
            print(f"  {cat:15s} top1={m['top1_acc']*100:5.1f}%  "
                  f"top5={m['top5_acc']*100:5.1f}%  "
                  f"name_rate={m['name_placeholder_rate']*100:5.1f}%  "
                  f"mean_lp={m['mean_target_log_prob']:+.2f}")

        print("\nC6 anti-suppression prompt ...")
        t = time.time()
        c6 = run_C6_anti_suppression(base, tokenizer, items, device)
        (RESULTS_DIR / "results_C6.json").write_text(json.dumps(c6, indent=2))
        agg = aggregate(c6, name_id)
        all_aggregates["C6_anti_suppression"] = agg
        print(f"  ({time.time() - t:.0f}s)")
        for cat, m in agg.items():
            print(f"  {cat:15s} top1={m['top1_acc']*100:5.1f}%  "
                  f"top5={m['top5_acc']*100:5.1f}%  "
                  f"name_rate={m['name_placeholder_rate']*100:5.1f}%  "
                  f"mean_lp={m['mean_target_log_prob']:+.2f}")

    (RESULTS_DIR / "six_condition_aggregates.json").write_text(
        json.dumps(all_aggregates, indent=2),
    )
    print("\nwrote six_condition_aggregates.json")


if __name__ == "__main__":
    main()
