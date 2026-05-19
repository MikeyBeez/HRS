"""Phase 03 (D2L) — Six-condition cloze evaluation with measurement fixes.

Two measurement additions over Phase 02, applied uniformly to all conditions:

  1. Multi-token target acceptance. Phase 02 marked an item "miss" if the
     model's top-1 token ID differed from the target token ID — but the
     Phase 02 failure-mode analysis revealed that all four C2 RAG character
     "misses" were sub-word boundary mismatches where the model emitted
     correct output text via a different BPE split (target ` Richard`,
     model `Rich`+`ard`). Here we additionally greedy-generate up to 5
     tokens and report a "content_hit" if the target word appears as a
     prefix of the generated text. Both metrics are recorded.

  2. <NAME>-banned scoring. Phase 02 plot category went to 0% across every
     condition because StarCoder2's PII-anonymization training fires the
     literal token "<NAME>" as top-1 for "Mr. " / "Lady " / "Inspector "
     cues, even with the answer in context (the correct token was at
     rank 91-1116 underneath). Here we additionally compute top-1 and
     top-5 after masking the <NAME> token's logit to -inf. Both metrics
     are recorded so the C_plot question — "is the failure just the
     suppression token, or is it deeper?" — gets a measurable answer.

Usage
-----
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/d2l/phase03_eval_six_conditions.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.d2l.phase03_perceiver_train import (
    Perceiver, install_lora_wrappers, set_lora, clear_lora,
    MODEL_ID, extract_passage_hidden,
)


CHECKPOINT_PATH = Path("results/d2l/phase03/perceiver_checkpoint.pt")
BLEAK_PATH = Path("results/d2l/phase01/bleak_house.txt")
CLOZE_PATH = Path("results/d2l/phase01/cloze_items.json")
PHASE01_RESULTS_PATH = Path("results/d2l/phase01/baseline_results.json")
RESULTS_DIR = Path("results/d2l/phase03")

ANTI_SUPPRESSION_PREFIX = (
    "In the following passage, fill in the missing word with the actual name "
    "from the book, not the literal token <NAME>. The passage continues:\n\n"
)

ENGRAM_LAYER = 16
PASSAGE_WINDOW = 150
MAX_PREFIX_TOKENS = 1500
GEN_TOKENS = 5
NAME_PLACEHOLDER = "<NAME>"


# ============================================================
# Helpers
# ============================================================
def find_source_passage(bleak_text, sentence):
    if not sentence:
        return ""
    idx = bleak_text.find(sentence[:80])
    if idx < 0:
        return sentence
    start = max(0, idx - PASSAGE_WINDOW)
    end = min(len(bleak_text), idx + len(sentence) + PASSAGE_WINDOW)
    return bleak_text[start:end]


def name_placeholder_id(tokenizer):
    ids = tokenizer.encode(NAME_PLACEHOLDER, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


# ============================================================
# Scoring
# ============================================================
@torch.no_grad()
def score_item(base, tokenizer, prefix_ids, item, device, name_id,
                prefix_embeds=None):
    """Returns dict with strict (top-1 ID exact match), content (target word
    as prefix of greedy gen), and NAME-banned variants of both.
    """
    if prefix_embeds is not None:
        out = base(inputs_embeds=prefix_embeds, use_cache=False)
        first_logits = out.logits[0, -1].float()
    else:
        ids_t = torch.tensor([prefix_ids[-MAX_PREFIX_TOKENS:]], dtype=torch.long, device=device)
        out = base(ids_t, use_cache=False)
        first_logits = out.logits[0, -1].float()

    # Strict top-1/top-5/rank/log-prob on target token ID
    target_id = item["target_token_id"]
    sorted_ids = torch.argsort(first_logits, descending=True)
    rank_strict = int((sorted_ids == target_id).nonzero(as_tuple=True)[0].item())
    top1_strict = int(sorted_ids[0].item())
    top5_strict = [int(x) for x in sorted_ids[:5].tolist()]
    lp = F.log_softmax(first_logits, dim=-1)
    target_lp = float(lp[target_id].item())

    # NAME-banned variant
    banned_logits = first_logits.clone()
    if name_id is not None:
        banned_logits[name_id] = float("-inf")
    sorted_b = torch.argsort(banned_logits, descending=True)
    rank_nameban = int((sorted_b == target_id).nonzero(as_tuple=True)[0].item())
    top1_nameban = int(sorted_b[0].item())
    top5_nameban = [int(x) for x in sorted_b[:5].tolist()]

    # Multi-token content match: greedy-extend GEN_TOKENS tokens and check
    # whether the target word appears as a (case-insensitive) prefix of the
    # generated text.
    if prefix_embeds is not None:
        # We have to extend in embedding space, which is messy; fall back
        # to embedding-then-token extension: get the next-token argmax,
        # then continue in token space.
        first_id = int(first_logits.argmax().item())
        cur_ids = torch.tensor([[first_id]], dtype=torch.long, device=device)
        gen_ids = [first_id]
        for _ in range(GEN_TOKENS - 1):
            full = torch.cat([prefix_embeds, base.get_input_embeddings()(cur_ids)], dim=1)
            o = base(inputs_embeds=full, use_cache=False)
            nxt = int(o.logits[0, -1].argmax().item())
            gen_ids.append(nxt)
            cur_ids = torch.cat([cur_ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    else:
        cur = torch.tensor([prefix_ids[-MAX_PREFIX_TOKENS:]], dtype=torch.long, device=device)
        gen_ids = []
        for _ in range(GEN_TOKENS):
            o = base(cur[:, -MAX_PREFIX_TOKENS:], use_cache=False)
            nxt = int(o.logits[0, -1].argmax().item())
            gen_ids.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    # Content hit: target_word (stripped, case-insensitive) is a prefix of
    # gen_text (also stripped, case-insensitive). target_string includes the
    # leading space when applicable; strip it.
    target_word = item["target_string"].strip()
    gen_text_strip = gen_text.lstrip()
    content_hit = bool(
        target_word
        and target_word.lower() == gen_text_strip[:len(target_word)].lower()
    )

    # NAME-banned content hit: take the model's NAME-banned argmax for the
    # first position, then greedy from there.
    nameban_first = int(banned_logits.argmax().item())
    cur2 = (
        torch.tensor([prefix_ids[-MAX_PREFIX_TOKENS:] + [nameban_first]],
                     dtype=torch.long, device=device)
        if prefix_embeds is None
        else None
    )
    gen_nameban_ids = [nameban_first]
    if prefix_embeds is None:
        for _ in range(GEN_TOKENS - 1):
            o = base(cur2[:, -MAX_PREFIX_TOKENS:], use_cache=False)
            nxt = int(o.logits[0, -1].argmax().item())
            gen_nameban_ids.append(nxt)
            cur2 = torch.cat([cur2, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
        gen_nameban_text = tokenizer.decode(gen_nameban_ids, skip_special_tokens=False)
    else:
        # For embed-prefix path skip the multi-step NAME-banned continuation
        gen_nameban_text = tokenizer.decode([nameban_first], skip_special_tokens=False)
    content_hit_nameban = bool(
        target_word
        and target_word.lower() == gen_nameban_text.lstrip()[:len(target_word)].lower()
    )

    return {
        "target_string": item["target_string"],
        "target_word": target_word,
        "category": item["category"],
        "passage_source": item.get("passage_source", ""),
        "full_sentence_for_context": item.get("full_sentence_for_context", ""),
        "prefix": item["prefix"],
        # strict (Phase 02 metrics)
        "rank": rank_strict,
        "top1_hit": rank_strict == 0,
        "top5_hit": rank_strict < 5,
        "target_log_prob": target_lp,
        "model_top1_id": top1_strict,
        "model_top1_str": tokenizer.decode([top1_strict]),
        "model_top5_strs": [tokenizer.decode([t]) for t in top5_strict],
        # NAME-banned strict
        "rank_nameban": rank_nameban,
        "top1_hit_nameban": rank_nameban == 0,
        "top5_hit_nameban": rank_nameban < 5,
        "model_top1_nameban_str": tokenizer.decode([top1_nameban]),
        "model_top5_nameban_strs": [tokenizer.decode([t]) for t in top5_nameban],
        # Multi-token content
        "greedy_gen": gen_text,
        "content_hit": content_hit,
        "greedy_gen_nameban": gen_nameban_text,
        "content_hit_nameban": content_hit_nameban,
    }


# ============================================================
# Condition runners
# ============================================================
def run_C2_rag(base, tokenizer, items, bleak_text, device, name_id):
    out = []
    for it in items:
        passage = find_source_passage(bleak_text, it.get("full_sentence_for_context", ""))
        full_prefix = passage + "\n\n" + it["prefix"]
        ids = tokenizer.encode(full_prefix, add_special_tokens=False)
        out.append(score_item(base, tokenizer, ids, it, device, name_id))
    return out


def run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak_text,
                      device, name_id, mode):
    out = []
    for it in items:
        passage = find_source_passage(bleak_text, it.get("full_sentence_for_context", ""))
        passage_ids = tokenizer.encode(passage, add_special_tokens=False)[:512]
        passage_hidden = extract_passage_hidden(base, tokenizer, passage_ids, device)
        passage_hidden = passage_hidden.float()
        A_mats, B_mats = perceiver(passage_hidden)
        A_list = [A_mats[i].to(torch.float32) for i in range(A_mats.size(0))]
        B_list = [B_mats[i].to(torch.float32) for i in range(B_mats.size(0))]
        set_lora(wrappers, A_list, B_list)

        if mode == "adapter":
            ids = tokenizer.encode(it["prefix"], add_special_tokens=False)
            r = score_item(base, tokenizer, ids, it, device, name_id)
        elif mode == "adapter_engram":
            with torch.no_grad():
                pids = torch.tensor([passage_ids], dtype=torch.long, device=device)
                base_out = base(pids, output_hidden_states=True, use_cache=False)
                h_layer = base_out.hidden_states[ENGRAM_LAYER]
                engram = h_layer.mean(dim=1, keepdim=True)
                pre_ids = tokenizer.encode(it["prefix"], add_special_tokens=False)
                pre_ids_t = torch.tensor([pre_ids], dtype=torch.long, device=device)
                pre_embs = base.get_input_embeddings()(pre_ids_t)
                full = torch.cat([engram.to(pre_embs.dtype), pre_embs], dim=1)
                r = score_item(base, tokenizer, pre_ids, it, device, name_id,
                                prefix_embeds=full)
        elif mode == "adapter_context":
            full_prefix = passage + "\n\n" + it["prefix"]
            ids = tokenizer.encode(full_prefix, add_special_tokens=False)
            r = score_item(base, tokenizer, ids, it, device, name_id)
        else:
            raise ValueError(mode)

        clear_lora(wrappers)
        out.append(r)
    return out


def run_C6(base, tokenizer, items, device, name_id):
    out = []
    for it in items:
        full_prefix = ANTI_SUPPRESSION_PREFIX + it["prefix"]
        ids = tokenizer.encode(full_prefix, add_special_tokens=False)
        out.append(score_item(base, tokenizer, ids, it, device, name_id))
    return out


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
        out[cat] = {
            "n": n,
            "top1_acc": sum(int(r["top1_hit"]) for r in rows) / n,
            "top5_acc": sum(int(r["top5_hit"]) for r in rows) / n,
            "content_acc": sum(int(r["content_hit"]) for r in rows) / n,
            "top1_acc_nameban": sum(int(r["top1_hit_nameban"]) for r in rows) / n,
            "top5_acc_nameban": sum(int(r["top5_hit_nameban"]) for r in rows) / n,
            "content_acc_nameban": sum(int(r["content_hit_nameban"]) for r in rows) / n,
            "mean_target_log_prob": sum(r["target_log_prob"] for r in rows) / n,
            "mean_target_rank": sum(r["rank"] for r in rows) / n,
            "mean_target_rank_nameban": sum(r["rank_nameban"] for r in rows) / n,
            "name_placeholder_rate": (
                sum(1 for r in rows if r["model_top1_id"] == name_id) / n
                if name_id is not None else None
            ),
        }
    return out


def fmt_cat(cat, m):
    return (f"  {cat:15s} n={m['n']:3d}  "
            f"top1={m['top1_acc']*100:5.1f}%  "
            f"content={m['content_acc']*100:5.1f}%  "
            f"nameban_top1={m['top1_acc_nameban']*100:5.1f}%  "
            f"nameban_content={m['content_acc_nameban']*100:5.1f}%  "
            f"mean_rank={m['mean_target_rank']:7.0f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Phase 03 — Six-condition evaluation with measurement fixes", flush=True)
    print(f"  base: {MODEL_ID}", flush=True)
    print(f"  checkpoint: {CHECKPOINT_PATH}", flush=True)

    t_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16,
    ).to(device).eval()
    for p in base.parameters():
        p.requires_grad = False
    print(f"  base loaded in {time.time() - t_load:.1f}s", flush=True)
    wrappers = install_lora_wrappers(base)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    perceiver = Perceiver(
        base_hidden=cfg["base_hidden"], latent_n=cfg["latent_n"], latent_d=cfg["latent_d"],
        n_cross=cfg["n_cross"], n_self=cfg["n_self"],
        n_layers=cfg["n_layers"], rank=cfg["rank"],
        dim_in=cfg["ffn_in"], dim_out=cfg["ffn_out"],
    ).to(device).float().eval()
    perceiver.load_state_dict(ckpt["perceiver_state_dict"])
    print(f"  Perceiver loaded (step {ckpt['step']}, alpha {cfg.get('alpha', '?')}, "
          f"fp32_lora {cfg.get('fp32_lora', False)})", flush=True)

    bleak = BLEAK_PATH.read_text(encoding="utf-8")
    items = json.load(open(CLOZE_PATH))
    print(f"  cloze items: {len(items)}", flush=True)

    name_id = name_placeholder_id(tokenizer)
    print(f"  <NAME> token id: {name_id}", flush=True)

    aggregates = {}

    # ---- C1: re-score Phase 01 items with the same scoring infrastructure
    # so multi-token / NAME-banned numbers are apples-to-apples.
    print("\nC1 baseline (re-scored) ...", flush=True)
    t = time.time()
    with torch.no_grad():
        c1 = [score_item(base, tokenizer, tokenizer.encode(it["prefix"], add_special_tokens=False),
                          it, device, name_id) for it in items]
    (RESULTS_DIR / "results_C1.json").write_text(json.dumps(c1, indent=2))
    aggregates["C1_baseline"] = aggregate(c1, name_id)
    print(f"  ({time.time() - t:.0f}s)", flush=True)
    for cat, m in aggregates["C1_baseline"].items():
        print(fmt_cat(cat, m), flush=True)

    with torch.no_grad():
        for cond, label, runner in [
            ("C2", "C2 RAG", lambda: run_C2_rag(base, tokenizer, items, bleak, device, name_id)),
            ("C3", "C3 D2L adapter only",
             lambda: run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak,
                                       device, name_id, mode="adapter")),
            ("C4", "C4 adapter + engram",
             lambda: run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak,
                                       device, name_id, mode="adapter_engram")),
            ("C5", "C5 adapter + context",
             lambda: run_with_adapter(base, wrappers, perceiver, tokenizer, items, bleak,
                                       device, name_id, mode="adapter_context")),
            ("C6", "C6 anti-suppression",
             lambda: run_C6(base, tokenizer, items, device, name_id)),
        ]:
            print(f"\n{label} ...", flush=True)
            t = time.time()
            rows = runner()
            (RESULTS_DIR / f"results_{cond}.json").write_text(json.dumps(rows, indent=2))
            agg = aggregate(rows, name_id)
            aggregates[label.replace(" ", "_").replace("+", "plus")] = agg
            print(f"  ({time.time() - t:.0f}s)", flush=True)
            for cat, m in agg.items():
                print(fmt_cat(cat, m), flush=True)

    (RESULTS_DIR / "six_condition_aggregates.json").write_text(
        json.dumps(aggregates, indent=2),
    )
    print("\nwrote six_condition_aggregates.json", flush=True)

    # ============================================================
    # Latency benchmark — how much overhead does the adapter add?
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("LATENCY BENCHMARK", flush=True)
    print("=" * 60, flush=True)
    bench_items = items[:30]   # subset, but reliable averages
    bleak = BLEAK_PATH.read_text(encoding="utf-8")

    def cuda_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def time_one_forward(prefix_ids):
        ids = torch.tensor([prefix_ids[-MAX_PREFIX_TOKENS:]], dtype=torch.long, device=device)
        cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            base(ids, use_cache=False)
        cuda_sync()
        return time.perf_counter() - t0

    def time_perceiver(passage_ids):
        with torch.no_grad():
            ph = extract_passage_hidden(base, tokenizer, passage_ids, device)
            ph = ph.float()
            cuda_sync()
            t0 = time.perf_counter()
            A, B = perceiver(ph)
            A_list = [A[i].to(torch.float32) for i in range(A.size(0))]
            B_list = [B[i].to(torch.float32) for i in range(B.size(0))]
            set_lora(wrappers, A_list, B_list)
        cuda_sync()
        return time.perf_counter() - t0

    # Warmup
    sample = bench_items[0]
    p = find_source_passage(bleak, sample.get("full_sentence_for_context", ""))
    p_ids = tokenizer.encode(p, add_special_tokens=False)[:512]
    for _ in range(2):
        time_one_forward(tokenizer.encode(sample["prefix"], add_special_tokens=False))
        time_perceiver(p_ids)
    clear_lora(wrappers)

    timings = {
        "base_only_cloze_prefix":       [],
        "base_only_rag_prefix":         [],
        "perceiver_passage_to_adapter": [],
        "base_with_adapter_cloze":      [],
        "base_with_adapter_rag":        [],
        "anti_suppression_prefix":      [],
    }
    prefix_token_counts = {"cloze_prefix": [], "rag_prefix": [], "anti_supp_prefix": []}

    for it in bench_items:
        passage = find_source_passage(bleak, it.get("full_sentence_for_context", ""))
        passage_ids = tokenizer.encode(passage, add_special_tokens=False)[:512]
        cloze_ids = tokenizer.encode(it["prefix"], add_special_tokens=False)
        rag_ids = tokenizer.encode(passage + "\n\n" + it["prefix"], add_special_tokens=False)
        anti_ids = tokenizer.encode(ANTI_SUPPRESSION_PREFIX + it["prefix"],
                                     add_special_tokens=False)

        prefix_token_counts["cloze_prefix"].append(len(cloze_ids))
        prefix_token_counts["rag_prefix"].append(len(rag_ids))
        prefix_token_counts["anti_supp_prefix"].append(len(anti_ids))

        # Base only on cloze (C1, C6 baseline operation; ignore anti-supp prepend)
        clear_lora(wrappers)
        timings["base_only_cloze_prefix"].append(time_one_forward(cloze_ids))

        # Base only on RAG prefix (C2 operation)
        timings["base_only_rag_prefix"].append(time_one_forward(rag_ids))

        # Anti-suppression prefix (C6 operation)
        timings["anti_suppression_prefix"].append(time_one_forward(anti_ids))

        # Perceiver inference: passage hidden -> LoRA matrices + install
        timings["perceiver_passage_to_adapter"].append(time_perceiver(passage_ids))

        # Base with adapter loaded — cloze prefix (C3 op given adapter is set)
        timings["base_with_adapter_cloze"].append(time_one_forward(cloze_ids))

        # Base with adapter loaded — RAG prefix (C5 op given adapter is set)
        timings["base_with_adapter_rag"].append(time_one_forward(rag_ids))

        clear_lora(wrappers)

    import statistics as stats
    timing_summary = {}
    for k, vs in timings.items():
        timing_summary[k] = {
            "n": len(vs),
            "mean_ms": stats.mean(vs) * 1000,
            "median_ms": stats.median(vs) * 1000,
            "stdev_ms": stats.stdev(vs) * 1000 if len(vs) > 1 else 0.0,
            "min_ms": min(vs) * 1000,
            "max_ms": max(vs) * 1000,
        }
    timing_summary["prefix_token_counts"] = {
        k: {"mean": stats.mean(v), "median": stats.median(v),
             "min": min(v), "max": max(v)}
        for k, v in prefix_token_counts.items()
    }

    # Derived: end-to-end per-query latency by condition
    p_amort = timing_summary["perceiver_passage_to_adapter"]["mean_ms"]
    base_cloze = timing_summary["base_only_cloze_prefix"]["mean_ms"]
    base_rag = timing_summary["base_only_rag_prefix"]["mean_ms"]
    adapter_cloze = timing_summary["base_with_adapter_cloze"]["mean_ms"]
    adapter_rag = timing_summary["base_with_adapter_rag"]["mean_ms"]
    anti = timing_summary["anti_suppression_prefix"]["mean_ms"]

    timing_summary["per_query_latency_ms"] = {
        "C1_baseline_cloze":          base_cloze,
        "C2_rag":                     base_rag,
        "C3_adapter_per_query_cold":  p_amort + adapter_cloze,
        "C3_adapter_amortized":       adapter_cloze,
        "C5_adapter_plus_rag_cold":   p_amort + adapter_rag,
        "C5_adapter_plus_rag_amort":  adapter_rag,
        "C6_anti_suppression":        anti,
    }
    timing_summary["per_query_overhead_vs_base_cloze"] = {
        "C2_rag":                     base_rag / base_cloze,
        "C3_adapter_per_query_cold":  (p_amort + adapter_cloze) / base_cloze,
        "C3_adapter_amortized":       adapter_cloze / base_cloze,
        "C5_adapter_plus_rag_cold":   (p_amort + adapter_rag) / base_cloze,
        "C5_adapter_plus_rag_amort":  adapter_rag / base_cloze,
        "C6_anti_suppression":        anti / base_cloze,
    }

    print("\nPer-call latency (ms):", flush=True)
    for k, m in timing_summary.items():
        if not isinstance(m, dict) or "mean_ms" not in m:
            continue
        print(f"  {k:35s}  mean={m['mean_ms']:7.2f}  median={m['median_ms']:7.2f}  "
              f"std={m['stdev_ms']:6.2f}  range=[{m['min_ms']:6.2f}, {m['max_ms']:6.2f}]",
              flush=True)
    print("\nPer-query end-to-end latency by condition (ms):", flush=True)
    for k, v in timing_summary["per_query_latency_ms"].items():
        print(f"  {k:35s}  {v:7.2f} ms", flush=True)
    print("\nLatency overhead vs C1 base-cloze (x):", flush=True)
    for k, v in timing_summary["per_query_overhead_vs_base_cloze"].items():
        print(f"  {k:35s}  {v:5.2f}x", flush=True)

    (RESULTS_DIR / "timing_benchmark.json").write_text(
        json.dumps(timing_summary, indent=2),
    )
    print("\nwrote timing_benchmark.json", flush=True)


if __name__ == "__main__":
    main()
