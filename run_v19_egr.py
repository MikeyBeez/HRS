"""V19-EGR: Run engram store population, MAUVE, and NIAH for V19.

Reuses existing infrastructure with V19 checkpoint.

Usage:
    python run_v19_egr.py [--phase 1] [--device cuda]
    python run_v19_egr.py --phase all
"""

import argparse
import json
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore
from entropy_monitor import EntropyMonitor
from retrieval_engine import RetrievalEngine


ABLATION = "v19_exp_kernel"
RESULTS_DIR = Path("results/v19_exp_kernel")
STORE_DIR = Path("v19_engram_store_data")


def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V19_EXP_KERNEL)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(RESULTS_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded V19 (step {step}, val_ppl {val_ppl:.2f})")
    return model, cfg


# ============================================================
# Phase 1: Populate Store
# ============================================================

def phase1_populate(device):
    """Populate engram store from WikiText-103 validation set."""
    print("\n" + "=" * 60)
    print("PHASE 1: Populate Engram Store")
    print("=" * 60)

    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    store = EngramStore(d_model=cfg.model.d_model)
    engine = RetrievalEngine(model=model, store=store, write_threshold=4.0)

    from datasets import load_dataset
    from populate_store import extract_documents
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    documents = extract_documents(raw["validation"]["text"])
    print(f"Found {len(documents)} validation documents")

    t0 = time.time()
    n_segments = 0
    n_stored = 0
    for doc_idx, (title, text) in enumerate(documents):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 32:
            continue
        for seg_start in range(0, len(ids), 512):
            seg_ids = ids[seg_start:seg_start + 512]
            if len(seg_ids) < 32:
                continue
            seg_text = tokenizer.decode(seg_ids, skip_special_tokens=True)
            n_segments += 1
            stored, entropy = engine.process_segment(
                text=seg_text, tokenizer=tokenizer,
                condition="isolated", source=f"{title}:{seg_start}",
            )
            if stored:
                n_stored += 1
            if seg_start > 0:
                ctx_start = max(0, seg_start - 512)
                ctx_ids = torch.tensor(ids[ctx_start:seg_start], dtype=torch.long).unsqueeze(0)
                stored_ctx, _ = engine.process_segment(
                    text=seg_text, tokenizer=tokenizer,
                    condition="full_context", source=f"{title}:{seg_start}",
                    context_ids=ctx_ids,
                )
                if stored_ctx:
                    n_stored += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s: {n_segments} segments, {n_stored} stored")
    print(f"Store stats: {store.stats()}")
    store.save(str(STORE_DIR))
    return store


# ============================================================
# Phase 2: MAUVE with EGR
# ============================================================

def phase2_mauve(device):
    """Run MAUVE benchmark with EGR."""
    print("\n" + "=" * 60)
    print("PHASE 2: MAUVE with Entropy-Gated Retrieval")
    print("=" * 60)

    import mauve
    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    store = EngramStore.load(str(STORE_DIR))
    engine = RetrievalEngine(
        model=model, store=store,
        read_threshold=4.0, read_window=10,
        top_k=1, min_similarity=0.3,
    )

    from data import load_wikitext
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    # Use sliding-window generation matching the original benchmark
    from benchmark_mauve_v18 import generate_continuations, extract_prompts_and_refs

    n_samples = 1000
    continuation_len = 256
    temperature = 0.9
    top_k = 50

    results = []
    for prompt_len in [50, 500]:
        print(f"\n  Condition: {prompt_len}-tok + EGR")
        prompts, ref_ids = extract_prompts_and_refs(
            test_tokens, prompt_len, continuation_len, n_samples
        )
        ref_texts = [tokenizer.decode(ref_ids[i], skip_special_tokens=True)
                     for i in range(n_samples)]

        gen_texts = []
        t0 = time.time()
        for i in range(n_samples):
            gen_ids, stats = engine.generate_with_retrieval(
                prompts[i], max_new_tokens=continuation_len,
                temperature=temperature, top_k_sampling=top_k,
            )
            gen_texts.append(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{n_samples} ({elapsed:.0f}s)")

        print("    Computing MAUVE...")
        out = mauve.compute_mauve(
            p_text=ref_texts, q_text=gen_texts,
            device_id=0 if device.type == "cuda" else -1, verbose=False,
        )
        print(f"    MAUVE: {out.mauve:.4f}")
        results.append({"prompt_len": prompt_len, "mauve": out.mauve})

    # Compare to V18-EGR
    print(f"\n  V18-EGR comparison:")
    print(f"    50-tok:  V18=0.926, V19={results[0]['mauve']:.4f}")
    print(f"    500-tok: V18=0.950, V19={results[1]['mauve']:.4f}")

    out_path = RESULTS_DIR / "mauve_egr_results.json"
    with open(out_path, "w") as f:
        json.dump({"conditions": results}, f, indent=2)
    return results


# ============================================================
# Phase 3: Needle in a Haystack
# ============================================================

def phase3_niah(device):
    """Run NIAH test."""
    print("\n" + "=" * 60)
    print("PHASE 3: Needle in a Haystack")
    print("=" * 60)

    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    from niah_egr import (
        NEEDLES, DISTRACTORS, build_store_with_needle,
        test_retrieval, test_generation,
    )

    distractors = DISTRACTORS[:20]
    results = []

    for needle in NEEDLES:
        print(f"\n  Needle: {needle.category}")
        store = EngramStore(d_model=cfg.model.d_model)
        engine = RetrievalEngine(
            model=model, store=store, write_threshold=4.0, read_threshold=4.0,
        )
        build_store_with_needle(engine, tokenizer, needle, distractors)
        result = test_retrieval(engine, tokenizer, needle, top_k=5)

        found = "FOUND" if result["needle_found_in_top_k"] else "NOT FOUND"
        print(f"    {found} rank={result['needle_rank']} sim={result['needle_similarity']:.4f}")
        results.append(result)

    n_found = sum(1 for r in results if r["needle_found_in_top_k"])
    ranks = [r["needle_rank"] for r in results if r["needle_found_in_top_k"]]
    sims = [r["needle_similarity"] for r in results if r["needle_found_in_top_k"]]

    print(f"\n  Summary: {n_found}/5 found")
    if ranks:
        print(f"  Mean rank: {sum(ranks)/len(ranks):.1f}")
        print(f"  Mean similarity: {sum(sims)/len(sims):.4f}")
    print(f"  (V18-EGR: 5/5 found, mean rank 1.2, mean sim 0.51)")

    out_path = RESULTS_DIR / "niah_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_found": n_found}, f, indent=2, default=str)
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                        help="Phase to run: 1, 2, 3, or all")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    STORE_DIR.mkdir(exist_ok=True)

    phases = args.phase
    if phases == "all":
        phases = "123"

    if "1" in phases:
        phase1_populate(device)

    if "2" in phases:
        phase2_mauve(device)

    if "3" in phases:
        phase3_niah(device)

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
