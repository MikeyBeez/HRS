"""Populate engram store from WikiText-103 validation set using V18 checkpoint.

Processes documents through V18, computes entropy, and stores engrams for
high-entropy segments. Stores both full-context and isolated engrams.

Usage:
    python populate_store.py [--threshold 4.0] [--output engram_store_data]
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore
from retrieval_engine import RetrievalEngine


def load_model(device, ablation="v18_cross_attn"):
    """Load trained model from best checkpoint."""
    ablation_map = {a.value: a for a in AblationConfig}
    cfg = ExperimentConfig.from_ablation(ablation_map[ablation])
    model = HRSTransformer(cfg).to(device)

    ckpt_path = Path(f"results/{ablation}/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True

    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded V18 (step {step}, val_ppl {val_ppl:.2f})")
    return model, cfg


def extract_documents(texts: list) -> list:
    """Extract documents from WikiText-103 raw text.

    Returns list of (title, text) tuples.
    """
    import re
    title_pattern = re.compile(r'^\s*=\s+([^=]+?)\s+=\s*$')
    documents = []
    current_title = None
    current_lines = []

    for line in texts:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('= ='):
            current_lines.append(stripped)
            continue
        m = title_pattern.match(stripped)
        if m:
            if current_title and current_lines:
                documents.append((current_title.strip(), "\n".join(current_lines)))
            current_title = m.group(1)
            current_lines = []
        else:
            current_lines.append(stripped)

    if current_title and current_lines:
        documents.append((current_title.strip(), "\n".join(current_lines)))

    return documents


def main():
    parser = argparse.ArgumentParser(description="Populate engram store from WikiText-103")
    parser.add_argument("--threshold", type=float, default=4.0, help="Entropy threshold for storage")
    parser.add_argument("--output", type=str, default="engram_store_data", help="Output directory")
    parser.add_argument("--max-docs", type=int, default=None, help="Max documents to process")
    parser.add_argument("--segment-len", type=int, default=512, help="Segment length in tokens")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablation", type=str, default="v18_cross_attn",
                        help="Ablation config (e.g., v18_cross_attn, v19_exp_kernel)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model, cfg = load_model(device, ablation=args.ablation)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create store and engine
    store = EngramStore(d_model=cfg.model.d_model)
    engine = RetrievalEngine(
        model=model,
        store=store,
        write_threshold=args.threshold,
    )

    # Load WikiText-103 validation set
    print("\nLoading WikiText-103 validation set...")
    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    documents = extract_documents(raw["validation"]["text"])
    print(f"Found {len(documents)} documents in validation set")

    if args.max_docs:
        documents = documents[:args.max_docs]
        print(f"Processing first {len(documents)} documents")

    # Process documents
    print(f"\nPopulating store (threshold={args.threshold} bits)...")
    t0 = time.time()
    n_segments = 0
    n_stored = 0

    for doc_idx, (title, text) in enumerate(documents):
        # Tokenize document
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 32:  # skip very short documents
            continue

        # Process in segments of segment_len tokens
        for seg_start in range(0, len(ids), args.segment_len):
            seg_ids = ids[seg_start:seg_start + args.segment_len]
            if len(seg_ids) < 32:
                continue

            seg_text = tokenizer.decode(seg_ids, skip_special_tokens=True)
            n_segments += 1

            # Condition A: isolated engram (just this segment)
            stored, entropy = engine.process_segment(
                text=seg_text,
                tokenizer=tokenizer,
                condition="isolated",
                source=f"{title}:{seg_start}",
            )
            if stored:
                n_stored += 1

            # Condition B: full-context engram (with preceding context)
            if seg_start > 0:
                ctx_start = max(0, seg_start - args.segment_len)
                ctx_ids = torch.tensor(ids[ctx_start:seg_start], dtype=torch.long).unsqueeze(0)
                stored_ctx, entropy_ctx = engine.process_segment(
                    text=seg_text,
                    tokenizer=tokenizer,
                    condition="full_context",
                    source=f"{title}:{seg_start}",
                    context_ids=ctx_ids,
                )
                if stored_ctx:
                    n_stored += 1

        if (doc_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {doc_idx + 1}/{len(documents)} docs | "
                  f"{n_segments} segments | {n_stored} stored | "
                  f"{elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Processed {n_segments} segments, stored {n_stored} engrams")
    print(f"Storage rate: {n_stored / max(n_segments, 1) * 100:.1f}%")

    # Print stats
    stats = store.stats()
    print(f"\nStore statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Save
    store.save(args.output)


if __name__ == "__main__":
    main()
