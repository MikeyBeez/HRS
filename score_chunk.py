"""Score a SlimPajama chunk: Shannon entropy via Mistral 7B.

Processes one chunk at a time on the NVMe for speed, saves results back to HDD.
Topic labeling is handled separately by label_topics.py.

Usage:
    python score_chunk.py <chunk_number>
    python score_chunk.py 0          # process chunk_0000.jsonl
    python score_chunk.py all        # process all unscored chunks
"""

import sys
import os
import json
import time
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Paths
HDD_DIR = Path("/mnt/data/Code/HRS/datasets/slimpajama")
NVME_WORK = Path("/tmp/slimpajama_work")
HDD_SCORED = Path("/mnt/data/Code/HRS/datasets/slimpajama_scored")

NVME_WORK.mkdir(exist_ok=True)
HDD_SCORED.mkdir(exist_ok=True)

# Scoring config
MAX_SCORE_TOKENS = 512   # score first 512 tokens for entropy
BATCH_SIZE = 4           # documents per batch for entropy scoring


def load_scorer():
    """Load Mistral 7B in 4-bit quantization."""
    print("Loading Mistral 7B (4-bit)...")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded, VRAM: {vram:.1f}GB")
    return model, tokenizer


@torch.no_grad()
def score_entropy(model, tokenizer, texts, max_tokens=MAX_SCORE_TOKENS):
    """Compute mean Shannon entropy for a batch of texts."""
    encodings = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=max_tokens, padding=True,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # (B, T-1, V)
    mask = attention_mask[:, 1:]  # (B, T-1)

    # Shannon entropy in bits
    probs = F.softmax(logits, dim=-1)
    log_probs = (probs + 1e-10).log2()
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, T-1)

    results = []
    for i in range(len(texts)):
        valid = mask[i].bool()
        if valid.sum() == 0:
            results.append({"mean_entropy": 0.0, "max_entropy": 0.0, "n_tokens": 0})
            continue
        ent = entropy[i][valid]
        results.append({
            "mean_entropy": ent.mean().item(),
            "max_entropy": ent.max().item(),
            "n_tokens": valid.sum().item(),
        })
    return results


def process_chunk(chunk_num, model, tokenizer):
    """Process a single chunk: copy to NVMe, score, save back to HDD."""
    chunk_name = f"chunk_{chunk_num:04d}.jsonl"
    src = HDD_DIR / chunk_name
    dst_scored = HDD_SCORED / f"scored_{chunk_num:04d}.jsonl"

    if not src.exists():
        print(f"  Chunk {chunk_name} not found, skipping")
        return False

    if dst_scored.exists():
        print(f"  Chunk {chunk_num} already scored, skipping")
        return True

    print(f"\n{'='*60}")
    print(f"Processing chunk {chunk_num}: {chunk_name}")
    print(f"{'='*60}")

    # Copy to NVMe
    work_file = NVME_WORK / chunk_name
    print(f"  Copying to NVMe...")
    shutil.copy2(src, work_file)

    # Load documents
    docs = []
    with open(work_file) as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"  Loaded {len(docs)} documents")

    # Score entropy in batches, saving every 1000 docs
    print(f"  Scoring entropy (batch_size={BATCH_SIZE})...")
    scored_work = NVME_WORK / f"scored_{chunk_num:04d}.jsonl"
    t0 = time.time()
    n_saved = 0

    # Resume from partial progress
    if scored_work.exists():
        n_saved = sum(1 for _ in open(scored_work))
        if n_saved > 0:
            print(f"  Resuming from {n_saved} already scored docs")
            docs = docs[n_saved:]

    out_f = open(scored_work, "a")
    buffer = []

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        texts = [d["text"][:4000] for d in batch]
        scores = score_entropy(model, tokenizer, texts)

        for j, score in enumerate(scores):
            doc = docs[i + j]
            doc.update(score)
            buffer.append(json.dumps({
                "text": doc["text"],
                "mean_entropy": doc.get("mean_entropy", 0),
                "max_entropy": doc.get("max_entropy", 0),
                "n_tokens": doc.get("n_tokens", 0),
                "topic": "",
                "source": doc.get("source", "unknown"),
            }))

        # Flush every 100 docs
        total_done = n_saved + min(i + BATCH_SIZE, len(docs))
        if len(buffer) >= 100 or (i + BATCH_SIZE) >= len(docs):
            out_f.write("\n".join(buffer) + "\n")
            out_f.flush()
            buffer = []

        if total_done % 1000 == 0 or (i + BATCH_SIZE) >= len(docs):
            elapsed = time.time() - t0
            rate = (i + BATCH_SIZE) / elapsed if elapsed > 0 else 0
            remaining = len(docs) - min(i + BATCH_SIZE, len(docs))
            eta = remaining / rate if rate > 0 else 0
            print(f"    {total_done}/{n_saved + len(docs)} scored ({rate:.1f} docs/s, ETA {eta/60:.0f}m)")

    out_f.close()

    entropy_time = time.time() - t0
    print(f"  Entropy scoring done in {entropy_time/60:.1f}m")

    # Move to HDD
    shutil.move(str(scored_work), str(dst_scored))
    print(f"  Saved to {dst_scored}")

    # Clean up NVMe
    work_file.unlink(missing_ok=True)

    # Stats
    entropies = [d.get("mean_entropy", 0) for d in docs]
    import numpy as np
    ent = np.array(entropies)
    print(f"\n  Chunk {chunk_num} stats:")
    print(f"    Documents: {len(docs)}")
    print(f"    Entropy: mean={ent.mean():.3f}, median={np.median(ent):.3f}, "
          f"std={ent.std():.3f}, min={ent.min():.3f}, max={ent.max():.3f}")
    print(f"    Time: {entropy_time/60:.1f}m")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_chunk.py <chunk_number|all>")
        sys.exit(1)

    model, tokenizer = load_scorer()

    if sys.argv[1] == "all":
        # Process all unscored chunks
        chunk_files = sorted(HDD_DIR.glob("chunk_*.jsonl"))
        for cf in chunk_files:
            num = int(cf.stem.split("_")[1])
            process_chunk(num, model, tokenizer)
    else:
        chunk_num = int(sys.argv[1])
        process_chunk(chunk_num, model, tokenizer)

    print("\nDone.")


if __name__ == "__main__":
    main()
