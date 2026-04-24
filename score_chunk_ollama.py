"""Score a SlimPajama chunk using Ollama (DeepSeek R1 or any model).

Designed for the Mac mini with Ollama. Connects to Ollama's local API
for topic labeling. Entropy scoring uses a lightweight approach based
on the Ollama response rather than requiring GPU forward passes.

Usage:
    python score_chunk_ollama.py <chunk_number>
    python score_chunk_ollama.py 16          # process chunk 16
    python score_chunk_ollama.py 16 30       # process chunks 16 through 30

Environment:
    OLLAMA_HOST: Ollama API URL (default: http://localhost:11434)
    OLLAMA_MODEL: model name (default: deepseek-r1)
"""

import sys
import os
import json
import time
import shutil
import urllib.request
import urllib.error
from pathlib import Path

# Paths — works whether run from main workstation or Mac mini via NFS
HDD_DIR = Path("/mnt/data/Code/HRS/datasets/slimpajama")
NVME_WORK = Path("/tmp/slimpajama_work")
HDD_SCORED = Path("/mnt/data/Code/HRS/datasets/slimpajama_scored")

NVME_WORK.mkdir(exist_ok=True)
HDD_SCORED.mkdir(exist_ok=True)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1")

MAX_TEXT_CHARS = 2000  # chars of document to send to Ollama


def ollama_generate(prompt, model=OLLAMA_MODEL, max_tokens=100):
    """Call Ollama API for text generation."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.3,
        }
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("response", "").strip()
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    Ollama error: {e}")
        return ""


def score_document(text):
    """Score a single document: get topic + quality assessment via Ollama.

    Since we can't do a forward pass for entropy, we ask the model to
    assess quality directly alongside the topic label.
    """
    truncated = text[:MAX_TEXT_CHARS]

    prompt = f"""Analyze this text and respond with EXACTLY two lines:
Line 1: The primary topic in 2-5 words
Line 2: A quality score from 1-10 (1=garbled/noisy, 5=average, 10=exceptionally clear and well-written)

Text: {truncated}

Topic:"""

    response = ollama_generate(prompt, max_tokens=50)

    # Parse response
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]

    topic = ""
    quality = 5  # default

    if len(lines) >= 1:
        topic = lines[0].strip().strip('"').strip("'")
        # Remove common prefixes
        for prefix in ["Topic:", "topic:", "1.", "1:"]:
            if topic.startswith(prefix):
                topic = topic[len(prefix):].strip()

    if len(lines) >= 2:
        # Try to extract number from second line
        quality_line = lines[-1]
        for prefix in ["Quality:", "quality:", "Score:", "score:", "2.", "2:"]:
            if quality_line.startswith(prefix):
                quality_line = quality_line[len(prefix):].strip()
        try:
            # Extract first number found
            import re
            nums = re.findall(r'\d+', quality_line)
            if nums:
                q = int(nums[0])
                if 1 <= q <= 10:
                    quality = q
        except (ValueError, IndexError):
            pass

    return topic, quality


def process_chunk(chunk_num):
    """Process a single chunk."""
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

    # Copy to local temp for faster reads
    work_file = NVME_WORK / chunk_name
    print(f"  Copying to temp dir...")
    shutil.copy2(src, work_file)

    # Load documents
    docs = []
    with open(work_file) as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"  Loaded {len(docs)} documents")

    # Test Ollama connection
    print(f"  Testing Ollama ({OLLAMA_HOST}, model={OLLAMA_MODEL})...")
    test = ollama_generate("Say OK", max_tokens=5)
    if not test:
        print("  ERROR: Cannot connect to Ollama. Is it running?")
        work_file.unlink(missing_ok=True)
        return False
    print(f"  Ollama OK (response: {test[:20]})")

    # Score all documents
    print(f"  Scoring {len(docs)} documents...")
    t0 = time.time()
    scored_docs = []

    for i, doc in enumerate(docs):
        topic, quality = score_document(doc["text"])

        scored = {
            "text": doc["text"],
            "mean_entropy": (10 - quality) / 10.0 * 8.0,  # map quality 1-10 to pseudo-entropy 0-8
            "max_entropy": 0.0,
            "n_tokens": min(len(doc["text"].split()), 512),
            "topic": topic,
            "quality": quality,
            "source": doc.get("source", "unknown"),
        }
        scored_docs.append(scored)

        if (i + 1) % 100 == 0 or (i + 1) == len(docs):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(docs) - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{len(docs)} scored ({rate:.1f} docs/s, ETA {eta/60:.0f}m)")

    total_time = time.time() - t0

    # Save scored version
    scored_work = NVME_WORK / f"scored_{chunk_num:04d}.jsonl"
    with open(scored_work, "w") as f:
        for doc in scored_docs:
            f.write(json.dumps(doc) + "\n")

    # Move to HDD
    shutil.move(str(scored_work), str(dst_scored))
    print(f"  Saved to {dst_scored}")

    # Clean up
    work_file.unlink(missing_ok=True)

    # Stats
    qualities = [d["quality"] for d in scored_docs]
    import numpy as np
    q = np.array(qualities)
    print(f"\n  Chunk {chunk_num} stats:")
    print(f"    Documents: {len(scored_docs)}")
    print(f"    Quality: mean={q.mean():.1f}, median={np.median(q):.1f}, "
          f"std={q.std():.1f}")
    print(f"    Low quality (<=3): {(q <= 3).sum()} ({(q <= 3).mean()*100:.1f}%)")
    print(f"    High quality (>=7): {(q >= 7).sum()} ({(q >= 7).mean()*100:.1f}%)")
    print(f"    Time: {total_time/60:.1f}m ({len(scored_docs)/total_time:.1f} docs/s)")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_chunk_ollama.py <start_chunk> [end_chunk]")
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2]) if len(sys.argv) > 2 else start

    for chunk_num in range(start, end + 1):
        process_chunk(chunk_num)

    print("\nDone.")


if __name__ == "__main__":
    main()
