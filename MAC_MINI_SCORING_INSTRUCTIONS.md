# Mac Mini Scoring Instructions

## Overview

The Mac mini scores SlimPajama chunks using DeepSeek R1 via Ollama. The main workstation (192.168.12.174) uses Mistral 7B on GPU for its chunks. Each machine processes different chunks.

## Setup

### 1. Mount the NFS drive

```bash
sudo mkdir -p /mnt/data
sudo mount -t nfs -o resvport 192.168.12.174:/mnt/data /mnt/data
ls /mnt/data/Code/HRS/  # verify
```

### 2. Verify Ollama is running with DeepSeek R1

```bash
ollama list                    # should show deepseek-r1
curl http://localhost:11434/   # should respond "Ollama is running"
```

If DeepSeek R1 isn't pulled yet:
```bash
ollama pull deepseek-r1
```

### 3. Install Python dependencies

```bash
cd /mnt/data/Code/HRS
pip3 install numpy
```

That's it — the Ollama script only needs numpy and stdlib. No torch, no transformers, no GPU libraries.

### 4. Run scoring

```bash
cd /mnt/data/Code/HRS

# Score chunks 16-30 (Mac mini's assignment)
python3 score_chunk_ollama.py 16 30

# Or one at a time if you prefer
python3 score_chunk_ollama.py 16
python3 score_chunk_ollama.py 17
# ...
```

### 5. Chunk assignment

| Machine | Chunks | Script |
|---------|--------|--------|
| Main workstation (192.168.12.174) | 0-15 | score_chunk.py (Mistral 7B, GPU) |
| Mac mini (192.168.12.125) | 16-30 | score_chunk_ollama.py (DeepSeek R1, Ollama) |

The scripts auto-skip already-scored chunks, so no collision risk.

### 6. Monitor progress

```bash
ls /mnt/data/Code/HRS/datasets/slimpajama_scored/
```

### 7. What the scoring does

For each document in the chunk, it asks DeepSeek R1:
- **Topic:** a 2-5 word topic label
- **Quality:** a 1-10 score (1=garbled, 5=average, 10=excellent)

Quality replaces the entropy score from the GPU-based script. Low quality (1-3) = will be filtered out. High quality (7-10) = keeper.

### 8. Expected speed

DeepSeek R1 via Ollama on Apple Silicon:
- ~2-5 docs/second depending on model size and RAM
- ~100K docs per chunk
- ~6-14 hours per chunk
- 15 chunks total: ~4-9 days

### 9. Output format

Each line in scored_NNNN.jsonl:
```json
{
  "text": "full document text",
  "mean_entropy": 4.8,
  "max_entropy": 0.0,
  "n_tokens": 512,
  "topic": "marine biology",
  "quality": 7,
  "source": "CommonCrawl"
}
```

### 10. Troubleshooting

- **NFS mount drops:** `sudo mount -t nfs -o resvport 192.168.12.174:/mnt/data /mnt/data`
- **Ollama not responding:** `ollama serve` (restart it)
- **Script crashes mid-chunk:** just re-run, it skips completed chunks
- **Use a different model:** `OLLAMA_MODEL=llama3 python3 score_chunk_ollama.py 16 30`
