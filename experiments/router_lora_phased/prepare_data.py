"""BPE-tokenize Tiny Shakespeare + Great Expectations, split GE by chapters
into LoRA-train / router-train / held-out-eval tertiles.

Output:
  data/shakespeare_train.pt, shakespeare_val.pt — 1D token tensors
  data/dickens_lora.pt        (chapters I-XXX, for Phase 2 LoRA training)
  data/dickens_router.pt      (chapters XXXI-L, for Phase 3 router training)
  data/dickens_eval.pt        (chapters LI-LIX, for held-out eval)
  data/dickens_eval_text.txt  (held-out chapters as text — for query auth)
  data/info.json              (vocab info, sizes)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]


def _normalize(text: str) -> str:
    """Light normalization: keep typography intact for BPE."""
    text = text.replace("\r", "")
    return text


def _shakespeare() -> str:
    p = REPO / "datasets/tiny_shakespeare.txt"
    return _normalize(p.read_text())


def _ge_full() -> str:
    p = REPO / "experiments/router_lora/data/great_expectations_full.txt"
    return _normalize(p.read_text())


def split_ge_by_chapters(text: str):
    """Return (lora_text, router_text, eval_text) by chapter ranges.

    Chapters I-XXX -> lora (first half = phase-2 training).
    Chapters XXXI-L -> router (second half, used for Phase 3 router training).
    Chapters LI-LIX -> eval (held-out).
    """
    # Find every chapter heading and its line offset.
    chapter_re = re.compile(r"^Chapter ([IVXL]+)\.\s*$", re.MULTILINE)
    matches = list(chapter_re.finditer(text))
    assert len(matches) >= 59, f"expected 59 chapters, found {len(matches)}"

    # Boundaries for our three groups
    ch1_start = matches[0].start()           # start of Chapter I
    ch31_start = matches[30].start()         # start of Chapter XXXI
    ch51_start = matches[50].start()         # start of Chapter LI
    end = text.find("*** END", ch51_start)
    if end < 0:
        end = len(text)

    lora_text = text[ch1_start:ch31_start]
    router_text = text[ch31_start:ch51_start]
    eval_text = text[ch51_start:end]
    return lora_text, router_text, eval_text


def main():
    out_dir = REPO / "experiments/router_lora_phased/data"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPT-2 tokenizer...")
    tok = AutoTokenizer.from_pretrained("gpt2")
    print(f"vocab_size = {tok.vocab_size}")

    print("\n--- Shakespeare ---")
    sh_text = _shakespeare()
    n_chars = len(sh_text)
    sh_ids = tok.encode(sh_text)
    n = len(sh_ids)
    split = int(0.9 * n)
    sh_train_ids = sh_ids[:split]
    sh_val_ids = sh_ids[split:]
    print(f"  chars={n_chars:,}  tokens={n:,}  train={len(sh_train_ids):,}  val={len(sh_val_ids):,}")
    torch.save(torch.tensor(sh_train_ids, dtype=torch.long), out_dir / "shakespeare_train.pt")
    torch.save(torch.tensor(sh_val_ids, dtype=torch.long), out_dir / "shakespeare_val.pt")

    print("\n--- Great Expectations ---")
    ge_text = _ge_full()
    lora_text, router_text, eval_text = split_ge_by_chapters(ge_text)

    for name, txt in [("dickens_lora", lora_text),
                       ("dickens_router", router_text),
                       ("dickens_eval", eval_text)]:
        ids = tok.encode(txt)
        torch.save(torch.tensor(ids, dtype=torch.long), out_dir / f"{name}.pt")
        print(f"  {name}: chars={len(txt):,}  tokens={len(ids):,}")

    # Save the eval text as plaintext for query authoring
    (out_dir / "dickens_eval_text.txt").write_text(eval_text)

    info = {
        "vocab_size": tok.vocab_size,
        "shakespeare_train_tokens": len(sh_train_ids),
        "shakespeare_val_tokens": len(sh_val_ids),
        "dickens_lora_tokens": len(tok.encode(lora_text)),
        "dickens_router_tokens": len(tok.encode(router_text)),
        "dickens_eval_tokens": len(tok.encode(eval_text)),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(f"\nSaved info.json: {info}")


if __name__ == "__main__":
    main()
