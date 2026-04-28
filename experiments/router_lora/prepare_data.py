"""Extract Great Expectations Ch 1 from the downloaded Project Gutenberg text,
normalize to the 65-char Shakespeare vocabulary, split into ~50 passages.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The exact 65-char Shakespeare vocabulary
SHAKESPEARE_CHARS = set("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# Char-level normalization: map Dickens chars to Shakespeare charset
NORMALIZE = {
    "‘": "'", "’": "'", "“": "'", "”": "'",   # smart quotes -> apostrophe (no `"` in vocab)
    "—": "-", "–": "-",                          # em/en dash -> hyphen
    "…": "...",                                  # ellipsis
    " ": " ", " ": " ", "​": "",  # whitespace forms
    "_": "",                                     # italic markers; drop
    "\r": "",                                    # carriage returns
    "0": " zero ", "1": " one ", "2": " two ",
    "4": " four ", "5": " five ", "6": " six ",
    "7": " seven ", "8": " eight ", "9": " nine ",
    # Shakespeare keeps the digit `3`, so only drop the others.
    "(": ",", ")": ",",                          # no parens; comma is closest
    "[": "", "]": "",                            # square brackets (illustration markers)
    "*": "",                                     # asterisks
    "/": " ",                                    # slash
    "é": "e",                               # é
    '"': "'",                                    # ASCII double quote -> apostrophe
    "%": " percent ",
    "+": " plus ",
    "=": " equals ",
}


def normalize(text: str) -> str:
    out = []
    for c in text:
        if c in NORMALIZE:
            out.append(NORMALIZE[c])
        elif c in SHAKESPEARE_CHARS:
            out.append(c)
        else:
            # Drop anything else
            pass
    s = "".join(out)
    # Collapse runs of whitespace (keep newlines)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_chapters_1_and_2(full_path: Path) -> str:
    """Extract Ch 1 + Ch 2 (combined) of Great Expectations."""
    text = full_path.read_text(encoding="utf-8")
    m = re.search(r"^Chapter I\.\s*$", text, flags=re.MULTILINE)
    assert m is not None, "Chapter I marker not found"
    start = m.end()
    m3 = re.search(r"^Chapter III\.\s*$", text, flags=re.MULTILINE)
    assert m3 is not None, "Chapter III marker not found"
    end = m3.start()
    return text[start:end].strip()


def split_long_sentence(sent: str, max_chars: int) -> list[str]:
    """If a sentence exceeds max_chars, hard-split it on commas/semicolons."""
    if len(sent) <= max_chars:
        return [sent]
    # Split on punctuation followed by whitespace
    parts = re.split(r"(?<=[,;:])\s+", sent)
    out = []
    cur = ""
    for p in parts:
        candidate = (cur + " " + p).strip() if cur else p
        if len(candidate) > max_chars and cur:
            out.append(cur)
            cur = p
        else:
            cur = candidate
    if cur:
        out.append(cur)
    # If still too long, hard-split on space
    final = []
    for o in out:
        while len(o) > max_chars:
            cut = o.rfind(" ", 0, max_chars)
            if cut < 0:
                cut = max_chars
            final.append(o[:cut].strip())
            o = o[cut:].strip()
        if o:
            final.append(o)
    return final


def split_into_passages(text: str, target_chars: int = 300, min_chars: int = 200,
                          max_chars: int = 400) -> list[str]:
    """Greedily concatenate sentences into passages of target_chars length.
    Hard caps: every passage in [min_chars, max_chars]."""
    sentences_raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for s in sentences_raw:
        sentences.extend(split_long_sentence(s.strip(), max_chars))

    passages = []
    cur = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = (cur + " " + sent).strip() if cur else sent
        if len(candidate) > max_chars and cur:
            passages.append(cur)
            cur = sent
        elif len(candidate) >= target_chars:
            passages.append(candidate)
            cur = ""
        else:
            cur = candidate
    if cur:
        if passages and len(passages[-1]) + len(cur) <= max_chars:
            passages[-1] = passages[-1] + " " + cur
        else:
            passages.append(cur)
    return [p for p in passages if len(p) >= min_chars]


def main():
    full = REPO / "experiments/router_lora/data/great_expectations_full.txt"
    raw = extract_chapters_1_and_2(full)
    norm = normalize(raw)
    passages = split_into_passages(norm)

    # Verify all chars in Shakespeare vocab
    for i, p in enumerate(passages):
        bad = [c for c in p if c not in SHAKESPEARE_CHARS]
        if bad:
            print(f"  Warning: passage {i} has unknown chars {bad[:10]!r}")
    bad_total = sum(1 for c in "".join(passages) if c not in SHAKESPEARE_CHARS)
    print(f"Passages: {len(passages)}")
    print(f"Total chars: {sum(len(p) for p in passages):,}")
    print(f"Avg passage length: {sum(len(p) for p in passages)/len(passages):.0f} chars")
    print(f"Min/max passage length: {min(len(p) for p in passages)}/{max(len(p) for p in passages)}")
    print(f"OOV chars total: {bad_total}")

    out_dir = REPO / "experiments/router_lora/data"
    out_passages = out_dir / "ge_ch12_passages.txt"
    with open(out_passages, "w") as f:
        for p in passages:
            f.write(p + "\n---\n")
    out_passages_json = out_dir / "ge_ch12_passages.json"
    with open(out_passages_json, "w") as f:
        json.dump({"passages": passages, "vocab_size": 65}, f, indent=2)
    print(f"\nSaved {out_passages} and {out_passages_json}")


if __name__ == "__main__":
    main()
