"""Phase 01 (D2L) — Baseline cloze on Bleak House for the code-only base.

GO/NO-GO check. Does a code-only-pretrained ~1B model have any meaningful prior
on Dickens-specific content? If not, we have headroom to install a Bleak House
signal via D2L-style adapters in Phase 02.

Setup deviation from spec
-------------------------
Spec asks for bigcode/starcoder-1b. All bigcode/starcoder* and
bigcode/starcoderbase* checkpoints are gated (HTTP 401 in this env, no
HF_TOKEN). bigcode/santacoder (1.1B, the closest non-gated size match) ships
custom modeling code that imports `transformers.onnx`, which was removed in
transformers 5.x.

Substituted bigcode/starcoder2-3b: same BigCode family, code-only training on
The Stack v2 (no broad web/literature data), non-gated, native transformers
support. Size is 3B rather than 1B; if anything this *biases against* the
"floor-level Dickens knowledge" hypothesis (a larger model is more likely to
have incidental Dickens exposure), making the test more conservative.
Substitution called out in the README headline.

The cloze infrastructure is the same regardless of base.

Procedure
---------
1. Load tokenizer + model (fp16, frozen, eval).
2. Read cleaned Bleak House body (`results/d2l/phase01/bleak_house.txt`).
3. Build ~50 items per category:
   A. Character names (proper noun masked at first sub-token).
   B. Possessions / attributive nouns ("his ___", "a ___ of ...").
   C. Plot facts (event/relation tokens — e.g. "Inspector ___ Bucket").
   D. Code idioms (`import ___`, `def __init__(self, ___):`, etc.) as positive
      control — if SantaCoder fails this, the cloze infra is broken.
4. For each item: top-1, top-5, target log-prob, target rank.
5. Per category: aggregate metrics + spot-check generations.
6. Write cloze_items.json, baseline_results.json.
7. README is written separately.

Cloze design notes
------------------
* Prefix is built up to but not including the target sub-token. We measure the
  *next-token* distribution at the last position of the prefix.
* Target is the first sub-token of the target string under SantaCoder's
  tokenizer with a leading space (since the target is mid-text). This is the
  standard way to handle BPE-mid-text cloze.
* Category A character list is the central cast described in the spec. The
  builder scans Bleak House for sentence boundaries that contain "<Title>?
  <FirstName> <LastName>" patterns (or "<Title> <LastName>") followed by a
  verb, and masks the first sub-token of the first name (or last name when no
  first name appears).
* Category B uses templated regex over the body: " his|her|the <NOUN>",
  " a (small|black|white|...) <NOUN>". We sample ~50 instances and mask the
  noun's first sub-token.
* Category C uses Bleak-House-specific phrases: "Jarndyce and Jarndyce",
  "Court of Chancery", "Tom-all-Alone's", etc. We pull sentences containing
  these phrases and mask the second proper noun's first sub-token.
* Category D is hand-curated.

Usage
-----
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/d2l/phase01_starcoder_baseline.py
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "bigcode/starcoder2-3b"   # substituted for gated bigcode/starcoder-1b (see module docstring)
N_PER_CATEGORY = 50
MAX_PREFIX_TOKENS = 512


# ============================================================
# Cloze construction
# ============================================================

CENTRAL_CAST = [
    "Esther Summerson", "John Jarndyce", "Lady Dedlock", "Sir Leicester",
    "Mr. Tulkinghorn", "Inspector Bucket", "Richard Carstone", "Ada Clare",
    "Harold Skimpole", "Mr. Guppy", "Krook", "Mr. Snagsby", "Mrs. Jellyby",
    "Caddy Jellyby", "Mr. Boythorn", "Allan Woodcourt", "Hortense",
    "George Rouncewell", "Mrs. Rouncewell", "Tulkinghorn",
]

# Category B keywords — common nouns that show up in physical descriptions.
ATTR_NOUNS = [
    "bonnet", "shawl", "gloves", "cloak", "hat", "boots", "ribbon", "kerchief",
    "umbrella", "stick", "spectacles", "fan", "watch", "ring", "purse",
    "candle", "lantern", "letter", "parcel", "book",
]

# Category C — Bleak-House-specific named entities (the answer-side token).
PLOT_FACTS = [
    # (prefix-style cue, target after the cue)
    ("the case of ", "Jarndyce"),
    ("the Court of ", "Chancery"),
    ("Mr. Krook keeps a ", "rag"),         # rag and bottle shop
    ("Tom-all-", "Alone"),
    ("the slum known as ", "Tom"),
    ("the law-writer ", "Nemo"),
    ("Lady Dedlock's place is called ", "Chesney"),  # Chesney Wold
    ("Esther was raised by her ", "godmother"),
    ("the ward in Chancery, Richard ", "Carstone"),
    ("the disease that scars Esther is ", "smallpox"),
]

CODE_IDIOMS = [
    ("import ", "numpy"),
    ("import ", "os"),
    ("from collections import ", "defaultdict"),
    ("from typing import ", "List"),
    ("def __init__(self, ", "*"),
    ("if __name__ == '__", "main"),
    ("class Foo(", "object"),
    ("for i in ", "range"),
    ("with open(path, ", "'r"),
    ("return ", "None"),
    ("raise ", "ValueError"),
    ("self.", "_"),
    ("logger = logging.get", "Logger"),
    ("torch.nn.", "Linear"),
    ("np.", "array"),
    ("plt.", "plot"),
    ("os.path.", "join"),
    ("subprocess.", "run"),
    ("json.", "dumps"),
    ("re.", "compile"),
    ("'.split(", "'"),
    ("super().__", "init"),
    ("@property\ndef ", "name"),
    ("try:\n    ", "x"),
    ("except Exception as ", "e"),
    ("assert ", "x"),
    ("yield ", "x"),
    ("async def ", "main"),
    ("await asyncio.", "sleep"),
    ("pytest.", "fixture"),
    ("@pytest.", "fixture"),
    ("typing.", "Optional"),
    ("Path(__", "file"),
    ("logging.", "info"),
    ("argparse.", "ArgumentParser"),
    ("dataclasses.", "dataclass"),
    ("functools.", "partial"),
    ("itertools.", "chain"),
    ("collections.", "OrderedDict"),
    ("string.", "ascii_lowercase"),
    ("math.", "pi"),
    ("time.", "sleep"),
    ("random.", "seed"),
    ("hashlib.", "sha256"),
    ("base64.", "b64encode"),
    ("urllib.", "request"),
    ("requests.", "get"),
    ("flask.", "Flask"),
    ("django.", "db"),
    ("nn.", "Linear"),
]


def split_sentences(text: str) -> list[str]:
    # Light-weight sentence splitter; good enough for cloze mining.
    text = re.sub(r"\s+", " ", text)
    return re.findall(r"[^.!?]+[.!?]", text)


def build_character_items(text: str, tokenizer, n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items = []
    sentences = split_sentences(text)
    # For each central character, find sentences mentioning them mid-sentence.
    by_char = {c: [] for c in CENTRAL_CAST}
    for s in sentences:
        for char in CENTRAL_CAST:
            # mention must not be at the very start (need a prefix to mask after)
            idx = s.find(char)
            if idx > 20:        # at least a small prefix
                by_char[char].append((s, idx))
                break
    # Round-robin across characters until we have n items.
    keys = [c for c in CENTRAL_CAST if by_char[c]]
    rng.shuffle(keys)
    cursor = {c: 0 for c in keys}
    while len(items) < n and keys:
        progress = False
        for c in keys:
            if cursor[c] >= len(by_char[c]):
                continue
            s, idx = by_char[c][cursor[c]]
            cursor[c] += 1
            # Target = first sub-token of leading proper-noun word
            first_word = c.split()[0].lstrip("Mr.").lstrip("Mrs.").strip(". ")
            if not first_word:
                first_word = c.split()[-1]
            target_text = " " + first_word
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            if not target_ids:
                continue
            target_id = target_ids[0]
            target_str = tokenizer.decode([target_id])
            # Prefix: sentence text up to the character mention, minus the leading space we baked into the target.
            # We want the prefix to end with whatever precedes the target, so feed s[:idx] (and let the leading
            # space live on the target side).
            prefix = s[:idx].rstrip()
            items.append({
                "category": "A_character",
                "prefix": prefix,
                "target_token_id": target_id,
                "target_string": target_str,
                "passage_source": f"character={c!r}",
                "full_sentence_for_context": s.strip(),
            })
            progress = True
            if len(items) >= n:
                break
        if not progress:
            break
    return items[:n]


def build_attribute_items(text: str, tokenizer, n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items = []
    sentences = split_sentences(text)
    # Look for "his|her|the <NOUN>" patterns where NOUN ∈ ATTR_NOUNS.
    found = []
    for s in sentences:
        for noun in ATTR_NOUNS:
            m = re.search(r"\b(his|her|the|a|an)\s+(" + re.escape(noun) + r")\b", s, re.IGNORECASE)
            if m and m.start() > 20:
                found.append((s, m.start(2), noun))
    rng.shuffle(found)
    seen_per_noun = {}
    for s, start, noun in found:
        if seen_per_noun.get(noun, 0) >= 4:    # limit per-noun to keep diversity
            continue
        target_text = " " + noun
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]
        target_str = tokenizer.decode([target_id])
        prefix = s[:start].rstrip()
        items.append({
            "category": "B_possession",
            "prefix": prefix,
            "target_token_id": target_id,
            "target_string": target_str,
            "passage_source": f"noun={noun!r}",
            "full_sentence_for_context": s.strip(),
        })
        seen_per_noun[noun] = seen_per_noun.get(noun, 0) + 1
        if len(items) >= n:
            break
    return items[:n]


def build_plot_items(text: str, tokenizer, n: int, seed: int = 0) -> list[dict]:
    """Plot-fact cloze items derived from book-content phrases that the model
    has to *know* in order to predict (e.g. ``Jarndyce and ___`` → ``Jarndyce``).

    For each book-grounded phrase we find every occurrence in the text and
    construct one cloze item per occurrence (up to n total)."""
    rng = random.Random(seed)
    items = []
    # Concrete book-grounded phrases. Each entry: (prefix_in_text, target_word).
    phrase_targets = [
        ("Jarndyce and ", "Jarndyce"),
        ("Court of ", "Chancery"),
        ("Chesney ", "Wold"),
        ("Sir Leicester ", "Dedlock"),
        ("Lady ", "Dedlock"),
        ("Mr. ", "Tulkinghorn"),
        ("Inspector ", "Bucket"),
        ("Mrs. ", "Jellyby"),
        ("Mr. ", "Krook"),
        ("Mr. ", "Snagsby"),
        ("Mr. ", "Guppy"),
        ("Allan ", "Woodcourt"),
        ("Harold ", "Skimpole"),
        ("Esther ", "Summerson"),
        ("Richard ", "Carstone"),
        ("Ada ", "Clare"),
        ("Mr. ", "Boythorn"),
        ("George ", "Rouncewell"),
        ("Mrs. ", "Rouncewell"),
        ("Caddy ", "Jellyby"),
    ]
    locs = []
    for cue, target in phrase_targets:
        full = cue + target
        # find every occurrence
        start = 0
        while True:
            i = text.find(full, start)
            if i == -1:
                break
            locs.append((i, cue, target))
            start = i + len(full)
    rng.shuffle(locs)
    seen_per_pair = {}
    for i, cue, target in locs:
        key = (cue, target)
        if seen_per_pair.get(key, 0) >= 4:
            continue
        # Build a prefix: up to and including the cue, but starting from some context before.
        start = max(0, i - 220)
        prefix = text[start:i + len(cue)]
        # Strip leading partial word
        if prefix and prefix[0] not in " \n":
            first_ws = prefix.find(" ")
            if first_ws != -1:
                prefix = prefix[first_ws + 1:]
        # Target: first sub-token of the target word. Because the cue ends
        # with a space (e.g. "Jarndyce and "), encoding the target *without*
        # a leading space gives us the right sub-token at the next position.
        if cue.endswith(" "):
            target_text = target
        else:
            target_text = " " + target
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]
        target_str = tokenizer.decode([target_id])
        items.append({
            "category": "C_plot",
            "prefix": prefix,
            "target_token_id": target_id,
            "target_string": target_str,
            "passage_source": f"phrase={cue!r}->{target!r}",
            "full_sentence_for_context": text[start:i + len(cue) + len(target)].strip(),
        })
        seen_per_pair[key] = seen_per_pair.get(key, 0) + 1
        if len(items) >= n:
            break
    return items[:n]


def build_code_items(tokenizer) -> list[dict]:
    items = []
    for prefix, target in CODE_IDIOMS:
        # Always probe the next sub-token at the prefix's last position.
        # Encode target WITHOUT a leading space, since prefixes in CODE_IDIOMS
        # already end with whatever punctuation/space the idiom calls for.
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            continue
        target_id = target_ids[0]
        target_str = tokenizer.decode([target_id])
        items.append({
            "category": "D_code",
            "prefix": prefix,
            "target_token_id": target_id,
            "target_string": target_str,
            "passage_source": f"idiom={prefix!r}->{target!r}",
            "full_sentence_for_context": prefix + target,
        })
    return items


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def score_items(model, tokenizer, items, device, batch_size=8):
    results = []
    eos_id = tokenizer.eos_token_id or 0
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start:batch_start + batch_size]
        # Tokenize each prefix separately, right-pad with EOS.
        token_lists = []
        for it in batch:
            ids = tokenizer.encode(it["prefix"], add_special_tokens=False)
            ids = ids[-MAX_PREFIX_TOKENS:]
            token_lists.append(ids)
        max_len = max(len(t) for t in token_lists)
        input_ids = torch.full((len(batch), max_len), eos_id, dtype=torch.long)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long)
        last_idx = []
        for i, ids in enumerate(token_lists):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[i, :len(ids)] = 1
            last_idx.append(len(ids) - 1)
        input_ids = input_ids.to(device)
        attention = attention.to(device)
        out = model(input_ids=input_ids, attention_mask=attention)
        logits = out.logits   # [B, T, V]
        for i, it in enumerate(batch):
            last = last_idx[i]
            v = logits[i, last].float()
            log_probs = torch.log_softmax(v, dim=-1)
            target_id = it["target_token_id"]
            target_lp = float(log_probs[target_id].item())
            # rank
            sorted_ids = torch.argsort(v, descending=True)
            rank = int((sorted_ids == target_id).nonzero(as_tuple=True)[0].item())
            top1 = int(sorted_ids[0].item())
            top5 = sorted_ids[:5].tolist()
            top5_strs = [tokenizer.decode([int(t)]) for t in top5]
            results.append({
                **it,
                "rank": rank,
                "top1_hit": rank == 0,
                "top5_hit": rank < 5,
                "target_log_prob": target_lp,
                "model_top1_id": top1,
                "model_top1_str": tokenizer.decode([top1]),
                "model_top5_strs": top5_strs,
            })
    return results


@torch.no_grad()
def generate_greedy(model, tokenizer, prefix, device, n_tokens=20):
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    ids = ids[-MAX_PREFIX_TOKENS:]
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(n_tokens):
        out = model(input_ids=input_ids[:, -MAX_PREFIX_TOKENS:])
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, nxt], dim=1)
    gen_ids = input_ids[0, len(ids):].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# ============================================================
# Main
# ============================================================

def aggregate(category_results):
    if not category_results:
        return {}
    n = len(category_results)
    top1 = sum(int(r["top1_hit"]) for r in category_results) / n
    top5 = sum(int(r["top5_hit"]) for r in category_results) / n
    mean_lp = sum(r["target_log_prob"] for r in category_results) / n
    mean_rank = sum(r["rank"] for r in category_results) / n
    return {
        "n": n,
        "top1_acc": top1,
        "top5_acc": top5,
        "mean_target_log_prob": mean_lp,
        "mean_target_rank": mean_rank,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path("results/d2l/phase01")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 01 (D2L) — Baseline cloze on Bleak House")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print()

    print("Loading tokenizer + model (fp16) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time() - t0:.1f}s   "
          f"params={n_params / 1e9:.3f}B  vocab={tokenizer.vocab_size}  "
          f"context={tokenizer.model_max_length}")

    bleak = Path("results/d2l/phase01/bleak_house.txt").read_text(encoding="utf-8")
    print(f"  Bleak House: {len(bleak)} chars, {bleak.count(chr(10))} lines")

    # ---- Sanity forward ----
    sanity_prefix = "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n %"
    sanity_gen = generate_greedy(model, tokenizer, sanity_prefix, device, n_tokens=10)
    print(f"  sanity gen: {sanity_gen!r}")

    # ---- Build cloze items ----
    print("\nBuilding cloze items ...")
    items_a = build_character_items(bleak, tokenizer, N_PER_CATEGORY, seed=1)
    items_b = build_attribute_items(bleak, tokenizer, N_PER_CATEGORY, seed=2)
    items_c = build_plot_items(bleak, tokenizer, N_PER_CATEGORY, seed=3)
    items_d = build_code_items(tokenizer)
    print(f"  A character: {len(items_a)}")
    print(f"  B possession: {len(items_b)}")
    print(f"  C plot: {len(items_c)}")
    print(f"  D code: {len(items_d)}")

    all_items = items_a + items_b + items_c + items_d
    (results_dir / "cloze_items.json").write_text(json.dumps(all_items, indent=2))
    print(f"  wrote cloze_items.json ({len(all_items)} items)")

    # ---- Score ----
    print("\nScoring ...")
    t_score = time.time()
    scored = score_items(model, tokenizer, all_items, device, batch_size=8)
    print(f"  scored {len(scored)} items in {time.time() - t_score:.1f}s")

    # ---- Per-category aggregation ----
    by_cat = {}
    for r in scored:
        by_cat.setdefault(r["category"], []).append(r)

    print("\nPer-category metrics:")
    metrics = {}
    for cat in ["A_character", "B_possession", "C_plot", "D_code"]:
        rows = by_cat.get(cat, [])
        agg = aggregate(rows)
        metrics[cat] = agg
        if agg:
            print(f"  {cat:15s} n={agg['n']:3d}  "
                  f"top1={agg['top1_acc']*100:5.1f}%  "
                  f"top5={agg['top5_acc']*100:5.1f}%  "
                  f"mean_lp={agg['mean_target_log_prob']:+.3f}  "
                  f"mean_rank={agg['mean_target_rank']:7.0f}")

    # ---- Spot-check generations (5 per category) ----
    print("\nSpot-check generations:")
    spot = {}
    rng = random.Random(0)
    for cat in ["A_character", "B_possession", "C_plot", "D_code"]:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        sample = rng.sample(rows, min(5, len(rows)))
        spot[cat] = []
        for r in sample:
            gen = generate_greedy(model, tokenizer, r["prefix"], device, n_tokens=20)
            spot[cat].append({
                "prefix_tail": r["prefix"][-120:],
                "target": r["target_string"],
                "model_top1": r["model_top1_str"],
                "model_top5": r["model_top5_strs"],
                "rank": r["rank"],
                "target_log_prob": r["target_log_prob"],
                "greedy_continuation": gen,
            })
            print(f"  [{cat}] target={r['target_string']!r}  "
                  f"top1={r['model_top1_str']!r}  rank={r['rank']}")

    # ---- Save full results ----
    out = {
        "model_id": MODEL_ID,
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "metrics": metrics,
        "spot_check": spot,
        "items": scored,
    }
    (results_dir / "baseline_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote baseline_results.json")

    # ---- Headline summary ----
    print("\n" + "=" * 60)
    print("HEADLINE")
    print("=" * 60)
    a = metrics.get("A_character", {})
    d = metrics.get("D_code", {})
    print(f"  character-name top-1: {a.get('top1_acc', 0) * 100:.1f}%  "
          f"(spec target < 15%; floor band 2-5%)")
    print(f"  code idiom top-1:     {d.get('top1_acc', 0) * 100:.1f}%  "
          f"(spec sanity > 50%)")
    rec = "GREEN"
    if a.get("top1_acc", 0) >= 0.30 or d.get("top1_acc", 0) < 0.50:
        rec = "RED"
    elif a.get("top1_acc", 0) >= 0.15:
        rec = "YELLOW"
    print(f"  recommendation: {rec}")


if __name__ == "__main__":
    main()
