"""Phase B: Generate ground-truth linguistic annotations via Ollama (DeepSeek R1).

Annotates three data sources:
  1. Passkey benchmark prompts (passages + prompts + paraphrases)
  2. WikiText samples (500 random 200-token passages)
  3. Diverse generated queries (500 spanning query types)

Each annotation follows the schema in schema.json. R1 fills in the form;
a small tagger later learns to reproduce these annotations.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/preprocessing/generate_ground_truth.py

Requires Ollama running locally with deepseek-r1 pulled.
"""

import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")

# -------------------------------------------------------------------
# Prompt sources
# -------------------------------------------------------------------

def passkey_prompts():
    """Collect all passkey benchmark prompts + passages + paraphrases."""
    from experiments.identity_ae.phase10_passkey import generate_passkeys
    from experiments.identity_ae.phase22_engram_key import stratified_tests
    from experiments.identity_ae.phase25_paraphrase import paraphrase as train_paraphrase
    from experiments.identity_ae.phase27_held_out import held_out_paraphrase

    tests = generate_passkeys(50)
    strat = stratified_tests()

    prompts = []
    for t in tests:
        prompts.append({"text": t["passage"], "source": "passkey_passage",
                        "meta": {"type": t["type"], "id": t["id"]}})
        prompts.append({"text": t["prompt"], "source": "passkey_prompt",
                        "meta": {"type": t["type"], "id": t["id"]}})

    for t in strat:
        for p in train_paraphrase(t):
            prompts.append({"text": p, "source": "passkey_train_para",
                            "meta": {"type": t["type"], "id": t["id"]}})
        for p in held_out_paraphrase(t):
            prompts.append({"text": p, "source": "passkey_held_out_para",
                            "meta": {"type": t["type"], "id": t["id"]}})

    return prompts


def wikitext_samples(n=500, max_tokens=200):
    """Random WikiText passages."""
    from transformers import AutoTokenizer
    from data import load_wikitext

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    splits, _ = load_wikitext("wikitext/wikitext-103-raw-v1", 512)
    texts = splits["train"]

    random.seed(42)
    indices = random.sample(range(len(texts)), min(n, len(texts)))
    prompts = []
    for idx in indices:
        item = texts[idx]
        # WikiTextDataset returns (x, y) tuples; we want x
        x = item[0] if isinstance(item, (list, tuple)) else item
        ids = x.tolist() if hasattr(x, "tolist") else list(x)
        ids = ids[:max_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if len(text) > 20:
            prompts.append({"text": text, "source": "wikitext",
                            "meta": {"index": idx}})
    return prompts[:n]


def diverse_queries(n=500):
    """Programmatically generated diverse query prompts."""
    random.seed(123)
    templates = {
        "factual": [
            "What is the capital of {country}?",
            "Who invented the {invention}?",
            "What year was {event} first recorded?",
        ],
        "numeric": [
            "How many {unit} does a {object} have?",
            "What is the population of {city}?",
            "What is the atomic number of {element}?",
        ],
        "entity": [
            "Who is the current CEO of {company}?",
            "What river flows through {city}?",
            "Which {role} won the {award} in {year}?",
        ],
        "technical": [
            "What is the boiling point of {substance} in {unit}?",
            "What algorithm does {system} use for {task}?",
            "What is the time complexity of {algorithm}?",
        ],
        "procedural": [
            "How do you {action} a {object}?",
            "What are the steps to {process}?",
            "Explain how to {task} using {tool}.",
        ],
        "compositional": [
            "What is the capital of {country1} and the population of {country2}?",
            "Who invented {invention1} and when was {invention2} first used?",
            "Compare the {property} of {thing1} and {thing2}.",
        ],
        "declarative": [
            "The {object} is located in {place}.",
            "{person} discovered {thing} in {year}.",
            "The protocol requires {number} signatories.",
        ],
    }
    fillers = {
        "country": ["France", "Japan", "Brazil", "Kenya", "Norway",
                     "Thailand", "Chile", "Egypt", "Canada", "India"],
        "country1": ["France", "Japan", "Brazil"],
        "country2": ["Kenya", "Norway", "Thailand"],
        "invention": ["the telephone", "penicillin", "the transistor",
                       "the printing press", "dynamite"],
        "invention1": ["the telephone", "penicillin"],
        "invention2": ["the transistor", "dynamite"],
        "event": ["the Olympics", "the census", "a solar eclipse"],
        "unit": ["legs", "chambers", "cylinders", "floors"],
        "object": ["spider", "heart", "engine", "building", "violin"],
        "city": ["Paris", "Tokyo", "Cairo", "Sydney", "Mumbai"],
        "element": ["oxygen", "gold", "iron", "neon", "lithium"],
        "company": ["Apple", "Toyota", "Samsung", "Novartis", "Shell"],
        "role": ["director", "actor", "scientist", "author"],
        "award": ["Nobel Prize", "Pulitzer Prize", "Oscar", "Fields Medal"],
        "year": ["2020", "1999", "1985", "2010", "1972"],
        "substance": ["water", "ethanol", "mercury", "nitrogen"],
        "system": ["GPT", "PageRank", "BERT", "MapReduce"],
        "task": ["ranking", "classification", "sorting", "compression"],
        "algorithm": ["quicksort", "dijkstra", "FFT", "gradient descent"],
        "action": ["calibrate", "assemble", "initialize", "configure"],
        "process": ["distillation", "fermentation", "crystallization"],
        "tool": ["Python", "a spectrometer", "a lathe", "MATLAB"],
        "place": ["the northern hemisphere", "Building 7", "sector 4"],
        "person": ["Dr. Voss", "Commander Petrov", "Agent Thornhill"],
        "thing": ["the anomaly", "Protocol X", "the resonance effect"],
        "thing1": ["steel", "aluminum", "copper"],
        "thing2": ["titanium", "glass", "ceramic"],
        "property": ["melting point", "tensile strength", "conductivity"],
        "number": ["7", "12", "22", "35"],
    }

    prompts = []
    for _ in range(n):
        qtype = random.choice(list(templates.keys()))
        template = random.choice(templates[qtype])
        text = template
        for key in re.findall(r"\{(\w+)\}", template):
            if key in fillers:
                text = text.replace("{" + key + "}", random.choice(fillers[key]), 1)
        prompts.append({"text": text, "source": f"diverse_{qtype}",
                        "meta": {"query_type": qtype}})
    return prompts


# -------------------------------------------------------------------
# Annotation generation via Ollama
# -------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).parent / "schema.json"

SYSTEM_PROMPT = """You are a linguistic annotation tool. You will receive a JSON schema and a text prompt. Fill in every field of the schema for the given prompt. Output ONLY valid JSON, no preamble, no markdown fences, no explanation.

Rules:
- tokenize by splitting on whitespace (keep punctuation attached to the preceding word if no space separates them, otherwise treat as its own token)
- every token gets an annotation in token_annotations, indexed from 0
- salience is 0.0-1.0: how much this token helps distinguish this prompt from other similar prompts
- key_discriminator is the single most distinctive element (usually a proper noun or rare term)
- content_type: "content" = carries meaning specific to this prompt, "template" = common phrasing reusable across prompts, "punctuation" = punctuation-only tokens
- entity_type: "none" for non-entity tokens
- be precise about query_type classification
- for passages (not questions), query_type should be "declarative"
- entities list should include all named entities, numbers, and technical terms mentioned
- salience should sum to roughly 1.0 across all tokens (distribute the budget)"""


def annotate_one(text, schema_str, max_retries=3):
    """Call Ollama and parse the JSON response."""
    user_msg = f"Schema:\n{schema_str}\n\nText to annotate:\n{text}"

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 4096},
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()

            # R1 may include <think>...</think> blocks — strip them
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if "```" in raw:
                    raw = raw[:raw.rfind("```")]
            raw = raw.strip()

            result = json.loads(raw)

            # Basic validation
            if "tokens" not in result or "token_annotations" not in result:
                raise ValueError("missing required fields")
            if len(result["tokens"]) != len(result["token_annotations"]):
                raise ValueError(
                    f"token count mismatch: {len(result['tokens'])} tokens "
                    f"vs {len(result['token_annotations'])} annotations")
            return result

        except (json.JSONDecodeError, ValueError, KeyError,
                requests.RequestException) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    return None


def main():
    # Check Ollama connectivity
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            print(f"WARNING: model '{OLLAMA_MODEL}' not found in Ollama. "
                  f"Available: {models}")
            print(f"Pull it with: ollama pull {OLLAMA_MODEL}")
            return
    except requests.RequestException:
        print(f"ERROR: cannot reach Ollama at {OLLAMA_HOST}")
        return

    out_dir = Path(__file__).parent / "ground_truth"
    out_dir.mkdir(exist_ok=True)
    schema_str = SCHEMA_PATH.read_text()

    # Collect all prompts
    print("Collecting prompts...")
    random.seed(0)
    all_prompts = []

    pk = passkey_prompts()
    print(f"  passkey prompts: {len(pk)}")
    all_prompts.extend(pk)

    wt = wikitext_samples(500)
    print(f"  wikitext samples: {len(wt)}")
    all_prompts.extend(wt)

    dq = diverse_queries(500)
    print(f"  diverse queries: {len(dq)}")
    all_prompts.extend(dq)

    print(f"  total: {len(all_prompts)}")

    # Check for already-completed annotations (resume support)
    manifest_path = out_dir / "manifest.csv"
    done = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(int(row["prompt_id"]))
        print(f"  already annotated: {len(done)}")

    # Open manifest for appending
    write_header = not manifest_path.exists() or len(done) == 0
    manifest_f = open(manifest_path, "a", newline="")
    writer = csv.DictWriter(manifest_f,
                            fieldnames=["prompt_id", "filename", "source",
                                        "text_preview"])
    if write_header:
        writer.writeheader()

    # Generate annotations
    n_success = len(done)
    n_fail = 0
    n_new = 0
    t0 = time.time()

    for i, prompt in enumerate(all_prompts):
        if i in done:
            continue

        result = annotate_one(prompt["text"], schema_str)
        if result is None:
            n_fail += 1
            print(f"  [{i+1}/{len(all_prompts)}] FAILED: "
                  f"{prompt['text'][:60]!r}")
            continue

        # Save annotation
        fname = f"prompt_{i:04d}.json"
        with open(out_dir / fname, "w") as f:
            json.dump({"prompt_id": i, "text": prompt["text"],
                       "source": prompt["source"], "meta": prompt["meta"],
                       "annotation": result}, f, indent=2)

        writer.writerow({
            "prompt_id": i, "filename": fname,
            "source": prompt["source"],
            "text_preview": prompt["text"][:80],
        })
        manifest_f.flush()
        n_success += 1
        n_new += 1

        if n_new % 25 == 0:
            elapsed = time.time() - t0
            rate = n_new / max(elapsed, 1)
            remaining = len(all_prompts) - i - 1
            eta = remaining / max(rate, 0.01)
            print(f"  [{i+1}/{len(all_prompts)}] "
                  f"success={n_success} fail={n_fail} new={n_new}  "
                  f"{rate:.2f}/s  ETA {eta/60:.0f}m")

    manifest_f.close()
    elapsed = time.time() - t0
    print(f"\nDone: {n_success} success, {n_fail} fail, "
          f"{n_new} new in {elapsed:.0f}s")
    print(f"Annotations saved to {out_dir}")


if __name__ == "__main__":
    main()
