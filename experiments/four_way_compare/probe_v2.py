"""Refined substrate probe with a few-shot extractive Q/A format.

Two-shot prompt with simple extractive questions, then the test passage +
test question. Tests whether better prompt engineering raises Mistral-7B
base above the 50% probe floor.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")


FEWSHOT = """Read the passage and answer the question with a short extracted answer.

Passage: The cat was named Whiskers and lived in a small blue house on Elm Street.
Question: What was the cat's name?
Answer: Whiskers

Passage: The lighthouse on the cliff was 90 feet tall and painted red and white.
Question: How tall was the lighthouse?
Answer: 90 feet

"""


def main():
    candidate = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-v0.1"
    print(f"=== Refined probe with few-shot: {candidate} ===")

    library = json.loads((REPO / "experiments/per_passage_dickens/data/library.json").read_text())
    chosen_ids = [0, 2, 16, 17, 22, 30, 31, 36, 41, 44]
    by_id = {e["id"]: e for e in library}

    QUESTIONS = {
        0:  ("What was Pip's father's family name?", "Pirrip"),
        2:  ("What was Joe Gargery's profession?", "blacksmith"),
        16: ("What was Miss Havisham's adopted daughter named?", "Estella"),
        17: ("What was the name of Mr. Jaggers's clerk?", "Wemmick"),
        22: ("What was the name of Estella's father?", "Provis"),
        30: ("Who was Pip's roommate at Barnard's Inn?", "Herbert"),
        31: ("Who was Pip's secret benefactor?", "Magwitch"),
        36: ("Who did Estella marry?", "Drummle"),
        41: ("Who was the swindler that jilted Miss Havisham?", "Compeyson"),
        44: ("Where did Pip end up working in a merchant house?", "Cairo"),
    }

    print(f"Loading {candidate} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(candidate)
    model = AutoModelForCausalLM.from_pretrained(
        candidate, torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    n = 0; n_hit = 0; details = []
    for cid, (q, ans) in QUESTIONS.items():
        e = by_id[cid]
        passage = e["passage"]
        prompt = FEWSHOT + f"Passage: {passage}\nQuestion: {q}\nAnswer:"

        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        # Stop at first newline
        gen_first_line = gen.split("\n")[0].strip()
        hit = ans.lower() in gen.lower()
        details.append({
            "id": cid, "question": q, "answer": ans,
            "gen_first_line": gen_first_line,
            "hit": hit,
        })
        n += 1
        if hit: n_hit += 1
        print(f"  [id={cid}] q={q[:50]!r}  ans={ans!r}  gen={gen_first_line!r}  hit={hit}")

    print(f"\nFew-shot Q/A probe accuracy: {n_hit}/{n} = {n_hit/n:.2f}")
    print(f"Pass (>=0.50): {'YES' if n_hit/n >= 0.5 else 'NO'}")

    out_path = REPO / "experiments/four_way_compare/results/substrate_probe_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "candidate": candidate,
        "format": "few-shot extractive",
        "n": n, "n_hit": n_hit, "rate": n_hit / n,
        "details": details,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
