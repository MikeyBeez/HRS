"""Substrate validation: does the candidate base do extractive Q/A?

Procedure: 10 passage+probe pairs from per_passage_dickens. For each:
  prompt = passage + "\n\n" + probe_with_QA_format
  Generate, check substring match for the answer.

Pass criterion: >= 50% retrieval (5/10).

Run on Mistral-7B-v0.1 first; if it fails, try Mistral-7B-Instruct
(would need HF download).

Uses HuggingFace transformers directly — no custom HRS infrastructure.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))


def main():
    candidate = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-v0.1"
    print(f"=== Substrate probe: {candidate} ===")

    library = json.loads((REPO / "experiments/per_passage_dickens/data/library.json").read_text())
    chosen_ids = [0, 2, 16, 17, 22, 30, 31, 36, 41, 36]  # 10 probes (some repeats ok)
    chosen_ids = [0, 2, 16, 17, 22, 30, 31, 36, 41, 44]
    by_id = {e["id"]: e for e in library}

    print(f"Loading {candidate} (fp16) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(candidate)
    model = AutoModelForCausalLM.from_pretrained(
        candidate, torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s, params: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    # Use a simple Q/A format that prompts extractive behavior
    def make_prompt(passage, probe_subject):
        return (f"Passage: {passage}\n\n"
                f"Question: What is {probe_subject}?\n"
                f"Answer:")

    n = 0; n_hit = 0; details = []
    for cid in chosen_ids:
        if cid not in by_id:
            continue
        e = by_id[cid]
        passage = e["passage"]
        # Try to extract a question subject from the fact (loose)
        probe_subject = None
        # Use a simple template
        for ho in e.get("paraphrases_held_out", []):
            if ho.startswith("What is "):
                probe_subject = ho[len("What is "):].rstrip(" ?=")
                break
        if probe_subject is None:
            # Fall back: take held-out paraphrase as probe directly
            probe = e["paraphrases_held_out"][0] if e.get("paraphrases_held_out") else e["fact"]
            prompt = f"Passage: {passage}\n\n{probe}"
        else:
            prompt = make_prompt(passage, probe_subject)

        ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        hit = e["answer"].lower() in gen.lower()
        details.append({
            "id": cid, "answer": e["answer"],
            "prompt_first_120": prompt[:120].replace("\n", " ↵ "),
            "gen": gen[:80],
            "hit": hit,
        })
        n += 1
        if hit: n_hit += 1
        print(f"  [id={cid}] ans={e['answer']!r}  gen={gen[:60]!r}  hit={hit}")

    print(f"\nQ/A probe accuracy: {n_hit}/{n} = {n_hit/n:.2f}")
    print(f"Pass (>=0.50): {'YES' if n_hit/n >= 0.5 else 'NO'}")

    out_path = REPO / "experiments/four_way_compare/results/substrate_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "candidate": candidate,
        "n": n, "n_hit": n_hit, "rate": n_hit / n,
        "details": details,
    }, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
