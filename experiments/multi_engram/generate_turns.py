"""Generate 100 prompt-response pairs about Civil War subtopics using
Mistral-7B-v0.1 base with a 2-shot prompt.

Each generated turn = "Q: {prompt}\nA: {response}"

Mistral base isn't instruction-tuned, but few-shot extractive Q/A
elicits factual content reasonably well (this was validated in the
four_way_compare experiment).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/multi_engram"
sys.path.insert(0, str(REPO))

from experiments.multi_engram.topics import ALL_TOPICS

BASE = "mistralai/Mistral-7B-v0.1"


FEWSHOT = """Below are short Q&A pairs about the U.S. Civil War.

Q: Tell me about the Battle of Antietam.
A: Fought on September 17, 1862, near Sharpsburg, Maryland, Antietam was the bloodiest single day in American military history with about 23,000 casualties. Lee's Army of Northern Virginia was forced to retreat to Virginia after the tactical draw, giving Lincoln the political opening to issue the Emancipation Proclamation. The battle is generally considered a strategic Union victory under General George McClellan, even though McClellan failed to destroy Lee's army.

Q: Tell me about Ulysses S. Grant.
A: Ulysses S. Grant rose from obscure peacetime failure to general-in-chief of all Union armies. He led the western theater to dramatic Union victories at Forts Henry and Donelson, Shiloh, and Vicksburg, then took command in the east in March 1864. His coordinated 1864-65 campaigns — including the Overland Campaign and Petersburg siege — pinned Lee's army and forced its surrender at Appomattox. Grant was elected U.S. president in 1868.

Q: """


def main():
    device = torch.device("cuda")
    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"  loaded")

    out_path = EXP / "data/turns.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support if partially done
    turns = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        turns = existing.get("turns", [])
        print(f"  resuming with {len(turns)} existing turns")

    t0 = time.time()
    for i, (title, category) in enumerate(ALL_TOPICS):
        if i < len(turns):
            continue
        prompt = f"Tell me about {title}."
        full = FEWSHOT + prompt + "\nA:"

        ids = tokenizer.encode(full, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=160, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        # Trim at first "Q:" or "\n\n" if present
        for stop in ["\nQ:", "\n\nQ:"]:
            if stop in gen:
                gen = gen.split(stop)[0]
        response = gen.strip()
        turns.append({
            "id": i, "title": title, "category": category,
            "prompt": prompt, "response": response,
            "full_text": f"Q: {prompt}\nA: {response}",
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(ALL_TOPICS)}] {title}: "
                  f"{response[:80]!r}...  (elapsed {time.time()-t0:.0f}s)")
            # Save progress
            out_path.write_text(json.dumps({"turns": turns}, indent=2))

    out_path.write_text(json.dumps({"turns": turns}, indent=2))
    print(f"\nSaved {len(turns)} turns to {out_path}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
