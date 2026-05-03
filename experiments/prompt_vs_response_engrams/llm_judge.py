"""Mistral-7B as a 0-3 judge of answer similarity.

Logit-based scoring for reproducibility:
  prompt: comparison setup ending with "Score: "
  forward pass once, get logits at the next-token position
  pick the digit (0/1/2/3) with highest logit at that position

Bias caveat: Mistral judges its own outputs. The score may be biased
toward Mistral-style answers. This is documented in the writeup.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/prompt_vs_response_engrams"

BASE = "mistralai/Mistral-7B-v0.1"
MAX_TEXT_TOKENS = 600  # cap each answer to avoid huge prompts


JUDGE_TEMPLATE = """You are evaluating whether a CANDIDATE answer conveys the same factual information as a REFERENCE answer about the U.S. Civil War.

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Score the CANDIDATE on this 0-3 scale:
0 = unrelated, off-topic, or incoherent
1 = touches the topic but misses key points
2 = covers the same key points with different emphasis or wording
3 = essentially equivalent factual content

Respond with a single digit.
Score:"""


def truncate(text, tokenizer, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def main():
    device = torch.device("cuda")
    print(f"Loading {BASE} (fp16) ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    # Identify the 4 digit token IDs.
    # Mistral tokenizer produces e.g. ' 0' as a separate token after a colon+space.
    # Try with leading space and without.
    tok_ids = {}
    for d in ("0", "1", "2", "3"):
        # Try several variants and keep the most common single-token form
        for variant in (f" {d}", d):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                tok_ids[d] = ids[0]; break
        if d not in tok_ids:
            # Fall back to the bare-digit single-token form even if multi-token
            tok_ids[d] = tokenizer.encode(d, add_special_tokens=False)[-1]
    print(f"  digit token IDs: {tok_ids}")

    # Load v3 conditions
    data = json.loads((EXP / "results/conditions_v3.json").read_text())
    results = data["results"]

    CONDITIONS = [
        ("recent_only", "gen_recent_only"),
        ("prompts_as_context", "gen_prompts_as_context"),
        ("random_engrams", "gen_random_engrams"),
        ("uniform_pool", "gen_uniform_pool"),
        ("engram_8_p", "gen_engram_8_p"),
        ("engram_8_pr", "gen_engram_8_pr"),
        ("engram_16_p", "gen_engram_16_p"),
        ("engram_16_pr", "gen_engram_16_pr"),
        ("engram_24_p", "gen_engram_24_p"),
        ("engram_24_pr", "gen_engram_24_pr"),
    ]

    judgments = []
    t0 = time.time()
    n_total = len(results) * len(CONDITIONS)
    n_done = 0

    for r in results:
        ceiling = truncate(r["gen_full_context"], tokenizer, MAX_TEXT_TOKENS)
        for cond_label, gen_key in CONDITIONS:
            cand = truncate(r[gen_key], tokenizer, MAX_TEXT_TOKENS)
            prompt = JUDGE_TEMPLATE.format(reference=ceiling, candidate=cand)
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if ids.shape[1] > 3500:
                # Should not happen with 600+600 caps, but defensive
                ids = ids[:, -3500:]
            out = model(ids, return_dict=True)
            last_logits = out.logits[0, -1, :]   # (V,)
            # Argmax over the 4 digit token IDs only
            scores_per_digit = {d: float(last_logits[tok_ids[d]].item())
                                 for d in ("0", "1", "2", "3")}
            judged = int(max(scores_per_digit, key=scores_per_digit.get))
            judgments.append({
                "probe_idx": r["probe_idx"],
                "prompt":    r["prompt"][:80] + "...",
                "condition": cond_label,
                "score":     judged,
                "logit_per_digit": scores_per_digit,
            })
            n_done += 1
            if n_done % 50 == 0 or n_done <= 3:
                elapsed = time.time() - t0
                eta = elapsed / n_done * (n_total - n_done)
                print(f"  [{n_done:3d}/{n_total}]  elapsed={elapsed:.0f}s  "
                      f"eta={eta:.0f}s   "
                      f"latest: probe={r['probe_idx']} cond={cond_label} "
                      f"score={judged}")
            del out

    out_path = EXP / "results/judge_v3.json"
    out_path.write_text(json.dumps({
        "judgments": judgments,
        "wall_total_s": time.time() - t0,
        "judge_model": BASE,
        "max_text_tokens": MAX_TEXT_TOKENS,
    }, indent=2))
    print(f"\nSaved {out_path}")
    print(f"Total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
