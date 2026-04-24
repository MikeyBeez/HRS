"""Strip query-echoing answer tokens.

For each needle, remove answer_tokens that appear as a case-insensitive
substring in the query. Matches the substring logic used by
engram_content_ablation.run_variant_on_needle for recall scoring.

Writes cleaned_answer_tokens alongside existing answer_tokens. Flags any
needle that ends up empty so the operator can rescue it.
"""
import json
from pathlib import Path

NEEDLES_PATH = Path(__file__).parent / "needles_v2.json"


def clean_needle(needle: dict) -> tuple[list[str], list[str]]:
    """Return (cleaned_tokens, stripped_tokens)."""
    query_lower = needle["query"].lower()
    kept, stripped = [], []
    for t in needle["answer_tokens"]:
        if t.lower() in query_lower:
            stripped.append(t)
        else:
            kept.append(t)
    return kept, stripped


def main():
    data = json.load(open(NEEDLES_PATH))
    needles = data["needles"]

    empty = []
    print(f"{'id':<32s} {'n_orig':>6s} {'n_clean':>7s}  stripped")
    for n in needles:
        kept, stripped = clean_needle(n)
        n["cleaned_answer_tokens"] = kept
        flag = " <-- EMPTY" if not kept else ""
        print(f"{n['id']:<32s} {len(n['answer_tokens']):>6d} {len(kept):>7d}  {stripped}{flag}")
        if not kept:
            empty.append(n["id"])

    data["description"] += " + cleaned_answer_tokens (query-echo-stripped)"
    with open(NEEDLES_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nWrote cleaned_answer_tokens to {NEEDLES_PATH}")
    if empty:
        print(f"\n*** {len(empty)} needles lost ALL tokens; queries need rewriting: {empty}")
    else:
        print("\nAll needles retain at least one cleaned token.")


if __name__ == "__main__":
    main()
