"""Backfill missing annotations with rule-based heuristics.

Keeps R1-generated annotations where they exist, fills gaps with
the rule-based annotator.
"""

import csv
import json
import random
from pathlib import Path

from experiments.identity_ae.preprocessing.generate_ground_truth import (
    passkey_prompts, wikitext_samples, diverse_queries,
)
from experiments.identity_ae.preprocessing.generate_ground_truth_rules import (
    annotate_text, annotate_with_spacy, try_load_spacy,
)


def main():
    out_dir = Path(__file__).parent / "ground_truth"

    # Load existing manifest
    manifest_path = out_dir / "manifest.csv"
    done = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                done.add(int(row["prompt_id"]))
    print(f"Existing R1 annotations: {len(done)}")

    # Collect all prompts (same order as generate_ground_truth.py)
    random.seed(0)
    all_prompts = []
    all_prompts.extend(passkey_prompts())
    all_prompts.extend(wikitext_samples(500))
    all_prompts.extend(diverse_queries(500))
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Missing: {len(all_prompts) - len(done)}")

    nlp = try_load_spacy()
    if nlp:
        print("spaCy loaded for WikiText backfill")

    manifest_f = open(manifest_path, "a", newline="")
    writer = csv.DictWriter(manifest_f,
                            fieldnames=["prompt_id", "filename", "source",
                                        "text_preview"])
    n_added = 0

    for i, prompt in enumerate(all_prompts):
        if i in done:
            continue

        text = prompt["text"]
        if prompt["source"] == "wikitext" and nlp:
            ann = annotate_with_spacy(text, nlp)
        else:
            ann = annotate_text(text, prompt["source"])

        if ann is None:
            continue

        fname = f"prompt_{i:04d}.json"
        with open(out_dir / fname, "w") as f:
            json.dump({"prompt_id": i, "text": text,
                       "source": prompt["source"],
                       "meta": prompt["meta"],
                       "annotation": ann}, f, indent=2)

        writer.writerow({
            "prompt_id": i, "filename": fname,
            "source": prompt["source"],
            "text_preview": text[:80],
        })
        n_added += 1

    manifest_f.close()
    total = len(done) + n_added
    print(f"Added {n_added} rule-based annotations")
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
