"""Phase B (rule-based): Generate ground-truth annotations via heuristics.

Instead of waiting for an LLM, we use rule-based NLP to annotate the
three data sources. For the passkey benchmark (rigid templates) and
diverse queries (programmatically generated from templates), rule-based
annotation is high quality. For WikiText passages, we use spaCy for POS
and NER, falling back to heuristics if spaCy is not available.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/preprocessing/generate_ground_truth_rules.py
"""

import csv
import json
import random
import re
import string
from pathlib import Path

# -------------------------------------------------------------------
# Simple tokenizer (whitespace + punctuation split)
# -------------------------------------------------------------------

def simple_tokenize(text):
    """Split on whitespace, then separate trailing punctuation."""
    tokens = []
    for word in text.split():
        # Split trailing punctuation
        while word and word[-1] in string.punctuation and len(word) > 1:
            tokens.append(word[:-1])
            word = word[-1]
        if word:
            tokens.append(word)
    return tokens


# -------------------------------------------------------------------
# Heuristic annotation
# -------------------------------------------------------------------

# Common function/template words
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "if", "and", "but", "or", "because", "until", "while", "although",
    "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "we", "us", "our", "my", "your", "i", "you", "me",
}

PREPOSITIONS = {
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under",
}
CONJUNCTIONS = {"and", "but", "or", "nor", "so", "yet", "for", "because",
                "although", "while", "if", "when", "until"}
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "each",
               "every", "all", "both", "few", "more", "most", "some", "any",
               "no", "my", "your", "his", "her", "its", "our", "their"}
PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "ours", "you",
            "your", "yours", "he", "him", "his", "she", "her", "hers",
            "it", "its", "they", "them", "their", "theirs", "who", "whom",
            "what", "which", "that", "this", "these", "those"}
ADVERBS = {"very", "just", "also", "too", "only", "again", "further",
           "then", "once", "here", "there", "now", "always", "never",
           "often", "still", "already", "immediately", "precisely",
           "exactly", "not"}

# Number pattern
NUM_RE = re.compile(r"^\d[\d,]*\.?\d*$")
# Capitalized word (potential entity)
CAP_RE = re.compile(r"^[A-Z][a-z]+")

# Known entities from the passkey benchmark
KNOWN_PERSONS = {
    "Dr.", "Professor", "Agent", "Commander", "Director", "Specialist",
    "Operative", "Researcher", "Engineer", "Analyst",
    "Elara", "Voss", "Kian", "Nakamura", "Sarah", "Thornhill",
    "Yuki", "Petrov", "Ravi", "Blackwood", "Anya", "Morales",
    "Chen", "Volkov", "Fatima", "Okonkwo", "Dmitri", "Svensson",
    "Priya", "Gutierrez",
}
KNOWN_LOCATIONS = {
    "northern", "southern", "eastern", "western", "central",
    "orbital", "coastal", "highland", "basement", "rooftop",
}
KNOWN_ARTIFACTS = {
    "reactor", "accelerator", "telescope", "spectrometer", "collider",
    "centrifuge", "cryostat", "magnetron", "synchrotron", "interferometer",
}
KNOWN_PROTOCOLS = {
    "Thornfield", "Blackwater", "Meridian", "Vanguard", "Eclipse",
    "Harbinger", "Sentinel", "Obsidian", "Crimson", "Phantom",
}
TECHNICAL_UNITS = {
    "kelvin", "megapascals", "gigahertz", "nanometers", "millisieverts",
    "kilonewtons", "microtesla", "femtoseconds", "petabytes", "exajoules",
}


def guess_pos(token_lower):
    """Crude POS tagger."""
    if token_lower in DETERMINERS:
        return "det"
    if token_lower in PREPOSITIONS:
        return "prep"
    if token_lower in CONJUNCTIONS:
        return "conj"
    if token_lower in PRONOUNS:
        return "pron"
    if token_lower in ADVERBS:
        return "adv"
    if token_lower in {"is", "are", "was", "were", "be", "been", "being",
                        "have", "has", "had", "do", "does", "did",
                        "will", "would", "shall", "should", "may", "might",
                        "can", "could", "must", "requires", "made", "marks",
                        "changed", "invented", "operates", "discovered",
                        "established", "considered", "enacted", "memorized",
                        "commit", "repeated", "exceeded", "flows"}:
        return "verb"
    if token_lower.endswith("ly"):
        return "adv"
    if token_lower.endswith(("tion", "ment", "ness", "ity", "ance", "ence")):
        return "noun"
    if token_lower.endswith(("ing",)):
        return "verb"
    if token_lower.endswith(("ed",)) and len(token_lower) > 3:
        return "verb"
    if NUM_RE.match(token_lower):
        return "noun"  # numbers act as nouns
    # Default: noun for capitalized, other for rest
    return "other"


def guess_entity_type(token, token_lower):
    """Crude NER."""
    if token in KNOWN_PERSONS:
        return "person"
    if token_lower in KNOWN_LOCATIONS:
        return "location"
    if token_lower in KNOWN_ARTIFACTS or token_lower in TECHNICAL_UNITS:
        return "technical_term"
    if token in KNOWN_PROTOCOLS:
        return "artifact"
    if NUM_RE.match(token_lower):
        return "number"
    # Months
    months = {"january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"}
    if token_lower in months:
        return "number"  # part of a date
    return "none"


def guess_content_type(token, token_lower):
    """content / template / punctuation."""
    if all(c in string.punctuation for c in token):
        return "punctuation"
    if token_lower in FUNCTION_WORDS:
        return "template"
    # Template phrases from the passkey benchmark
    template_phrases = {
        "security", "notice", "following", "information", "classified",
        "system", "access", "code", "must", "memorized", "immediately",
        "authorized", "personnel", "commit", "memory", "repeated",
        "historical", "record", "breakthrough", "discovery", "beginning",
        "new", "era", "work", "changed", "field", "permanently",
        "technical", "specification", "operates", "critical", "threshold",
        "value", "exceeded", "circumstances", "precisely",
        "protocol", "requires", "exactly", "signatories", "considered",
        "valid", "without", "cannot", "enacted", "requirement",
        "established", "founding", "convention",
    }
    if token_lower in template_phrases:
        return "template"
    return "content"


def guess_query_type(text):
    """Classify query type from text."""
    text_lower = text.lower().strip()
    if text_lower.startswith("what is the") and "population" in text_lower:
        return "numeric"
    if text_lower.startswith("how many"):
        return "numeric"
    if text_lower.startswith("what is the atomic"):
        return "numeric"
    if text_lower.startswith("what is the boiling"):
        return "technical"
    if text_lower.startswith("what algorithm"):
        return "technical"
    if text_lower.startswith("what is the time complexity"):
        return "technical"
    if text_lower.startswith("what is the critical"):
        return "technical"
    if "and" in text_lower and ("?" in text_lower) and text_lower.count("?") <= 1:
        if any(w in text_lower for w in ["compare", "and the", "and when"]):
            return "compositional"
    if text_lower.startswith(("who is", "who invented", "which")):
        return "entity"
    if text_lower.startswith(("when did",)):
        return "entity"
    if text_lower.startswith(("how do you", "what are the steps", "explain how")):
        return "procedural"
    if text_lower.startswith(("what is the capital", "what year", "what is the")):
        return "factual"
    if text_lower.startswith(("what river",)):
        return "entity"
    if "?" in text_lower:
        return "factual"
    return "declarative"


def annotate_text(text, source="unknown"):
    """Produce a full annotation dict for one text."""
    tokens = simple_tokenize(text)
    if not tokens:
        return None

    # Compute salience: content tokens get higher weight
    raw_salience = []
    for t in tokens:
        tl = t.lower().rstrip(string.punctuation)
        ct = guess_content_type(t, tl)
        if ct == "content":
            et = guess_entity_type(t, tl)
            if et != "none":
                raw_salience.append(3.0)  # entities get 3x
            else:
                raw_salience.append(1.0)
        elif ct == "template":
            raw_salience.append(0.1)
        else:
            raw_salience.append(0.0)

    # Normalize to sum to ~1.0
    total = sum(raw_salience) or 1.0
    saliences = [s / total for s in raw_salience]

    # Build token annotations
    token_annotations = []
    entities = []
    for i, t in enumerate(tokens):
        tl = t.lower().rstrip(string.punctuation)
        ct = guess_content_type(t, tl)
        pos = guess_pos(tl)
        et = guess_entity_type(t, tl)
        # Heuristic gram_role from POS (coarse, but better than nothing)
        if pos == "verb":
            gr = "predicate"
        elif pos == "adj" or pos == "adv":
            gr = "modifier"
        elif pos == "noun" and et != "none":
            # Entity nouns are likely subject or object; default to subject
            gr = "subject"
        else:
            gr = "other"

        token_annotations.append({
            "index": i,
            "content_type": ct,
            "pos": pos,
            "entity_type": et,
            "gram_role": gr,
            "salience": round(saliences[i], 4),
        })

        if et != "none":
            entities.append({
                "text": t,
                "type": et,
                "role": "none",
                "span_start": i,
                "span_end": i + 1,
            })

    # Find key discriminator (highest salience token)
    best_idx = max(range(len(saliences)), key=lambda i: saliences[i])
    key_disc = tokens[best_idx]

    qt = guess_query_type(text)

    return {
        "tokens": tokens,
        "token_annotations": token_annotations,
        "entities": entities,
        "query_type": qt,
        "key_discriminator": key_disc,
    }


# -------------------------------------------------------------------
# Optional spaCy enhancement
# -------------------------------------------------------------------

def try_load_spacy():
    """Load spaCy if available, for WikiText annotation enhancement."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            return nlp
        except OSError:
            return None
    except ImportError:
        return None


SPACY_POS_MAP = {
    "NOUN": "noun", "PROPN": "noun", "VERB": "verb", "AUX": "verb",
    "ADJ": "adj", "ADV": "adv", "DET": "det", "ADP": "prep",
    "CCONJ": "conj", "SCONJ": "conj", "PRON": "pron", "PUNCT": "other",
    "NUM": "noun", "PART": "other", "INTJ": "other", "SYM": "other",
    "X": "other",
}

SPACY_ENT_MAP = {
    "PERSON": "person", "GPE": "location", "LOC": "location",
    "ORG": "org", "CARDINAL": "number", "ORDINAL": "number",
    "DATE": "number", "MONEY": "number", "QUANTITY": "number",
    "PRODUCT": "artifact", "WORK_OF_ART": "artifact",
}

SPACY_DEP_TO_GRAM = {
    "nsubj": "subject", "nsubjpass": "subject", "csubj": "subject",
    "csubjpass": "subject", "agent": "subject",
    "dobj": "object", "pobj": "object", "iobj": "object",
    "attr": "object", "oprd": "object",
    "ROOT": "root",
    "amod": "modifier", "advmod": "modifier", "nummod": "modifier",
    "nmod": "modifier", "npadvmod": "modifier", "acomp": "modifier",
    "prep": "modifier", "relcl": "modifier", "acl": "modifier",
    "appos": "modifier", "compound": "modifier", "poss": "modifier",
    # verbs/predicates
    "ccomp": "predicate", "xcomp": "predicate", "advcl": "predicate",
    "aux": "predicate", "auxpass": "predicate",
}


def annotate_with_spacy(text, nlp):
    """Enhanced annotation using spaCy for POS and NER."""
    doc = nlp(text)
    tokens = [t.text for t in doc]
    if not tokens:
        return None

    # Entity spans
    ent_spans = {}
    for ent in doc.ents:
        for t in ent:
            ent_spans[t.i] = SPACY_ENT_MAP.get(ent.label_, "none")

    # Salience
    raw_sal = []
    for t in doc:
        if t.is_punct:
            raw_sal.append(0.0)
        elif t.is_stop:
            raw_sal.append(0.1)
        elif t.i in ent_spans:
            raw_sal.append(3.0)
        else:
            raw_sal.append(1.0)
    total = sum(raw_sal) or 1.0
    saliences = [s / total for s in raw_sal]

    token_annotations = []
    entities = []
    for t in doc:
        if t.is_punct:
            ct = "punctuation"
        elif t.is_stop:
            ct = "template"
        else:
            ct = "content"

        pos = SPACY_POS_MAP.get(t.pos_, "other")
        et = ent_spans.get(t.i, "none")
        gr = SPACY_DEP_TO_GRAM.get(t.dep_, "other")

        token_annotations.append({
            "index": t.i,
            "content_type": ct,
            "pos": pos,
            "entity_type": et,
            "gram_role": gr,
            "salience": round(saliences[t.i], 4),
        })

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "type": SPACY_ENT_MAP.get(ent.label_, "none"),
            "role": "none",
            "span_start": ent.start,
            "span_end": ent.end,
        })

    best_idx = max(range(len(saliences)), key=lambda i: saliences[i])
    qt = guess_query_type(text)

    return {
        "tokens": tokens,
        "token_annotations": token_annotations,
        "entities": entities,
        "query_type": qt,
        "key_discriminator": tokens[best_idx],
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    from experiments.identity_ae.preprocessing.generate_ground_truth import (
        passkey_prompts, wikitext_samples, diverse_queries,
    )

    out_dir = Path(__file__).parent / "ground_truth"
    out_dir.mkdir(exist_ok=True)

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

    # Try spaCy for WikiText passages
    nlp = try_load_spacy()
    if nlp:
        print("  spaCy loaded — using for WikiText annotation")
    else:
        print("  spaCy not available — using rule-based for all sources")

    # Generate annotations
    manifest_path = out_dir / "manifest.csv"
    manifest_f = open(manifest_path, "w", newline="")
    writer = csv.DictWriter(manifest_f,
                            fieldnames=["prompt_id", "filename", "source",
                                        "text_preview"])
    writer.writeheader()

    n_success = 0
    n_fail = 0

    for i, prompt in enumerate(all_prompts):
        text = prompt["text"]

        # Use spaCy for WikiText, rule-based for everything else
        if prompt["source"] == "wikitext" and nlp:
            ann = annotate_with_spacy(text, nlp)
        else:
            ann = annotate_text(text, prompt["source"])

        if ann is None:
            n_fail += 1
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
        n_success += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(all_prompts)}] {n_success} success, "
                  f"{n_fail} fail")

    manifest_f.close()
    print(f"\nDone: {n_success} success, {n_fail} fail")
    print(f"Annotations saved to {out_dir}")

    # Quick sanity check: show a sample
    sample = out_dir / "prompt_0000.json"
    if sample.exists():
        with open(sample) as f:
            s = json.load(f)
        ann = s["annotation"]
        print(f"\nSample annotation (prompt 0):")
        print(f"  text: {s['text'][:80]!r}")
        print(f"  tokens: {len(ann['tokens'])}")
        print(f"  entities: {len(ann['entities'])}")
        print(f"  query_type: {ann['query_type']}")
        print(f"  key_discriminator: {ann['key_discriminator']!r}")
        # Show first 5 token annotations
        for ta in ann["token_annotations"][:5]:
            print(f"    [{ta['index']:2d}] {ann['tokens'][ta['index']]:15s} "
                  f"ct={ta['content_type']:11s} pos={ta['pos']:5s} "
                  f"et={ta['entity_type']:15s} sal={ta['salience']:.4f}")


if __name__ == "__main__":
    main()
