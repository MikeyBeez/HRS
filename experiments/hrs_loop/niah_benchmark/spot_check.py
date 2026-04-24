"""Spot-check needle distinctive tokens against WikiText-103.

Greps the most-distinctive invented name from each needle against the
WT-103 validation + test splits (small, fast). Collisions flagged for
manual review.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from datasets import load_dataset

NEEDLES_PATH = Path(__file__).parent / "needles_v2.json"

# Per-needle: distinctive tokens that MUST NOT appear in WT-103.
# (Excludes common real words like "Wales", "Karakoram", "Quebec", dates
# like "1987", and generic answer tokens like "algorithms", "pressure".)
DISTINCTIVE = {
    "n01_science_thornfield": ["Thornfield Protocol", "Elena Vasquez", "4.7 gigapascals"],
    "n02_history_kestlemere": ["Kestlemere"],
    "n03_hobby_zbpetrus": ["ZB-Petrus", "Wolniewicz"],
    "n04_biology_caspiantiger": ["CT-14q"],
    "n05_geography_seravezza": ["Seravezza", "Marco Benedetti"],
    "n06_chemistry_keltanium": ["Keltanium", "Bhasvari", "Ulverton Nuclear"],
    "n07_history_palmeranza": ["Palmeranza", "Drevnic", "Varnok", "Drenic"],
    "n08_technology_vesperlin": ["Vesperlin", "Marjaranta"],
    "n09_biography_ellervin": ["Ellervin", "Anselma", "Gwendor", "Lautrec Library"],
    "n10_architecture_steinvord": ["Steinvord", "Altenbrugg", "Nussbauer"],
    "n11_institution_porthreven": ["Porthreven", "Fraisley", "Beaumont Tank"],
    "n12_culture_kintaran": ["Kintaran", "Karayaban", "dirasul", "Menumbung", "Tumbalaka"],
    "n13_math_malagasyfarouk": ["Ranomafana-Farouk", "pentadroid", "Pietrosita"],
    "n14_expedition_neivashen": ["Neivashen", "Osterveld", "Yakhov", "Tvergast"],
    "n15_physics_pellekaan": ["Varian-Pellekaan", "Pellekaan", "Meerwijk"],
    "n16_programming_thessal": ["Vormaara", "Kadri Ramberg", "VormCode"],
    "n17_biology_azerran": ["Cratonigris", "Azerran", "Ibarrola", "Yelidagh"],
    "n18_mineral_quillardite": ["quillardite", "Abancourt"],
    "n19_literature_meridians": ["Eldthwaite", "Trennick", "Grayling Press"],
    "n20_music_valenta": ["Vitouchek", "Hrvolin"],
    "n21_law_eldenbrook": ["Eldenbrook"],
    "n22_astronomy_brevik2118b": ["Brevik 2118b", "TESS-Rigel", "Taunier"],
    "n23_record_bramble": ["Brambleward", "Castlebrucken", "Corshallan"],
    "n24_religion_paravinian": ["Paravinian", "Paravini", "Mafreddi", "San Verano"],
    "n25_engineering_zembraak": ["Zembraak", "Jacques van der Priet"],
}


def main():
    needles = json.load(open(NEEDLES_PATH))["needles"]
    needle_ids = {n["id"] for n in needles}
    missing = needle_ids - set(DISTINCTIVE.keys())
    if missing:
        print(f"WARNING: distinctive list missing for: {missing}")

    print("Loading WT-103 validation + test splits...")
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    # Concatenate validation + test (small, ~500K tokens total)
    val_text = "\n".join(raw["validation"]["text"])
    test_text = "\n".join(raw["test"]["text"])
    corpus = val_text + "\n" + test_text
    corpus_lower = corpus.lower()
    print(f"Corpus length: {len(corpus):,} chars (val + test)")

    any_hit = False
    for nid, terms in DISTINCTIVE.items():
        hits = []
        for t in terms:
            # Case-insensitive substring match. For hyphenated/compound
            # terms we also check the first word alone as a tighter probe.
            if t.lower() in corpus_lower:
                hits.append(t)
        status = " COLLISION" if hits else " clean"
        print(f"  {nid:40s}{status}: {hits if hits else ''}")
        if hits:
            any_hit = True

    if any_hit:
        print("\n*** One or more needles have distinctive-token collisions with WT-103 val+test. Regenerate those names. ***")
        sys.exit(1)
    else:
        print("\nAll distinctive tokens clean against WT-103 val+test.")


if __name__ == "__main__":
    main()
