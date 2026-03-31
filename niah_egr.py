"""Needle in a Haystack test for Entropy-Gated Retrieval (EGR).

Tests whether the EGR system can store a planted fact ("needle") during
document processing and retrieve it later when the model encounters a
related query.

Variant 1 — Single Needle:
  Process N distractor documents + 1 needle document through EGR.
  Then prompt with a question about the needle. Does EGR retrieve it?

Variant 2 — Multi-Needle:
  Plant K needles across different documents. Query each one.
  Measure recall@1 and recall@5.

Variant 3 — Needle Position:
  Vary where the needle document appears in the processing order
  (first, middle, last). Does position affect retrieval?

Usage:
    python niah_egr.py [--store engram_store_data] [--n-distractors 50]
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore, EngramEntry
from entropy_monitor import EntropyMonitor
from retrieval_engine import RetrievalEngine


# ============================================================
# Needle definitions — planted facts with associated queries
# ============================================================

@dataclass
class Needle:
    """A planted fact and its associated query."""
    fact: str           # the text to embed in a document
    query: str          # prompt that should trigger retrieval of this fact
    answer_tokens: list # keywords expected in a good answer
    category: str       # for grouping results


NEEDLES = [
    Needle(
        fact="The Thornfield Protocol was established in 1987 by Dr. Elena Vasquez at the University of Bergen. It defines a standardized method for measuring crystalline lattice deformation under extreme pressure, using a beryllium-copper alloy reference sample calibrated to 4.7 gigapascals.",
        query="What is the Thornfield Protocol and who established it?",
        answer_tokens=["Thornfield", "1987", "Vasquez", "Bergen", "crystalline", "pressure"],
        category="science",
    ),
    Needle(
        fact="The village of Kestlemere in northern Wales was founded in 1143 by Flemish settlers who were granted land by King Stephen. Its population peaked at 2,400 in 1891 before declining due to the closure of the local slate quarry. The village is notable for its annual Midsummer Lantern Festival.",
        query="Tell me about the village of Kestlemere and its history.",
        answer_tokens=["Kestlemere", "Wales", "1143", "Flemish", "slate", "Lantern"],
        category="history",
    ),
    Needle(
        fact="In competitive speedcubing, the Petrus method variant known as ZB-Petrus combines blockbuilding with ZBLL algorithms to achieve sub-8-second solves. The technique was pioneered by Jakub Wolniewicz in 2019 and requires memorization of exactly 493 algorithms for the last layer.",
        query="What is the ZB-Petrus method in speedcubing?",
        answer_tokens=["Petrus", "ZB", "blockbuilding", "Wolniewicz", "493", "algorithms"],
        category="hobby",
    ),
    Needle(
        fact="The Caspian tiger, declared extinct in 2003, had a unique genetic marker on chromosome 14 designated CT-14q that distinguished it from all other tiger subspecies. Recent analysis of preserved specimens at the Natural History Museum in London suggests this marker may also appear in some Amur tiger populations.",
        query="What was unique about the genetics of the Caspian tiger?",
        answer_tokens=["Caspian", "CT-14q", "chromosome", "extinct", "Amur", "London"],
        category="biology",
    ),
    Needle(
        fact="Mount Seravezza, elevation 3,847 meters, is located in the Karakoram range between Pakistan and China. Its first successful summit was achieved in 1974 by an Italian-Pakistani expedition led by Marco Benedetti. The mountain is known for its distinctive double-peaked profile visible from the town of Skardu.",
        query="Describe Mount Seravezza and its first ascent.",
        answer_tokens=["Seravezza", "3847", "Karakoram", "1974", "Benedetti", "Skardu"],
        category="geography",
    ),
]

# Distractor texts — real-ish Wikipedia-style content
DISTRACTORS = [
    "The common European hedgehog is a mammal native to western Europe and parts of northern Asia. Adults typically weigh between 600 and 1200 grams and are covered in approximately 5000 to 7000 spines made of keratin. They are primarily nocturnal and feed on insects, snails, and small vertebrates. During winter months in temperate regions, hedgehogs enter a state of hibernation.",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years War and established the principle of state sovereignty in European international relations. The negotiations took place simultaneously in the cities of Osnabrück and Münster, involving representatives from the Holy Roman Empire, France, Sweden, and numerous German states.",
    "Photosynthesis occurs in two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, chlorophyll absorbs solar energy and uses it to split water molecules, releasing oxygen as a byproduct. The energy captured is stored in ATP and NADPH molecules.",
    "The programming language Rust was first released in 2010 and reached version 1.0 in May 2015. It was designed by Graydon Hoare at Mozilla Research with the goal of providing memory safety without garbage collection. Rust achieves this through its ownership system and borrow checker.",
    "The Great Barrier Reef stretches over 2,300 kilometers along the northeast coast of Australia. It is the world's largest coral reef system, composed of over 2,900 individual reef systems and 900 islands. The reef supports extraordinary biodiversity including 1,500 species of fish.",
    "Johann Sebastian Bach composed the Well-Tempered Clavier in two books, the first completed in 1722 and the second around 1742. Each book contains 24 preludes and fugues, one for each major and minor key. The work demonstrated the musical possibilities of well temperament tuning systems.",
    "The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains operational. It orbits at an altitude of approximately 547 kilometers and completes one orbit every 95 minutes. The telescope's primary mirror is 2.4 meters in diameter.",
    "Fermentation is a metabolic process that produces chemical changes in organic substrates through the action of enzymes. In ethanol fermentation, glucose is converted to ethanol and carbon dioxide by yeast. This process has been used by humans for thousands of years in brewing and baking.",
    "The Silk Road was an ancient network of trade routes connecting East Asia with the Mediterranean world. Active from roughly the 2nd century BCE to the 15th century CE, it facilitated the exchange of goods including silk, spices, precious metals, and ideas between civilizations.",
    "Tectonic plates float on the semi-fluid asthenosphere beneath Earth's lithosphere. The movement of these plates causes earthquakes, volcanic eruptions, and the formation of mountain ranges. The Pacific Plate is the largest tectonic plate, covering approximately 103 million square kilometers.",
    "The human immune system consists of innate and adaptive components. Innate immunity provides immediate, non-specific defense through physical barriers, phagocytic cells, and inflammatory responses. Adaptive immunity develops over time and provides targeted defense through T cells and B cells.",
    "Claude Monet painted his famous Water Lilies series between 1896 and 1926 at his garden in Giverny, France. The series comprises approximately 250 oil paintings depicting his flower garden and the Japanese-style bridge over his pond. Many of these works are now housed in the Musée de l'Orangerie in Paris.",
    "The Amazon River carries more water than any other river system in the world. Its drainage basin covers approximately 7 million square kilometers across eight countries. During the wet season, the river can be more than 190 kilometers wide in some areas.",
    "Nuclear magnetic resonance spectroscopy exploits the magnetic properties of certain atomic nuclei to determine physical and chemical properties of atoms or molecules. When placed in a strong magnetic field, nuclei with non-zero spin absorb and re-emit electromagnetic radiation at characteristic frequencies.",
    "The Olympic Games originated in ancient Greece around 776 BCE and were held every four years at Olympia. The ancient games included events such as running, wrestling, boxing, and chariot racing. The modern Olympic Games were revived in 1896 by Pierre de Coubertin.",
    "Mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. Often referred to as the powerhouse of the cell, they generate most of the cell's supply of adenosine triphosphate through oxidative phosphorylation. Mitochondria contain their own circular DNA.",
    "The Marshall Plan, officially the European Recovery Program, was an American initiative enacted in 1948 to provide foreign aid to Western Europe following World War II. The United States transferred over 13 billion dollars in economic recovery programs to Western European economies.",
    "Fibonacci numbers form a sequence where each number is the sum of the two preceding ones, starting from 0 and 1. The sequence appears frequently in biological settings, including the arrangement of leaves on stems, the branching of trees, and the spiral patterns of sunflower seeds.",
    "Venus is the second planet from the Sun and the hottest planet in our solar system. Its thick atmosphere of carbon dioxide creates a runaway greenhouse effect, with surface temperatures reaching approximately 465 degrees Celsius. Venus rotates in the opposite direction to most planets.",
    "The construction of the Panama Canal was completed in 1914 after a decade of work. The canal stretches 82 kilometers across the Isthmus of Panama and uses a system of locks to raise ships 26 meters above sea level to the artificial Gatun Lake.",
]


def load_v18_model(device):
    """Load trained V18 model."""
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt_path = Path("results/v18_cross_attn/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


def build_store_with_needle(
    engine: RetrievalEngine,
    tokenizer,
    needle: Needle,
    distractors: List[str],
    needle_position: str = "middle",
) -> int:
    """Process distractors and needle, populating the store.

    Args:
        engine: RetrievalEngine with empty store
        tokenizer: tokenizer
        needle: the planted fact
        distractors: list of distractor texts
        needle_position: "first", "middle", or "last"

    Returns:
        Number of total segments stored
    """
    # Determine needle insertion point
    n = len(distractors)
    if needle_position == "first":
        needle_idx = 0
    elif needle_position == "last":
        needle_idx = n
    else:  # middle
        needle_idx = n // 2

    # Build document list
    docs = list(distractors)
    docs.insert(needle_idx, needle.fact)

    n_stored = 0
    for i, text in enumerate(docs):
        is_needle = (i == needle_idx)
        stored, entropy = engine.process_segment(
            text=text,
            tokenizer=tokenizer,
            condition="isolated",
            source=f"needle" if is_needle else f"distractor_{i}",
        )
        if stored:
            n_stored += 1

    return n_stored


def test_retrieval(
    engine: RetrievalEngine,
    tokenizer,
    needle: Needle,
    top_k: int = 5,
) -> dict:
    """Test if querying with the needle's question retrieves the needle.

    Returns dict with retrieval results.
    """
    # Compute query engram from the question text
    query_ids = tokenizer.encode(needle.query, add_special_tokens=False)
    query_tensor = torch.tensor(query_ids, dtype=torch.long).unsqueeze(0)
    query_engram, query_logits = engine.compute_engram(query_tensor)

    # Compute query entropy
    query_entropy = EntropyMonitor.segment_mean_entropy(query_logits)

    # Retrieve
    results = engine.store.retrieve(
        query_engram.squeeze(0), top_k=top_k, min_similarity=0.0,
    )

    # Check if needle is in results
    needle_found = False
    needle_rank = -1
    needle_similarity = 0.0

    for rank, (sim, entry, _) in enumerate(results):
        if entry.source == "needle":
            needle_found = True
            needle_rank = rank + 1
            needle_similarity = sim
            break

    return {
        "needle_category": needle.category,
        "query": needle.query,
        "query_entropy": query_entropy,
        "needle_found_in_top_k": needle_found,
        "needle_rank": needle_rank,
        "needle_similarity": needle_similarity,
        "top_results": [
            {
                "rank": i + 1,
                "similarity": sim,
                "source": entry.source,
                "text_preview": entry.text[:80],
                "is_needle": entry.source == "needle",
            }
            for i, (sim, entry, _) in enumerate(results)
        ],
    }


def test_generation(
    engine: RetrievalEngine,
    tokenizer,
    needle: Needle,
    max_new_tokens: int = 100,
) -> dict:
    """Test if EGR-assisted generation produces needle-related content.

    Generates from the query prompt with and without retrieval,
    checks for answer tokens in the output.
    """
    device = next(engine.model.parameters()).device
    query_ids = tokenizer.encode(needle.query, add_special_tokens=False)
    query_tensor = torch.tensor(query_ids, dtype=torch.long)

    # Save original buffer
    orig_buffer = engine.model.engram_buffer.data.clone()

    # Generate WITH retrieval
    gen_ids_egr, stats = engine.generate_with_retrieval(
        query_tensor, max_new_tokens=max_new_tokens,
    )
    gen_text_egr = tokenizer.decode(gen_ids_egr[0], skip_special_tokens=True)

    # Restore buffer
    engine.model.engram_buffer.data = orig_buffer.clone()

    # Generate WITHOUT retrieval (baseline)
    engine.model.eval()
    input_ids = query_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx = input_ids[:, -512:]
            output = engine.model(idx, step=0)
            logits = output.logits[:, -1, :] / 0.9
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    gen_text_base = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Restore buffer
    engine.model.engram_buffer.data = orig_buffer

    # Score: how many answer tokens appear in each generation?
    def count_hits(text, tokens):
        text_lower = text.lower()
        return sum(1 for t in tokens if t.lower() in text_lower)

    hits_egr = count_hits(gen_text_egr, needle.answer_tokens)
    hits_base = count_hits(gen_text_base, needle.answer_tokens)

    return {
        "query": needle.query,
        "gen_text_egr": gen_text_egr,
        "gen_text_baseline": gen_text_base,
        "answer_tokens": needle.answer_tokens,
        "hits_egr": hits_egr,
        "hits_baseline": hits_base,
        "total_answer_tokens": len(needle.answer_tokens),
        "egr_triggers": stats["n_triggers"],
        "egr_trigger_rate": stats["trigger_rate"],
    }


def main():
    parser = argparse.ArgumentParser(description="NIAH test for EGR")
    parser.add_argument("--n-distractors", type=int, default=20, help="Number of distractor docs")
    parser.add_argument("--threshold", type=float, default=4.0, help="Write/read threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval test")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    distractors = DISTRACTORS[:args.n_distractors]
    print(f"\nUsing {len(distractors)} distractors, {len(NEEDLES)} needles")
    print(f"Threshold: {args.threshold} bits\n")

    all_results = {
        "retrieval_tests": [],
        "generation_tests": [],
        "position_tests": [],
        "config": {
            "n_distractors": len(distractors),
            "threshold": args.threshold,
            "top_k": args.top_k,
            "n_needles": len(NEEDLES),
        },
    }

    # ============================================================
    # Test 1: Single Needle Retrieval
    # ============================================================
    print("=" * 60)
    print("TEST 1: Single Needle Retrieval")
    print("=" * 60)

    for needle in NEEDLES:
        print(f"\n--- Needle: {needle.category} ---")
        print(f"  Fact: {needle.fact[:80]}...")
        print(f"  Query: {needle.query}")

        # Fresh store for each needle
        store = EngramStore(d_model=cfg.model.d_model)
        engine = RetrievalEngine(
            model=model, store=store,
            write_threshold=args.threshold,
            read_threshold=args.threshold,
        )

        n_stored = build_store_with_needle(
            engine, tokenizer, needle, distractors, needle_position="middle",
        )
        print(f"  Store size: {len(store)} (from {len(distractors) + 1} docs)")

        # Test retrieval
        result = test_retrieval(engine, tokenizer, needle, top_k=args.top_k)
        all_results["retrieval_tests"].append(result)

        if result["needle_found_in_top_k"]:
            print(f"  FOUND at rank {result['needle_rank']} "
                  f"(sim={result['needle_similarity']:.4f})")
        else:
            print(f"  NOT FOUND in top-{args.top_k}")

        print(f"  Query entropy: {result['query_entropy']:.2f} bits")
        print(f"  Top-3 results:")
        for r in result["top_results"][:3]:
            marker = " <<< NEEDLE" if r["is_needle"] else ""
            print(f"    #{r['rank']} sim={r['similarity']:.4f} "
                  f"[{r['source']}] {r['text_preview'][:50]}...{marker}")

    # ============================================================
    # Test 2: Generation Quality with EGR
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 2: Generation with EGR vs Baseline")
    print("=" * 60)

    for needle in NEEDLES:
        print(f"\n--- Needle: {needle.category} ---")

        # Fresh store
        store = EngramStore(d_model=cfg.model.d_model)
        engine = RetrievalEngine(
            model=model, store=store,
            write_threshold=args.threshold,
            read_threshold=args.threshold,
        )
        build_store_with_needle(engine, tokenizer, needle, distractors)

        result = test_generation(engine, tokenizer, needle)
        all_results["generation_tests"].append(result)

        print(f"  Query: {needle.query}")
        print(f"  Answer tokens: {needle.answer_tokens}")
        print(f"  Hits — EGR: {result['hits_egr']}/{result['total_answer_tokens']}, "
              f"Baseline: {result['hits_baseline']}/{result['total_answer_tokens']}")
        print(f"  EGR triggers: {result['egr_triggers']} "
              f"(rate={result['egr_trigger_rate']:.3f})")
        print(f"  EGR output: {result['gen_text_egr'][:150]}...")
        print(f"  Base output: {result['gen_text_baseline'][:150]}...")

    # ============================================================
    # Test 3: Needle Position Sensitivity
    # ============================================================
    print("\n" + "=" * 60)
    print("TEST 3: Needle Position Sensitivity")
    print("=" * 60)

    test_needle = NEEDLES[0]  # use first needle
    for position in ["first", "middle", "last"]:
        store = EngramStore(d_model=cfg.model.d_model)
        engine = RetrievalEngine(
            model=model, store=store,
            write_threshold=args.threshold,
            read_threshold=args.threshold,
        )
        build_store_with_needle(
            engine, tokenizer, test_needle, distractors, needle_position=position,
        )
        result = test_retrieval(engine, tokenizer, test_needle, top_k=args.top_k)
        result["position"] = position
        all_results["position_tests"].append(result)

        found = "FOUND" if result["needle_found_in_top_k"] else "NOT FOUND"
        rank = result["needle_rank"] if result["needle_found_in_top_k"] else "N/A"
        sim = result["needle_similarity"]
        print(f"  Position={position:6s}: {found} rank={rank} sim={sim:.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Retrieval accuracy
    retrieval_results = all_results["retrieval_tests"]
    n_found = sum(1 for r in retrieval_results if r["needle_found_in_top_k"])
    print(f"\nRetrieval (top-{args.top_k}):")
    print(f"  Found: {n_found}/{len(retrieval_results)} "
          f"({n_found / len(retrieval_results) * 100:.0f}%)")
    ranks = [r["needle_rank"] for r in retrieval_results if r["needle_found_in_top_k"]]
    if ranks:
        print(f"  Mean rank: {sum(ranks) / len(ranks):.1f}")
        print(f"  Rank distribution: {sorted(ranks)}")

    # Generation quality
    gen_results = all_results["generation_tests"]
    total_egr_hits = sum(r["hits_egr"] for r in gen_results)
    total_base_hits = sum(r["hits_baseline"] for r in gen_results)
    total_possible = sum(r["total_answer_tokens"] for r in gen_results)
    print(f"\nGeneration answer token hits:")
    print(f"  EGR:      {total_egr_hits}/{total_possible} "
          f"({total_egr_hits / total_possible * 100:.0f}%)")
    print(f"  Baseline: {total_base_hits}/{total_possible} "
          f"({total_base_hits / total_possible * 100:.0f}%)")

    # Position sensitivity
    print(f"\nPosition sensitivity (needle: {test_needle.category}):")
    for r in all_results["position_tests"]:
        found = "Y" if r["needle_found_in_top_k"] else "N"
        print(f"  {r['position']:6s}: found={found} rank={r['needle_rank']} "
              f"sim={r['needle_similarity']:.4f}")

    # Save
    out_path = Path("engram_store_data") / "niah_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate long generation texts for JSON
    for r in all_results["generation_tests"]:
        r["gen_text_egr"] = r["gen_text_egr"][:500]
        r["gen_text_baseline"] = r["gen_text_baseline"][:500]
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
