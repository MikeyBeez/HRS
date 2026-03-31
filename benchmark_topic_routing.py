"""Hard benchmarks for topic-routed context assembly.

Tests failure modes: centroid drift, semantic overlap resolution,
adversarial vocabulary, and topic forking.

Usage:
    python benchmark_topic_routing.py [--device cuda] [--threshold 0.4]
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import torch
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from topic_context import TopicContextManager


# ============================================================
# Test data
# ============================================================

DRIFT_BOMB = [
    "The Roman Empire dominated the Mediterranean for centuries.",
    "Roman roads connected distant provinces to the capital.",
    "Roman engineering influenced infrastructure across Europe.",
    "European infrastructure evolved significantly in the medieval period.",
    "Medieval European cities developed complex trade networks.",
    "Trade networks in the medieval period shaped modern economics.",
    "Modern economics relies on computational models.",
    "Computational models use iterative optimization.",
    "Iterative optimization is central to training neural networks.",
    "Neural networks learn hierarchical representations from data.",
]

DRIFT_BOMB_PROBE = "The Roman Empire fell in 476 AD when the last emperor was deposed."

SEMANTIC_OVERLAP = {
    0: {
        "name": "Trivial (Roman history vs JavaScript)",
        "topic_a": [
            "The Roman Empire expanded through military conquest and strategic alliances.",
            "Roman legions were organized into cohorts and centuries under strict discipline.",
            "The Pax Romana was a period of relative peace lasting about 200 years.",
            "Roman law formed the basis of many modern European legal systems.",
            "The Roman Senate held significant power during the Republican period.",
        ],
        "topic_b": [
            "JavaScript uses prototypal inheritance rather than classical inheritance.",
            "The async/await syntax simplifies handling asynchronous operations in JavaScript.",
            "Node.js allows JavaScript to run on the server side outside the browser.",
            "The DOM API lets JavaScript manipulate HTML elements dynamically.",
            "TypeScript adds static type checking to JavaScript for larger codebases.",
        ],
    },
    1: {
        "name": "Distant overlap (Roman military engineering vs modern civil engineering)",
        "topic_a": [
            "Roman military engineers built fortified camps called castra at each overnight stop.",
            "Siege engines like the ballista and onager gave Roman armies ranged capability.",
            "Roman military roads were built in layers: rubble, gravel, fitted stone slabs.",
            "Roman engineers diverted rivers to flood enemy positions during sieges.",
            "The pontoon bridge allowed Roman legions to cross rivers rapidly during campaigns.",
        ],
        "topic_b": [
            "Modern civil engineers use finite element analysis to model structural loads.",
            "Prestressed concrete allows bridges to span greater distances with less material.",
            "Environmental impact assessments are required before major construction projects.",
            "Geotechnical engineering determines soil bearing capacity for building foundations.",
            "Traffic flow modeling uses queuing theory to design highway interchanges.",
        ],
    },
    2: {
        "name": "Moderate overlap (Roman bread baking vs medieval bread baking)",
        "topic_a": [
            "Roman bakers ground wheat using large rotary mills called molae driven by donkeys.",
            "The pistrina were commercial bakeries in Rome that produced bread for the public.",
            "Roman bread was often made with emmer wheat, producing a dense, dark loaf.",
            "Bakers in ancient Rome formed the collegium pistorum, one of the earliest trade guilds.",
            "Roman soldiers received a daily grain ration called frumentum to make their own bread.",
        ],
        "topic_b": [
            "Medieval bakers used communal ovens called bannalités controlled by the local lord.",
            "Rye bread was more common than wheat bread in northern medieval Europe.",
            "The Assize of Bread and Ale in 1266 regulated bread prices and weights in England.",
            "Medieval monasteries developed brewing and baking techniques that advanced fermentation.",
            "Dark bread made from maslin, a mix of wheat and rye, was the staple of medieval peasants.",
        ],
    },
    3: {
        "name": "High overlap (Roman Republic vs Roman Empire governance)",
        "topic_a": [
            "The Roman Republic was governed by elected magistrates and the Senate after 509 BC.",
            "Two consuls served as joint heads of state during the Republic, each with veto power.",
            "The tribune of the plebs could block legislation that harmed common citizens' interests.",
            "Republican Rome expanded through a system of alliances and granted citizenship gradually.",
            "The Senate's authority in the Republic derived from custom and prestige, not formal law.",
        ],
        "topic_b": [
            "The Roman Empire centralized power under a single princeps beginning with Augustus.",
            "Imperial governors administered provinces with greater autonomy than Republican proconsuls.",
            "The Praetorian Guard served as the emperor's personal bodyguard and political kingmaker.",
            "Imperial Rome maintained control through a professional standing army loyal to the emperor.",
            "The emperor held tribunicia potestas, absorbing the tribune's power into the imperial office.",
        ],
    },
    4: {
        "name": "Near-identical (Caesar's military vs Caesar's politics)",
        "topic_a": [
            "Caesar's conquest of Gaul between 58 and 50 BC brought vast territory under Roman control.",
            "The siege of Alesia demonstrated Caesar's engineering genius with double circumvallation.",
            "Caesar's legions in Gaul developed fierce loyalty through shared hardship and generous rewards.",
            "The crossing of the Rubicon in 49 BC was a military act that triggered civil war.",
            "Caesar's victory at Pharsalus against Pompey decided the civil war in his favor.",
        ],
        "topic_b": [
            "Caesar's political career began with his election as quaestor in 69 BC.",
            "The First Triumvirate was an informal political alliance between Caesar, Pompey, and Crassus.",
            "Caesar's land reform laws redistributed public land to veterans and the urban poor.",
            "As dictator perpetuo, Caesar enacted a calendar reform that created the Julian calendar.",
            "Caesar's assassination on the Ides of March was motivated by senators fearing tyranny.",
        ],
    },
}

ADVERSARIAL_PAIRS = [
    {
        "name": "Baking metaphor for ML",
        "metaphorical": "The layers of a cake are like the layers of a neural network — each one transforms the input into something richer and more complex.",
        "straight": "Neural networks use stacked layers where each layer applies a learned transformation to its input.",
        "true_topic": "machine_learning",
    },
    {
        "name": "ML vocabulary for history",
        "metaphorical": "Napoleon's defeat at Waterloo was a catastrophic loss function for the French Empire, with no gradient path back to dominance.",
        "straight": "Napoleon's defeat at Waterloo in 1815 ended French imperial ambitions and reshaped European politics.",
        "true_topic": "history",
    },
    {
        "name": "CS vocabulary for Roman history",
        "metaphorical": "The Roman Senate operated like a distributed system with no single point of failure, using consensus protocols to maintain stability.",
        "straight": "The Roman Senate distributed power across multiple elected officials to prevent any one person from gaining absolute control.",
        "true_topic": "roman_history",
    },
    {
        "name": "Relationship vocabulary for physics",
        "metaphorical": "Quantum entanglement is like a long-distance relationship — two particles remain correlated no matter how far apart, yet neither sends a message.",
        "straight": "Quantum entanglement creates correlations between spatially separated particles that persist regardless of distance.",
        "true_topic": "physics",
    },
    {
        "name": "Cooking vocabulary for chemistry",
        "metaphorical": "Polymerization is like making a chain of paper clips — you keep snapping monomers together until you have a long, flexible strand.",
        "straight": "Polymerization links monomer molecules into long chain-like macromolecules through repeated chemical bonding.",
        "true_topic": "chemistry",
    },
]

TOPIC_FORK = {
    "unified": [
        "Machine learning models learn patterns from training data.",
        "Neural networks use layers of learned transformations to process inputs.",
        "Gradient descent optimizes model parameters by following the loss surface.",
    ],
    "fork_a_name": "computer_vision",
    "fork_a": [
        "Convolutional layers detect spatial features like edges and textures in images.",
        "Object detection requires predicting bounding boxes and class labels simultaneously.",
        "Image segmentation assigns a class label to every pixel in the image.",
    ],
    "fork_b_name": "nlp",
    "fork_b": [
        "Attention mechanisms let models weigh the importance of different input tokens.",
        "Language models predict the next token given a sequence of preceding tokens.",
        "Transformers replaced recurrent networks for most sequence modeling tasks.",
    ],
}


# ============================================================
# Benchmark implementations
# ============================================================

def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


def run_benchmark_7_drift_bomb(model, tokenizer, threshold, device):
    """Benchmark 7: Centroid Drift Bomb."""
    print("\n" + "=" * 70)
    print("BENCHMARK 7: Centroid Drift Bomb")
    print("=" * 70)

    mgr = TopicContextManager(model, tokenizer, similarity_threshold=threshold)

    # Feed drift sequence
    centroids_over_time = []
    initial_centroid = None

    for i, prompt in enumerate(DRIFT_BOMB):
        result = mgr.process_prompt(prompt)
        centroid = mgr.clusters[0].centroid.clone()
        if initial_centroid is None:
            initial_centroid = centroid.clone()
        drift = 1.0 - (initial_centroid @ centroid).item()
        centroids_over_time.append(drift)
        print(f"  Step {i}: \"{prompt[:55]}...\"")
        print(f"    cluster={result['cluster_id']}, n_clusters={result['n_clusters']}, "
              f"drift_from_original={drift:.4f}")

    # Probe: does the original topic still match?
    print(f"\n  Probe: \"{DRIFT_BOMB_PROBE[:55]}...\"")
    probe_engram = mgr.extract_engram(
        torch.tensor(tokenizer.encode(DRIFT_BOMB_PROBE), dtype=torch.long)
    )
    nearest, sim = mgr.find_nearest_cluster(probe_engram)
    probe_matches_original = nearest is not None and nearest.cluster_id == 0

    print(f"    Nearest cluster: {nearest.cluster_id if nearest else 'None'}, "
          f"sim={sim:.4f}, matches_original={probe_matches_original}")
    print(f"    Cluster title: \"{mgr.clusters[0].title}\"")
    print(f"    Final centroid drift: {centroids_over_time[-1]:.4f}")

    return {
        "benchmark": "drift_bomb",
        "drift_trajectory": centroids_over_time,
        "final_drift": centroids_over_time[-1],
        "probe_matches_original": probe_matches_original,
        "probe_similarity": sim,
        "final_title": mgr.clusters[0].title,
        "n_clusters_final": len(mgr.clusters),
    }


def run_benchmark_1_semantic_overlap(model, tokenizer, threshold, device):
    """Benchmark 1: Semantic Overlap Gradient."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Semantic Overlap Gradient")
    print("=" * 70)

    results_by_level = {}

    for level in sorted(SEMANTIC_OVERLAP.keys()):
        data = SEMANTIC_OVERLAP[level]
        print(f"\n  Level {level}: {data['name']}")

        mgr = TopicContextManager(model, tokenizer, similarity_threshold=threshold)

        # Interleave topics
        labels = []  # ground truth: 'a' or 'b'
        assigned = []  # cluster assignments

        for a, b in zip(data["topic_a"], data["topic_b"]):
            result_a = mgr.process_prompt(a)
            labels.append("a")
            assigned.append(result_a["cluster_id"])

            result_b = mgr.process_prompt(b)
            labels.append("b")
            assigned.append(result_b["cluster_id"])

        # Compute purity
        n_clusters = len(mgr.clusters)
        cluster_labels = {}
        for label, cluster_id in zip(labels, assigned):
            if cluster_id not in cluster_labels:
                cluster_labels[cluster_id] = []
            cluster_labels[cluster_id].append(label)

        # Purity: fraction of majority label in each cluster
        total_correct = 0
        for cid, clabels in cluster_labels.items():
            majority = max(clabels.count("a"), clabels.count("b"))
            total_correct += majority
        purity = total_correct / len(labels)

        # Misrouting: did any 'a' and 'b' end up in the same cluster?
        mixed_clusters = sum(
            1 for clabels in cluster_labels.values()
            if "a" in clabels and "b" in clabels
        )

        # Centroid distance
        if n_clusters >= 2:
            centroids = [c.centroid for c in mgr.clusters[:2]]
            separation = 1.0 - (centroids[0] @ centroids[1]).item()
        else:
            separation = 0.0

        print(f"    Clusters: {n_clusters}, Purity: {purity:.3f}, "
              f"Separation: {separation:.4f}, Mixed: {mixed_clusters}")
        for c in mgr.clusters:
            c_labels = cluster_labels.get(c.cluster_id, [])
            a_count = c_labels.count("a")
            b_count = c_labels.count("b")
            print(f"      Cluster {c.cluster_id} \"{c.title}\": "
                  f"{a_count}a + {b_count}b = {len(c_labels)} prompts")

        results_by_level[level] = {
            "name": data["name"],
            "n_clusters": n_clusters,
            "purity": purity,
            "separation": separation,
            "mixed_clusters": mixed_clusters,
        }

    return {"benchmark": "semantic_overlap", "levels": results_by_level}


def run_benchmark_4_adversarial(model, tokenizer, threshold, device):
    """Benchmark 4: Adversarial Prompts."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Adversarial Prompts")
    print("=" * 70)

    results = []

    for pair in ADVERSARIAL_PAIRS:
        print(f"\n  {pair['name']}:")

        mgr = TopicContextManager(model, tokenizer, similarity_threshold=threshold)

        # Process straight version first (establishes the "correct" cluster)
        result_straight = mgr.process_prompt(pair["straight"])
        straight_cluster = result_straight["cluster_id"]

        # Process metaphorical version
        result_meta = mgr.process_prompt(pair["metaphorical"])
        meta_cluster = result_meta["cluster_id"]

        same_cluster = straight_cluster == meta_cluster

        # Also check cosine similarity between the two engrams
        eng_straight = mgr.clusters[straight_cluster].prompts[0].engram
        eng_meta = None
        for c in mgr.clusters:
            for p in c.prompts:
                if p.text == pair["metaphorical"]:
                    eng_meta = p.engram
                    break

        if eng_meta is not None:
            from torch.nn.functional import normalize
            sim = (normalize(eng_straight.float(), dim=0) @ normalize(eng_meta.float(), dim=0)).item()
        else:
            sim = 0.0

        print(f"    Straight -> cluster {straight_cluster}")
        print(f"    Metaphor -> cluster {meta_cluster}")
        print(f"    Same cluster: {same_cluster}, Engram similarity: {sim:.4f}")

        results.append({
            "name": pair["name"],
            "true_topic": pair["true_topic"],
            "same_cluster": same_cluster,
            "engram_similarity": sim,
            "n_clusters": len(mgr.clusters),
        })

    correct = sum(1 for r in results if r["same_cluster"])
    print(f"\n  Routing accuracy: {correct}/{len(results)} "
          f"({correct/len(results)*100:.0f}%)")

    return {"benchmark": "adversarial", "pairs": results, "accuracy": correct / len(results)}


def run_benchmark_3_topic_fork(model, tokenizer, threshold, device):
    """Benchmark 3: Topic Fork."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Topic Fork")
    print("=" * 70)

    mgr = TopicContextManager(model, tokenizer, similarity_threshold=threshold)

    # Phase 1: unified prompts
    print("\n  Phase 1: Unified (machine learning)")
    for p in TOPIC_FORK["unified"]:
        result = mgr.process_prompt(p)
        print(f"    \"{p[:55]}\" -> cluster {result['cluster_id']}")

    unified_cluster_count = len(mgr.clusters)
    print(f"    Clusters after unified phase: {unified_cluster_count}")

    # Phase 2: interleaved fork
    print("\n  Phase 2: Forking (CV and NLP interleaved)")
    fork_labels = []
    fork_clusters = []

    for a, b in zip(TOPIC_FORK["fork_a"], TOPIC_FORK["fork_b"]):
        result_a = mgr.process_prompt(a)
        fork_labels.append("cv")
        fork_clusters.append(result_a["cluster_id"])
        print(f"    [CV]  \"{a[:50]}\" -> cluster {result_a['cluster_id']}")

        result_b = mgr.process_prompt(b)
        fork_labels.append("nlp")
        fork_clusters.append(result_b["cluster_id"])
        print(f"    [NLP] \"{b[:50]}\" -> cluster {result_b['cluster_id']}")

    final_cluster_count = len(mgr.clusters)
    did_split = final_cluster_count > unified_cluster_count

    # Check if CV and NLP ended up in different clusters
    cv_clusters = set(c for l, c in zip(fork_labels, fork_clusters) if l == "cv")
    nlp_clusters = set(c for l, c in zip(fork_labels, fork_clusters) if l == "nlp")
    clean_split = len(cv_clusters) == 1 and len(nlp_clusters) == 1 and cv_clusters != nlp_clusters

    print(f"\n    Clusters after fork: {final_cluster_count}")
    print(f"    Did split: {did_split}")
    print(f"    Clean split (CV vs NLP): {clean_split}")
    print(f"    CV in clusters: {cv_clusters}")
    print(f"    NLP in clusters: {nlp_clusters}")
    print(f"\n    {mgr.describe_clusters()}")

    return {
        "benchmark": "topic_fork",
        "unified_clusters": unified_cluster_count,
        "final_clusters": final_cluster_count,
        "did_split": did_split,
        "clean_split": clean_split,
        "cv_clusters": list(cv_clusters),
        "nlp_clusters": list(nlp_clusters),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Topic routing benchmarks")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print(f"Similarity threshold: {args.threshold}")

    all_results = {"threshold": args.threshold, "benchmarks": {}}

    # Run benchmarks in priority order
    t0 = time.time()

    r = run_benchmark_7_drift_bomb(model, tokenizer, args.threshold, device)
    all_results["benchmarks"]["drift_bomb"] = r

    r = run_benchmark_1_semantic_overlap(model, tokenizer, args.threshold, device)
    all_results["benchmarks"]["semantic_overlap"] = r

    r = run_benchmark_4_adversarial(model, tokenizer, args.threshold, device)
    all_results["benchmarks"]["adversarial"] = r

    r = run_benchmark_3_topic_fork(model, tokenizer, args.threshold, device)
    all_results["benchmarks"]["topic_fork"] = r

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Drift bomb
    db = all_results["benchmarks"]["drift_bomb"]
    print(f"\nDrift Bomb:")
    print(f"  Final drift from original: {db['final_drift']:.4f}")
    print(f"  Probe matches original cluster: {db['probe_matches_original']}")
    print(f"  Probe similarity: {db['probe_similarity']:.4f}")

    # Semantic overlap
    so = all_results["benchmarks"]["semantic_overlap"]
    print(f"\nSemantic Overlap Gradient:")
    for level, data in so["levels"].items():
        print(f"  Level {level} ({data['name'][:40]}): "
              f"purity={data['purity']:.3f}, separation={data['separation']:.4f}, "
              f"clusters={data['n_clusters']}")

    # Adversarial
    adv = all_results["benchmarks"]["adversarial"]
    print(f"\nAdversarial Prompts:")
    print(f"  Routing accuracy: {adv['accuracy']*100:.0f}%")
    for p in adv["pairs"]:
        status = "PASS" if p["same_cluster"] else "FAIL"
        print(f"    [{status}] {p['name']}: sim={p['engram_similarity']:.4f}")

    # Topic fork
    tf = all_results["benchmarks"]["topic_fork"]
    print(f"\nTopic Fork:")
    print(f"  Did split: {tf['did_split']}")
    print(f"  Clean split: {tf['clean_split']}")
    print(f"  Clusters: {tf['unified_clusters']} -> {tf['final_clusters']}")

    print(f"\nTotal time: {elapsed:.0f}s")

    # Save
    out_path = Path("results/v18_cross_attn/topic_routing_benchmarks.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
