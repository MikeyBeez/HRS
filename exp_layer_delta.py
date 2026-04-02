"""Layer-Delta Engram experiment: isolating novel context from the residual.

Standard engrams capture what the model already knows. Layer deltas
(layer[i+1] - layer[i]) isolate what each layer specifically adds.

Tests whether delta-based engrams improve retrieval and grounding
over standard mean-pooled engrams.

Usage:
    python exp_layer_delta.py [--device cuda]
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore, EngramEntry
from niah_egr import NEEDLES, DISTRACTORS


def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


@torch.no_grad()
def compute_all_hidden_states(model, token_ids, device):
    """Run forward pass and capture hidden states at every layer.

    Returns list of (1, T, D) tensors, one per layer output,
    plus the embedding output as layer 0.
    """
    ids = token_ids[:512].unsqueeze(0).to(device)

    # Hook every block to capture outputs
    hidden_states = []

    # Capture post-embedding (layer 0)
    x = model.drop(model.tok_emb(ids))
    hidden_states.append(x.detach().clone())

    # Run through each block
    for i, block in enumerate(model.blocks):
        # V18 cross-attn needs engram_buffer
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        x, _, _, _ = block(x, step=0, engram_buffer=eb)
        hidden_states.append(x.detach().clone())

    return hidden_states  # list of (1, T, D), length = n_layers + 1


@torch.no_grad()
def compute_layer_deltas(model, token_ids, device):
    """Compute per-layer deltas and their norms.

    Returns:
        deltas: list of (D,) normalized delta vectors (one per layer transition)
        delta_norms: list of scalar norms (magnitude of change per layer)
        standard_engram: (D,) mean-pooled hidden state from second-to-last layer
    """
    hidden_states = compute_all_hidden_states(model, token_ids, device)
    n_layers = len(hidden_states) - 1  # subtract embedding

    deltas = []
    delta_norms = []

    for i in range(n_layers):
        delta = hidden_states[i + 1] - hidden_states[i]  # (1, T, D)
        # Mean pool across positions
        pooled = delta.mean(dim=1).squeeze(0)  # (D,)
        # Record norm before normalizing
        delta_norms.append(pooled.norm().item())
        # L2 normalize
        normalized = F.normalize(pooled, dim=0)
        deltas.append(normalized.cpu())

    # Standard engram: mean-pool from second-to-last layer
    standard = hidden_states[-2].mean(dim=1).squeeze(0)  # (D,)
    standard = F.normalize(standard, dim=0).cpu()

    return deltas, delta_norms, standard


def build_stores_with_needle(model, tokenizer, needle, distractors, device):
    """Process needle + distractors, building stores for each delta layer.

    Returns:
        stores: dict mapping layer_idx -> EngramStore with all docs
        standard_store: EngramStore with standard engrams
        needle_deltas: list of delta vectors for the needle
        needle_norms: delta norms for the needle
        distractor_norms: list of delta norm lists for distractors
    """
    d_model = model.cfg.model.d_model
    n_layers = model.cfg.model.n_layers

    # One store per delta layer + one for standard + one for concatenated
    stores = {i: EngramStore(d_model) for i in range(n_layers)}
    standard_store = EngramStore(d_model)
    concat_store = EngramStore(d_model * n_layers)

    all_docs = [(needle.fact, "needle")] + [(d, f"distractor_{i}") for i, d in enumerate(distractors)]

    needle_deltas = None
    needle_norms = None
    distractor_norms = []

    for text, source in all_docs:
        ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
        if ids.shape[0] < 5:
            continue

        deltas, norms, standard = compute_layer_deltas(model, ids, device)

        entry = EngramEntry(text=text, mean_entropy=0.0, condition="delta", source=source)

        # Store in each per-layer store
        for layer_idx, delta in enumerate(deltas):
            stores[layer_idx].store(delta, EngramEntry(
                text=text, mean_entropy=0.0, condition=f"delta_{layer_idx}", source=source,
            ))

        # Store standard engram
        standard_store.store(standard, EngramEntry(
            text=text, mean_entropy=0.0, condition="standard", source=source,
        ))

        # Store concatenated deltas
        concat = torch.cat(deltas)  # (n_layers * D,)
        concat = F.normalize(concat, dim=0)
        concat_store.store(concat, EngramEntry(
            text=text, mean_entropy=0.0, condition="concat_delta", source=source,
        ))

        if source == "needle":
            needle_deltas = deltas
            needle_norms = norms
        else:
            distractor_norms.append(norms)

    return stores, standard_store, concat_store, needle_deltas, needle_norms, distractor_norms


def test_retrieval(store, query_engram, top_k=5):
    """Test if needle is retrieved."""
    results = store.retrieve(query_engram, top_k=top_k, min_similarity=0.0)
    for rank, (sim, entry, _) in enumerate(results):
        if entry.source == "needle":
            return True, rank + 1, sim
    return False, -1, 0.0


@torch.no_grad()
def test_generation_grounding(model, query_ids, engram_vector, device,
                              answer_tokens, max_new_tokens=150):
    """Generate with engram injected and measure answer token recall."""
    model.eval()
    d_model = model.cfg.model.d_model
    orig_buffer = model.engram_buffer.data.clone()
    orig_init = model._engram_buffer_initialized

    # Expand engram to fill buffer
    if engram_vector.shape[0] == d_model:
        buffer = engram_vector.unsqueeze(0).unsqueeze(0).expand(1, 32, d_model).contiguous()
    else:
        # Truncate or pad to 32 slots
        ev = engram_vector.view(-1, d_model)
        if ev.shape[0] >= 32:
            buffer = ev[:32].unsqueeze(0)
        else:
            repeats = (32 + ev.shape[0] - 1) // ev.shape[0]
            buffer = ev.repeat(repeats, 1)[:32].unsqueeze(0)

    model.engram_buffer = torch.nn.Parameter(buffer.to(device), requires_grad=False)
    model._engram_buffer_initialized = True

    input_ids = query_ids.unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        idx = input_ids[:, -512:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / 0.9
        v, _ = torch.topk(logits, 50)
        logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    model.engram_buffer = torch.nn.Parameter(orig_buffer, requires_grad=False)
    model._engram_buffer_initialized = orig_init

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gen_text = tokenizer.decode(input_ids[0, query_ids.shape[0]:], skip_special_tokens=True)
    text_lower = gen_text.lower()
    hits = sum(1 for t in answer_tokens if t.lower() in text_lower)
    return hits, len(answer_tokens), gen_text[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    n_layers = cfg.model.n_layers
    d_model = cfg.model.d_model

    distractors = DISTRACTORS[:20]
    results_dir = Path("results/v18_cross_attn/layer_delta")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{n_layers} layers → {n_layers} deltas per document")
    print(f"Testing {len(NEEDLES)} needles with {len(distractors)} distractors\n")

    all_results = {
        "per_layer_retrieval": {i: [] for i in range(n_layers)},
        "standard_retrieval": [],
        "concat_retrieval": [],
        "delta_magnitude": {"needle": [], "distractor": []},
        "generation_grounding": {},
    }

    for needle in NEEDLES:
        print(f"{'='*60}")
        print(f"Needle: {needle.category} — {needle.fact[:60]}...")

        # Build stores
        stores, std_store, cat_store, needle_deltas, needle_norms, dist_norms = \
            build_stores_with_needle(model, tokenizer, needle, distractors, device)

        # Record delta magnitudes
        all_results["delta_magnitude"]["needle"].append({
            "category": needle.category,
            "norms": needle_norms,
        })
        for dn in dist_norms:
            all_results["delta_magnitude"]["distractor"].append(dn)

        print(f"  Delta magnitudes: {['%.2f' % n for n in needle_norms]}")
        avg_dist_norms = [sum(d[i] for d in dist_norms) / len(dist_norms) for i in range(n_layers)]
        print(f"  Distractor mean:  {['%.2f' % n for n in avg_dist_norms]}")

        # Query engrams
        query_ids = torch.tensor(
            tokenizer.encode(needle.query, add_special_tokens=False), dtype=torch.long
        )
        query_deltas, query_norms, query_standard = compute_layer_deltas(model, query_ids, device)
        query_concat = F.normalize(torch.cat(query_deltas), dim=0)

        # Test retrieval per layer
        print(f"\n  Retrieval by delta layer:")
        for layer_idx in range(n_layers):
            found, rank, sim = test_retrieval(stores[layer_idx], query_deltas[layer_idx])
            status = f"FOUND rank={rank} sim={sim:.4f}" if found else "NOT FOUND"
            print(f"    Delta {layer_idx} (layer {layer_idx}→{layer_idx+1}): {status}")
            all_results["per_layer_retrieval"][layer_idx].append({
                "needle": needle.category, "found": found, "rank": rank, "sim": sim,
            })

        # Standard engram retrieval
        found_std, rank_std, sim_std = test_retrieval(std_store, query_standard)
        status = f"FOUND rank={rank_std} sim={sim_std:.4f}" if found_std else "NOT FOUND"
        print(f"    Standard engram: {status}")
        all_results["standard_retrieval"].append({
            "needle": needle.category, "found": found_std, "rank": rank_std, "sim": sim_std,
        })

        # Concatenated deltas retrieval
        found_cat, rank_cat, sim_cat = test_retrieval(cat_store, query_concat)
        status = f"FOUND rank={rank_cat} sim={sim_cat:.4f}" if found_cat else "NOT FOUND"
        print(f"    Concat deltas:   {status}")
        all_results["concat_retrieval"].append({
            "needle": needle.category, "found": found_cat, "rank": rank_cat, "sim": sim_cat,
        })

    # ============================================================
    # Generation grounding with best delta
    # ============================================================
    print(f"\n{'='*60}")
    print("GENERATION GROUNDING — Best delta vs standard")
    print(f"{'='*60}")

    # Find which delta layer gives best retrieval
    layer_scores = {}
    for layer_idx in range(n_layers):
        results = all_results["per_layer_retrieval"][layer_idx]
        n_found = sum(1 for r in results if r["found"])
        mean_rank = sum(r["rank"] for r in results if r["found"]) / max(n_found, 1)
        layer_scores[layer_idx] = (n_found, -mean_rank)  # higher found, lower rank = better
    best_layer = max(layer_scores, key=lambda k: layer_scores[k])
    print(f"\n  Best retrieval layer: delta {best_layer}")

    for needle in NEEDLES:
        query_ids = torch.tensor(
            tokenizer.encode(needle.query, add_special_tokens=False), dtype=torch.long
        )
        needle_ids = torch.tensor(
            tokenizer.encode(needle.fact, add_special_tokens=False), dtype=torch.long
        )

        # Get the best-layer delta for this needle
        deltas, _, standard = compute_layer_deltas(model, needle_ids, device)
        best_delta = deltas[best_layer]

        # Test generation with best delta
        hits_d, total_d, text_d = test_generation_grounding(
            model, query_ids, best_delta, device, needle.answer_tokens
        )
        # Test generation with standard engram
        hits_s, total_s, text_s = test_generation_grounding(
            model, query_ids, standard, device, needle.answer_tokens
        )

        print(f"\n  {needle.category}:")
        print(f"    Standard engram: {hits_s}/{total_s} — {text_s[:100]}...")
        print(f"    Delta {best_layer}:        {hits_d}/{total_d} — {text_d[:100]}...")

        all_results["generation_grounding"][needle.category] = {
            "standard": {"hits": hits_s, "total": total_s},
            "best_delta": {"hits": hits_d, "total": total_d, "layer": best_layer},
        }

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n  Retrieval accuracy by engram type:")
    print(f"  {'Type':<25} {'Found':>6} {'Mean Rank':>10} {'Mean Sim':>10}")
    print(f"  {'-'*55}")

    # Standard
    std_found = sum(1 for r in all_results["standard_retrieval"] if r["found"])
    std_ranks = [r["rank"] for r in all_results["standard_retrieval"] if r["found"]]
    std_sims = [r["sim"] for r in all_results["standard_retrieval"] if r["found"]]
    print(f"  {'Standard engram':<25} {std_found:>6}/5 "
          f"{sum(std_ranks)/len(std_ranks) if std_ranks else 0:>10.1f} "
          f"{sum(std_sims)/len(std_sims) if std_sims else 0:>10.4f}")

    # Per layer
    for layer_idx in range(n_layers):
        results = all_results["per_layer_retrieval"][layer_idx]
        n_found = sum(1 for r in results if r["found"])
        ranks = [r["rank"] for r in results if r["found"]]
        sims = [r["sim"] for r in results if r["found"]]
        marker = " ← best" if layer_idx == best_layer else ""
        print(f"  {f'Delta {layer_idx} (L{layer_idx}→L{layer_idx+1})':<25} {n_found:>6}/5 "
              f"{sum(ranks)/len(ranks) if ranks else 0:>10.1f} "
              f"{sum(sims)/len(sims) if sims else 0:>10.4f}{marker}")

    # Concat
    cat_found = sum(1 for r in all_results["concat_retrieval"] if r["found"])
    cat_ranks = [r["rank"] for r in all_results["concat_retrieval"] if r["found"]]
    cat_sims = [r["sim"] for r in all_results["concat_retrieval"] if r["found"]]
    print(f"  {'Concat all deltas':<25} {cat_found:>6}/5 "
          f"{sum(cat_ranks)/len(cat_ranks) if cat_ranks else 0:>10.1f} "
          f"{sum(cat_sims)/len(cat_sims) if cat_sims else 0:>10.4f}")

    # Delta magnitude comparison
    print(f"\n  Delta magnitude by layer (needle vs distractor mean):")
    print(f"  {'Layer':<15} {'Needle':>10} {'Distractor':>12} {'Ratio':>8}")
    all_needle_norms = all_results["delta_magnitude"]["needle"]
    all_dist_norms = all_results["delta_magnitude"]["distractor"]
    for i in range(n_layers):
        n_mean = sum(nn["norms"][i] for nn in all_needle_norms) / len(all_needle_norms)
        d_mean = sum(dn[i] for dn in all_dist_norms) / len(all_dist_norms)
        ratio = n_mean / d_mean if d_mean > 0 else 0
        print(f"  {f'Delta {i}':<15} {n_mean:>10.2f} {d_mean:>12.2f} {ratio:>8.2f}x")

    # Generation grounding
    print(f"\n  Generation grounding (answer token recall):")
    total_std = sum(v["standard"]["hits"] for v in all_results["generation_grounding"].values())
    total_delta = sum(v["best_delta"]["hits"] for v in all_results["generation_grounding"].values())
    total_possible = sum(v["standard"]["total"] for v in all_results["generation_grounding"].values())
    print(f"    Standard engram: {total_std}/{total_possible} ({total_std/total_possible:.0%})")
    print(f"    Best delta (L{best_layer}): {total_delta}/{total_possible} ({total_delta/total_possible:.0%})")

    # Save
    out_path = results_dir / "layer_delta_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
