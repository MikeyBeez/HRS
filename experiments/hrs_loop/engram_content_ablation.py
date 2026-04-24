"""Engram content ablation for V18.

Tests four engram-construction strategies against the V18 cross-attention
buffer, using the existing needle-in-haystack setup (5 planted facts + 20
WikiText-103-style distractors). Weights are frozen; only the engram
representation changes.

Variants:
  v1_baseline         : final-layer mean-pooled (the production engram)
  v2_multi_layer      : layers 1, 3, 5 each mean-pooled, stacked as 3 slots
  v3_last_n_tokens    : last 32 final-layer token hidden states, no pooling
  v4_first_and_last   : first 16 + last 16 final-layer token hidden states

For each variant x each needle:
  1. Build retrieval key + injection tensor for needle and all distractors.
  2. Rank the needle against distractors using the variant's retrieval key.
  3. Inject the top-retrieved document's injection tensor into the buffer
     and run greedy-gated sampling (temperature=0.9, top-k=50).
  4. Count answer-token hits in the generation.

No entropy gating: top-1 retrieval is forced before each generation so we
measure the engram-content ceiling, not the entropy monitor.

Outputs per-variant and combined JSON under
experiments/hrs_loop/results/engram_content_ablation/.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer
from niah_egr import NEEDLES, DISTRACTORS, Needle


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_v18_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load(
        "results/v18_cross_attn/best.pt", map_location=device, weights_only=False
    )
    # V18 predates the V21 gate_scalar addition; load non-strict and neutralize.
    import math
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    inv_softplus_1 = math.log(math.e - 1.0)  # softplus(x) = 1 at x = log(e-1)
    patched = 0
    for name, p in model.named_parameters():
        if name.endswith(".cross_attn.gate_scalar") and name.split(".")[0] == "blocks":
            if any(name == m for m in missing):
                with torch.no_grad():
                    p.fill_(inv_softplus_1)
                patched += 1
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)} "
          f"patched gate_scalar on {patched} blocks (→ softplus=1)")
    if unexpected:
        print(f"  unexpected keys (first 5): {unexpected[:5]}")

    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18 (step {ckpt.get('step', '?')}, val_ppl "
          f"{ckpt.get('val_ppl', float('nan')):.2f})")
    return model, cfg


# ------------------------------------------------------------
# Hidden-state extraction
# ------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states(model, input_ids, layer_indices):
    """Run a forward pass and return hidden states at the requested layers.

    Returns a dict {layer_idx: (B, T, D)}.
    """
    captures = {}
    handles = []

    def make_hook(idx):
        def hook(module, _inputs, outputs):
            captures[idx] = outputs[0].detach()
        return hook

    for idx in layer_indices:
        handles.append(model.blocks[idx].register_forward_hook(make_hook(idx)))
    try:
        _ = model(input_ids, step=0)
    finally:
        for h in handles:
            h.remove()
    return captures


# ------------------------------------------------------------
# Variants: each returns (retrieval_key (D,), injection_tensor (N_SLOTS, D))
# N_SLOTS is always 32 (buffer shape). Short-doc handling described inline.
# ------------------------------------------------------------

N_BUFFER_SLOTS = 32


def _pad_or_repeat_slots(vecs: torch.Tensor, n_slots: int = N_BUFFER_SLOTS) -> torch.Tensor:
    """Fill a (k, D) tensor into (n_slots, D) by cyclic repeat if needed."""
    k = vecs.shape[0]
    if k == n_slots:
        return vecs
    if k > n_slots:
        return vecs[:n_slots]
    reps = (n_slots + k - 1) // k
    return vecs.repeat(reps, 1)[:n_slots]


def v1_baseline(model, input_ids, extract_layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    caps = extract_hidden_states(model, input_ids, [extract_layer])
    h = caps[extract_layer][0]  # (T, D)
    key = h.mean(dim=0)  # (D,)
    injection = key.unsqueeze(0).expand(N_BUFFER_SLOTS, -1).contiguous()
    return key, injection


def v2_multi_layer(model, input_ids, layers=(1, 3, 5)) -> Tuple[torch.Tensor, torch.Tensor]:
    caps = extract_hidden_states(model, input_ids, list(layers))
    layer_means = torch.stack([caps[l][0].mean(dim=0) for l in layers], dim=0)  # (K, D)
    key = layer_means.mean(dim=0)  # (D,) for cosine retrieval
    injection = _pad_or_repeat_slots(layer_means)  # (32, D): [l1,l3,l5,l1,...]
    return key, injection


def v3_last_n_tokens(model, input_ids, extract_layer: int, n: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    caps = extract_hidden_states(model, input_ids, [extract_layer])
    h = caps[extract_layer][0]  # (T, D)
    t = h.shape[0]
    last = h[-min(n, t):]  # (k, D), k = min(n, t)
    key = last.mean(dim=0)  # (D,)
    injection = _pad_or_repeat_slots(last)
    return key, injection


def v4_first_and_last(model, input_ids, extract_layer: int, n: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    caps = extract_hidden_states(model, input_ids, [extract_layer])
    h = caps[extract_layer][0]  # (T, D)
    t = h.shape[0]
    if t <= 2 * n:
        selected = h  # doc shorter than 2N — take everything, still deduplicated
    else:
        selected = torch.cat([h[:n], h[-n:]], dim=0)  # (2N, D)
    key = selected.mean(dim=0)
    injection = _pad_or_repeat_slots(selected)
    return key, injection


VARIANTS = {
    "v1_baseline": lambda m, ids, cfg: v1_baseline(m, ids, cfg["extract_layer"]),
    "v2_multi_layer": lambda m, ids, cfg: v2_multi_layer(m, ids, cfg["v2_layers"]),
    "v3_last_n_tokens": lambda m, ids, cfg: v3_last_n_tokens(m, ids, cfg["extract_layer"], cfg["v3_n"]),
    "v4_first_and_last": lambda m, ids, cfg: v4_first_and_last(m, ids, cfg["extract_layer"], cfg["v4_n"]),
}


# ------------------------------------------------------------
# Retrieval + generation
# ------------------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt_text: str, max_new_tokens: int,
             temperature: float, top_k: int, device) -> str:
    ids = torch.tensor(
        tokenizer.encode(prompt_text, add_special_tokens=False),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx = ids[:, -512:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def run_variant_on_needle(
    model, tokenizer, device, variant_name: str, build_fn: Callable,
    build_cfg: dict, needle: Needle, distractors: List[str],
    temperature: float, top_k: int, max_new_tokens: int,
) -> dict:
    docs = list(distractors) + [needle.fact]  # needle at last position
    needle_idx = len(docs) - 1

    # Encode each doc and build (key, injection) pairs.
    keys = []
    injections = []
    for text in docs:
        ids = torch.tensor(
            tokenizer.encode(text, add_special_tokens=False)[:512],
            dtype=torch.long, device=device,
        ).unsqueeze(0)
        key, inj = build_fn(model, ids, build_cfg)
        keys.append(F.normalize(key.float(), dim=0))
        injections.append(inj.detach())
    keys = torch.stack(keys, dim=0)  # (n_docs, D)

    # Build query key using the SAME variant construction (query-side consistency).
    q_ids = torch.tensor(
        tokenizer.encode(needle.query, add_special_tokens=False)[:512],
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    q_key, _ = build_fn(model, q_ids, build_cfg)
    q_key = F.normalize(q_key.float(), dim=0)

    # Cosine similarity and rank the needle.
    sims = keys @ q_key  # (n_docs,)
    order = sims.argsort(descending=True)
    rank = int((order == needle_idx).nonzero(as_tuple=True)[0].item()) + 1
    top_doc_idx = int(order[0].item())
    top_is_needle = top_doc_idx == needle_idx

    # Inject top-retrieved injection tensor and generate.
    orig_buffer = model.engram_buffer.data.clone()
    orig_flag = model._engram_buffer_initialized
    inj = injections[top_doc_idx].to(device).unsqueeze(0)  # (1, 32, D)
    assert inj.shape == model.engram_buffer.shape, (
        f"inj shape {tuple(inj.shape)} != buffer {tuple(model.engram_buffer.shape)}"
    )
    model.engram_buffer.data.copy_(inj)
    model._engram_buffer_initialized = True
    try:
        gen_text = generate(
            model, tokenizer, needle.query, max_new_tokens,
            temperature, top_k, device,
        )
    finally:
        model.engram_buffer.data.copy_(orig_buffer)
        model._engram_buffer_initialized = orig_flag

    gen_tail = gen_text[len(needle.query):]  # only the continuation
    gen_lower = gen_tail.lower()
    hits = [t for t in needle.answer_tokens if t.lower() in gen_lower]

    return {
        "variant": variant_name,
        "needle_category": needle.category,
        "query": needle.query,
        "rank": rank,
        "top_sim": float(sims[top_doc_idx].item()),
        "needle_sim": float(sims[needle_idx].item()),
        "top_is_needle": top_is_needle,
        "hits": hits,
        "n_hits": len(hits),
        "n_answer_tokens": len(needle.answer_tokens),
        "gen_text": gen_tail[:400],
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--n-distractors", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--v3-n", type=int, default=32)
    ap.add_argument("--v4-n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str,
                    default="experiments/hrs_loop/results/engram_content_ablation")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # extract_layer resolved positively: n_layers=6, cross_attn_engram.extract_layer=-2 → 4
    ca_cfg = cfg.cross_attn_engram
    extract_layer = ca_cfg.extract_layer if ca_cfg.extract_layer >= 0 \
        else cfg.model.n_layers + ca_cfg.extract_layer

    build_cfg = {
        "extract_layer": extract_layer,
        "v2_layers": (1, 3, 5),
        "v3_n": args.v3_n,
        "v4_n": args.v4_n,
    }

    distractors = DISTRACTORS[:args.n_distractors]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for vname in args.variants:
        if vname not in VARIANTS:
            print(f"Unknown variant: {vname}")
            continue
        print(f"\n{'='*60}\nVariant: {vname}\n{'='*60}")
        build_fn = VARIANTS[vname]

        per_needle = []
        t0 = time.time()
        # Deterministic per-variant seed so sampling is comparable across variants
        torch.manual_seed(args.seed)
        for needle in NEEDLES:
            r = run_variant_on_needle(
                model, tokenizer, device, vname, build_fn, build_cfg,
                needle, distractors,
                args.temperature, args.top_k, args.max_new_tokens,
            )
            per_needle.append(r)
            print(f"  {needle.category:10s} rank={r['rank']} "
                  f"sim_needle={r['needle_sim']:.3f} sim_top={r['top_sim']:.3f} "
                  f"top_is_needle={r['top_is_needle']} "
                  f"hits={r['n_hits']}/{r['n_answer_tokens']} {r['hits']}")
        elapsed = time.time() - t0

        ranks = [r["rank"] for r in per_needle]
        acc1 = sum(1 for r in ranks if r == 1)
        total_hits = sum(r["n_hits"] for r in per_needle)
        total_possible = sum(r["n_answer_tokens"] for r in per_needle)
        summary = {
            "variant": vname,
            "n_needles": len(per_needle),
            "mean_rank": sum(ranks) / len(ranks),
            "acc_at_1": acc1 / len(ranks),
            "answer_recall": total_hits / total_possible,
            "total_hits": total_hits,
            "total_possible": total_possible,
            "elapsed_s": elapsed,
        }
        print(f"\n  summary: mean_rank={summary['mean_rank']:.2f} "
              f"acc@1={summary['acc_at_1']:.2f} "
              f"recall={summary['answer_recall']:.2f} "
              f"({total_hits}/{total_possible}) in {elapsed:.0f}s")

        all_results[vname] = {"summary": summary, "per_needle": per_needle}
        with open(out_dir / f"{vname}.json", "w") as f:
            json.dump(all_results[vname], f, indent=2)

    combined_path = out_dir / "combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    print(f"{'variant':<22} {'mean_rank':>10} {'acc@1':>8} {'recall':>10}")
    for vname, res in all_results.items():
        s = res["summary"]
        print(f"{vname:<22} {s['mean_rank']:>10.2f} {s['acc_at_1']:>8.2f} "
              f"{s['answer_recall']:>10.2f}")
    print(f"\nSaved combined JSON to {combined_path}")


if __name__ == "__main__":
    main()
