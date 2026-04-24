"""Sharper follow-up: on the composite task, *which* positions have high
cos(e[b,i,:], m_up_T[b,:])? Are they systematically the ANS position (the one
that must predict the terminal), or scattered randomly?

If high-cosine positions concentrate at ANS or ANS-adjacent positions, the
MPAR is doing learned targeted addressing. If they're uniform across
position types, the selective alignment is structural noise, not semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.hrs_loop.loop_block import HRSLoop, HRSLoopConfig
from experiments.hrs_loop.tasks.compositional_lookup import (
    CompositionalConfig, CompositionalDataset,
    ARROW_TOK, SEP_TOK, QUERY_TOK, ANS_TOK, PAD_TOK, TERM_START, N_KEYS,
)


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def _classify_positions(tokens: torch.Tensor) -> dict:
    """For one sequence's token tensor (L,), return a dict of category -> index list."""
    cats = {
        "key": [],        # tokens 0..99
        "arrow": [],      # token 100
        "terminal": [],   # tokens 105..124 (appearing before ANS, i.e. inside pairs)
        "sep": [],        # token 101
        "query": [],      # token 102
        "ans": [],        # token 103
        "pad": [],        # token 104
        "terminal_at_ans_plus_1": [],  # the terminal right after ANS
    }
    ans_idx = None
    for i, t in enumerate(tokens.tolist()):
        if t == ANS_TOK:
            ans_idx = i
            cats["ans"].append(i)
        elif t == QUERY_TOK:
            cats["query"].append(i)
        elif t == ARROW_TOK:
            cats["arrow"].append(i)
        elif t == SEP_TOK:
            cats["sep"].append(i)
        elif t == PAD_TOK:
            cats["pad"].append(i)
        elif TERM_START <= t < TERM_START + 20:
            cats["terminal"].append(i)
        elif 0 <= t < N_KEYS:
            cats["key"].append(i)
    if ans_idx is not None and ans_idx + 1 < len(tokens):
        cats["terminal_at_ans_plus_1"] = [ans_idx + 1]
        # Remove from "terminal" list so it doesn't double-count
        cats["terminal"] = [i for i in cats["terminal"] if i != ans_idx + 1]
    return cats, ans_idx


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = CKPT_DIR / "composite_B_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = HRSLoopConfig(**ckpt["cfg"])
    assert cfg.variant == "B"
    model = HRSLoop(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Val dataset at k=4 (the hardest trained case).
    task_cfg = CompositionalConfig(**ckpt["task_cfg"])
    ds = CompositionalDataset(task_cfg, n_samples=200, k_fixed=4, seed=7)

    xs = torch.stack([ds[i][0] for i in range(200)]).to(device)
    ans_positions = torch.stack([ds[i][2] for i in range(200)]).to(device)
    B, L = xs.shape

    # Forward: Prelude → recurrent (capture m_T)
    pos = torch.arange(L, device=device)
    h = model.tok_emb(xs) + model.pos_emb(pos)[None]
    for blk in model.prelude:
        h = blk(h)
    e = h.float()                                     # (B, L, d)

    rec = model.recurrent
    m = torch.zeros(B, rec.rank_m, device=device, dtype=e.dtype)
    for t in range(4):
        h_t = e + rec.project_up(m).float()
        block_out = rec.block(h_t.to(h.dtype)).float()
        lora_out = rec.loras[t](h_t.to(h.dtype)).float()
        h_out = block_out + lora_out
        m = rec.project_down(h_out.to(h.dtype)).float()
    W_up = rec.project_up.up.weight.detach().float()
    m_up_T = m @ W_up.T                                # (B, d)

    per_pos_cos = torch.nn.functional.cosine_similarity(
        e, m_up_T.unsqueeze(1), dim=-1
    )                                                  # (B, L)

    # Classify positions per sequence.
    cat_buckets: dict[str, list[float]] = {
        "ans": [], "query": [], "key": [], "arrow": [], "sep": [],
        "pad": [], "terminal": [], "terminal_at_ans_plus_1": [],
        "ans_minus_1": [], "ans_minus_2": [], "ans_minus_3": [],
    }
    for b in range(B):
        toks = xs[b].cpu()
        cats, ans_i = _classify_positions(toks)
        for cat, idxs in cats.items():
            for i in idxs:
                cat_buckets[cat].append(per_pos_cos[b, i].item())
        if ans_i is not None:
            for offset in (1, 2, 3):
                j = ans_i - offset
                if 0 <= j < L:
                    cat_buckets[f"ans_minus_{offset}"].append(
                        per_pos_cos[b, j].item()
                    )

    # Also: top-k positions by cosine, are they concentrated at specific token types?
    # Compute fraction of positions in top-5% that fall in each category.
    flat_cos = per_pos_cos.flatten()
    top_thresh = torch.quantile(flat_cos, 0.95).item()
    top_mask = per_pos_cos >= top_thresh             # (B, L)
    top_cat_counts = {k: 0 for k in cat_buckets}
    top_total = 0
    for b in range(B):
        toks = xs[b].cpu()
        cats, ans_i = _classify_positions(toks)
        for i in range(L):
            if top_mask[b, i].item():
                top_total += 1
                # Determine which single category this position belongs to
                t = int(toks[i])
                if t == ANS_TOK: top_cat_counts["ans"] += 1
                elif t == QUERY_TOK: top_cat_counts["query"] += 1
                elif t == ARROW_TOK: top_cat_counts["arrow"] += 1
                elif t == SEP_TOK: top_cat_counts["sep"] += 1
                elif t == PAD_TOK: top_cat_counts["pad"] += 1
                elif TERM_START <= t < TERM_START + 20:
                    if ans_i is not None and i == ans_i + 1:
                        top_cat_counts["terminal_at_ans_plus_1"] += 1
                    else:
                        top_cat_counts["terminal"] += 1
                elif 0 <= t < N_KEYS:
                    top_cat_counts["key"] += 1

    print(f"Composite Variant B, k=4, n={B} sequences, seq_len={L}")
    print(f"Overall cos(e[b,i], m_up_T[b]):  "
          f"mean={flat_cos.mean().item():+.3f}  std={flat_cos.std().item():.3f}  "
          f"top-5% threshold={top_thresh:+.3f}")
    print()
    print(f"{'category':>24}  {'n':>6}  {'mean':>7}  {'std':>7}  "
          f"{'min':>7}  {'max':>7}")
    print("-" * 68)
    for cat in ["ans", "terminal_at_ans_plus_1", "ans_minus_1", "ans_minus_2",
                 "ans_minus_3", "query", "key", "arrow", "sep", "terminal",
                 "pad"]:
        v = cat_buckets[cat]
        if not v:
            continue
        t = torch.tensor(v)
        print(f"{cat:>24}  {len(v):>6d}  {t.mean().item():+7.3f}  "
              f"{t.std().item():>7.3f}  {t.min().item():+7.3f}  "
              f"{t.max().item():+7.3f}")

    print()
    print(f"Top-5% positions (cos ≥ {top_thresh:+.3f}):  total={top_total}")
    for cat, c in top_cat_counts.items():
        if c > 0:
            pct = 100.0 * c / top_total
            # baseline prevalence in the full sequence?
            baseline = len(cat_buckets[cat])
            baseline_pct = 100.0 * baseline / (B * L) if baseline else 0.0
            enrichment = pct / baseline_pct if baseline_pct > 0 else float('inf')
            print(f"  {cat:>24}:  {c:>5}  ({pct:>5.1f}% of top-5%, "
                  f"baseline prevalence {baseline_pct:.1f}%, "
                  f"enrichment ×{enrichment:.1f})")

    out = {
        "n_sequences": B,
        "seq_len": L,
        "overall": {
            "mean": flat_cos.mean().item(),
            "std": flat_cos.std().item(),
            "top5pct_threshold": top_thresh,
        },
        "by_category": {
            cat: {
                "n": len(v),
                "mean": float(torch.tensor(v).mean().item()) if v else None,
                "std": float(torch.tensor(v).std().item()) if len(v) > 1 else None,
                "min": float(torch.tensor(v).min().item()) if v else None,
                "max": float(torch.tensor(v).max().item()) if v else None,
            }
            for cat, v in cat_buckets.items()
        },
        "top5pct_category_counts": top_cat_counts,
        "top5pct_total": top_total,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stage3_address_composite_positions.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
