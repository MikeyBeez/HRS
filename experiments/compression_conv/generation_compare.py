"""Side-by-side qualitative generation comparison: baseline vs compressed.

Important caveat: this architecture predicts every COMPRESSION_RATIO-th
token (e.g., every 16th), not arbitrary next tokens. There is no natural
"generate continuously" mode. We do two honest comparisons:

  1. Top-K next-token distributions on N held-out prompts. Each prompt
     is length T (divisible by COMPRESSION_RATIO). Both models predict
     the very next token (position T). Print top-5 from each + ground
     truth.

  2. Coarsened greedy continuation. From a length-T prefix:
       a. Predict next token at position T (one of the model's
          aligned predictions).
       b. Append the ground-truth tokens [T..T+CR-1] from val data so
          the prefix grows to T+CR (still aligned).
       c. Repeat for K cycles, recording each cycle's predicted token
          and the ground-truth token at that position.
     This compares what each model "anchors on" given identical
     unfolding context. Not free-form generation, but a fair
     side-by-side at the architecture's natural cadence.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/compression_conv"
sys.path.insert(0, str(REPO))

from experiments.compression_conv.model import (
    CompressedConfig, CompressedTransformer,
)
from experiments.compression_conv.train import COMPRESSION_RATIO


def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    known = {f for f in CompressedConfig.__dataclass_fields__}
    cfg = CompressedConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    model = CompressedTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def top_k_predict(model, ids, k=5):
    """Return top-k (token_id, prob) from the last predict-able position."""
    logits = model(ids)
    next_logits = logits[0, -1, :].float()
    probs = torch.softmax(next_logits, dim=-1)
    vals, idxs = torch.topk(probs, k)
    return [(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)]


@torch.no_grad()
def predict_argmax(model, ids):
    logits = model(ids)
    return int(logits[0, -1, :].argmax().item())


def fmt_topk(topk, tokenizer):
    parts = []
    for tid, p in topk:
        s = tokenizer.decode([tid])
        s_repr = repr(s)[1:-1]   # strip outer quotes
        parts.append(f"  {p:.3f} {tid:>6d}  {s_repr!r}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ck", required=True)
    ap.add_argument("--compressed-ck", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--continuation-cycles", type=int, default=8)
    ap.add_argument("--out", default=str(EXP / "results/generation_compare.json"))
    ap.add_argument("--out-md", default=str(EXP / "results/generation_compare.md"))
    args = ap.parse_args()

    assert args.ctx % COMPRESSION_RATIO == 0

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    c = torch.load(REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt",
                     weights_only=False)
    val_tokens = c["splits"]["validation"].tokens
    print(f"Val tokens: {len(val_tokens):,}")

    baseline = load_model(args.baseline_ck, device)
    compressed = load_model(args.compressed_ck, device)

    # Initial prefix length is shorter so cycles can extend without
    # exceeding ctx_train (pos_emb size). After K cycles, prefix length =
    # initial_prefix_len + K * CR, capped at args.ctx.
    initial_prefix_len = args.ctx - args.continuation_cycles * COMPRESSION_RATIO
    assert initial_prefix_len > 0 and initial_prefix_len % COMPRESSION_RATIO == 0, \
        f"initial_prefix_len {initial_prefix_len} invalid"
    print(f"Initial prefix length: {initial_prefix_len} (ctx={args.ctx}, "
          f"cycles={args.continuation_cycles})")

    # Fixed seed for reproducibility of which prompts we sample
    rng = random.Random(0)
    prompt_starts = sorted(rng.sample(
        range(len(val_tokens) - args.ctx),
        args.n_prompts,
    ))

    md_lines = ["# Generation comparison — baseline vs compressed (16x)", ""]
    md_lines.append(f"Architecture caveat: this model predicts every "
                     f"{COMPRESSION_RATIO}th token, not arbitrary next tokens. "
                     f"Side-by-side comparisons below are at the model's natural cadence.")
    md_lines.append("")

    json_out = {"prompts": []}

    for prompt_idx, start in enumerate(prompt_starts):
        prefix = val_tokens[start:start + initial_prefix_len].tolist()
        prefix_ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)

        # ------ Top-K at the natural next position ------
        gt_next = int(val_tokens[start + initial_prefix_len].item())
        gt_next_str = tokenizer.decode([gt_next])

        b_top = top_k_predict(baseline, prefix_ids, k=5)
        c_top = top_k_predict(compressed, prefix_ids, k=5)

        # ------ Coarsened continuation ------
        # At each cycle k = 0 .. K-1: prefix has length args.ctx + k * CR.
        # Both models predict token at position args.ctx + k * CR.
        # Ground truth for that position is val_tokens[start + args.ctx + k*CR].
        # Then we append val_tokens[start + args.ctx + k*CR : start + args.ctx + (k+1)*CR]
        # to grow prefix to args.ctx + (k+1) * CR (still aligned).
        baseline_anchors = []
        compressed_anchors = []
        gt_anchors = []
        cycle_prefix = prefix_ids.clone()
        for k in range(args.continuation_cycles):
            target_pos_in_val = start + initial_prefix_len + k * COMPRESSION_RATIO
            gt_tok = int(val_tokens[target_pos_in_val].item())
            b_pred = predict_argmax(baseline, cycle_prefix)
            c_pred = predict_argmax(compressed, cycle_prefix)
            gt_anchors.append((gt_tok, tokenizer.decode([gt_tok])))
            baseline_anchors.append((b_pred, tokenizer.decode([b_pred]),
                                     b_pred == gt_tok))
            compressed_anchors.append((c_pred, tokenizer.decode([c_pred]),
                                       c_pred == gt_tok))
            # Extend prefix by next CR ground-truth tokens
            chunk = val_tokens[target_pos_in_val:
                               target_pos_in_val + COMPRESSION_RATIO]
            chunk = chunk.to(device).unsqueeze(0)
            cycle_prefix = torch.cat([cycle_prefix, chunk], dim=1)

        # Markdown writeup
        md_lines.append(f"## Prompt {prompt_idx} (val position {start}, ctx {args.ctx})")
        md_lines.append("")
        md_lines.append("**Last 240 chars of prompt:**")
        md_lines.append("```")
        md_lines.append(tokenizer.decode(prefix[-80:]))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append(f"**Ground-truth next token:** `{gt_next}` = `{gt_next_str!r}`")
        md_lines.append("")
        md_lines.append("**Baseline top-5:**")
        md_lines.append("```")
        md_lines.append(fmt_topk(b_top, tokenizer))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("**Compressed top-5:**")
        md_lines.append("```")
        md_lines.append(fmt_topk(c_top, tokenizer))
        md_lines.append("```")
        md_lines.append("")

        md_lines.append(f"**Coarsened anchor predictions over {args.continuation_cycles} cycles** "
                         f"(each cycle predicts a token {COMPRESSION_RATIO} positions ahead, "
                         f"using ground-truth tokens to fill between):")
        md_lines.append("")
        md_lines.append("| cycle | ground truth | baseline (hit?) | compressed (hit?) |")
        md_lines.append("|---|---|---|---|")
        for k in range(args.continuation_cycles):
            gt_id, gt_str = gt_anchors[k]
            b_id, b_str, b_hit = baseline_anchors[k]
            c_id, c_str, c_hit = compressed_anchors[k]
            md_lines.append(f"| {k} | `{gt_str!r}` | `{b_str!r}` "
                             f"({'✓' if b_hit else '✗'}) | "
                             f"`{c_str!r}` ({'✓' if c_hit else '✗'}) |")
        md_lines.append("")
        b_hits = sum(1 for _, _, h in baseline_anchors if h)
        c_hits = sum(1 for _, _, h in compressed_anchors if h)
        md_lines.append(f"Baseline hits: {b_hits}/{args.continuation_cycles}  "
                         f"|  Compressed hits: {c_hits}/{args.continuation_cycles}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        json_out["prompts"].append({
            "prompt_idx": prompt_idx,
            "val_start": start,
            "ground_truth_next_id": gt_next,
            "ground_truth_next_str": gt_next_str,
            "baseline_top5": [{"id": i, "prob": p,
                               "str": tokenizer.decode([i])} for i, p in b_top],
            "compressed_top5": [{"id": i, "prob": p,
                                 "str": tokenizer.decode([i])} for i, p in c_top],
            "coarsened_continuation": [
                {"cycle": k,
                 "gt": {"id": gt_anchors[k][0], "str": gt_anchors[k][1]},
                 "baseline": {"id": baseline_anchors[k][0],
                               "str": baseline_anchors[k][1],
                               "hit": baseline_anchors[k][2]},
                 "compressed": {"id": compressed_anchors[k][0],
                                 "str": compressed_anchors[k][1],
                                 "hit": compressed_anchors[k][2]}}
                for k in range(args.continuation_cycles)
            ],
            "baseline_hits": b_hits, "compressed_hits": c_hits,
        })

    # Aggregate across all prompts
    total_b_hits = sum(p["baseline_hits"] for p in json_out["prompts"])
    total_c_hits = sum(p["compressed_hits"] for p in json_out["prompts"])
    total_cycles = args.n_prompts * args.continuation_cycles
    md_lines.insert(2, f"Aggregate coarsened-anchor accuracy across {args.n_prompts} prompts × "
                       f"{args.continuation_cycles} cycles = {total_cycles} predictions:")
    md_lines.insert(3, f"  - **baseline:** {total_b_hits}/{total_cycles} = {total_b_hits/total_cycles:.3f}")
    md_lines.insert(4, f"  - **compressed:** {total_c_hits}/{total_cycles} = {total_c_hits/total_cycles:.3f}")
    md_lines.insert(5, "")

    json_out["aggregate"] = {
        "n_prompts": args.n_prompts,
        "cycles_per_prompt": args.continuation_cycles,
        "total_predictions": total_cycles,
        "baseline_acc": total_b_hits / total_cycles,
        "compressed_acc": total_c_hits / total_cycles,
    }

    Path(args.out).write_text(json.dumps(json_out, indent=2))
    Path(args.out_md).write_text("\n".join(md_lines))
    print(f"Saved {args.out}")
    print(f"Saved {args.out_md}")
    print(f"\nCoarsened anchor accuracy:")
    print(f"  baseline:    {total_b_hits}/{total_cycles} = {total_b_hits/total_cycles:.3f}")
    print(f"  compressed:  {total_c_hits}/{total_cycles} = {total_c_hits/total_cycles:.3f}")


if __name__ == "__main__":
    main()
