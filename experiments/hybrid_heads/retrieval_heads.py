"""Retrieval-head detection per Wu et al. 2024.

For each attention head in a trained model, measure the strength of "copy-paste"
attention pattern on NIAH-style prompts. The detection metric: when the model
processes "...The magic number is XX...The magic number is", the attention from
the final query position to the needle's "XX" position. Heads that strongly
attend to the needle (large attention weight, "copy" pattern) are retrieval
heads.

For HybridAttention, only the FULL-attention heads can in principle act as
retrieval heads since the compressed heads have averaged-away K/V at the
needle position. We measure both anyway and report the contrast.

Output: per-(layer, head) attention-from-query-to-needle weight, averaged
over a small set of NIAH probes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/hybrid_heads"
sys.path.insert(0, str(REPO))

from experiments.hybrid_heads.model import (
    HybridConfig, HybridTransformer, HybridAttention,
)
from experiments.hybrid_heads.eval_niah import (
    load_val_tokens, make_haystack, QUERY, NEEDLES,
)


@torch.no_grad()
def measure_retrieval_attention(model, val_tokens, tokenizer, ctx, n_probes=8,
                                  device=None):
    """For each layer × head, measure the maximum attention weight from the last
    query position to the position of the needle answer token.

    Returns: dict mapping (layer_idx, head_idx, kind) -> avg attention weight
             where kind ∈ {'full', 'compressed'}.
    """
    cfg = model.cfg
    n_full = cfg.n_full_heads
    n_comp = cfg.n_heads - n_full

    # Tokenize once
    query_ids = tokenizer.encode(QUERY)
    answer_id_for_token = {}
    for n in NEEDLES:
        ids = tokenizer.encode(n)
        if len(ids) == 1:
            answer_id_for_token[n] = ids[0]

    rng = torch.Generator(); rng.manual_seed(0)
    fill_starts = torch.randint(0, max(1, len(val_tokens) - ctx),
                                  (n_probes,), generator=rng).tolist()

    # We need to capture attention weights. SDPA doesn't return them natively.
    # Walk into the attention module and reproduce the attention with
    # explicit softmax instead of SDPA, just for these probes.
    attention_weights_per_head = {}  # (layer, head) -> list of weights

    def hook_attn(layer_idx):
        def hook(module: HybridAttention, inputs, output):
            x = inputs[0]   # (B, T, D)
            B, T, D = x.shape
            qkv = module.qkv(x).reshape(B, T, 3, module.n_heads, module.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
            head_dim = module.head_dim

            # Full heads: standard attention; we want attention[B=0, h, T-1, :]
            if module.n_full > 0:
                q_f = q[:, :module.n_full]
                k_f = k[:, :module.n_full]
                # attention scores at the LAST query position
                scores_last = torch.einsum("bhd,bhtd->bht",
                                              q_f[:, :, -1, :],
                                              k_f) / (head_dim ** 0.5)
                # Causal mask: last position can attend to all earlier
                # softmax over T
                attn_last = F.softmax(scores_last, dim=-1)   # (B, n_full, T)
                for h in range(module.n_full):
                    attention_weights_per_head.setdefault(
                        (layer_idx, h, "full"), []
                    ).append(attn_last[0, h].detach().cpu())

            # Compressed heads: attention scores at last query position over
            # T_kv = T/16 compressed positions
            if module.n_comp > 0:
                q_c = q[:, module.n_full:]
                k_full = k[:, module.n_full:]
                kc_in = k_full.reshape(B * module.n_comp, T, head_dim)
                k_c = module.compress_k(kc_in).reshape(
                    B, module.n_comp, T // module.compression_ratio, head_dim
                )
                scores_last = torch.einsum("bhd,bhtd->bht",
                                              q_c[:, :, -1, :],
                                              k_c) / (head_dim ** 0.5)
                attn_last = F.softmax(scores_last, dim=-1)   # (B, n_comp, T_kv)
                for h in range(module.n_comp):
                    attention_weights_per_head.setdefault(
                        (layer_idx, module.n_full + h, "compressed"), []
                    ).append(attn_last[0, h].detach().cpu())
        return hook

    handles = []
    for li, block in enumerate(model.blocks):
        h = block.attn.register_forward_hook(hook_attn(li))
        handles.append(h)

    needle_locations = []
    for fs in fill_starts:
        for n_str, ans_id in answer_id_for_token.items():
            needle_text = f"{QUERY}{n_str}. "
            needle_ids = tokenizer.encode(needle_text)
            depth = 0.5
            ids = make_haystack(val_tokens, fs, ctx, needle_ids,
                                  query_ids, depth)
            # Find the position of ans_id within the input (the needle's answer token)
            ids_list = ids[0].tolist()
            try:
                # The answer is at the last occurrence of ans_id BEFORE the
                # final query (which doesn't include the answer).
                # Scan forward looking for the needle pattern.
                ans_pos = ids_list.index(ans_id)
            except ValueError:
                continue
            needle_locations.append(ans_pos)
            ids = ids.to(next(model.parameters()).device)
            _ = model(ids)
            # Last query position is T - 1; we'll process attention weights below

    for h in handles:
        h.remove()

    # Aggregate: for each (layer, head, kind), average attention weight at the
    # needle's position across all probes.
    n_locations = len(needle_locations)
    if n_locations == 0:
        return {}, n_locations
    summary = {}
    for (li, hi, kind), weights_list in attention_weights_per_head.items():
        # weights_list[i] is (T,) for full or (T_kv,) for compressed
        # Pair with needle_locations[i].
        T_kv = weights_list[0].shape[0]
        ratio = (model.cfg.ctx_len // T_kv) if T_kv != model.cfg.ctx_len else 1
        ws = []
        for w, loc in zip(weights_list, needle_locations):
            if kind == "full":
                # Direct lookup at original position
                if loc < w.shape[0]:
                    ws.append(float(w[loc].item()))
            else:
                # Compressed: needle's compressed bucket
                cloc = loc // ratio
                if cloc < w.shape[0]:
                    ws.append(float(w[cloc].item()))
        if ws:
            summary[(li, hi, kind)] = {
                "mean_attention_to_needle": sum(ws) / len(ws),
                "max_attention_to_needle": max(ws),
                "n_probes": len(ws),
            }
    return summary, n_locations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["ts", "wt103"])
    ap.add_argument("--arch", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ck_path = EXP / f"checkpoints/{args.corpus}_{args.arch}_seed{args.seed}.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    cfg = HybridConfig(**{k: v for k, v in cfg_dict.items()
                          if k in HybridConfig.__dataclass_fields__})
    model = HybridTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    val_tokens = load_val_tokens(args.corpus)
    summary, n_loc = measure_retrieval_attention(
        model, val_tokens, tokenizer, args.ctx, n_probes=8, device=device,
    )
    print(f"Probes used: {n_loc}")
    print(f"Per (layer, head, kind):")
    rows = []
    for (li, hi, kind), s in sorted(summary.items()):
        print(f"  L{li:2d} H{hi:2d} {kind:>10s}  "
              f"mean_attn={s['mean_attention_to_needle']:.4f}  "
              f"max_attn={s['max_attention_to_needle']:.4f}")
        rows.append({"layer": li, "head": hi, "kind": kind, **s})

    out = {
        "corpus": args.corpus, "arch": args.arch, "seed": args.seed,
        "n_probes": n_loc,
        "rows": rows,
    }
    out_path = EXP / f"results/retrieval_heads_{args.corpus}_{args.arch}_seed{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
