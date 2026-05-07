"""NIAH evaluation for trained hybrid models.

NIAH protocol: insert a known short sentence at uniformly distributed positions
in N validation prompts of length CTX. Model is queried with the prefix
followed by a short query token; we measure whether the model's argmax
prediction recovers the planted answer.

Per the user's note: at 5000 steps both architectures may score zero on NIAH.
We treat NIAH as a "regime check" — if all configurations fail equally, the
test cannot discriminate among them, but it does tell us whether any model
in this experiment has reached the in-context-retrieval regime.

Protocol:
  Pick N=32 random ctx-length spans from val data.
  For each, choose a relative position p in {0.1, 0.3, 0.5, 0.7, 0.9}.
  Insert the needle "The magic number is XX." at position p where XX is a
  single-token answer.
  Append the query "The magic number is" so the answer is the next token.
  Predict next token; check argmax against expected.

  Total trials = 32 × 5 needle positions × 5 answers = 800 per model.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/hybrid_heads"
sys.path.insert(0, str(REPO))

from experiments.hybrid_heads.model import HybridConfig, HybridTransformer


WT103_CACHE = REPO / "experiments/hrs_loop/cache/wt103_seqlen512_ncat50.pt"
TS_VAL = REPO / "experiments/router_lora_phased/data/shakespeare_val.pt"
TS_TRAIN = REPO / "experiments/router_lora_phased/data/shakespeare_train.pt"

DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_PROMPTS = 32

QUERY = " The magic number is"
NEEDLES = [" 7", " 13", " 42", " 99", " 256"]


def load_val_tokens(corpus):
    if corpus == "ts":
        # TS val is small; pad with train if needed
        v = torch.load(TS_VAL, weights_only=False)
        t = torch.load(TS_TRAIN, weights_only=False)
        # Concatenate so we have plenty for ctx-length sampling
        return torch.cat([v, t[:200_000]])
    if corpus == "wt103":
        c = torch.load(WT103_CACHE, weights_only=False)
        return c["splits"]["validation"].tokens
    raise ValueError(corpus)


def load_model(ck_path, device):
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg_dict = ck["config"]
    cfg = HybridConfig(**{k: v for k, v in cfg_dict.items()
                          if k in HybridConfig.__dataclass_fields__})
    model = HybridTransformer(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, cfg


def make_haystack(val_tokens, fill_start, total_len, needle_ids,
                   query_ids, depth):
    K = len(query_ids)
    body_len = total_len - K
    insert_pos = max(0, min(body_len - len(needle_ids),
                              int(round(depth * body_len))))
    fill_size = body_len - len(needle_ids)
    fill = val_tokens[fill_start: fill_start + fill_size]
    if len(fill) < fill_size:
        fill = torch.cat([fill, val_tokens[:fill_size - len(fill)]])
    fill = fill.tolist()
    body = fill[:insert_pos] + list(needle_ids) + fill[insert_pos:]
    full = body + list(query_ids)
    return torch.tensor(full, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def predict_next(model, ids):
    logits = model(ids)
    return int(logits[0, -1, :].argmax().item())


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
    model, cfg = load_model(ck_path, device)

    val_tokens = load_val_tokens(args.corpus)
    print(f"[{args.corpus}/{args.arch}/seed{args.seed}] val tokens: "
          f"{len(val_tokens):,}")

    query_ids = tokenizer.encode(QUERY)
    answer_ids = []
    for n in NEEDLES:
        ids = tokenizer.encode(n)
        if len(ids) == 1:
            answer_ids.append((n, ids[0]))

    rng = torch.Generator(); rng.manual_seed(0)
    fill_starts = torch.randint(0, max(1, len(val_tokens) - args.ctx),
                                  (N_PROMPTS,), generator=rng).tolist()

    n_total = 0; n_hits = 0
    by_depth = defaultdict(list)
    by_answer = defaultdict(list)
    for ans_str, ans_id in answer_ids:
        needle_text = f"{QUERY}{ans_str}. "
        needle_ids = tokenizer.encode(needle_text)
        for depth in DEPTHS:
            for fs in fill_starts:
                ids = make_haystack(val_tokens, fs, args.ctx, needle_ids,
                                      query_ids, depth)
                ids = ids.to(device)
                pred = predict_next(model, ids)
                hit = pred == ans_id
                n_total += 1
                if hit:
                    n_hits += 1
                by_depth[depth].append(int(hit))
                by_answer[ans_str].append(int(hit))

    overall = n_hits / max(1, n_total)
    print(f"NIAH overall: {n_hits}/{n_total} = {overall:.3f}")

    out = {
        "corpus": args.corpus, "arch": args.arch, "seed": args.seed,
        "ctx": args.ctx,
        "n_total": n_total, "n_hits": n_hits, "accuracy": overall,
        "per_depth": {d: sum(v)/len(v) for d, v in by_depth.items()},
        "per_answer": {a: sum(v)/len(v) for a, v in by_answer.items()},
    }
    out_path = EXP / f"results/niah_{args.corpus}_{args.arch}_seed{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
