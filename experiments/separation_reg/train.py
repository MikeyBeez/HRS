"""Train Procedure A (baseline) and Procedure B (with contrastive
regularizer) adapters on the 200-entry library.

Stored engram definition for THIS experiment (different from canonical
Phase 47): adapter-active L5-mean of the training paraphrases (averaged).
Adapter-active means we run forward with the trained adapter's LoRA on.
Phase 47's canonical engrams are base-model; using adapter-active is
required so the contrastive regularizer can actually shape them.

Query engrams stay base-model (L0-mean, no LoRA) — uniform across
adapters, no need to know the right adapter at routing time. W projects
L0 → L5 (now adapter-active L5).

Procedure A: standard Phase 47 LoRA training (passage + paraphrase+answer
sources, 150 steps, HIGH_LR -> BASE_LR with StepLR).

Procedure B: adds a regularizer per training step:
  reg = lambda * sum_{j < k} softplus(cos(h_k^active, h_j^stored))
where h_k^active is computed in the same forward pass we already do
(extract L5 of the sampled training source, mean-pooled).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict,
)

PPD = REPO / "experiments/per_passage_dickens"
SR = REPO / "experiments/separation_reg"

RANK = 128
ALPHA = RANK * 2
N_STEPS = 150


def hidden_at_layer_grad(model, ids_t, layer_idx):
    """Like phase22_engram_key.hidden_at_layer but allows gradients through."""
    h = model.drop(model.tok_emb(ids_t))
    for i, block in enumerate(model.blocks):
        eb = model.engram_buffer if model._engram_buffer_initialized else None
        h, _, _, _ = block(h, step=0, engram_buffer=eb)
        if i == layer_idx:
            return h
    return h


def build_sources(entry, tokenizer, device, ctx=512):
    """Phase 47 sources: passage + (paraphrase+answer) pairs."""
    sources = []
    p_ids = tokenizer.encode(entry["passage"], add_special_tokens=False)[:ctx]
    if len(p_ids) >= 2:
        sources.append(torch.tensor(p_ids, dtype=torch.long).unsqueeze(0).to(device))
    answer = entry["answer"]
    for p in entry["paraphrases_train"]:
        full = f"{p}{answer}"
        ids = tokenizer.encode(full, add_special_tokens=False)[:ctx]
        if len(ids) >= 2:
            sources.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return sources


def build_engram_inputs(entry, tokenizer, device, ctx=512):
    """For computing engrams: tokenize the training paraphrases (no answer).
    These are what we'll mean-pool over to get the engram vector."""
    out = []
    for p in entry["paraphrases_train"]:
        ids = tokenizer.encode(p, add_special_tokens=False)[:ctx]
        if len(ids) >= 2:
            out.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return out


def compute_active_engram(model, paraphrase_inputs, layer=5):
    """Compute the adapter-active L{layer}-mean engram, averaged over the
    given paraphrase inputs. Differentiable w.r.t. current LoRA params."""
    means = []
    for ids_t in paraphrase_inputs:
        h = hidden_at_layer_grad(model, ids_t, layer)
        means.append(h.mean(dim=1).squeeze(0))
    return torch.stack(means).mean(dim=0)


def train_adapter(model, sources, n_steps, high_lr, base_lr, seed,
                  *, regularizer=False,
                  reg_lambda=0.0,
                  reg_paraphrase_inputs=None,
                  prior_engrams=None,
                  reg_freq=1):
    """Train one adapter. If regularizer=True, add the contrastive term.

    prior_engrams: (k, D) tensor of FIXED prior adapter engrams.
    reg_paraphrase_inputs: list of (1, T) tensors of training paraphrases
       used to compute the new engram for the regularizer.
    """
    rng = random.Random(seed)
    params = [p for n, p in model.named_parameters()
              if "lora_" in n and p.requires_grad]
    opt = torch.optim.Adam(params, lr=high_lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, n_steps // 2), gamma=base_lr / high_lr,
    )
    history = []
    model.train()
    for step in range(n_steps):
        ids_t = sources[rng.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        ce = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                              ids_t[:, 1:].reshape(-1))
        loss = ce
        reg_val = 0.0
        if regularizer and prior_engrams is not None and prior_engrams.shape[0] > 0 and (step % reg_freq == 0):
            # Compute current adapter-active engram on a single sampled
            # paraphrase (cheaper than averaging all paraphrases each step).
            inp = reg_paraphrase_inputs[rng.randint(0, len(reg_paraphrase_inputs) - 1)]
            h5 = hidden_at_layer_grad(model, inp, 5).mean(dim=1).squeeze(0)
            # Cosine to all prior engrams
            h5n = h5 / (h5.norm() + 1e-8)
            prior_n = prior_engrams / (prior_engrams.norm(dim=-1, keepdim=True) + 1e-8)
            cos = (prior_n @ h5n)  # (k,)
            # softplus pushes high cos values down strongly, low / negative
            # cos values weakly. Sum over prior adapters.
            reg = F.softplus(cos).sum()
            loss = loss + reg_lambda * reg
            reg_val = float(reg.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0 or step == 0:
            history.append({"step": step + 1, "ce": float(ce.item()),
                            "reg": reg_val, "total": float(loss.item())})
    model.eval()
    return history


@torch.no_grad()
def compute_active_engram_eval(model, paraphrase_inputs, layer=5):
    """No-grad version for storing the final engram after training."""
    means = []
    for ids_t in paraphrase_inputs:
        h = hidden_at_layer_grad(model, ids_t, layer)
        means.append(h.mean(dim=1).squeeze(0).detach())
    return torch.stack(means).mean(dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", choices=["A", "B"], required=True)
    ap.add_argument("--lambda_", type=float, default=1.0)
    ap.add_argument("--n_max", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    library = json.loads((SR / "data/library_200.json").read_text())[:args.n_max]

    print(f"Loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    out_dir = SR / ("adapters_a" if args.procedure == "A" else "adapters_b")
    out_dir.mkdir(parents=True, exist_ok=True)

    log = []
    engrams_so_far = []  # list of (D,) tensors on GPU
    t_total = time.time()

    for k, entry in enumerate(library):
        sources = build_sources(entry, tokenizer, device)
        para_inputs = build_engram_inputs(entry, tokenizer, device)

        reset_lora_to_zero(model)
        prior_tensor = None
        if args.procedure == "B" and len(engrams_so_far) > 0:
            prior_tensor = torch.stack(engrams_so_far, dim=0).detach()

        t0 = time.time()
        hist = train_adapter(
            model, sources, n_steps=N_STEPS,
            high_lr=HIGH_LR, base_lr=BASE_LR, seed=0,
            regularizer=(args.procedure == "B"),
            reg_lambda=args.lambda_,
            reg_paraphrase_inputs=para_inputs,
            prior_engrams=prior_tensor,
            reg_freq=1,
        )
        wall = time.time() - t0

        # Compute and store the final engram (adapter-active)
        active_engram = compute_active_engram_eval(model, para_inputs, layer=5)
        engrams_so_far.append(active_engram)

        # Save adapter
        sd = {k_: v.detach().cpu().clone()
              for k_, v in get_lora_state_dict(model).items()}
        torch.save(sd, out_dir / f"adapter_{k:03d}.pt")

        log.append({
            "k": k, "entry_id": entry["id"],
            "answer": entry["answer"],
            "wall_s": wall,
            "ce_init": hist[0]["ce"], "ce_final": hist[-1]["ce"],
            "reg_init": hist[0]["reg"] if hist else 0.0,
            "reg_final": hist[-1]["reg"] if hist else 0.0,
        })
        if (k + 1) % 10 == 0 or k == 0:
            print(f"  [{k+1}/{len(library)}] proc={args.procedure} "
                  f"ce={hist[-1]['ce']:.3f} reg={hist[-1]['reg']:.3f} "
                  f"wall={wall:.1f}s  total_elapsed={time.time()-t_total:.0f}s")

    # Save engrams stack
    engrams_path = SR / f"results/engrams_{args.procedure}.pt"
    torch.save({
        "engrams": torch.stack(engrams_so_far).cpu(),
        "ids":     [e["id"] for e in library],
        "lambda":  args.lambda_,
        "procedure": args.procedure,
        "n":       len(library),
    }, engrams_path)
    print(f"\nSaved engrams to {engrams_path}")

    log_path = SR / f"results/train_log_{args.procedure}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "procedure": args.procedure, "lambda": args.lambda_,
        "log": log,
        "wall_total_s": time.time() - t_total,
    }, indent=2))
    print(f"Saved log to {log_path}")
    print(f"\nProc {args.procedure} total wall: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
