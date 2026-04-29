"""Train baseline + selectivity adapters.

Baseline (Phase 47 protocol): positive examples only, next-token CE.
Selectivity: positive CE + λ * KL(adapter, base) on negative examples.

KL is computed per-position on probe-only input (no answer token), where
"base" means the adapter weights zeroed out. Base logits are precomputed
once at the start of each adapter training (depends only on input tokens
and frozen base, not on the LoRA being trained).

We sweep λ ∈ {0.5, 1.0, 2.0} for each domain (A, B). Plus 2 baselines.
Total: 8 adapters trained.
"""
from __future__ import annotations

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
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)
from experiments.selectivity.data import domain_data

PPD = REPO / "experiments/per_passage_dickens"
RANK = 128
ALPHA = RANK * 2
N_STEPS = 200  # slightly bumped from 150 because dual objective takes longer to converge
CTX = 512


def encode(tokenizer, text, device, max_len=64):
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


def build_pos_sources(examples, tokenizer, device):
    """Each positive: tokenize 'probe + answer' as one sequence."""
    out = []
    for ex in examples:
        full = f"{ex['probe']} {ex['answer']}"
        ids = tokenizer.encode(full, add_special_tokens=False)[:CTX]
        if len(ids) >= 2:
            out.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return out


def build_neg_sources(examples, tokenizer, device):
    """Each negative: tokenize probe only (no answer)."""
    out = []
    for ex in examples:
        ids = tokenizer.encode(ex["probe"], add_special_tokens=False)[:CTX]
        if len(ids) >= 2:
            out.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return out


@torch.no_grad()
def precompute_base_logits(model, neg_sources):
    """Run model with LoRA zeroed on each neg source; return list of logits.

    Saves and restores the current LoRA state so this is safe to call mid-
    training (though we only call it once at the beginning).
    """
    saved = {n: p.detach().clone()
             for n, p in model.named_parameters() if "lora_" in n}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data.zero_()
    out = []
    for ids in neg_sources:
        out.append(model(ids[:, :-1], step=0).logits.detach())
    cur = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in saved.items():
            cur[n].data.copy_(v)
    return out


def train_dual(model, pos_sources, neg_sources, n_steps, high_lr, base_lr,
               lambda_=0.0, seed=0, log_every=50):
    """If lambda_=0, this is the Phase 47 baseline (positive only).
    If lambda_>0, dual objective: pos CE + lambda * KL(adapter, base) on negatives.

    Base logits are precomputed once at step 0 (after freezing LoRA target=0).
    """
    rng = random.Random(seed)
    params = [p for n, p in model.named_parameters()
              if "lora_" in n and p.requires_grad]
    opt = torch.optim.Adam(params, lr=high_lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, n_steps // 2), gamma=base_lr / high_lr,
    )
    base_logits = []
    if lambda_ > 0 and len(neg_sources) > 0:
        # Precompute base logits BEFORE LoRA contains anything.
        # Adapter starts at LoRA=zeros (reset_lora_to_zero called by caller),
        # so the model now equals the base; precompute returns this.
        base_logits = precompute_base_logits(model, neg_sources)

    history = []
    model.train()
    for step in range(n_steps):
        # Positive step
        pos_ids = pos_sources[rng.randint(0, len(pos_sources) - 1)]
        if pos_ids.shape[1] < 2:
            continue
        out = model(pos_ids[:, :-1], step=0)
        pos_logits = out.logits
        pos_loss = F.cross_entropy(pos_logits.reshape(-1, pos_logits.shape[-1]),
                                    pos_ids[:, 1:].reshape(-1))
        loss = pos_loss

        kl_val = 0.0
        if lambda_ > 0 and len(neg_sources) > 0:
            j = rng.randint(0, len(neg_sources) - 1)
            neg_ids = neg_sources[j]
            base_log = base_logits[j]
            neg_logits = model(neg_ids[:, :-1], step=0).logits
            log_p = F.log_softmax(neg_logits, dim=-1)
            p_base = F.softmax(base_log, dim=-1)
            kl = F.kl_div(log_p, p_base, reduction="batchmean")
            loss = loss + lambda_ * kl
            kl_val = float(kl.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if (step + 1) % log_every == 0 or step == 0:
            history.append({
                "step": step + 1,
                "pos_loss": float(pos_loss.item()),
                "kl": kl_val,
                "total": float(loss.item()),
            })
    model.eval()
    return history


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = domain_data()

    print("Loading V22-Dickens base ...")
    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)

    pos_A = build_pos_sources(data["A"]["train"], tokenizer, device)
    pos_B = build_pos_sources(data["B"]["train"], tokenizer, device)
    # Negatives: probe-only sequences from the OTHER domain.
    neg_A = build_neg_sources(data["B"]["train"], tokenizer, device)  # B's probes for A's negatives
    neg_B = build_neg_sources(data["A"]["train"], tokenizer, device)
    print(f"pos_A: {len(pos_A)}  pos_B: {len(pos_B)}")
    print(f"neg_A (B probes): {len(neg_A)}  neg_B (A probes): {len(neg_B)}")

    out_dir = REPO / "experiments/selectivity/adapters"
    out_dir.mkdir(parents=True, exist_ok=True)

    LAMBDAS = [0.05, 0.1, 0.5, 1.0, 2.0]

    configs = [
        ("baseline_A", "A", pos_A, [], 0.0),
        ("baseline_B", "B", pos_B, [], 0.0),
    ]
    for lam in LAMBDAS:
        configs.append((f"sel_A_lam{lam:.1f}", "A", pos_A, neg_A, lam))
        configs.append((f"sel_B_lam{lam:.1f}", "B", pos_B, neg_B, lam))

    train_log = {}
    t_total = time.time()

    for name, dom, pos_src, neg_src, lam in configs:
        print(f"\n=== Training {name} (domain {dom}, λ={lam}) ===")
        reset_lora_to_zero(model)
        t0 = time.time()
        history = train_dual(model, pos_src, neg_src, N_STEPS,
                              HIGH_LR, BASE_LR, lambda_=lam, seed=0)
        wall = time.time() - t0
        sd = {k: v.detach().cpu().clone()
              for k, v in get_lora_state_dict(model).items()}
        torch.save(sd, out_dir / f"{name}.pt")
        # Print loss curve
        for h in history[-3:]:
            print(f"  step {h['step']}: pos_loss={h['pos_loss']:.3f} "
                  f"kl={h['kl']:.4f} total={h['total']:.3f}")
        print(f"  wall: {wall:.0f}s")
        train_log[name] = {"domain": dom, "lambda": lam, "history": history,
                           "wall_s": wall}

    out_path = REPO / "experiments/selectivity/results/train_log.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(train_log, indent=2))
    print(f"\nTotal training wall: {time.time()-t_total:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
