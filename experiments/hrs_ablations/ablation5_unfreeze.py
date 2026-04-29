"""Ablation 5: partial base unfreezing (last block) during adapter training.

Each adapter is trained with both its LoRA params AND the last block (block 5)
of the base model unfrozen. We save per-adapter snapshots of both.

At inference: routing uses the *original* frozen base (so engrams stay
canonical and the projection W is reusable). When we generate, we:
  1. Restore the last-block to that adapter's snapshot
  2. Load that adapter's LoRA
  3. Generate
  4. Restore last-block to the canonical state for the next routing step

We report routing_acc, retrieval_acc, and a "forgetting" probe: pick 5
adapters at random; for each, evaluate the LoRA-routed retrieval when the
WRONG adapter's last-block snapshot is loaded. Catastrophic? Or does the
LoRA still dominate?
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.hrs_ablations.util import (
    PPD, D, get_tokenizer, get_hidden, pool, held_out_queries,
    generate, check_match, GEN_TOKENS,
)
from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


RANK = 128
N_STEPS = 150
LAST_BLOCK_LR = 1e-5  # conservative — full base lr would be too aggressive


def train_adapter_with_unfreeze(model, passage, prompts_with_answers,
                                 tokenizer, device, last_block_params,
                                 n_steps=N_STEPS):
    """Train: LoRA params at HIGH_LR/BASE_LR (Phase 47 schedule) AND
    last-block params at LAST_BLOCK_LR (constant). Step-LR halve at half."""
    sources = []
    p_ids = tokenizer.encode(passage, add_special_tokens=False)
    if len(p_ids) >= 2:
        sources.append(torch.tensor(p_ids, dtype=torch.long).unsqueeze(0).to(device))
    for s in prompts_with_answers:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) >= 2:
            sources.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))

    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    opt = torch.optim.AdamW(
        [{"params": lora_params, "lr": HIGH_LR},
         {"params": last_block_params, "lr": LAST_BLOCK_LR}],
        betas=(0.9, 0.95), weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, n_steps // 2), gamma=BASE_LR / HIGH_LR,
    )
    model.train()
    import random as _r
    rng = _r.Random(0)
    for step in range(n_steps):
        ids_t = sources[rng.randint(0, len(sources)-1)]
        out = model(ids_t[:, :-1], step=0)
        logits = out.logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # clip on lora and last block separately
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        torch.nn.utils.clip_grad_norm_(last_block_params, 0.5)
        opt.step()
        sched.step()
    model.eval()


def block5_state(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.blocks[5].named_parameters()}


def restore_block5(model, sd, device):
    cur = dict(model.blocks[5].named_parameters())
    with torch.no_grad():
        for n, v in sd.items():
            cur[n].data.copy_(v.to(device))


def main():
    device = torch.device("cuda")
    tokenizer = get_tokenizer()
    library = json.loads((PPD / "data/library.json").read_text())
    keys = json.loads((PPD / "results/library_keys.json").read_text())
    queries = held_out_queries(library)

    library_l5 = torch.tensor(
        np.stack([np.array(e["l5_aggregate"]) for e in keys]),
        device=device, dtype=torch.float32,
    )
    library_l5_n = F.normalize(library_l5, dim=-1)
    proj_ck = torch.load(PPD / "results/projection_W.pt",
                         map_location=device, weights_only=False)
    W = nn.Linear(D, D, bias=False).to(device)
    W.load_state_dict(proj_ck["W_state"])
    W.eval()

    model, cfg = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=RANK, alpha=RANK * 2, target_modules=L45_TARGETS)

    # Save the canonical block-5 state to restore between adapters.
    canonical_block5 = block5_state(model)

    # Mark block 5's base params as trainable.
    for n, p in model.named_parameters():
        if n.startswith("blocks.5.") and "lora_" not in n:
            p.requires_grad = True

    last_block_params = [p for n, p in model.named_parameters()
                          if n.startswith("blocks.5.") and "lora_" not in n]
    print(f"Last-block params unfrozen: "
          f"{sum(p.numel() for p in last_block_params):,}")
    print(f"LoRA params: "
          f"{sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n):,}")

    # Train all 50 adapters with last-block unfrozen. Save (lora_sd, block5_sd).
    print("\nTraining 50 adapters with block-5 unfrozen ...")
    out_dir = REPO / "experiments/hrs_ablations/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_pkgs = {}  # id -> {"lora": sd, "block5": sd}
    t_train = time.time()
    for entry in library:
        i = entry["id"]
        # Restore canonical block5 + zero LoRA before each adapter.
        restore_block5(model, canonical_block5, device)
        reset_lora_to_zero(model)
        prompts_with_answers = [f"{p}{entry['answer']}" for p in entry["paraphrases_train"]]
        train_adapter_with_unfreeze(model, entry["passage"],
                                     prompts_with_answers, tokenizer, device,
                                     last_block_params, n_steps=N_STEPS)
        adapter_pkgs[i] = {
            "lora": {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()},
            "block5": block5_state(model),
        }
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/50] trained ({time.time()-t_train:.0f}s)")
    print(f"  Total adapter training wall: {time.time()-t_train:.0f}s")

    # Move LoRA pkgs to device for fast eval
    for i in adapter_pkgs:
        adapter_pkgs[i]["lora"] = {k: v.to(device) for k, v in adapter_pkgs[i]["lora"].items()}

    # ----- Eval: routing + retrieval, with per-adapter block5 swap -----
    print("\nEvaluating routing+retrieval (3 seeds) ...")
    t_eval = time.time()
    n = 0; n_routing = 0; n_retrieval = 0
    for seed in (0, 1, 2):
        for qi, q in enumerate(queries):
            # Routing on canonical base
            restore_block5(model, canonical_block5, device)
            reset_lora_to_zero(model)
            ids = tokenizer.encode(q["probe"], add_special_tokens=False)
            ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                l0 = pool(get_hidden(model, ids_t, "L0"), "mean")
                proj = W(l0.unsqueeze(0))
                proj_n = F.normalize(proj, dim=-1)
                sim = (proj_n @ library_l5_n.T).squeeze(0)
                routed = sim.argmax().item()
            # Generate with that adapter's full pkg
            restore_block5(model, adapter_pkgs[routed]["block5"], device)
            load_lora_state_dict(model, adapter_pkgs[routed]["lora"])
            gen = generate(model, ids_t, GEN_TOKENS, gen_seed=seed*10000+qi)
            full = tokenizer.decode(gen[0], skip_special_tokens=True)
            cont = full[len(q["probe"]):]
            n += 1
            if routed == q["adapter_id"]: n_routing += 1
            if check_match(q["answer"], cont): n_retrieval += 1
    eval_wall = time.time() - t_eval

    summary = {
        "rank": RANK,
        "n_steps": N_STEPS,
        "last_block_lr": LAST_BLOCK_LR,
        "routing_acc": n_routing / n,
        "retrieval_acc": n_retrieval / n,
        "n": n,
        "train_wall_s": time.time() - t_train - eval_wall,
        "eval_wall_s": eval_wall,
    }
    print(f"\n[unfreeze_last]  routing={summary['routing_acc']:.3f}  "
          f"retrieval={summary['retrieval_acc']:.3f}  "
          f"eval_wall={eval_wall:.0f}s")

    # Forgetting probe: load adapter A's lora but adapter B's block5 (B != A)
    # for 5 (A, B) pairs. Measure retrieval drop.
    print("\nForgetting probe: lora=A but block5=B (mismatched) ...")
    import random as _r
    _r.seed(42)
    pairs = []
    while len(pairs) < 5:
        a = _r.randrange(len(library)); b = _r.randrange(len(library))
        if a != b: pairs.append((a, b))
    forgetting_results = []
    for (a, b) in pairs:
        # Use one held-out paraphrase from adapter A
        entry_a = library[a]
        probe = entry_a["paraphrases_held_out"][0]
        answer = entry_a["answer"]
        ids = tokenizer.encode(probe, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        # Match: lora=A, block5=A
        restore_block5(model, adapter_pkgs[a]["block5"], device)
        load_lora_state_dict(model, adapter_pkgs[a]["lora"])
        gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
        match = check_match(answer, tokenizer.decode(gen[0], skip_special_tokens=True)[len(probe):])
        # Mismatched: lora=A, block5=B
        restore_block5(model, adapter_pkgs[b]["block5"], device)
        # Note: lora is still A since we didn't reset it
        gen = generate(model, ids_t, GEN_TOKENS, gen_seed=0)
        mismatch = check_match(answer, tokenizer.decode(gen[0], skip_special_tokens=True)[len(probe):])
        forgetting_results.append({"a": a, "b": b, "match_match": match,
                                    "mismatch_match": mismatch})
        print(f"  a={a} b={b}  matched={match}  mismatched(b's block5)={mismatch}")

    summary["forgetting_probe"] = forgetting_results

    out_path = out_dir / "ablation5_unfreeze.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nAblation 5 wall: {time.time()-t_eval:.0f}s eval; total in main")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
