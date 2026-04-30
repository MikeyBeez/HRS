"""Train 10 combination + 8 single-passage LoRA adapters on Mistral-7B-v0.1.

LoRA via PEFT, rank 128, alpha 256, on attn (q_proj, v_proj) and FFN
(gate_proj, down_proj) of the last 2 transformer layers (layers 30, 31)
of Mistral. This mirrors Phase 47's L45_TARGETS choice scaled to a
larger model (Phase 47 used the last 2 of 6 layers; here last 2 of 32).

Each adapter is saved as its own LoRA state-dict pickle so we can
manually compose at inference time.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.combo_adapter.combos import COMBINATIONS, CHOSEN_IDS

PPD = REPO / "experiments/per_passage_dickens"
FWC = REPO / "experiments/four_way_compare"
ADAPTERS = FWC / "adapters_mistral"
ADAPTERS.mkdir(parents=True, exist_ok=True)

BASE = "mistralai/Mistral-7B-v0.1"
RANK = 128
ALPHA = RANK * 2
N_STEPS_PER_PASSAGE = 150
HIGH_LR = 3e-4
BASE_LR = 1e-4
LAYERS = [30, 31]  # last 2 of 32

# PEFT target_modules: list of substrings — matches q_proj, v_proj,
# gate_proj, down_proj on layers 30, 31 (Mistral has all of these named
# uniformly across layers, so we filter via the layers_to_transform arg).
TARGET_MODULES = ["q_proj", "v_proj", "gate_proj", "down_proj"]


def make_peft_model(base_model):
    cfg = LoraConfig(
        r=RANK, lora_alpha=ALPHA,
        target_modules=TARGET_MODULES,
        layers_to_transform=LAYERS,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, cfg)


def reset_lora_to_zero(peft_model):
    """Zero out LoRA B (so initial contribution = 0). PEFT inits B=0
    already, but we may have run a prior training cycle. Re-init A=randn,
    B=0 for fresh start."""
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.zero_()
            elif "lora_A" in n:
                p.normal_(std=0.02)


def get_lora_state_dict(peft_model):
    return {n: p.detach().cpu().clone()
            for n, p in peft_model.named_parameters() if "lora_" in n}


def build_sources(constituent_entries, tokenizer, device, ctx=512):
    sources = []
    for entry in constituent_entries:
        ids = tokenizer.encode(entry["passage"], add_special_tokens=False)[:ctx]
        if len(ids) >= 2:
            sources.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
        for p in entry["paraphrases_train"]:
            full = f"{p} {entry['answer']}"
            ids = tokenizer.encode(full, add_special_tokens=False)[:ctx]
            if len(ids) >= 2:
                sources.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device))
    return sources


def train_loop(model, sources, n_steps, high_lr=HIGH_LR, base_lr=BASE_LR,
               seed=0, log_every=50):
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
        out = model(ids_t[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step == 0:
            history.append({"step": step + 1, "loss": float(loss.item())})
    model.eval()
    return history


def main():
    device = torch.device("cuda")
    print(f"Loading {BASE} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16,
    ).to(device)
    print(f"  base loaded in {time.time()-t0:.0f}s")

    # Wrap with PEFT once. We'll reset LoRA between adapter trainings.
    peft_model = make_peft_model(base_model)
    n_lora = sum(p.numel() for n, p in peft_model.named_parameters()
                  if "lora_" in n and p.requires_grad)
    print(f"  LoRA trainable params: {n_lora:,}")

    library = json.loads((PPD / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    train_log = []
    t_total = time.time()

    # ----- Combo adapters -----
    print("\n=== Training combo adapters ===")
    for combo in COMBINATIONS:
        name = combo["name"]
        constituents = [by_id[i] for i in combo["constituents"]]
        sources = build_sources(constituents, tokenizer, device)
        n_steps = N_STEPS_PER_PASSAGE * combo["k"]
        print(f"  [{name}] k={combo['k']} sources={len(sources)} steps={n_steps}")

        reset_lora_to_zero(peft_model)
        t = time.time()
        hist = train_loop(peft_model, sources, n_steps, seed=0)
        wall = time.time() - t
        sd = get_lora_state_dict(peft_model)
        torch.save(sd, ADAPTERS / f"combo_{name}.pt")
        train_log.append({
            "kind": "combo", "name": name, "k": combo["k"],
            "n_sources": len(sources), "n_steps": n_steps,
            "loss_init": hist[0]["loss"], "loss_final": hist[-1]["loss"],
            "wall_s": wall,
        })
        print(f"    loss {hist[0]['loss']:.2f} -> {hist[-1]['loss']:.2f}  wall={wall:.0f}s")

    # ----- Single-passage adapters -----
    print("\n=== Training single-passage adapters ===")
    for cid in CHOSEN_IDS:
        entry = by_id[cid]
        sources = build_sources([entry], tokenizer, device)
        n_steps = N_STEPS_PER_PASSAGE
        print(f"  [single_{cid}] ans={entry['answer']!r} sources={len(sources)}")
        reset_lora_to_zero(peft_model)
        t = time.time()
        hist = train_loop(peft_model, sources, n_steps, seed=0)
        wall = time.time() - t
        sd = get_lora_state_dict(peft_model)
        torch.save(sd, ADAPTERS / f"single_{cid}.pt")
        train_log.append({
            "kind": "single", "cid": cid, "answer": entry["answer"],
            "n_sources": len(sources), "n_steps": n_steps,
            "loss_init": hist[0]["loss"], "loss_final": hist[-1]["loss"],
            "wall_s": wall,
        })
        print(f"    loss {hist[0]['loss']:.2f} -> {hist[-1]['loss']:.2f}  wall={wall:.0f}s")

    out = {"train_log": train_log, "wall_total_s": time.time() - t_total}
    out_path = FWC / "results/train_log_mistral.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTotal training wall: {time.time()-t_total:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
