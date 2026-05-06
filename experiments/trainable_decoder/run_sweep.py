"""Trainable-decoder capacity sweep on Dickens-50.

Sequential training of N=50 single-passage adapters with a shared
trainable decoder (lm_head). Snapshots taken at N ∈ {1, 2, 4, 8, 16, 32, 50}.
At each snapshot, every adapter trained so far is evaluated under the
current decoder state to measure forgetting.

This single linear sweep is equivalent to running 6 independent sweeps
from scratch (because the seed-42 passage order is preserved as a prefix
across all sweep points) but ~10x cheaper compute.

Output:
  results/sweep_results.csv          per-(snapshot_N, adapter_position) retrieval
  results/sweep_summary.json         aggregates per snapshot
  decoder_states/decoder_N{NN}.pt    decoder snapshot at each sweep point
  adapters/adapter_pos{NN}.pt        adapter saved after its training step
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/trainable_decoder"
DICKENS = REPO / "experiments/per_passage_dickens"
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase16_combined import HIGH_LR, BASE_LR
from experiments.identity_ae.phase22_engram_key import reset_lora_to_zero
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)
from experiments.trainable_decoder.decoder import (
    install_switching_lm_head, get_decoder_state,
    trainable_decoder_params,
)


RANK = 128
ALPHA = RANK * 2
N_STEPS_PER_ADAPTER = 150
GEN_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 50
MAX_CTX_POS = 512
SNAPSHOTS = [1, 2, 4, 8, 16, 32, 50]
PASSAGE_SEED = 42


def select_passage_order(library, n=50, seed=PASSAGE_SEED):
    """Sample n distinct passage indices in a fixed order (seed=42)."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(library)), n)
    return indices


def build_sources(passage_entry, tokenizer, device):
    """passage text + (paraphrase + answer) strings."""
    sources = []
    p_ids = tokenizer.encode(passage_entry["passage"], add_special_tokens=False)
    sources.append(torch.tensor(p_ids, dtype=torch.long)[:512]
                    .unsqueeze(0).to(device))
    for p in passage_entry["paraphrases_train"]:
        ids = tokenizer.encode(f"{p}{passage_entry['answer']}",
                                 add_special_tokens=False)
        sources.append(torch.tensor(ids, dtype=torch.long)[:512]
                        .unsqueeze(0).to(device))
    return sources


def train_one_adapter(model, sources, n_steps=N_STEPS_PER_ADAPTER,
                       high_lr=HIGH_LR, base_lr=BASE_LR):
    """Train both the LoRA params and the trainable decoder for n_steps.

    The trainable decoder is shared across all adapters; its state at
    the start of this call is whatever the previous adapter's training
    left it at.
    """
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=high_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=n_steps // 2, gamma=base_lr / high_lr,
    )
    model.train()
    losses = []
    for _ in range(n_steps):
        ids_t = sources[random.randint(0, len(sources) - 1)]
        if ids_t.shape[1] < 2:
            continue
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]),
            ids_t[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))
    model.eval()
    return sum(losses[-30:]) / 30


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed,
              temperature=TEMPERATURE, top_k=TOP_K):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -MAX_CTX_POS:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :].float() / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, generation):
    if answer.lower() in generation.lower():
        return True
    clean_a = answer.replace(",", "").replace(" ", "").lower()
    clean_g = generation.replace(",", "").replace(" ", "").lower()
    if clean_a and clean_a in clean_g:
        return True
    return False


def evaluate_adapter(model, tokenizer, entry, device):
    """Run all 3 held-out paraphrases × 3 seeds for one adapter.
    Returns mean retrieval and a per-fact-type record."""
    hits = []
    for probe in entry["paraphrases_held_out"]:
        ids = tokenizer.encode(probe, add_special_tokens=False)
        ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        for seed in [0, 1, 2]:
            gen_seed = seed * 100000 + entry["id"] * 100
            gen_ids = generate(model, ids_t, GEN_TOKENS, gen_seed=gen_seed)
            full = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            cont = full[len(probe):]
            hits.append(check_match(entry["answer"], cont))
    return sum(hits) / len(hits), entry["fact_type"]


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    library = json.loads((DICKENS / "data/library.json").read_text())
    by_id = {e["id"]: e for e in library}

    # Model + Dickens checkpoint
    model, cfg = load_model(device)
    dickens_ck = torch.load(
        DICKENS / "results/v22_dickens_base.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)

    # Install switching lm_head BEFORE applying LoRA (so apply_lora's
    # freeze pass sees both heads as non-LoRA and freezes them; we'll
    # then unfreeze trainable.)
    sw = install_switching_lm_head(model)
    print(f"Installed SwitchingLMHead. d_model={sw.frozen.in_features} "
          f"vocab={sw.frozen.out_features}")

    # Apply LoRA (freezes everything except lora_*)
    n_lora = apply_lora(model, rank=RANK, alpha=ALPHA, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    print(f"LoRA params per adapter: {n_lora:,}")

    # Unfreeze the trainable decoder (apply_lora froze it)
    for p in sw.trainable.parameters():
        p.requires_grad = True
    n_dec = sum(p.numel() for p in sw.trainable.parameters())
    print(f"Trainable decoder params: {n_dec:,}")
    sw.use_trainable = True   # activate trainable path for training/eval

    # Passage order
    order = select_passage_order(library, n=50, seed=PASSAGE_SEED)
    print(f"Passage order (seed {PASSAGE_SEED}): {order[:10]}... ({len(order)} total)")

    # Sweep: linear, with snapshots
    saved_adapters = []  # (passage_id, fact_type, lora_state_dict_cpu)
    sweep_records = []   # rows: snapshot_N, position, passage_id, retrieval, fact_type
    snapshot_summary = []
    final_losses = []

    t_total = time.time()
    for position, pid in enumerate(order):
        entry = by_id[pid]
        sources = build_sources(entry, tokenizer, device)

        # Reset LoRA params for the new adapter (fresh A noise, B = 0)
        reset_lora_to_zero(model)

        t_train = time.time()
        loss = train_one_adapter(model, sources)
        final_losses.append({"position": position, "passage_id": pid,
                              "final_loss": loss,
                              "train_time_s": time.time() - t_train})
        # Save adapter state
        sd_cpu = {k: v.detach().cpu().clone()
                  for k, v in get_lora_state_dict(model).items()}
        torch.save({
            "passage_id": pid, "position": position,
            "fact_type": entry["fact_type"],
            "lora_state_dict": sd_cpu,
        }, EXP / f"adapters/adapter_pos{position:02d}.pt")
        saved_adapters.append({"pid": pid, "fact_type": entry["fact_type"],
                                "sd": sd_cpu})

        # Snapshot?
        N = position + 1
        if N in SNAPSHOTS:
            print(f"\n--- snapshot N={N}  (after training {N} adapters)  "
                  f"elapsed={time.time()-t_total:.0f}s ---")
            # Save decoder snapshot
            torch.save({
                "snapshot_N": N,
                "decoder_weight": get_decoder_state(model),
            }, EXP / f"decoder_states/decoder_N{N:02d}.pt")

            # Eval every adapter trained so far under current decoder state
            t_eval = time.time()
            per_pos_results = []
            per_type = defaultdict(list)
            for j, ad in enumerate(saved_adapters):
                load_lora_state_dict(model, ad["sd"])
                ret, ft = evaluate_adapter(model, tokenizer, by_id[ad["pid"]], device)
                per_pos_results.append({
                    "snapshot_N": N, "position": j, "passage_id": ad["pid"],
                    "fact_type": ft, "retrieval": ret,
                })
                sweep_records.append(per_pos_results[-1])
                per_type[ft].append(ret)

            # Aggregate for this snapshot
            rets = [r["retrieval"] for r in per_pos_results]
            ad1 = per_pos_results[0]["retrieval"] if per_pos_results else None
            adN = per_pos_results[-1]["retrieval"] if per_pos_results else None
            mean_ret = sum(rets) / len(rets)
            std_ret = float(np.std(rets))
            min_ret = min(rets); max_ret = max(rets)
            snap = {
                "N": N,
                "mean_retrieval": mean_ret,
                "std_retrieval": std_ret,
                "min_retrieval": min_ret,
                "max_retrieval": max_ret,
                "adapter_1_retrieval": ad1,
                "adapter_N_retrieval": adN,
                "per_fact_type": {ft: sum(v)/len(v) for ft, v in per_type.items()},
                "per_fact_type_n": {ft: len(v) for ft, v in per_type.items()},
                "eval_wall_s": time.time() - t_eval,
            }
            snapshot_summary.append(snap)
            print(f"  N={N:2d}  mean={mean_ret:.3f}  std={std_ret:.3f}  "
                  f"min={min_ret:.3f}  max={max_ret:.3f}  "
                  f"adapter_1={ad1:.3f}  adapter_N={adN:.3f}  "
                  f"eval_wall={snap['eval_wall_s']:.0f}s")
            for ft in ["entity", "numeric", "place", "relation"]:
                if ft in snap["per_fact_type"]:
                    print(f"    {ft:>10s}: {snap['per_fact_type'][ft]:.3f} "
                          f"(n={snap['per_fact_type_n'][ft]})")

        # Concise progress log (every adapter)
        if N <= 10 or N % 5 == 0:
            print(f"  trained pos={position:2d} pid={pid:2d} "
                  f"({entry['fact_type']:>8s})  loss={loss:.3f}")

    # Save outputs
    with (EXP / "results/sweep_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snapshot_N", "position", "passage_id", "fact_type",
                     "retrieval"])
        for r in sweep_records:
            w.writerow([r["snapshot_N"], r["position"], r["passage_id"],
                         r["fact_type"], f"{r['retrieval']:.3f}"])
    summary = {
        "snapshots": snapshot_summary,
        "passage_order": order,
        "final_losses": final_losses,
        "config": {
            "rank": RANK, "alpha": ALPHA,
            "n_steps_per_adapter": N_STEPS_PER_ADAPTER,
            "snapshots": SNAPSHOTS,
            "passage_seed": PASSAGE_SEED,
        },
        "wall_total_s": time.time() - t_total,
    }
    (EXP / "results/sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # Print final table
    print(f"\n{'='*82}")
    print(f"  SWEEP SUMMARY — wall total {time.time()-t_total:.0f}s")
    print(f"{'='*82}")
    print(f"  {'N':>3s}  {'mean':>6s}  {'std':>6s}  {'min':>6s}  {'max':>6s}  "
          f"{'adp_1':>6s}  {'adp_N':>6s}")
    for s in snapshot_summary:
        print(f"  {s['N']:3d}  {s['mean_retrieval']:6.3f}  "
              f"{s['std_retrieval']:6.3f}  {s['min_retrieval']:6.3f}  "
              f"{s['max_retrieval']:6.3f}  {s['adapter_1_retrieval']:6.3f}  "
              f"{s['adapter_N_retrieval']:6.3f}")
    print(f"\nSaved {EXP/'results/sweep_summary.json'}")


if __name__ == "__main__":
    main()
