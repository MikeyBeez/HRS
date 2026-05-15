"""Phase 69: Independent adapter composition via rank-dimension concatenation.

Test whether HRS adapters trained independently on different passages with
standard next-token prediction will compose cleanly when their LoRA matrices
are concatenated along the rank dimension. If so, we have a memory architecture
that scales without per-passage routing: store training data, train adapters
in parallel as needed, compose at inference time.

The motivating intuition: D2L (Charakorn et al., 2026) meta-trains its
hypernetwork-generated adapters for composability. But the orthogonality may
actually live in the data itself — different passages need different model
behaviors, so their adapters modify different regions of weight space, so
summation shouldn't produce destructive interference.

Procedure (autonomous; spec checked in at the top of the script for posterity):
  1. Select 8 passages + 1 held-out from the Phase 47/22 stratified passkey
     corpus (Phase 47/50 didn't have a Dickens corpus despite spec wording;
     the stratified corpus is the closest existing topical-diversity set,
     with 4 distinct topic types: numeric, entity, technical, fact).
  2. Train 8 independent rank-8 LoRA adapters on L45_TARGETS (attn + peer FFN
     on layers 4-5), one per passage. Save A, B matrices to disk.
  3. Verify each adapter's per-passage passkey retrieval at K=1 before
     proceeding to composition.
  4. Compose at K=1,2,4,6,8 by row-concat of A (along dim 1) and col-concat of
     B (along dim 0). Scaling is preserved per-component (effective scaling
     stays at alpha/rank per slice, so the composed forward equals the sum of
     per-adapter forwards).
  5. For each composition, evaluate retrieval (passkey hit) + perplexity
     (cross-entropy on the passage tokens) on each constituent passage AND
     on the held-out (control: should look like base).
  6. Record Frobenius norm of total weight perturbation as a function of K.

Composition installation: we don't allocate larger LoRA layers. Instead we
merge the composed delta into base_layer.weight (saving the original first),
with LoRA-B held at zero so the LoRALayer adds nothing further. After eval,
restore the saved base weights.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python \\
        experiments/identity_ae/phase69_adapter_composition.py
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.phase10_passkey import (
    check_passkey, load_model, generate_greedy,
)
from experiments.identity_ae.phase13_lora_scheduled import run_ttt_lora_scheduled
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, stratified_tests,
)
from experiments.identity_ae.lora_wrapper import (
    apply_lora, get_lora_state_dict, load_lora_state_dict,
)


# ============================================================
# Config
# ============================================================

# Two ranks are evaluated. Rank 8 is the spec primary (D2L per-chunk rank).
# Rank 128 is Phase 47's working setting; if rank 8 underperforms the
# "individual adapter ≈ 93-97% retrieval" debug gate from the spec, rank 128
# acts as a methodology control so the composition story is still legible.
RANK_PRIMARY = 8
RANK_CONTROL = 128

N_STEPS = 150       # Phase 47 setting
HIGH_LR = 3e-4
BASE_LR = 1e-4

N_PASSAGES = 8
K_VALUES = [1, 2, 4, 6, 8]


# Topical-diversity selection from the 20-passage stratified corpus
# (stratified_tests returns ids 0-4, 20-24, 30-34, 40-44 — first 5 of each type).
# Pick 2 of each topic type (numeric/entity/technical/fact), spread across
# distinct facilities, names, instruments, and protocols. Keep one held-out
# passage that is not part of any composition.
SELECTED_IDS = [0, 4, 20, 23, 30, 33, 40, 43]   # 8 training passages
HELDOUT_ID = 21                                 # entity, not in composition


# ============================================================
# Composition helpers
# ============================================================

def lora_layer_modules(model):
    """Return list of (name, LoRALayer) — every LoRA-wrapped module."""
    from experiments.identity_ae.lora_wrapper import LoRALayer
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALayer)]


def save_base_weights(model):
    """Snapshot the underlying nn.Linear weights of every LoRA-wrapped module."""
    snapshot = {}
    for name, mod in lora_layer_modules(model):
        snapshot[name] = mod.base_layer.weight.data.clone()
    return snapshot


def restore_base_weights(model, snapshot):
    for name, mod in lora_layer_modules(model):
        mod.base_layer.weight.data.copy_(snapshot[name])


def compute_per_module_delta(state_dict, scaling):
    """Given a single LoRA state_dict (lora_A/lora_B per module), return
    {layer_name: delta_W} where delta_W has shape (out_f, in_f), ready to
    add into the base nn.Linear.weight.

    LoRA forward adds (x @ A @ B) * scaling. As a weight delta on
    nn.Linear (which computes x @ W.T + b), this is W += (A @ B).T * scaling.
    """
    deltas = {}
    by_layer = {}
    for k, v in state_dict.items():
        # k looks like 'blocks.4.attn.qkv.lora_A'
        layer = k.rsplit('.', 1)[0]
        suffix = k.rsplit('.', 1)[1]
        by_layer.setdefault(layer, {})[suffix] = v
    for layer, parts in by_layer.items():
        A = parts['lora_A']
        B = parts['lora_B']
        delta = (A @ B).T * scaling
        deltas[layer] = delta
    return deltas


def sum_deltas(delta_list):
    """Sum a list of per-module-delta dicts into one delta dict."""
    if not delta_list:
        return {}
    out = {k: torch.zeros_like(v) for k, v in delta_list[0].items()}
    for d in delta_list:
        for k, v in d.items():
            out[k] = out[k] + v
    return out


def install_delta(model, delta):
    """Add delta into base_layer.weight for each LoRA-wrapped module."""
    for name, mod in lora_layer_modules(model):
        if name in delta:
            mod.base_layer.weight.data.add_(delta[name].to(mod.base_layer.weight.device,
                                                            dtype=mod.base_layer.weight.dtype))


def frobenius(delta):
    """Sum-of-squares Frobenius norm across all modules in delta dict."""
    total = 0.0
    for v in delta.values():
        total += float((v.float() ** 2).sum().item())
    return math.sqrt(total)


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def passage_ppl(model, ids_t):
    """Cross-entropy perplexity on the passage tokens."""
    out = model(ids_t[:, :-1], step=0)
    loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                            ids_t[:, 1:].reshape(-1))
    return float(torch.exp(loss).item())


def eval_passage(model, test, tokenizer, device):
    """Return dict with retrieval (passkey hit) and perplexity for one passage."""
    ids = tokenizer.encode(test["passage"], add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    ppl = passage_ppl(model, ids_t)
    gen = generate_greedy(model, test["prompt"], tokenizer, device, 50)
    hit = bool(check_passkey(gen, test["passkey"]))
    return {"ppl": ppl, "hit": hit, "gen": gen[:120]}


# ============================================================
# Main
# ============================================================

def run_rank(rank, train_tests, heldout_test, tokenizer, device, results_dir):
    """One full pipeline (train 8 adapters, K-sweep) at the specified rank.
    Returns (summary_dict, perturbation_dict, composition_results, k1_results)."""

    alpha = 2 * rank
    adapters_dir = results_dir / f"adapters_rank{rank}"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Phase 69: Independent adapter composition — RANK {rank}")
    print("=" * 60)
    print(f"Rank: {rank}  Alpha: {alpha}  Steps: {N_STEPS}  "
          f"LR: {HIGH_LR:.0e} -> {BASE_LR:.0e}")
    print(f"Targets: {L45_TARGETS}")
    print(f"Selected ids ({len(train_tests)}): {[t['id'] for t in train_tests]}  "
          f"types: {[t['type'] for t in train_tests]}")
    print(f"Held-out id: {heldout_test['id']}  type: {heldout_test['type']}")
    print()

    # Build model with LoRA structure.
    model, _ = load_model(device)
    n_lora = apply_lora(model, rank=rank, alpha=alpha, target_modules=L45_TARGETS)
    print(f"LoRA params per adapter: {n_lora:,}")

    # Snapshot pristine base weights for restoration after each eval condition.
    base_snapshot = save_base_weights(model)

    # ============================================================
    # PHASE A: Train 8 independent adapters
    # ============================================================
    print(f"\n{'=' * 60}\nPHASE A: TRAIN 8 ADAPTERS\n{'=' * 60}")
    adapter_states = []  # list of state_dicts
    t0 = time.time()
    for i, test in enumerate(train_tests):
        reset_lora_to_zero(model)
        run_ttt_lora_scheduled(model, test["passage"], tokenizer, device,
                                N_STEPS, HIGH_LR, BASE_LR)
        sd = {k: v.detach().cpu().clone() for k, v in get_lora_state_dict(model).items()}
        adapter_states.append(sd)
        torch.save({"state_dict": sd, "test": test, "rank": rank, "alpha": alpha,
                    "n_steps": N_STEPS, "high_lr": HIGH_LR, "base_lr": BASE_LR,
                    "targets": L45_TARGETS},
                    adapters_dir / f"passage_{i}.pt")
        elapsed = time.time() - t0
        print(f"  [{i + 1}/{N_PASSAGES}] id={test['id']:2d} {test['type']:9s}  "
              f"({elapsed:.0f}s)")

    # ============================================================
    # PHASE B: Verify per-adapter K=1 baseline
    # ============================================================
    print(f"\n{'=' * 60}\nPHASE B: K=1 BASELINE PER ADAPTER\n{'=' * 60}")
    k1_results = []
    scaling = alpha / rank
    for i, test in enumerate(train_tests):
        restore_base_weights(model, base_snapshot)
        reset_lora_to_zero(model)  # zero out lora_B so wrapper adds nothing
        delta = compute_per_module_delta(adapter_states[i], scaling)
        install_delta(model, delta)
        res = eval_passage(model, test, tokenizer, device)
        k1_results.append({"passage_idx": i, "id": test["id"], "type": test["type"],
                           **res})
        print(f"  [{i}] id={test['id']:2d} {test['type']:9s}  "
              f"ppl={res['ppl']:7.2f}  hit={int(res['hit'])}  "
              f"gen={res['gen'][:60]!r}")

    restore_base_weights(model, base_snapshot)

    n_hit_k1 = sum(int(r['hit']) for r in k1_results)
    print(f"  K=1 retrieval: {n_hit_k1}/{N_PASSAGES} "
          f"= {100 * n_hit_k1 / N_PASSAGES:.1f}%")

    # ============================================================
    # PHASE C: Composition at K=1, 2, 4, 6, 8
    # ============================================================
    print(f"\n{'=' * 60}\nPHASE C: COMPOSITION SWEEP\n{'=' * 60}")
    composition_results = {}    # K -> list of per-passage results (constituents + heldout)
    perturbation_magnitudes = {}  # K -> frobenius norm of total delta

    # Base-model held-out reference (zero perturbation) for context.
    restore_base_weights(model, base_snapshot)
    reset_lora_to_zero(model)
    base_heldout = eval_passage(model, heldout_test, tokenizer, device)
    print(f"  base-model held-out (id={heldout_test['id']}): "
          f"ppl={base_heldout['ppl']:.2f}  hit={int(base_heldout['hit'])}")

    for K in K_VALUES:
        # Compose first K adapters
        deltas_k = [compute_per_module_delta(adapter_states[i], scaling)
                    for i in range(K)]
        composed = sum_deltas(deltas_k)
        fnorm = frobenius(composed)
        perturbation_magnitudes[K] = fnorm

        restore_base_weights(model, base_snapshot)
        reset_lora_to_zero(model)
        install_delta(model, composed)

        per_passage = []
        for i in range(K):
            res = eval_passage(model, train_tests[i], tokenizer, device)
            per_passage.append({"passage_idx": i, "id": train_tests[i]['id'],
                                "type": train_tests[i]['type'],
                                "role": "constituent", **res})
        ho = eval_passage(model, heldout_test, tokenizer, device)
        per_passage.append({"passage_idx": -1, "id": heldout_test['id'],
                            "type": heldout_test['type'],
                            "role": "heldout", **ho})

        composition_results[K] = per_passage

        hits = [r['hit'] for r in per_passage if r['role'] == 'constituent']
        ppls = [r['ppl'] for r in per_passage if r['role'] == 'constituent']
        mean_hit = sum(int(h) for h in hits) / len(hits) if hits else 0.0
        mean_ppl = sum(ppls) / len(ppls) if ppls else 0.0
        print(f"  K={K}  ||delta||_F={fnorm:7.3f}  "
              f"constituent hit={mean_hit * 100:.1f}%  "
              f"constituent ppl={mean_ppl:.2f}  "
              f"heldout hit={int(ho['hit'])}  heldout ppl={ho['ppl']:.2f}")

    restore_base_weights(model, base_snapshot)

    summary = {
        "config": {
            "rank": rank, "alpha": alpha, "n_steps": N_STEPS,
            "high_lr": HIGH_LR, "base_lr": BASE_LR,
            "targets": L45_TARGETS,
            "n_passages": N_PASSAGES,
            "selected_ids": SELECTED_IDS,
            "heldout_id": HELDOUT_ID,
            "k_values": K_VALUES,
            "scaling": scaling,
            "n_lora_params": n_lora,
        },
        "k1_baseline": k1_results,
        "composition": {str(K): v for K, v in composition_results.items()},
        "base_heldout": base_heldout,
        "wall_time_s": time.time() - t0,
    }
    # Free the model + GPU memory before next rank
    del model
    torch.cuda.empty_cache()
    return summary, perturbation_magnitudes, composition_results, k1_results


def plot_curves(rank, composition_results, perturbation_magnitudes,
                 train_tests, results_dir, suffix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib missing — skipping plot)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    by_idx = {i: {"K": [], "hit": [], "ppl": []} for i in range(N_PASSAGES)}
    for K, rows in composition_results.items():
        for r in rows:
            if r['role'] == 'constituent' and r['passage_idx'] < N_PASSAGES:
                by_idx[r['passage_idx']]["K"].append(K)
                by_idx[r['passage_idx']]["hit"].append(int(r['hit']))
                by_idx[r['passage_idx']]["ppl"].append(r['ppl'])

    ax = axes[0]
    for i in range(N_PASSAGES):
        ax.plot(by_idx[i]["K"], by_idx[i]["hit"], "o-", alpha=0.55,
                label=f"id={train_tests[i]['id']}/{train_tests[i]['type']}")
    mean_curve = []
    for K in K_VALUES:
        hits = [int(r['hit']) for r in composition_results[K] if r['role'] == 'constituent']
        mean_curve.append(sum(hits) / len(hits))
    ax.plot(K_VALUES, mean_curve, "k-", linewidth=2.5, label="mean")
    ax.set_xlabel("K (composed adapters)")
    ax.set_ylabel("passkey hit (0/1)")
    ax.set_title(f"Retrieval per passage vs K (rank {rank})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i in range(N_PASSAGES):
        ax.plot(by_idx[i]["K"], by_idx[i]["ppl"], "o-", alpha=0.55,
                label=f"id={train_tests[i]['id']}")
    mean_ppl_curve = []
    for K in K_VALUES:
        ppls = [r['ppl'] for r in composition_results[K] if r['role'] == 'constituent']
        mean_ppl_curve.append(sum(ppls) / len(ppls))
    ax.plot(K_VALUES, mean_ppl_curve, "k-", linewidth=2.5, label="mean")
    ax.set_xlabel("K (composed adapters)")
    ax.set_ylabel("perplexity")
    ax.set_title(f"Perplexity per passage vs K (rank {rank})")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    fnorms = [perturbation_magnitudes[K] for K in K_VALUES]
    ax.plot(K_VALUES, fnorms, "o-", color="crimson", linewidth=2)
    linear_ref = [fnorms[0] * K for K in K_VALUES]
    ax.plot(K_VALUES, linear_ref, "--", color="gray", alpha=0.6, label="linear (K · ||δ_1||)")
    ax.set_xlabel("K (composed adapters)")
    ax.set_ylabel("||sum_i δ_i||_F")
    ax.set_title(f"Composed perturbation magnitude (rank {rank})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = results_dir / f"composition_curves{suffix}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    results_dir = Path("results/identity_ae/phase69")
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    all_tests = stratified_tests()
    by_id = {t["id"]: t for t in all_tests}
    train_tests = [by_id[i] for i in SELECTED_IDS]
    heldout_test = by_id[HELDOUT_ID]

    # ============================================================
    # Run primary (rank 8, spec) and control (rank 128, Phase 47) sweeps.
    # ============================================================
    runs = {}
    perturb = {}
    composition = {}
    k1 = {}

    for rank in [RANK_PRIMARY, RANK_CONTROL]:
        summary, perturb_r, comp_r, k1_r = run_rank(
            rank, train_tests, heldout_test, tokenizer, device, results_dir,
        )
        runs[rank] = summary
        perturb[rank] = perturb_r
        composition[rank] = comp_r
        k1[rank] = k1_r

        suffix = "" if rank == RANK_PRIMARY else f"_rank{rank}"
        plot_curves(rank, comp_r, perturb_r, train_tests, results_dir, suffix)

    # ============================================================
    # Save merged artifacts
    # ============================================================
    print(f"\n{'=' * 60}\nSAVING RESULTS\n{'=' * 60}")

    out = {
        "selected_ids": SELECTED_IDS,
        "heldout_id": HELDOUT_ID,
        "k_values": K_VALUES,
        "n_passages": N_PASSAGES,
        "runs": {str(r): runs[r] for r in runs},
    }
    (results_dir / "composition_results.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote composition_results.json")

    (results_dir / "perturbation_magnitudes.json").write_text(json.dumps(
        {str(r): {str(K): v for K, v in perturb[r].items()} for r in perturb},
        indent=2,
    ))
    print(f"  wrote perturbation_magnitudes.json")

    # Concise headline summary for README copy-paste.
    print("\nHEADLINE")
    for rank in [RANK_PRIMARY, RANK_CONTROL]:
        comp = composition[rank]
        for K in K_VALUES:
            hits = [int(r['hit']) for r in comp[K] if r['role'] == 'constituent']
            ppls = [r['ppl'] for r in comp[K] if r['role'] == 'constituent']
            ho = [r for r in comp[K] if r['role'] == 'heldout'][0]
            mh = 100 * sum(hits) / len(hits) if hits else 0.0
            mp = sum(ppls) / len(ppls) if ppls else 0.0
            print(f"  rank={rank} K={K}  hit={mh:5.1f}%  ppl={mp:7.2f}  "
                  f"||delta||_F={perturb[rank][K]:7.3f}  "
                  f"heldout hit={int(ho['hit'])} ppl={ho['ppl']:.2f}")


if __name__ == "__main__":
    main()
