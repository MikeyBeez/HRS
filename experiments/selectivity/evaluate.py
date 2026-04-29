"""Run Tests 1-4 on baseline and selectivity adapters.

Test 1: positive-content retrieval (substring match on held-out probes).
Test 2: passivity on training negative probes (KL/cosine/token-overlap vs
        base output).
Test 3: passivity on held-out negative content (Domain C: bread).
Test 4: K=2 composition with both adapters loaded (block-stacked).

Identity loss formulation = KL on output token distributions
(reduction='batchmean' = sum-over-vocab, mean-over-tokens).

Identity / passivity definition:
  KL = mean_per_token KL( P_adapter || P_base )
  Higher KL = more active. Closer to 0 = more passive.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

REPO = Path("/mnt/data/Code/HRS")
sys.path.insert(0, str(REPO))

from experiments.identity_ae.phase10_passkey import load_model
from experiments.identity_ae.phase14_lora_l45 import L45_TARGETS
from experiments.identity_ae.phase22_engram_key import (
    reset_lora_to_zero, hidden_at_layer,
)
from experiments.identity_ae.phase43_k_capacity import stack_k_state_dicts
from experiments.identity_ae.lora_wrapper import (
    apply_lora, load_lora_state_dict,
)
from experiments.selectivity.data import domain_data

PPD = REPO / "experiments/per_passage_dickens"
SEL = REPO / "experiments/selectivity"
RANK = 128
ALPHA = RANK * 2
GEN_TOKENS = 20
TEMPERATURE = 0.6
TOP_K = 20
SEEDS = (0, 1, 2)


def encode(tokenizer, text, device, ctx=512):
    ids = tokenizer.encode(text, add_special_tokens=False)[:ctx]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)


@torch.no_grad()
def generate(model, ids_t, n_tokens, gen_seed):
    rng = torch.Generator(device=ids_t.device); rng.manual_seed(gen_seed)
    for _ in range(n_tokens):
        idx = ids_t[:, -512:]
        out = model(idx, step=0)
        logits = out.logits[:, -1, :] / TEMPERATURE
        v, _ = torch.topk(logits, TOP_K)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=rng)
        ids_t = torch.cat([ids_t, nxt], dim=1)
    return ids_t


def check_match(answer, gen):
    if answer.lower() in gen.lower():
        return True
    a = answer.replace(",", "").replace(" ", "").lower()
    g = gen.replace(",", "").replace(" ", "").lower()
    return bool(a) and a in g


def load_adapter_to_model(model, sd_path, device):
    sd = torch.load(sd_path, map_location=device, weights_only=False)
    load_lora_state_dict(model, sd)


def build_model(device, rank=RANK):
    model, _ = load_model(device)
    dickens_ck = torch.load(PPD / "results/v22_dickens_base.pt",
                             map_location=device, weights_only=False)
    model.load_state_dict(dickens_ck["model_state_dict"], strict=False)
    apply_lora(model, rank=rank, alpha=rank * 2, target_modules=L45_TARGETS)
    reset_lora_to_zero(model)
    model.eval()
    return model


# ---------- Test 1: positive retrieval ----------
def test1_positive_retrieval(model, adapter_paths, held_out_examples,
                              tokenizer, device):
    """For each adapter, eval substring match on held-out positive probes."""
    out = {}
    for name, path in adapter_paths.items():
        reset_lora_to_zero(model)
        load_adapter_to_model(model, path, device)
        n = 0; n_hit = 0
        for ex in held_out_examples:
            for seed in SEEDS:
                ids_t = encode(tokenizer, ex["probe"], device)
                gen = generate(model, ids_t, GEN_TOKENS,
                                gen_seed=seed * 10000 + hash(ex["probe"]) % 1000)
                full = tokenizer.decode(gen[0], skip_special_tokens=True)
                cont = full[len(ex["probe"]):]
                n += 1
                if check_match(ex["answer"], cont):
                    n_hit += 1
        out[name] = {"n": n, "n_hit": n_hit, "rate": n_hit / n}
    return out


# ---------- Tests 2/3: passivity ----------
@torch.no_grad()
def passivity_metrics(model, base_logits_cache, neg_examples, tokenizer, device):
    """For each negative probe, compare adapter logits to base logits.
    Returns mean KL, mean cosine similarity at last hidden layer, mean
    token-argmax-overlap.
    """
    kls = []; cos_l5 = []; arg_overlaps = []
    for ex, base_logits in zip(neg_examples, base_logits_cache):
        ids_t = encode(tokenizer, ex["probe"], device)
        if ids_t.shape[1] < 2:
            continue
        adapter_logits = model(ids_t[:, :-1], step=0).logits
        # KL: mean over tokens
        log_p = F.log_softmax(adapter_logits, dim=-1)
        p_b = F.softmax(base_logits, dim=-1)
        kl = F.kl_div(log_p, p_b, reduction="batchmean").item()
        kls.append(kl)
        # Argmax overlap: fraction of tokens where the same token is most likely
        adapter_arg = adapter_logits.argmax(dim=-1).squeeze(0)
        base_arg = base_logits.argmax(dim=-1).squeeze(0)
        if adapter_arg.shape == base_arg.shape and adapter_arg.numel() > 0:
            arg_overlaps.append(float((adapter_arg == base_arg).float().mean().item()))
        # Hidden cosine at L5 for adapter; compare to L5 with LoRA off
        h_adapter = hidden_at_layer(model, ids_t[:, :-1], 5).mean(dim=1).squeeze(0)
        # We don't have base hidden cached; compute it now (toggle).
        cos_l5.append(None)  # skip — we'll add a separate pass

    return {
        "kl_mean": float(np.mean(kls)) if kls else float("nan"),
        "kl_std": float(np.std(kls)) if kls else float("nan"),
        "argmax_overlap_mean": float(np.mean(arg_overlaps)) if arg_overlaps else float("nan"),
    }


@torch.no_grad()
def precompute_base_artifacts(model, neg_examples, tokenizer, device):
    """With LoRA zeroed, precompute logits and L5-mean hidden for each neg probe."""
    saved = {n: p.detach().clone()
             for n, p in model.named_parameters() if "lora_" in n}
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data.zero_()
    base_logits_list = []
    base_h5_list = []
    for ex in neg_examples:
        ids_t = encode(tokenizer, ex["probe"], device)
        if ids_t.shape[1] < 2:
            base_logits_list.append(None); base_h5_list.append(None); continue
        base_logits_list.append(model(ids_t[:, :-1], step=0).logits.detach().clone())
        h5 = hidden_at_layer(model, ids_t[:, :-1], 5).mean(dim=1).squeeze(0).detach().clone()
        base_h5_list.append(h5)
    cur = dict(model.named_parameters())
    for n, v in saved.items():
        cur[n].data.copy_(v)
    return base_logits_list, base_h5_list


@torch.no_grad()
def passivity_with_h5(model, base_logits_list, base_h5_list, neg_examples,
                       tokenizer, device):
    """Full passivity metrics: KL, argmax overlap, L5 cosine."""
    kls = []; arg_overlaps = []; cos_h5 = []
    for ex, base_logits, base_h5 in zip(neg_examples, base_logits_list, base_h5_list):
        if base_logits is None:
            continue
        ids_t = encode(tokenizer, ex["probe"], device)
        adapter_logits = model(ids_t[:, :-1], step=0).logits
        log_p = F.log_softmax(adapter_logits, dim=-1)
        p_b = F.softmax(base_logits, dim=-1)
        kl = F.kl_div(log_p, p_b, reduction="batchmean").item()
        kls.append(kl)
        adapter_arg = adapter_logits.argmax(dim=-1).squeeze(0)
        base_arg = base_logits.argmax(dim=-1).squeeze(0)
        arg_overlaps.append(float((adapter_arg == base_arg).float().mean().item()))
        h5_a = hidden_at_layer(model, ids_t[:, :-1], 5).mean(dim=1).squeeze(0)
        cos = F.cosine_similarity(h5_a.unsqueeze(0), base_h5.unsqueeze(0)).item()
        cos_h5.append(cos)
    return {
        "kl_mean": float(np.mean(kls)) if kls else float("nan"),
        "argmax_overlap_mean": float(np.mean(arg_overlaps)) if arg_overlaps else float("nan"),
        "h5_cos_mean": float(np.mean(cos_h5)) if cos_h5 else float("nan"),
        "n": len(kls),
    }


def test_2_3(model, adapter_paths, neg_examples, tokenizer, device, label):
    """Run Test 2 (training negs) or Test 3 (held-out C bread)."""
    print(f"\n[{label}] Precomputing base logits + L5 ...")
    reset_lora_to_zero(model)
    base_logits, base_h5 = precompute_base_artifacts(
        model, neg_examples, tokenizer, device,
    )
    out = {}
    for name, path in adapter_paths.items():
        reset_lora_to_zero(model)
        load_adapter_to_model(model, path, device)
        out[name] = passivity_with_h5(model, base_logits, base_h5,
                                        neg_examples, tokenizer, device)
    return out


# ---------- Test 4: K=2 composition ----------
def test4_composition(adapter_a_path, adapter_b_path, baseline_a_path,
                       baseline_b_path, queries_by_kind, c_examples,
                       tokenizer, device):
    """For each pair (selectivity / baseline), block-stack at K=2:
       - retrieval rate on positive A, positive B, C (out-of-domain).
       - KL/h5_cos vs base on C (passivity-under-composition).
    """
    out = {}
    for label, paths in [
        ("selectivity_AB", (adapter_a_path, adapter_b_path)),
        ("baseline_AB",    (baseline_a_path, baseline_b_path)),
    ]:
        sd_a = torch.load(paths[0], map_location="cpu", weights_only=False)
        sd_b = torch.load(paths[1], map_location="cpu", weights_only=False)
        stacked = stack_k_state_dicts([sd_a, sd_b])

        model = build_model(device, rank=2 * RANK)
        load_lora_state_dict(model, {k: v.to(device) for k, v in stacked.items()})

        kind_results = {}
        for kind, examples in queries_by_kind.items():
            n = 0; n_hit = 0
            for ex in examples:
                for seed in SEEDS:
                    ids_t = encode(tokenizer, ex["probe"], device)
                    gen = generate(model, ids_t, GEN_TOKENS,
                                    gen_seed=seed * 10000 + hash(ex["probe"]) % 1000)
                    full = tokenizer.decode(gen[0], skip_special_tokens=True)
                    cont = full[len(ex["probe"]):]
                    n += 1
                    if check_match(ex["answer"], cont):
                        n_hit += 1
            kind_results[kind] = {"n": n, "n_hit": n_hit, "rate": n_hit / n}

        # Passivity under composition: KL/h5_cos vs base on C examples
        # First compute base artifacts (with LoRA off), then compute adapter ones.
        base_logits, base_h5 = precompute_base_artifacts(
            model, c_examples, tokenizer, device,
        )
        # Restore the stacked LoRA state (precompute zeros it then restores;
        # but defensively, reload it)
        load_lora_state_dict(model, {k: v.to(device) for k, v in stacked.items()})
        passivity = passivity_with_h5(model, base_logits, base_h5,
                                        c_examples, tokenizer, device)
        kind_results["C_passivity"] = passivity
        out[label] = kind_results
        del model
        torch.cuda.empty_cache()
    return out


# ---------- Main ----------
def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    data = domain_data()

    print("Building base model + LoRA structure (rank 128) ...")
    model = build_model(device, rank=RANK)

    # Adapter paths
    ad = SEL / "adapters"
    LAMS = [0.05, 0.1, 0.5, 1.0, 2.0]
    A_paths = {"baseline_A": ad / "baseline_A.pt"}
    B_paths = {"baseline_B": ad / "baseline_B.pt"}
    for lam in LAMS:
        A_paths[f"sel_A_lam{lam:.1f}"] = ad / f"sel_A_lam{lam:.1f}.pt"
        B_paths[f"sel_B_lam{lam:.1f}"] = ad / f"sel_B_lam{lam:.1f}.pt"
    all_paths = {**A_paths, **B_paths}

    out = {"tests": {}}
    t0 = time.time()

    # Test 1: positive retrieval
    print("\n=== Test 1: positive-content retrieval (held-out probes) ===")
    t1 = {}
    t1_A = test1_positive_retrieval(model, A_paths, data["A"]["held_out"],
                                      tokenizer, device)
    t1_B = test1_positive_retrieval(model, B_paths, data["B"]["held_out"],
                                      tokenizer, device)
    for k, v in t1_A.items():
        t1[k] = v; print(f"  {k}: rate={v['rate']:.3f} ({v['n_hit']}/{v['n']})")
    for k, v in t1_B.items():
        t1[k] = v; print(f"  {k}: rate={v['rate']:.3f} ({v['n_hit']}/{v['n']})")
    out["tests"]["test1_positive"] = t1

    # Test 2: passivity on training negatives.
    # Adapter A's training negs were B's probes; Adapter B's training negs were A's probes.
    print("\n=== Test 2: passivity on training negatives ===")
    t2 = {}
    t2_A = test_2_3(model, A_paths, data["B"]["train"], tokenizer, device,
                     "A vs B-train")
    t2_B = test_2_3(model, B_paths, data["A"]["train"], tokenizer, device,
                     "B vs A-train")
    for k, v in t2_A.items():
        t2[k] = v
        print(f"  {k}: KL={v['kl_mean']:.3f} h5_cos={v['h5_cos_mean']:.3f} "
              f"argmax_overlap={v['argmax_overlap_mean']:.3f}")
    for k, v in t2_B.items():
        t2[k] = v
        print(f"  {k}: KL={v['kl_mean']:.3f} h5_cos={v['h5_cos_mean']:.3f} "
              f"argmax_overlap={v['argmax_overlap_mean']:.3f}")
    out["tests"]["test2_train_neg_passivity"] = t2

    # Test 3: passivity on held-out C (bread)
    print("\n=== Test 3: passivity on held-out C (bread) ===")
    t3 = {}
    t3_A = test_2_3(model, A_paths, data["C"]["held_out"], tokenizer, device,
                     "A vs C")
    t3_B = test_2_3(model, B_paths, data["C"]["held_out"], tokenizer, device,
                     "B vs C")
    for k, v in t3_A.items():
        t3[k] = v
        print(f"  {k}: KL={v['kl_mean']:.3f} h5_cos={v['h5_cos_mean']:.3f}")
    for k, v in t3_B.items():
        t3[k] = v
        print(f"  {k}: KL={v['kl_mean']:.3f} h5_cos={v['h5_cos_mean']:.3f}")
    out["tests"]["test3_held_out_C_passivity"] = t3

    # Test 4: K=2 composition. Pick "Pareto-best" λ: the one with positive
    # retrieval >= 0.7 * baseline AND minimum KL among those that satisfy.
    def pareto_best(kind, t1_results, t2_results, baseline_key):
        baseline_pos = t1_results[baseline_key]["rate"]
        threshold = 0.7 * baseline_pos
        candidates = [k for k in t1_results
                       if k.startswith(f"sel_{kind}_")
                       and t1_results[k]["rate"] >= threshold]
        if not candidates:
            # Fall back to highest positive
            return max([k for k in t1_results if k.startswith(f"sel_{kind}_")],
                        key=lambda k: t1_results[k]["rate"])
        return min(candidates, key=lambda k: t2_results[k]["kl_mean"])

    best_A = pareto_best("A", t1, t2, "baseline_A")
    best_B = pareto_best("B", t1, t2, "baseline_B")
    print(f"\nPareto-best selectivity adapters (positive >= 0.7*baseline, "
          f"min KL among those): A={best_A}  B={best_B}")
    out["best_lambda"] = {"A": best_A, "B": best_B}

    print("\n=== Test 4: K=2 composition (A_loaded + B_loaded) ===")
    queries_by_kind = {
        "A_positive": data["A"]["held_out"],
        "B_positive": data["B"]["held_out"],
        "C_held_out": data["C"]["held_out"][:10],
    }
    t4 = test4_composition(
        A_paths[best_A], B_paths[best_B],
        A_paths["baseline_A"], B_paths["baseline_B"],
        queries_by_kind, data["C"]["held_out"][:20],
        tokenizer, device,
    )
    for label, kinds in t4.items():
        print(f"  {label}:")
        for kind, v in kinds.items():
            if "rate" in v:
                print(f"    {kind}: rate={v['rate']:.3f} ({v['n_hit']}/{v['n']})")
            else:
                print(f"    {kind}: KL={v['kl_mean']:.3f} "
                      f"h5_cos={v['h5_cos_mean']:.3f} "
                      f"argmax_overlap={v['argmax_overlap_mean']:.3f}")
    out["tests"]["test4_composition"] = t4

    out["wall_total_s"] = time.time() - t0
    out_path = REPO / "experiments/selectivity/results/tests.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTotal eval wall: {out['wall_total_s']:.0f}s  saved {out_path}")


if __name__ == "__main__":
    main()
