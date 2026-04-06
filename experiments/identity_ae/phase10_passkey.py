"""Phase 10: Passkey Retrieval from Weights.

Train on a passage containing a hidden passkey via TTT.
Remove passage from context. Ask for the passkey.
If the model produces it, knowledge lives in weights, not context.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS python experiments/identity_ae/phase10_passkey.py
"""

import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from experiments.identity_ae.lora_wrapper import apply_lora, reset_lora


random.seed(42)


# ============================================================
# Passkey generation
# ============================================================

def generate_passkeys(n=50):
    """Generate passkey test cases."""
    tests = []

    # Type 1: Simple numeric (20 tests)
    for i in range(20):
        length = random.choice([4, 6, 8])
        passkey = ''.join([str(random.randint(0, 9)) for _ in range(length)])
        facilities = ["northern", "southern", "eastern", "western", "central",
                       "orbital", "coastal", "highland", "basement", "rooftop"]
        facility = facilities[i % len(facilities)]
        passage = (
            f"SECURITY NOTICE: The following information is classified. "
            f"The system access code for the {facility} facility is {passkey}. "
            f"This code must be memorized immediately. "
            f"All authorized personnel must commit code {passkey} to memory. "
            f"The {facility} facility code {passkey} will not be repeated."
        )
        prompt = f"What is the system access code for the {facility} facility?"
        tests.append({"type": "numeric", "passage": passage, "passkey": passkey,
                       "prompt": prompt, "id": i})

    # Type 2: Named entity + date (10 tests)
    names = ["Dr. Elara Voss", "Professor Kian Nakamura", "Agent Sarah Thornhill",
             "Commander Yuki Petrov", "Director Ravi Blackwood",
             "Specialist Anya Morales", "Operative Chen Volkov",
             "Researcher Fatima Okonkwo", "Engineer Dmitri Svensson",
             "Analyst Priya Gutierrez"]
    for i in range(10):
        month = random.choice(["January", "February", "March", "April", "May",
                                "June", "July", "August", "September", "October"])
        day = random.randint(1, 28)
        year = random.randint(1947, 2019)
        passkey = f"{month} {day}, {year}"
        passage = (
            f"Historical record: {names[i]} made a breakthrough discovery on {passkey}. "
            f"The discovery date of {passkey} marks the beginning of a new era. "
            f"{names[i]}'s work on {passkey} changed the field permanently."
        )
        prompt = f"When did {names[i]} make their breakthrough discovery?"
        tests.append({"type": "entity", "passage": passage, "passkey": passkey,
                       "prompt": prompt, "id": 20 + i})

    # Type 3: Technical value (10 tests)
    for i in range(10):
        value = random.randint(100, 9999)
        units = ["kelvin", "megapascals", "gigahertz", "nanometers",
                  "millisieverts", "kilonewtons", "microtesla",
                  "femtoseconds", "petabytes", "exajoules"]
        thing = ["reactor", "accelerator", "telescope", "spectrometer",
                  "collider", "centrifuge", "cryostat", "magnetron",
                  "synchrotron", "interferometer"]
        passkey = str(value)
        passage = (
            f"Technical specification: The {thing[i]} operates at a critical "
            f"threshold of {value} {units[i]}. This value of {value} {units[i]} "
            f"must not be exceeded under any circumstances. "
            f"The {thing[i]} critical value is precisely {value} {units[i]}."
        )
        prompt = f"What is the critical threshold of the {thing[i]} in {units[i]}?"
        tests.append({"type": "technical", "passage": passage, "passkey": passkey,
                       "prompt": prompt, "id": 30 + i})

    # Type 4: Made-up fact (10 tests)
    protocols = ["Thornfield", "Blackwater", "Meridian", "Vanguard", "Eclipse",
                  "Harbinger", "Sentinel", "Obsidian", "Crimson", "Phantom"]
    for i in range(10):
        num = random.randint(3, 47)
        passkey = str(num)
        passage = (
            f"The {protocols[i]} Protocol requires exactly {num} signatories "
            f"to be considered valid. Without all {num} signatories, the "
            f"{protocols[i]} Protocol cannot be enacted. The requirement of "
            f"{num} signatories was established at the founding convention."
        )
        prompt = f"How many signatories does the {protocols[i]} Protocol require?"
        tests.append({"type": "fact", "passage": passage, "passkey": passkey,
                       "prompt": prompt, "id": 40 + i})

    return tests


# ============================================================
# Core functions
# ============================================================

def check_passkey(text, passkey):
    """Check if passkey appears in generated text."""
    if passkey in text:
        return True
    clean = passkey.replace(',', '').replace(' ', '')
    if clean in text.replace(',', '').replace(' ', ''):
        return True
    return False


def load_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v22_learned_kernel/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    for b in model.blocks:
        if hasattr(b, 'cross_attn') and b.use_cross_attn_engram and b.layer_idx == 3:
            b.use_cross_attn_engram = False
    return model, cfg


@torch.no_grad()
def generate_greedy(model, prompt, tokenizer, device, n_tokens=50):
    """Greedy decoding for reproducibility."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    for _ in range(n_tokens):
        idx = input_ids[:, -512:]
        out = model(idx, step=0)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0, len(ids):], skip_special_tokens=True)


def run_ttt_full(model, passage, tokenizer, device, n_steps, lr=1e-5):
    """Full-model TTT on a passage."""
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(n_steps):
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()


def run_ttt_lora(model, passage, tokenizer, device, n_steps, lr=1e-4):
    """LoRA TTT on a passage."""
    ids = tokenizer.encode(passage, add_special_tokens=False)
    ids_t = torch.tensor(ids, dtype=torch.long)[:512].unsqueeze(0).to(device)
    params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    model.train()
    for _ in range(n_steps):
        out = model(ids_t[:, :-1], step=0)
        loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                ids_t[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    model.eval()


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results_dir = Path("results/identity_ae/phase10")
    results_dir.mkdir(parents=True, exist_ok=True)

    tests = generate_passkeys(50)
    print(f"Generated {len(tests)} passkey tests")

    # Save test cases
    with open(results_dir / "passkeys.json", "w") as f:
        json.dump(tests, f, indent=2)

    # Methods to test
    methods = [
        ("full_20", "full", 20, 1e-5),
        ("full_50", "full", 50, 1e-5),
        ("full_100", "full", 100, 1e-5),
        ("lora_100", "lora", 100, 1e-4),
    ]

    all_results = {}

    for method_name, method_type, n_steps, lr in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        # Load fresh model for this method
        model, cfg = load_model(device)
        if method_type == "lora":
            apply_lora(model, rank=128, alpha=256)
        # Enable gradients for full-model methods
        if method_type == "full":
            for p in model.parameters():
                p.requires_grad = True

        base_state = copy.deepcopy(model.state_dict())

        results = {"method": method_name, "n_steps": n_steps, "lr": lr, "tests": []}
        n_baseline_found = 0
        n_context_found = 0
        n_weights_found = 0

        t0 = time.time()
        for ti, test in enumerate(tests):
            # Reload baseline
            model.load_state_dict(base_state)
            if method_type == "lora":
                reset_lora(model)

            passage = test["passage"]
            passkey = test["passkey"]
            prompt = test["prompt"]

            # Step 1: Baseline (no context, no TTT)
            gen_baseline = generate_greedy(model, prompt, tokenizer, device, 50)
            baseline_found = check_passkey(gen_baseline, passkey)
            if baseline_found: n_baseline_found += 1

            # Step 2: Context window (passage + prompt)
            gen_context = generate_greedy(model, passage + " " + prompt, tokenizer, device, 50)
            context_found = check_passkey(gen_context, passkey)
            if context_found: n_context_found += 1

            # Step 3: TTT on passage
            if method_type == "full":
                run_ttt_full(model, passage, tokenizer, device, n_steps, lr)
            else:
                run_ttt_lora(model, passage, tokenizer, device, n_steps, lr)

            # Step 4: Retrieval from weights (prompt only, NO passage)
            gen_weights = generate_greedy(model, prompt, tokenizer, device, 50)
            weights_found = check_passkey(gen_weights, passkey)
            if weights_found: n_weights_found += 1

            results["tests"].append({
                "id": test["id"],
                "type": test["type"],
                "passkey": passkey,
                "baseline_found": baseline_found,
                "context_found": context_found,
                "weights_found": weights_found,
                "gen_weights": gen_weights[:100],
            })

            if (ti + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{ti+1:2d}/50] baseline={n_baseline_found} context={n_context_found} "
                      f"WEIGHTS={n_weights_found} ({elapsed:.0f}s)")

        # Summary for this method
        results["summary"] = {
            "baseline_rate": n_baseline_found / len(tests),
            "context_rate": n_context_found / len(tests),
            "weights_rate": n_weights_found / len(tests),
        }

        # Per-type breakdown
        for ptype in ["numeric", "entity", "technical", "fact"]:
            type_tests = [t for t in results["tests"] if t["type"] == ptype]
            if type_tests:
                w_rate = sum(1 for t in type_tests if t["weights_found"]) / len(type_tests)
                results["summary"][f"{ptype}_weights_rate"] = w_rate

        print(f"\n  {method_name} RESULTS:")
        print(f"    Baseline retrieval: {n_baseline_found}/50 ({n_baseline_found/50:.0%})")
        print(f"    Context retrieval:  {n_context_found}/50 ({n_context_found/50:.0%})")
        print(f"    WEIGHTS retrieval:  {n_weights_found}/50 ({n_weights_found/50:.0%})")

        for ptype in ["numeric", "entity", "technical", "fact"]:
            if f"{ptype}_weights_rate" in results["summary"]:
                print(f"      {ptype}: {results['summary'][f'{ptype}_weights_rate']:.0%}")

        with open(results_dir / f"{method_name}.json", "w") as f:
            json.dump(results, f, indent=2)

        all_results[method_name] = results["summary"]
        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Final summary
    # ============================================================
    print(f"\n{'='*60}")
    print("PASSKEY RETRIEVAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Method':20s} {'Baseline':>10} {'Context':>10} {'WEIGHTS':>10}")
    print(f"  {'-'*52}")
    for method_name, summary in all_results.items():
        print(f"  {method_name:20s} {summary['baseline_rate']:>10.0%} "
              f"{summary['context_rate']:>10.0%} {summary['weights_rate']:>10.0%}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
