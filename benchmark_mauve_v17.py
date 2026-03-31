"""Quick MAUVE benchmark for V17 (vanilla PEER, no engrams)."""

import sys
import json
import time
from pathlib import Path

import torch
import mauve
from transformers import AutoTokenizer
from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext
from benchmark_mauve_v18 import extract_prompts_and_refs, run_condition


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path("runs/v17_peer_only/v17_peer_only")

    cfg = ExperimentConfig.from_ablation(AblationConfig.V17_PEER_ONLY)
    model = HRSTransformer(cfg).to(device)

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded V17 (step {step}, val_ppl {val_ppl})")

    print("\nLoading WikiText-103 test set...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    n_samples = 1000
    continuation_len = 256
    temperature = 0.9
    top_k = 50
    batch_size = 8

    results = []

    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=50, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device,
        label="50-tok prompt (vanilla PEER, no engram)",
    )
    results.append(r)

    r = run_condition(
        model, tokenizer, test_tokens,
        prompt_len=500, continuation_len=continuation_len,
        n_samples=n_samples, batch_size=batch_size,
        temperature=temperature, top_k=top_k,
        device=device,
        label="500-tok prompt (vanilla PEER, no engram)",
    )
    results.append(r)

    print("\n" + "=" * 70)
    print("V17 MAUVE RESULTS (Vanilla PEER baseline)")
    print("=" * 70)
    print(f"Model: v17_peer_only (step {step}, val_ppl {val_ppl})")
    print("-" * 70)
    for r in results:
        print(f"  {r['label']:<50} MAUVE: {r['mauve_score']:.4f}")
    print("=" * 70)

    out = {"version": "v17_peer_only", "step": step, "val_ppl": val_ppl, "conditions": results}
    out_path = run_dir / "mauve_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
