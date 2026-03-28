"""Quick generation script to sanity-check a checkpoint."""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext


def generate(model, tokenizer, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=40):
    """Generate from a token sequence (already on device)."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = prompt_ids.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx = input_ids[:, -512:]
            output = model(idx, step=0)
            logits = output.logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0]


def main():
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "v14_attn_sink"
    ablation_map = {a.value: a for a in AblationConfig}
    # Support output dirs that differ from ablation name (e.g. v16_peer_engram_v2)
    run_dir = Path(f"runs/{version}")
    # Find the ablation subdir inside
    subdirs = [d for d in run_dir.iterdir() if d.is_dir()] if run_dir.exists() else []
    if len(subdirs) == 1:
        run_dir = subdirs[0]
        ablation_name = subdirs[0].name
    else:
        run_dir = run_dir / version
        ablation_name = version
    cfg = ExperimentConfig.from_ablation(ablation_map[ablation_name])

    use_cpu = "--cpu" in sys.argv
    device = torch.device("cpu") if use_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRSTransformer(cfg).to(device)

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    step = ckpt.get("step", "?")
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"Loaded best.pt (step {step}, val_ppl {val_ppl})")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)

    # Load real WikiText validation data for context seeding
    print("Loading WikiText validation data for context seeding...")
    splits, _ = load_wikitext()
    val_dataset = splits["validation"]
    val_tokens = val_dataset.tokens

    # Use 256 tokens of real WikiText as context (fills 2 engram windows),
    # then generate 80 new tokens
    context_len = 256
    gen_tokens = 80

    # Pick 5 different starting positions
    starts = [0, 5000, 15000, 30000, 50000]

    print(f"\nUsing {context_len} real WikiText tokens as context, generating {gen_tokens} new tokens")
    print("="*70)

    for i, start in enumerate(starts):
        context = val_tokens[start:start + context_len]
        context_text = tokenizer.decode(context)

        input_ids = context.unsqueeze(0)  # (1, context_len)
        output_ids = generate(model, tokenizer, input_ids, max_new_tokens=gen_tokens, temperature=0.8, top_k=40)

        generated_ids = output_ids[context_len:]
        generated_text = tokenizer.decode(generated_ids)

        # Show last 50 chars of context for reference
        print(f"\nSample {i+1} (starting at token {start}):")
        print(f"CONTEXT (last 100 chars): ...{context_text[-100:]}")
        print(f"GENERATED: {generated_text}")
        print("="*70)

    # Also test with short prompts (no engram context) to show the contrast
    print("\n\nShort prompts (NO engram context — expect worse quality):")
    print("="*70)
    short_prompts = [
        "The city was founded in",
        "During the Second World War ,",
    ]
    for prompt in short_prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        output_ids = generate(model, tokenizer, ids, max_new_tokens=80, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(output_ids[ids.shape[1]:])
        print(f"\nPROMPT:    {prompt}")
        print(f"GENERATED: {generated_text}")
        print("="*70)


if __name__ == "__main__":
    main()
