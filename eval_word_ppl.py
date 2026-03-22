#!/usr/bin/env python3
"""
Word-level perplexity evaluation on WikiText-103 test set.

Computes perplexity that is directly comparable to published benchmarks
(Transformer-XL, kNN-LM, etc.) which report word-level perplexity.

The model uses GPT-2 BPE tokenization. To convert to word-level perplexity:
1. Tokenize the raw text preserving word boundaries
2. Run the model to get per-BPE-token log probabilities
3. Sum log probs for all BPE tokens within each word
4. Compute perplexity as exp(-1/N_words * sum(log_probs))

Following the standard protocol from Merity et al. / Transformer-XL:
- Evaluate on the full test set (no truncation)
- Use sliding window with context from previous segments
- Report perplexity over all words (not subwords)
"""

import argparse
import math
import re
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from config import AblationConfig, ExperimentConfig
from model import HRSTransformer


def load_model(checkpoint_path, ablation, device):
    """Load model from checkpoint."""
    cfg = ExperimentConfig.from_ablation(AblationConfig(ablation))
    model = HRSTransformer(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    step = ckpt.get("step", "unknown")
    print(f"Loaded {ablation} from {checkpoint_path} (step {step})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, cfg


def tokenize_with_word_boundaries(text, tokenizer):
    """Tokenize text and track which BPE tokens belong to which words.

    Returns:
        token_ids: list of BPE token IDs
        word_starts: list of indices into token_ids where each word begins
        n_words: number of words in the text

    Words are defined as whitespace-delimited tokens in the original text,
    matching the WikiText-103 word-level evaluation protocol.
    """
    # Split text into words (whitespace-delimited), preserving the original
    # word count that published benchmarks use.
    # WikiText raw format: words separated by spaces, with newlines between articles
    words = text.split()

    token_ids = []
    word_starts = []  # index into token_ids where each word starts
    n_words = 0

    for word in words:
        # Skip empty strings
        if not word:
            continue

        # Record where this word's BPE tokens start
        word_starts.append(len(token_ids))
        n_words += 1

        # Tokenize the word. Add a space prefix for words that aren't
        # sentence-initial, matching GPT-2's tokenization convention.
        # GPT-2 BPE expects spaces to be part of the token.
        word_tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        token_ids.extend(word_tokens)

    # Sentinel: marks end of last word
    word_starts.append(len(token_ids))

    return token_ids, word_starts, n_words


def evaluate_word_ppl(model, tokenizer, test_text, device, seq_len=512,
                      stride=512, amp_dtype=torch.bfloat16):
    """Compute word-level perplexity on the test set.

    Uses a sliding window approach with the model's sequence length.
    For each window, we compute log probabilities for all tokens,
    then aggregate them back to word boundaries.

    Args:
        model: the language model
        tokenizer: BPE tokenizer
        test_text: raw test set text
        device: torch device
        seq_len: model's max sequence length
        stride: how far to advance the window each step
        amp_dtype: mixed precision dtype
    """
    print("\nTokenizing test set with word boundaries...")
    token_ids, word_starts, n_words = tokenize_with_word_boundaries(
        test_text, tokenizer
    )
    n_tokens = len(token_ids)
    print(f"  {n_words:,} words -> {n_tokens:,} BPE tokens "
          f"(ratio: {n_tokens/n_words:.2f} tokens/word)")

    tokens = torch.tensor(token_ids, dtype=torch.long)

    # For each BPE token, accumulate its log probability
    token_log_probs = torch.zeros(n_tokens, dtype=torch.float64)
    token_counted = torch.zeros(n_tokens, dtype=torch.bool)

    # Sliding window evaluation
    n_windows = max(1, (n_tokens - seq_len) // stride + 1)
    # Make sure we cover the entire sequence
    if (n_windows - 1) * stride + seq_len < n_tokens:
        n_windows += 1

    print(f"\nEvaluating {n_windows} windows (seq_len={seq_len}, stride={stride})...")
    t0 = time.time()

    for i in range(n_windows):
        start = min(i * stride, n_tokens - seq_len)
        end = start + seq_len

        if end > n_tokens:
            end = n_tokens
            start = max(0, end - seq_len)

        input_ids = tokens[start:end].unsqueeze(0).to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(input_ids)
            logits = output.logits  # (1, T, V)

        # Compute log probs: log p(token_t | context)
        log_probs = F.log_softmax(logits[0].float(), dim=-1)  # (T, V)

        # For each position t, the model predicts token at t+1
        # So log_probs[t] gives the distribution for predicting tokens[start+t+1]
        for t in range(log_probs.shape[0] - 1):
            target_pos = start + t + 1  # position in the full sequence
            target_token = token_ids[target_pos]

            # Only count tokens in the non-overlapping part of the window
            # (except for the first window which counts everything)
            if i == 0 or (start + t + 1) >= (i * stride):
                if not token_counted[target_pos]:
                    token_log_probs[target_pos] = log_probs[t, target_token].item()
                    token_counted[target_pos] = True

        if (i + 1) % 100 == 0 or i == n_windows - 1:
            elapsed = time.time() - t0
            pct = (i + 1) / n_windows * 100
            counted = token_counted.sum().item()
            print(f"  Window {i+1}/{n_windows} ({pct:.1f}%) - "
                  f"{counted:,}/{n_tokens:,} tokens scored - "
                  f"{elapsed:.1f}s")

    # Aggregate BPE token log probs to word-level log probs
    total_word_log_prob = 0.0
    words_scored = 0

    for w in range(n_words):
        bpe_start = word_starts[w]
        bpe_end = word_starts[w + 1]

        # Sum log probs of all BPE tokens in this word
        word_log_prob = 0.0
        word_complete = True

        for t in range(bpe_start, bpe_end):
            if t == 0:
                # First token has no prediction (no context)
                continue
            if not token_counted[t]:
                word_complete = False
                break
            word_log_prob += token_log_probs[t].item()

        if word_complete and bpe_end > bpe_start and bpe_start > 0:
            total_word_log_prob += word_log_prob
            words_scored += 1

    avg_word_log_prob = total_word_log_prob / max(words_scored, 1)
    word_ppl = math.exp(-avg_word_log_prob)

    # Also compute standard BPE-level perplexity for comparison
    counted_mask = token_counted.clone()
    counted_mask[0] = False  # first token has no prediction
    bpe_log_probs = token_log_probs[counted_mask]
    bpe_ppl = math.exp(-bpe_log_probs.mean().item())

    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  BPE tokens scored:  {counted_mask.sum().item():,} / {n_tokens:,}")
    print(f"  Words scored:       {words_scored:,} / {n_words:,}")
    print(f"  BPE-level PPL:      {bpe_ppl:.2f}")
    print(f"  Word-level PPL:     {word_ppl:.2f}")
    print(f"{'='*60}")

    return {
        "word_ppl": word_ppl,
        "bpe_ppl": bpe_ppl,
        "n_words": n_words,
        "words_scored": words_scored,
        "n_bpe_tokens": n_tokens,
        "bpe_tokens_scored": int(counted_mask.sum().item()),
        "tokens_per_word": n_tokens / n_words,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Word-level perplexity evaluation on WikiText-103"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--ablation", type=str, default="v9_learnable",
        help="Ablation config name (default: v9_learnable)"
    )
    parser.add_argument(
        "--stride", type=int, default=256,
        help="Sliding window stride (default: 256, half of seq_len for overlap)"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext-103",
        help="Dataset name (default: wikitext-103)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, cfg = load_model(args.checkpoint, args.ablation, device)
    seq_len = cfg.model.max_seq_len

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load test set
    ds_config = "wikitext-2-raw-v1" if args.dataset == "wikitext-2" else "wikitext-103-raw-v1"
    print(f"\nLoading {args.dataset} test set...")
    raw = load_dataset("wikitext", ds_config)
    test_text = "\n".join(raw["test"]["text"])
    print(f"  Test set: {len(test_text):,} characters")

    # Evaluate
    results = evaluate_word_ppl(
        model, tokenizer, test_text, device,
        seq_len=seq_len, stride=args.stride,
    )

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON WITH PUBLISHED RESULTS")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Params':>10} {'Word PPL':>10}")
    print("-" * 60)
    print(f"{'HRS V9 (ours, step 50K)':.<35} {'176M':>10} {results['word_ppl']:>10.2f}")
    print(f"{'kNN-LM (Khandelwal+ 2020)':.<35} {'247M':>10} {'15.79':>10}")
    print(f"{'Routing Transformer (Roy+ 2021)':.<35} {'~250M':>10} {'15.80':>10}")
    print(f"{'Compressive Trans. (Rae+ 2020)':.<35} {'257M':>10} {'17.10':>10}")
    print(f"{'GPT-2 XL zero-shot (2019)':.<35} {'1558M':>10} {'17.48':>10}")
    print(f"{'MEGA (Ma+ 2023)':.<35} {'252M':>10} {'18.07':>10}")
    print(f"{'Transformer-XL (Dai+ 2019)':.<35} {'257M':>10} {'18.30':>10}")
    print(f"{'Adaptive Input (Baevski+ 2019)':.<35} {'247M':>10} {'18.70':>10}")
    print(f"{'S4 (Gu+ 2022)':.<35} {'249M':>10} {'20.95':>10}")
    print(f"{'GPT-2 Small zero-shot (2019)':.<35} {'117M':>10} {'37.50':>10}")
    print(f"{'='*60}")
    print(f"\nNote: BPE-level PPL was {results['bpe_ppl']:.2f} "
          f"({results['tokens_per_word']:.2f} BPE tokens per word)")


if __name__ == "__main__":
    main()
