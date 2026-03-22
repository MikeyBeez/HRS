#!/usr/bin/env python3
"""
Word-level perplexity evaluation on WikiText-103 test set (v2).

Fixes from v1:
- Tokenizes full text at once (preserves BPE cross-word-boundary behavior)
- Reconstructs word boundaries from the continuous token stream
- Adds sanity checks (shuffled text, no-overlap mode)
- Reports BPE PPL as the primary honest metric
- Word-level PPL reported with explicit caveats

The BPE aggregation advantage means word-level PPL from a BPE model is NOT
directly comparable to word-level PPL from a word-level model. The BPE PPL
is the honest metric. Word-level PPL is provided for rough reference only.
"""

import argparse
import math
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


def tokenize_continuous_with_word_map(text, tokenizer):
    """Tokenize full text continuously and map tokens back to words.

    Unlike v1 which tokenized each word independently, this preserves
    GPT-2 BPE's cross-word-boundary behavior by tokenizing the entire
    text as a single string.

    Returns:
        token_ids: list of BPE token IDs from continuous tokenization
        word_to_tokens: list of (start, end) index pairs into token_ids
        n_words: number of whitespace-delimited words
    """
    # Tokenize the full text at once — this is how the model actually sees it
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Now reconstruct word boundaries by re-decoding tokens and matching
    # against whitespace-delimited words
    words = text.split()
    n_words = len(words)

    # Decode each token to text to find word boundaries
    token_texts = [tokenizer.decode([tid]) for tid in token_ids]

    # Walk through tokens, accumulating text and matching to words
    word_to_tokens = []
    current_word_idx = 0
    current_word_start = 0
    accumulated = ""

    for tok_idx, tok_text in enumerate(token_texts):
        accumulated += tok_text

        # Check if we've completed the current word
        # Words are separated by spaces in the accumulated text
        while current_word_idx < n_words:
            target = words[current_word_idx]
            # Strip leading/trailing whitespace from accumulated text
            stripped = accumulated.strip()

            if stripped == target:
                # Found complete word
                word_to_tokens.append((current_word_start, tok_idx + 1))
                current_word_idx += 1
                current_word_start = tok_idx + 1
                accumulated = ""
                break
            elif len(stripped) > len(target) or (
                " " in stripped and stripped.split()[0] == target
            ):
                # We've gone past the word boundary
                # The word ended somewhere in the accumulated tokens
                # Use a simpler approach: find the split point
                word_to_tokens.append((current_word_start, tok_idx + 1))
                current_word_idx += 1

                # Check if remaining accumulated text contains more words
                remaining = accumulated.strip()
                if remaining.startswith(target):
                    remaining = remaining[len(target):].lstrip()
                    accumulated = remaining
                    current_word_start = tok_idx + 1
                else:
                    accumulated = ""
                    current_word_start = tok_idx + 1
                    break
            else:
                # Still accumulating
                break

    # Handle any remaining tokens
    if current_word_idx < n_words and current_word_start < len(token_ids):
        word_to_tokens.append((current_word_start, len(token_ids)))
        current_word_idx += 1

    return token_ids, word_to_tokens, len(word_to_tokens)


def tokenize_simple_with_word_map(text, tokenizer):
    """Simpler approach: tokenize full text, then use character offsets.

    This uses the tokenizer's offset mapping to precisely map tokens
    back to character positions, which we then align with word boundaries.
    """
    # Tokenize full text at once
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(token_ids)

    # Get character spans for each word
    words = text.split()
    n_words = len(words)

    # Build character-to-token mapping by decoding progressively
    # Each token decodes to some characters
    token_char_starts = []
    char_pos = 0
    full_decoded = ""
    for i, tid in enumerate(token_ids):
        token_char_starts.append(len(full_decoded))
        full_decoded += tokenizer.decode([tid])
    token_char_starts.append(len(full_decoded))  # sentinel

    # Find character positions of each word in the original text
    word_char_starts = []
    pos = 0
    for word in words:
        idx = text.find(word, pos)
        if idx == -1:
            break
        word_char_starts.append(idx)
        pos = idx + len(word)

    # Map words to token ranges using character positions
    # For each word, find which tokens overlap with its character span
    word_to_tokens = []
    tok_idx = 0

    for w_idx, w_start in enumerate(word_char_starts):
        word = words[w_idx]
        w_end = w_start + len(word)

        # Find first token that overlaps with this word
        # (token whose decoded chars include w_start)
        while tok_idx < n_tokens and token_char_starts[tok_idx + 1] <= w_start:
            tok_idx += 1

        t_start = tok_idx

        # Find last token that overlaps
        t_end = t_start
        while t_end < n_tokens and token_char_starts[t_end] < w_end:
            t_end += 1

        word_to_tokens.append((t_start, t_end))

    return token_ids, word_to_tokens, len(word_to_tokens)


def evaluate(model, token_ids, device, seq_len=512, stride=512,
             amp_dtype=torch.bfloat16):
    """Compute per-token log probabilities using sliding window.

    Returns:
        token_log_probs: tensor of log prob for each token (0 for first token)
        token_counted: boolean tensor of which tokens were scored
    """
    n_tokens = len(token_ids)
    tokens = torch.tensor(token_ids, dtype=torch.long)

    token_log_probs = torch.zeros(n_tokens, dtype=torch.float64)
    token_counted = torch.zeros(n_tokens, dtype=torch.bool)

    n_windows = max(1, (n_tokens - seq_len) // stride + 1)
    if (n_windows - 1) * stride + seq_len < n_tokens:
        n_windows += 1

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
            logits = output.logits

        log_probs = F.log_softmax(logits[0].float(), dim=-1)

        for t in range(log_probs.shape[0] - 1):
            target_pos = start + t + 1
            target_token = token_ids[target_pos]

            # Only count tokens in the non-overlapping region
            # (first window counts everything, subsequent windows
            # only count tokens past the overlap boundary)
            if i == 0 or target_pos >= (i * stride):
                if not token_counted[target_pos]:
                    token_log_probs[target_pos] = log_probs[t, target_token].item()
                    token_counted[target_pos] = True

        if (i + 1) % 200 == 0 or i == n_windows - 1:
            elapsed = time.time() - t0
            print(f"  Window {i+1}/{n_windows} ({(i+1)/n_windows*100:.1f}%) - "
                  f"{token_counted.sum().item():,}/{n_tokens:,} tokens - "
                  f"{elapsed:.1f}s")

    return token_log_probs, token_counted


def compute_metrics(token_log_probs, token_counted, token_ids,
                    word_to_tokens=None):
    """Compute BPE-level and word-level perplexity."""
    # BPE-level PPL (the honest metric)
    counted_mask = token_counted.clone()
    counted_mask[0] = False  # first token has no prediction
    bpe_log_probs = token_log_probs[counted_mask]
    bpe_ppl = math.exp(-bpe_log_probs.mean().item())
    bpe_scored = int(counted_mask.sum().item())

    results = {
        "bpe_ppl": bpe_ppl,
        "bpe_tokens_scored": bpe_scored,
        "bpe_tokens_total": len(token_ids),
    }

    # Word-level PPL (for rough reference only — NOT comparable to word-level models)
    if word_to_tokens is not None:
        total_word_log_prob = 0.0
        words_scored = 0

        for t_start, t_end in word_to_tokens:
            word_log_prob = 0.0
            word_complete = True

            for t in range(t_start, t_end):
                if t == 0:
                    continue
                if not token_counted[t]:
                    word_complete = False
                    break
                word_log_prob += token_log_probs[t].item()

            if word_complete and t_end > t_start and t_start > 0:
                total_word_log_prob += word_log_prob
                words_scored += 1

        if words_scored > 0:
            avg_word_log_prob = total_word_log_prob / words_scored
            word_ppl = math.exp(-avg_word_log_prob)
        else:
            word_ppl = float("inf")

        results["word_ppl"] = word_ppl
        results["words_scored"] = words_scored
        results["words_total"] = len(word_to_tokens)

    return results


def run_sanity_checks(model, tokenizer, test_text, device, seq_len, amp_dtype):
    """Run sanity checks to verify evaluation integrity."""
    import random

    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Normal eval (small sample for speed)
    sample = " ".join(test_text.split()[:2000])
    token_ids = tokenizer.encode(sample, add_special_tokens=False)
    lp, counted = evaluate(model, token_ids, device, seq_len, seq_len, amp_dtype)
    counted[0] = False
    normal_ppl = math.exp(-lp[counted].mean().item())
    print(f"\n  Normal (first 2K words):     BPE PPL = {normal_ppl:.2f}")

    # Shuffled tokens — should explode
    shuffled_ids = token_ids.copy()
    random.shuffle(shuffled_ids)
    lp_s, counted_s = evaluate(model, shuffled_ids, device, seq_len, seq_len, amp_dtype)
    counted_s[0] = False
    shuffled_ppl = math.exp(min(-lp_s[counted_s].mean().item(), 20))
    print(f"  Shuffled tokens:             BPE PPL = {shuffled_ppl:.2f}")

    # Shifted targets — use wrong targets
    shifted_ids = token_ids[1:] + [token_ids[0]]
    lp_shift = torch.zeros(len(shifted_ids), dtype=torch.float64)
    tokens_t = torch.tensor(token_ids, dtype=torch.long)
    shifted_t = torch.tensor(shifted_ids, dtype=torch.long)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
        inp = tokens_t[:seq_len].unsqueeze(0).to(device)
        out = model(inp)
        log_probs = F.log_softmax(out.logits[0].float(), dim=-1)
        for t in range(min(log_probs.shape[0] - 1, len(shifted_ids) - 1)):
            lp_shift[t + 1] = log_probs[t, shifted_ids[t + 1]].item()

    valid = lp_shift[2:seq_len]
    shifted_ppl = math.exp(min(-valid.mean().item(), 20))
    print(f"  Shifted targets (off by 1):  BPE PPL = {shifted_ppl:.2f}")

    print()
    if shuffled_ppl > 100:
        print("  [PASS] Shuffled text PPL exploded as expected")
    else:
        print("  [FAIL] Shuffled text PPL suspiciously low — possible leak!")

    if shifted_ppl > normal_ppl * 1.5:
        print("  [PASS] Shifted targets increased PPL as expected")
    else:
        print("  [FAIL] Shifted targets didn't increase PPL much — check eval!")

    print()
    return normal_ppl, shuffled_ppl, shifted_ppl


def main():
    parser = argparse.ArgumentParser(
        description="Word-level perplexity evaluation on WikiText-103 (v2)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--ablation", type=str, default="v9_learnable",
        help="Ablation config name"
    )
    parser.add_argument(
        "--stride", type=int, default=None,
        help="Sliding window stride (default: seq_len, i.e. no overlap)"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext-103",
        help="Dataset name"
    )
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Run sanity checks (shuffled text, shifted targets)"
    )
    parser.add_argument(
        "--no-overlap", action="store_true",
        help="Use stride=seq_len (no overlapping windows)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, args.ablation, device)
    seq_len = cfg.model.max_seq_len

    stride = args.stride or seq_len  # default: no overlap
    if args.no_overlap:
        stride = seq_len

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    ds_config = "wikitext-2-raw-v1" if args.dataset == "wikitext-2" else "wikitext-103-raw-v1"
    print(f"\nLoading {args.dataset} test set...")
    raw = load_dataset("wikitext", ds_config)
    test_text = "\n".join(raw["test"]["text"])
    print(f"  Test set: {len(test_text):,} characters")

    # Run sanity checks if requested
    if args.sanity_check:
        run_sanity_checks(model, tokenizer, test_text, device, seq_len,
                          torch.bfloat16)

    # Tokenize full text continuously (preserves BPE cross-word behavior)
    print(f"\nTokenizing full test set continuously...")
    token_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"  {len(token_ids):,} BPE tokens")

    # Build word mapping from continuous tokenization
    print("Building word-to-token mapping...")
    words = test_text.split()
    print(f"  {len(words):,} whitespace-delimited words")
    print(f"  Ratio: {len(token_ids)/len(words):.2f} BPE tokens per word")

    # Use simple character-offset based mapping
    _, word_to_tokens, n_words_mapped = tokenize_simple_with_word_map(
        test_text, tokenizer
    )
    print(f"  Mapped {n_words_mapped:,} words to token ranges")

    # Evaluate
    print(f"\nEvaluating (seq_len={seq_len}, stride={stride}, "
          f"overlap={'yes' if stride < seq_len else 'no'})...")
    token_log_probs, token_counted = evaluate(
        model, token_ids, device, seq_len, stride
    )

    # Compute metrics
    results = compute_metrics(token_log_probs, token_counted, token_ids,
                              word_to_tokens)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  BPE-level PPL:      {results['bpe_ppl']:.2f}  (PRIMARY METRIC)")
    print(f"  BPE tokens scored:  {results['bpe_tokens_scored']:,} / {results['bpe_tokens_total']:,}")
    if "word_ppl" in results:
        print(f"  Word-level PPL:     {results['word_ppl']:.2f}  (BPE aggregation — NOT comparable to word-level models)")
        print(f"  Words scored:       {results['words_scored']:,} / {results['words_total']:,}")
    print(f"{'='*60}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON (word-level PPL from word-level models)")
    print("NOTE: Our word-level PPL uses BPE aggregation and is NOT")
    print("directly comparable. Our BPE PPL is the honest metric.")
    print(f"{'='*60}")
    print(f"  HRS V12 BPE PPL:              {results['bpe_ppl']:.2f}")
    if "word_ppl" in results:
        print(f"  HRS V12 word PPL (BPE agg):   {results['word_ppl']:.2f}")
    print(f"  kNN-LM (word-level model):    15.79")
    print(f"  Routing Transformer:          15.80")
    print(f"  Compressive Transformer:      17.10")
    print(f"  Transformer-XL:               18.30")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
