"""Accuracy evaluation: are topic-routed generations actually better?

Three parts:
1. Automated: reference overlap (generated vs actual continuation)
2. LLM-as-judge: ollama model scores which generation is better
3. Human eval: blind A/B pairs for manual judgment

Usage:
    python eval_accuracy.py [--n-samples 50] [--threshold 0.5] [--device cuda]
    python eval_accuracy.py --ollama-host 192.168.12.174 --ollama-model llama3.1
"""

import argparse
import json
import random
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from data import load_wikitext, _extract_articles
from topic_context import TopicContextManager


import requests


def ollama_judge(prompt_text, gen_a_text, gen_b_text, reference_text,
                 host="localhost", port=11434, model="llama3.1"):
    """Use an ollama model as a blind judge between two generations.

    Returns: 'A', 'B', or 'TIE', plus the model's reasoning.
    """
    judge_prompt = f"""You are evaluating two text continuations of a prompt. Judge which continuation is better.

PROMPT:
{prompt_text[:400]}

REFERENCE (what actually came next — for context only):
{reference_text[:300]}

CONTINUATION A:
{gen_a_text[:400]}

CONTINUATION B:
{gen_b_text[:400]}

Judge on these criteria:
1. Coherence: Does it follow logically from the prompt?
2. Relevance: Is the content related to the prompt's topic?
3. Fluency: Is the text natural and readable?

Respond with EXACTLY one of these formats:
WINNER: A
WINNER: B
WINNER: TIE

Then briefly explain why (1-2 sentences)."""

    try:
        resp = requests.post(
            f"http://{host}:{port}/api/generate",
            json={"model": model, "prompt": judge_prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")

        # Parse verdict
        text_upper = text.upper()
        if "WINNER: A" in text_upper:
            verdict = "A"
        elif "WINNER: B" in text_upper:
            verdict = "B"
        elif "WINNER: TIE" in text_upper or "TIE" in text_upper:
            verdict = "TIE"
        else:
            verdict = "UNKNOWN"

        return verdict, text.strip()
    except Exception as e:
        return "ERROR", str(e)


def load_v18_model(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V18 (step {ckpt.get('step', '?')})")
    return model, cfg


@torch.no_grad()
def generate(model, context_ids, prompt_ids, num_tokens, device,
             temperature=0.9, top_k=50, max_seq_len=512):
    """Generate continuation with optional context."""
    model.eval()
    if context_ids is not None and context_ids.shape[0] > 0:
        combined = torch.cat([context_ids, prompt_ids])[-max_seq_len:]
    else:
        combined = prompt_ids[-max_seq_len:]

    input_ids = combined.unsqueeze(0).to(device)

    for _ in range(num_tokens):
        idx = input_ids[:, -max_seq_len:]
        output = model(idx, step=0)
        logits = output.logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Return generated tokens only (after the initial combined input)
    return input_ids[0, combined.shape[0]:]


def compute_token_overlap(generated_ids, reference_ids, ns=(1, 2, 3)):
    """Compute n-gram overlap between generated and reference token sequences.

    Returns dict with precision, recall, F1 for each n.
    """
    results = {}
    gen_list = generated_ids.tolist()
    ref_list = reference_ids.tolist()

    for n in ns:
        gen_ngrams = Counter(tuple(gen_list[i:i+n]) for i in range(len(gen_list) - n + 1))
        ref_ngrams = Counter(tuple(ref_list[i:i+n]) for i in range(len(ref_list) - n + 1))

        if not gen_ngrams or not ref_ngrams:
            results[f"overlap_{n}"] = {"precision": 0, "recall": 0, "f1": 0}
            continue

        overlap = sum((gen_ngrams & ref_ngrams).values())
        precision = overlap / sum(gen_ngrams.values())
        recall = overlap / sum(ref_ngrams.values())
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[f"overlap_{n}"] = {"precision": precision, "recall": recall, "f1": f1}

    return results


def compute_semantic_similarity(model, gen_ids, ref_ids, device, extract_layer):
    """Compute engram cosine similarity between generated and reference text."""
    if gen_ids.shape[0] < 5 or ref_ids.shape[0] < 5:
        return 0.0

    captured = {}
    def hook_fn(module, inp, out):
        captured['h'] = out[0].detach()

    handle = model.blocks[extract_layer].register_forward_hook(hook_fn)

    # Generated engram
    ids = gen_ids[:512].unsqueeze(0).to(device)
    _ = model(ids, step=0)
    gen_engram = captured['h'].mean(dim=1).squeeze(0)

    # Reference engram
    ids = ref_ids[:512].unsqueeze(0).to(device)
    _ = model(ids, step=0)
    ref_engram = captured['h'].mean(dim=1).squeeze(0)

    handle.remove()

    return F.cosine_similarity(gen_engram.unsqueeze(0), ref_engram.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--continuation-len", type=int, default=128)
    parser.add_argument("--prompt-len", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ollama-host", type=str, default=None,
                        help="Ollama host for LLM-as-judge (e.g., 192.168.12.174 or localhost)")
    parser.add_argument("--ollama-model", type=str, default="llama3.1",
                        help="Ollama model name for judging")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_v18_model(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    extract_layer = cfg.model.n_layers - 2

    # Load data
    print("\nLoading WikiText-103...")
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    from datasets import load_dataset
    raw = load_dataset("wikitext", "wikitext-103-raw-v1")
    val_articles = _extract_articles(raw["validation"]["text"])

    # Populate topic manager
    mgr = TopicContextManager(
        model, tokenizer,
        similarity_threshold=args.threshold,
        max_context_tokens=384,
        max_active=3,
    )
    for i, (title, text) in enumerate(val_articles[:60]):
        if len(text.strip()) < 100:
            continue
        mgr.process_prompt(text[:500].strip())
    print(f"  {len(mgr.clusters)} clusters")

    # Generate paired continuations
    total_per_sample = args.prompt_len + args.continuation_len
    max_start = len(test_tokens) - total_per_sample
    stride = max(1, max_start // args.n_samples)

    print(f"\nGenerating {args.n_samples} paired continuations...")
    print(f"  Prompt: {args.prompt_len} tokens, Continuation: {args.continuation_len} tokens")

    results = []
    overlap_baseline = {f"overlap_{n}": [] for n in (1, 2, 3)}
    overlap_topic = {f"overlap_{n}": [] for n in (1, 2, 3)}
    sem_sim_baseline = []
    sem_sim_topic = []

    t0 = time.time()
    for i in range(args.n_samples):
        start = i * stride
        if start + total_per_sample > len(test_tokens):
            start = len(test_tokens) - total_per_sample

        prompt_ids = test_tokens[start:start + args.prompt_len]
        ref_ids = test_tokens[start + args.prompt_len:start + total_per_sample]

        # Baseline generation
        gen_baseline = generate(model, None, prompt_ids, args.continuation_len, device)

        # Topic-routed generation
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        mgr.process_prompt(prompt_text)
        topic_ctx = mgr.get_context()
        gen_topic = generate(model, topic_ctx, prompt_ids, args.continuation_len, device)

        # Token overlap with reference
        ov_base = compute_token_overlap(gen_baseline, ref_ids)
        ov_topic = compute_token_overlap(gen_topic, ref_ids)
        for n in (1, 2, 3):
            overlap_baseline[f"overlap_{n}"].append(ov_base[f"overlap_{n}"]["f1"])
            overlap_topic[f"overlap_{n}"].append(ov_topic[f"overlap_{n}"]["f1"])

        # Semantic similarity to reference
        sim_base = compute_semantic_similarity(model, gen_baseline, ref_ids, device, extract_layer)
        sim_topic = compute_semantic_similarity(model, gen_topic, ref_ids, device, extract_layer)
        sem_sim_baseline.append(sim_base)
        sem_sim_topic.append(sim_topic)

        # Store for human eval
        prompt_text_full = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
        gen_base_text = tokenizer.decode(gen_baseline, skip_special_tokens=True)
        gen_topic_text = tokenizer.decode(gen_topic, skip_special_tokens=True)

        results.append({
            "id": i,
            "prompt": prompt_text_full,
            "reference": ref_text,
            "gen_baseline": gen_base_text,
            "gen_topic": gen_topic_text,
            "overlap_baseline": ov_base,
            "overlap_topic": ov_topic,
            "semantic_sim_baseline": sim_base,
            "semantic_sim_topic": sim_topic,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{args.n_samples} ({elapsed:.0f}s)")

    # ============================================================
    # Automated metrics summary
    # ============================================================
    print("\n" + "=" * 60)
    print("AUTOMATED ACCURACY METRICS")
    print("=" * 60)

    print(f"\nToken overlap with reference (F1):")
    print(f"  {'':>12} {'Baseline':>10} {'Topic':>10} {'Delta':>10}")
    for n in (1, 2, 3):
        base_mean = sum(overlap_baseline[f"overlap_{n}"]) / len(overlap_baseline[f"overlap_{n}"])
        topic_mean = sum(overlap_topic[f"overlap_{n}"]) / len(overlap_topic[f"overlap_{n}"])
        delta = topic_mean - base_mean
        print(f"  {f'{n}-gram':>12} {base_mean:>10.4f} {topic_mean:>10.4f} {delta:>+10.4f}")

    base_sem = sum(sem_sim_baseline) / len(sem_sim_baseline)
    topic_sem = sum(sem_sim_topic) / len(sem_sim_topic)
    print(f"\nSemantic similarity to reference (engram cosine):")
    print(f"  Baseline: {base_sem:.4f}")
    print(f"  Topic:    {topic_sem:.4f}")
    print(f"  Delta:    {topic_sem - base_sem:+.4f}")

    # Count wins
    base_wins = 0
    topic_wins = 0
    ties = 0
    for r in results:
        b = r["semantic_sim_baseline"]
        t = r["semantic_sim_topic"]
        if abs(b - t) < 0.01:
            ties += 1
        elif t > b:
            topic_wins += 1
        else:
            base_wins += 1
    print(f"\nSemantic similarity wins: baseline={base_wins}, topic={topic_wins}, ties={ties}")

    # ============================================================
    # Generate human eval file
    # ============================================================
    print("\n" + "=" * 60)
    print("GENERATING HUMAN EVALUATION FILE")
    print("=" * 60)

    # Randomize which is A and which is B for blind evaluation
    human_eval = []
    for r in results:
        coin = random.random() > 0.5
        if coin:
            gen_a, gen_b = r["gen_baseline"], r["gen_topic"]
            label_a, label_b = "baseline", "topic"
        else:
            gen_a, gen_b = r["gen_topic"], r["gen_baseline"]
            label_a, label_b = "topic", "baseline"

        human_eval.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "reference": r["reference"],
            "generation_A": gen_a,
            "generation_B": gen_b,
            # Answer key (don't peek during evaluation!)
            "_key_A": label_a,
            "_key_B": label_b,
        })

    # Write readable evaluation file
    eval_path = Path("results/v18_cross_attn/human_eval_pairs.txt")
    with open(eval_path, "w") as f:
        f.write("BLIND A/B EVALUATION — Topic-Routed Context\n")
        f.write("=" * 70 + "\n\n")
        f.write("Instructions: For each pair, read the prompt, then read Generation A\n")
        f.write("and Generation B. Judge which is better on:\n")
        f.write("  1. Coherence: Does the continuation follow from the prompt?\n")
        f.write("  2. Relevance: Is the content related to the prompt's topic?\n")
        f.write("  3. Fluency:   Is the text natural and readable?\n")
        f.write("Mark your choice: A, B, or TIE\n\n")
        f.write("The reference continuation (what WikiText actually says) is included\n")
        f.write("for context but don't judge based on similarity to reference.\n")
        f.write("=" * 70 + "\n\n")

        for item in human_eval:
            f.write(f"--- Pair {item['id'] + 1} ---\n\n")
            f.write(f"PROMPT:\n{item['prompt'][:300]}\n\n")
            f.write(f"REFERENCE (actual continuation):\n{item['reference'][:300]}\n\n")
            f.write(f"GENERATION A:\n{item['generation_A'][:300]}\n\n")
            f.write(f"GENERATION B:\n{item['generation_B'][:300]}\n\n")
            f.write(f"Your judgment (A / B / TIE): ________\n")
            f.write(f"Notes: \n\n")
            f.write("-" * 70 + "\n\n")

    print(f"  Written {len(human_eval)} pairs to {eval_path}")

    # Save answer key separately
    key_path = Path("results/v18_cross_attn/human_eval_key.json")
    with open(key_path, "w") as f:
        json.dump({
            "pairs": [{
                "id": h["id"],
                "A_is": h["_key_A"],
                "B_is": h["_key_B"],
            } for h in human_eval],
        }, f, indent=2)
    print(f"  Answer key saved to {key_path}")

    # ============================================================
    # LLM-as-judge (if ollama available)
    # ============================================================
    llm_judge_results = None
    if args.ollama_host:
        print("\n" + "=" * 60)
        print(f"LLM-AS-JUDGE (ollama {args.ollama_model} @ {args.ollama_host})")
        print("=" * 60)

        judge_wins = {"baseline": 0, "topic": 0, "tie": 0, "error": 0}
        judge_details = []

        for item in human_eval:
            verdict, reasoning = ollama_judge(
                item["prompt"], item["generation_A"], item["generation_B"],
                item["reference"],
                host=args.ollama_host, model=args.ollama_model,
            )

            # Map verdict back to actual condition
            if verdict == "A":
                winner = item["_key_A"]
            elif verdict == "B":
                winner = item["_key_B"]
            elif verdict == "TIE":
                winner = "tie"
            else:
                winner = "error"

            judge_wins[winner] = judge_wins.get(winner, 0) + 1
            judge_details.append({
                "id": item["id"],
                "verdict": verdict,
                "winner": winner,
                "reasoning": reasoning[:200],
            })

            done = len(judge_details)
            if done % 10 == 0:
                print(f"  {done}/{len(human_eval)}: "
                      f"baseline={judge_wins['baseline']} topic={judge_wins['topic']} "
                      f"tie={judge_wins['tie']} error={judge_wins['error']}")

        print(f"\n  LLM Judge Results:")
        print(f"    Baseline wins: {judge_wins['baseline']}")
        print(f"    Topic wins:    {judge_wins['topic']}")
        print(f"    Ties:          {judge_wins['tie']}")
        print(f"    Errors:        {judge_wins['error']}")

        total_judged = judge_wins["baseline"] + judge_wins["topic"] + judge_wins["tie"]
        if total_judged > 0:
            topic_rate = judge_wins["topic"] / total_judged
            base_rate = judge_wins["baseline"] / total_judged
            print(f"    Topic win rate: {topic_rate:.1%}")
            print(f"    Baseline win rate: {base_rate:.1%}")

        llm_judge_results = {
            "model": args.ollama_model,
            "host": args.ollama_host,
            "wins": judge_wins,
            "details": judge_details,
        }

        # Save judge results separately
        judge_path = Path("results/v18_cross_attn/llm_judge_results.json")
        with open(judge_path, "w") as f:
            json.dump(llm_judge_results, f, indent=2)
        print(f"  Judge details saved to {judge_path}")

    # Save full results
    out_path = Path("results/v18_cross_attn/eval_accuracy.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_samples": args.n_samples,
            "prompt_len": args.prompt_len,
            "continuation_len": args.continuation_len,
            "threshold": args.threshold,
            "overlap_f1": {
                n: {
                    "baseline": sum(overlap_baseline[f"overlap_{n}"]) / len(overlap_baseline[f"overlap_{n}"]),
                    "topic": sum(overlap_topic[f"overlap_{n}"]) / len(overlap_topic[f"overlap_{n}"]),
                }
                for n in (1, 2, 3)
            },
            "semantic_similarity": {
                "baseline": base_sem,
                "topic": topic_sem,
            },
            "wins": {"baseline": base_wins, "topic": topic_wins, "ties": ties},
            "llm_judge": llm_judge_results,
        }, f, indent=2)
    print(f"  Full results saved to {out_path}")


if __name__ == "__main__":
    main()
