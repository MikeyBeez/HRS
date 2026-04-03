"""V20 Evaluation Suite: NIAH, LLM Judge, Per-Head Analysis.

Run after MAUVE. Uses V20 best checkpoint.

Usage:
    python eval_v20.py [--device cuda] [--ollama-host 192.168.12.125]
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer, PerHeadBonsignoreAttention
from engram_store import EngramStore, EngramEntry
from niah_egr import NEEDLES, DISTRACTORS
from retrieval_engine import RetrievalEngine


def load_v20(device):
    cfg = ExperimentConfig.from_ablation(AblationConfig.V20_BONSIGNORE)
    model = HRSTransformer(cfg).to(device)
    ckpt = torch.load("results/v20_bonsignore/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if model.engram_buffer.norm() > 0:
        model._engram_buffer_initialized = True
    model.eval()
    print(f"Loaded V20 (step {ckpt.get('step', '?')}, val_ppl {ckpt.get('val_ppl', '?'):.2f})")
    return model, cfg


def run_niah(model, cfg, tokenizer, device):
    """NIAH with standard and Delta-5 engrams."""
    print(f"\n{'='*60}")
    print("NIAH RETRIEVAL")
    print(f"{'='*60}")

    extract_layer = cfg.model.n_layers - 2
    distractors = DISTRACTORS[:20]

    for engram_type in ["standard", "delta5"]:
        print(f"\n  Engram type: {engram_type}")
        results = []
        for needle in NEEDLES:
            store = EngramStore(d_model=cfg.model.d_model)
            all_docs = [(needle.fact, "needle")] + [(d, f"dist_{i}") for i, d in enumerate(distractors)]

            for text, source in all_docs:
                ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
                if ids.shape[0] < 5:
                    continue

                # Extract hidden states
                x = ids[:512].unsqueeze(0).to(device)
                hidden_states = []
                h = model.drop(model.tok_emb(x))
                hidden_states.append(h.detach())
                for block in model.blocks:
                    eb = model.engram_buffer if model._engram_buffer_initialized else None
                    h, _, _, _ = block(h, step=0, engram_buffer=eb)
                    hidden_states.append(h.detach())

                if engram_type == "standard":
                    engram = hidden_states[extract_layer + 1].mean(dim=1).squeeze(0)
                else:  # delta5
                    delta = hidden_states[-1] - hidden_states[-2]
                    engram = delta.mean(dim=1).squeeze(0)

                engram = F.normalize(engram.cpu(), dim=0)
                store.store(engram, EngramEntry(text=text, mean_entropy=0, condition=engram_type, source=source))

            # Query
            qids = torch.tensor(tokenizer.encode(needle.query, add_special_tokens=False), dtype=torch.long)
            qx = qids[:512].unsqueeze(0).to(device)
            qh = model.drop(model.tok_emb(qx))
            q_hidden = []
            q_hidden.append(qh.detach())
            for block in model.blocks:
                eb = model.engram_buffer if model._engram_buffer_initialized else None
                qh, _, _, _ = block(qh, step=0, engram_buffer=eb)
                q_hidden.append(qh.detach())

            if engram_type == "standard":
                q_eng = q_hidden[extract_layer + 1].mean(dim=1).squeeze(0)
            else:
                q_delta = q_hidden[-1] - q_hidden[-2]
                q_eng = q_delta.mean(dim=1).squeeze(0)
            q_eng = F.normalize(q_eng.cpu(), dim=0)

            retrieval = store.retrieve(q_eng, top_k=5, min_similarity=0.0)
            found, rank, sim = False, -1, 0.0
            for r, (s, entry, _) in enumerate(retrieval):
                if entry.source == "needle":
                    found, rank, sim = True, r + 1, s
                    break

            status = f"rank={rank} sim={sim:.4f}" if found else "NOT FOUND"
            print(f"    {needle.category}: {status}")
            results.append({"needle": needle.category, "found": found, "rank": rank, "sim": sim})

        n_found = sum(1 for r in results if r["found"])
        ranks = [r["rank"] for r in results if r["found"]]
        sims = [r["sim"] for r in results if r["found"]]
        print(f"  → {n_found}/5 found, mean rank {sum(ranks)/len(ranks):.1f}, mean sim {sum(sims)/len(sims):.4f}")

    return results


def run_llm_judge(model, tokenizer, device, ollama_host="192.168.12.125",
                  ollama_model="llama3.1:latest", n_prompts=50):
    """Blind A/B comparison: V20 vs V18 generations."""
    print(f"\n{'='*60}")
    print(f"LLM JUDGE ({n_prompts} prompts)")
    print(f"{'='*60}")

    import requests
    from data import load_wikitext
    splits, _ = load_wikitext()
    test_tokens = splits["test"].tokens

    # Load V18 for comparison
    cfg18 = ExperimentConfig.from_ablation(AblationConfig.V18_CROSS_ATTN)
    model18 = HRSTransformer(cfg18).to(device)
    ckpt18 = torch.load("results/v18_cross_attn/best.pt", map_location=device, weights_only=False)
    model18.load_state_dict(ckpt18["model_state_dict"])
    if model18.engram_buffer.norm() > 0:
        model18._engram_buffer_initialized = True
    model18.eval()

    prompt_len = 100
    gen_len = 150
    stride = max(1, (len(test_tokens) - prompt_len - gen_len) // n_prompts)

    @torch.no_grad()
    def gen(m, prompt_ids):
        ids = prompt_ids.unsqueeze(0).to(device)
        for _ in range(gen_len):
            x = ids[:, -512:]
            out = m(x, step=0)
            logits = out.logits[:, -1, :] / 0.9
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1)], dim=1)
        return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)

    v18_wins, v20_wins, ties = 0, 0, 0
    t0 = time.time()

    for i in range(n_prompts):
        start = i * stride
        prompt_ids = test_tokens[start:start + prompt_len]

        gen_v18 = gen(model18, prompt_ids)
        gen_v20 = gen(model, prompt_ids)
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)[-200:]

        # Randomize A/B
        coin = random.random() > 0.5
        if coin:
            a, b, la, lb = gen_v18[:400], gen_v20[:400], "V18", "V20"
        else:
            a, b, la, lb = gen_v20[:400], gen_v18[:400], "V20", "V18"

        try:
            resp = requests.post(
                f"http://{ollama_host}:11434/api/generate",
                json={"model": ollama_model, "prompt": f"""Rate two text continuations. Score 1-10 on human-likeness, informativeness, coherence, overall.

PROMPT: ...{prompt_text}

A: {a}

B: {b}

Reply EXACTLY: A: human=N info=N coherence=N overall=N
B: human=N info=N coherence=N overall=N
WINNER: A or B or TIE""", "stream": False},
                timeout=120,
            )
            text = resp.json()["response"].upper()
            if "WINNER: A" in text:
                winner = la
            elif "WINNER: B" in text:
                winner = lb
            else:
                winner = "TIE"

            if winner == "V18": v18_wins += 1
            elif winner == "V20": v20_wins += 1
            else: ties += 1
        except:
            ties += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_prompts}: V18={v18_wins} V20={v20_wins} TIE={ties} ({time.time()-t0:.0f}s)")

    del model18
    torch.cuda.empty_cache()

    print(f"\n  Final: V18={v18_wins}, V20={v20_wins}, TIE={ties}")
    return {"v18_wins": v18_wins, "v20_wins": v20_wins, "ties": ties}


def run_per_head_analysis(model, cfg, device):
    """Analyze per-head temperature and alpha specialization."""
    print(f"\n{'='*60}")
    print("PER-HEAD ANALYSIS")
    print(f"{'='*60}")

    for i, block in enumerate(model.blocks):
        if isinstance(block.attn, PerHeadBonsignoreAttention):
            diag = block.attn.get_diagnostics()
            print(f"\n  Layer {i}:")
            for h in range(cfg.model.n_heads):
                print(f"    Head {h}: τ={diag['taus'][h]:.1f}, α={diag['alphas'][h]:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ollama-host", type=str, default=None)
    parser.add_argument("--ollama-model", type=str, default="llama3.1:latest")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_v20(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Per-head analysis
    run_per_head_analysis(model, cfg, device)

    # NIAH
    niah = run_niah(model, cfg, tokenizer, device)

    # LLM Judge (if ollama available)
    judge = None
    if args.ollama_host:
        judge = run_llm_judge(model, tokenizer, device,
                              ollama_host=args.ollama_host,
                              ollama_model=args.ollama_model)

    # Save
    out_path = Path("results/v20_bonsignore/eval_results.json")
    results = {"niah": niah, "judge": judge}
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
