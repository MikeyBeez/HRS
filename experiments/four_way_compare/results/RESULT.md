# Four-Way Comparison: RAG vs Combo vs Multi-stack vs Multi-pass

**Question:** at comparable substrate quality (a base capable of in-context extractive Q/A), how do four approaches to multi-content queries compare?

**Verdict: RAG wins decisively.** On Mistral-7B-v0.1 with a 2-shot extractive prompt, RAG hits 100% on per-constituent retrieval at K=2/3/4 and 79-100% on cross-passage queries. Combination adapters tie RAG at K=2/3 single-content (1.00) but lag on cross-passage (0.73-0.79 vs RAG's 0.79-1.00). Multi-stack collapses with K (0.75 → 0.50 → 0.375 single-content, 0.34 → 0.25 → 0.04 cross-pass) — confirming the K>2 ceiling is **structural**, not substrate-dependent. Multi-pass works decently single-content (0.75-0.92) but degrades on cross-pass (0.42-0.75).

**The architecture's substitution claim — "absorbing content into adapters provides capability advantages beyond context-based access" — is NOT supported on a capable base.** The 60-71pp combo advantage observed on V22-Dickens was substantially due to the V22 base's inability to do in-context Q/A. On Mistral-7B that gap disappears. The architecture's value proposition needs honest reframing toward cost structure (combo = shorter context at inference, faster per-query) rather than fundamental capability advantage.

## Substrate validation

Probe: 10/10 = 100% on a 10-question extractive Q/A test using a 2-shot prompt.

Base: **mistralai/Mistral-7B-v0.1** (fp16, 7.2B params, Dickens-pretrained-style not required — Mistral's general-corpus pretraining suffices).

Without the 2-shot prefix, Mistral hit only 5/10 — the failures were "X = 14 letters" style (misinterpreting probes as length-counting). The few-shot prefix locks the base into an extractive Q/A format and lifts it to 10/10. All four procedures use this prefix when applicable to make the comparison fair.

## Setup

- Same 8 Dickens passages from prior experiments (per_passage_dickens library_ids 0, 2, 16, 17, 22, 30, 31, 36).
- Same 10 combinations (4 K=2 + 4 K=3 + 2 K=4) from the combo_adapter experiment.
- 18 LoRA adapters trained on Mistral-7B (10 combo + 8 single-passage).
- LoRA: rank 128 on q_proj, v_proj, gate_proj, down_proj of the **last 2 transformer layers** (layers 30 and 31 of 32). Phase 47 used last 2 of 6; analogous choice on Mistral.
- All 18 adapters converged to loss < 0.16 in 5-22s each. Total training wall: 199s (~3 min).
- For inference, a single PEFT model wrapped at rank 512 (alpha 1024 → scaling 2.0). Adapters block-stacked-and-padded into the rank-512 slot per procedure (Phase 43 stacking; rank-128 single adapters padded with zeros up to 512). This makes RAG / Combo / Multi-stack / Multi-pass all use the same model — no model swapping per procedure.
- Decoding: greedy (1 seed). 30 generated tokens for single-content questions, 80 for cross-passage queries.
- Same scoring: substring match for Test 1; fragment coverage for Test 2.

## Procedures

- **A: RAG.** LoRA disabled (zeroed). Prompt = few-shot extractive prefix + concatenated constituent passages + question.
- **B: Combo.** Combo adapter loaded (rank-128 padded to rank-512 slot). Prompt = few-shot prefix + question (no passages).
- **C: Multi-stack.** K constituent single-passage adapters block-stacked at rank K×128 in the rank-512 slot (zero-padded for the rest). Prompt = few-shot prefix + question.
- **D: Multi-pass.** K passes — each loads one constituent adapter (combo-style prompt) and captures the first generated line as a note. Then a synthesis pass with adapter disabled and the K notes prepended to the question.

## Test 1: per-constituent retrieval

For each combination, ask the question for each constituent passage. K constituents × 1 question × 1 seed = K evals per (combination, procedure).

### Per-combination averages

| combination | K | RAG | Combo | Multi-stack | Multi-pass |
|---|---:|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 1.000 | 1.000 | 1.000 | 0.500 |
| P2_Estella_Drummle | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| P3_Magwitch_Provis | 2 | 1.000 | 1.000 | 0.500 | 0.500 |
| P4_Herbert_Wemmick | 2 | 1.000 | 1.000 | 0.500 | 1.000 |
| T1_Pip_Estella_Magwitch | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| T2_Joe_Estella_Drummle | 3 | 1.000 | 1.000 | 0.667 | 1.000 |
| T3_Pip_Herbert_Wemmick | 3 | 1.000 | 1.000 | 0.000 | 1.000 |
| T4_Magwitch_Provis_Drummle | 3 | 1.000 | 1.000 | 0.333 | 0.667 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 1.000 | 1.000 | 0.750 | 0.750 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 1.000 | 0.750 | 0.000 | 1.000 |

### Aggregated by K

| K | RAG | Combo | Multi-stack | Multi-pass |
|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 0.750 | 0.750 |
| 3 | 1.000 | 1.000 | 0.500 | 0.917 |
| 4 | 1.000 | 0.875 | 0.375 | 0.875 |

**Read:** RAG hits 100% across all K. Combo hits 100% at K=2/3, drops to 87.5% at K=4. Multi-stack collapses: 0.75 → 0.50 → 0.375 — replicating the K>2 ceiling observed on V22-Dickens. Multi-pass is reasonable: 0.75/0.92/0.875.

## Test 2: cross-passage queries (fragment coverage)

Hand-crafted Phase-43-style chained probes per combination, each requiring K answer fragments in one generation. 3 probes per combination × 1 seed.

### Per-combination averages

| combination | K | RAG | Combo | Multi-stack | Multi-pass |
|---|---:|---:|---:|---:|---:|
| P1_Pip_Joe | 2 | 1.000 | 1.000 | 0.167 | 0.500 |
| P2_Estella_Drummle | 2 | 1.000 | 0.667 | 0.500 | 0.667 |
| P3_Magwitch_Provis | 2 | 1.000 | 0.500 | 0.500 | 0.833 |
| P4_Herbert_Wemmick | 2 | 1.000 | 0.833 | 0.167 | 1.000 |
| T1_Pip_Estella_Magwitch | 3 | 0.889 | 0.667 | 0.222 | 0.889 |
| T2_Joe_Estella_Drummle | 3 | 0.889 | 0.778 | 0.444 | 0.667 |
| T3_Pip_Herbert_Wemmick | 3 | 0.889 | 0.556 | 0.000 | 0.444 |
| T4_Magwitch_Provis_Drummle | 3 | 0.667 | 0.889 | 0.333 | 0.556 |
| Q1_Pip_Estella_Magwitch_Provis | 4 | 0.917 | 0.917 | 0.083 | 0.417 |
| Q2_Pip_Joe_Herbert_Wemmick | 4 | 0.667 | 0.667 | 0.000 | 0.417 |

### Aggregated by K

| K | RAG | Combo | Multi-stack | Multi-pass |
|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.750 | 0.333 | 0.750 |
| 3 | 0.833 | 0.722 | 0.250 | 0.639 |
| 4 | 0.792 | 0.792 | 0.042 | 0.417 |

**Read:** RAG dominates cross-passage: 1.00 / 0.84 / 0.79. Combo is competitive at K=4 (0.79, tying RAG) but lags at K=2/3 (0.75 / 0.73 vs RAG's 1.00 / 0.84). Multi-stack catastrophically fails (0.34 → 0.25 → 0.04). Multi-pass holds 0.42-0.75 — better than multi-stack, worse than RAG and Combo.

## Test 3: scaling behavior

How does each approach scale with combination size?

- **RAG**: flat at 100% on Test 1; decreases slightly on Test 2 (1.00 → 0.84 → 0.79). Slight degradation as the context gets longer and the question requires more fragments.
- **Combo**: flat at 100% on Test 1 through K=3; small drop at K=4 (0.875). On Test 2 roughly flat at 0.73-0.79.
- **Multi-stack**: monotonic collapse with K — both tests. The K>2 ceiling is **structural**, not substrate-dependent. Same phenomenon as on V22-Dickens but playing out at higher absolute levels.
- **Multi-pass**: roughly flat on Test 1 (0.75-0.92); decreasing on Test 2 (0.75 → 0.64 → 0.42). The synthesis pass struggles with more notes to combine.

## Test 4: cost structure

Forward passes per query and dominant cost factor:

| Procedure | Passes/query | Context per pass | Notes |
|---|---:|---|---|
| RAG | 1 | few-shot + K passages + question (~150-300 tokens) | longest context |
| Combo | 1 | few-shot + question (~70 tokens) | shortest context |
| Multi-stack | 1 | few-shot + question (~70 tokens) | LoRA stack costs ~10MB extra at rank K×128 |
| Multi-pass | K+1 | per-pass: few-shot + question; final: + K notes | K+1× compute |

RAG and Combo have the same number of passes (1) but different context lengths. Multi-stack has 1 pass with short context but pays for a wider LoRA. Multi-pass has K+1× the inference cost.

Wall-clock of the eval (Mistral-7B, fp16, RTX 5070 Ti):
- Test 1 (24 questions × 4 procedures): 142s
- Test 2 (30 cross-pass queries × 4 procedures): 288s
- Total: 429s for 240 = 240 question-procedure evaluations.

## Sample outputs

### T3_Pip_Herbert_Wemmick (K=3, multi-stack catastrophic)

**Q ('Pirrip'):** `What was Pip's father's family name?`

- **RAG:** `Pirrip` → hit=True
- **Combo:** `Pirrip.` → hit=True
- **Multi-stack:** `was was was was was was was was was was was was was was was was was was was was was was was was was was was was was` → hit=False
- **Multi-pass:** `Pirrip` → hit=True

**Q ('Herbert'):** `Who was Pip's roommate at Barnard's Inn?`

- **RAG:** `Herbert Pocket` → hit=True
- **Combo:** `Herbert Pip's roommate was  Herbert. Herbert was the roommate of Pip. Herbert was the roommate of Pip'` → hit=True
- **Multi-stack:** `' was was was was was was was was was was was was was was was was was was was was was was was was was was was was was` → hit=False
- **Multi-pass:** `Herbert Pip's roommate was was was was was was was was was was was was was was was was was was was was was was was` → hit=True

### Q1_Pip_Estella_Magwitch_Provis (K=4 cross-pass, Combo ties RAG)

**Probe (frags = ['Pirrip', 'Estella', 'Magwitch', 'Provis']):**
`Four facts: Pip's family name was . Havisham's adopted daughter was . Pip's secr...`

- **RAG** (frac=1.00): `Pirrip, Estella, Magwitch, Provis

Passage: The cat was named Whiskers and lived in a small blue house on Elm Street.
Qu`
- **Combo** (frac=1.00): ` Pirrip,  Estella,  Magwitch,   Provis.

Question: Pip's secret benefactor was a convict from the marshes who later retu`
- **Multi-stack** (frac=0.00): `ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss`
- **Multi-pass** (frac=1.00): `Pirrip, Estella, Magwitch, Provis

Passage: The cat was named Whiskers and lived in a small blue house on Elm Street.
Qu`

## Honest assessment of the architecture's claims

**Claim 1: "Parameter-space absorption beats context-space access at comparable substrate."**
**Status: NOT SUPPORTED.** On Mistral-7B, RAG beats or ties combo at every K on every test. The 60-71pp combo advantage observed on V22-Dickens was artifact of V22's lack of in-context Q/A capability, not of a fundamental architectural advantage.

**Claim 2: "K>2 multi-adapter composition fails due to structural cross-term interference."**
**Status: SUPPORTED.** The multi-stack curve replicates on Mistral-7B: 0.75 / 0.50 / 0.375 single-content, 0.34 / 0.25 / 0.04 cross-passage. Same shape as V22-Dickens. The ceiling is structural, not substrate-dependent. Combo adapters or multi-pass are needed for K>2 if one is committed to using LoRA-style adapters.

**Claim 3: "Combination adapters provide a viable deployment pattern."**
**Status: SUPPORTED.** Combo adapters tie RAG on Test 1 at K=2/3 and lag only modestly at K=4. They tie RAG at K=4 cross-passage. The cost-structure advantage is real: combo's 1-pass with ~70-token context is cheaper than RAG's 1-pass with 150-300-token context, and is **much** cheaper than multi-pass's K+1-pass overhead.

**Claim 4: "The architecture provides capabilities RAG doesn't."**
**Status: NOT SUPPORTED on this scale.** RAG produces outputs at least as good as combo on every test. The architecture's value proposition is cost structure, not capability.

## Implications

**For the recruitment ask / article**: the architecture is defensible on cost-structure grounds. "Combination adapters retrieve content with shorter context than RAG, at comparable accuracy." This is honest and useful — adapters are 70 tokens vs RAG's 300+, ~3-4× cheaper at long-context inference. But **the article should not claim parameter-space absorption provides capabilities RAG doesn't**, because at scale where RAG works, it works as well as or better than the architecture.

**For deployment planning**: the architecture has three viable patterns:
1. K=1 single-passage adapters for individual content (if the user is confident about routing).
2. Combo adapters for known content groupings (comparable accuracy to RAG, cheaper inference).
3. RAG for novel combinations or when adapter training isn't feasible.

**Multi-stack is dead.** The K>2 collapse replicates on every substrate tested. It should be removed from the architecture's deployment toolkit.

**Multi-pass is OK but expensive.** It's a fallback when neither combo adapters exist nor RAG can fit the content in context. K+1× inference cost is the reason to prefer combo when possible.

## Wall-clock totals

| Stage | Wall |
|---|---:|
| Substrate probe (Mistral-7B Q/A check) | ~30s |
| Training (10 combo + 8 single Mistral adapters) | 199s (~3 min) |
| Four-procedure evaluation (Tests 1+2) | 429s (~7 min) |
| **Total** | **~10 min** |

## Summary

This experiment was designed to remove the substrate caveat that confounded prior comparisons on V22-Dickens. With Mistral-7B as a base capable of in-context extractive Q/A, the four-way comparison produces the most informative result so far:

- **RAG works.** 100% on per-constituent retrieval across K=2/3/4. The base correctly extracts answers from concatenated source passages.
- **Combo adapters work too.** 87.5-100% on per-constituent retrieval, comparable to RAG. They also tie RAG at K=4 cross-passage.
- **Multi-stack still fails at K>2.** This is the structural finding: the K>2 ceiling isn't a substrate artifact.
- **Multi-pass works but pays K+1× compute.**

The architecture's substitution claim — that parameter-space absorption is fundamentally different from in-context access — is **NOT supported** on a capable base. The article needs to honestly reframe the architecture's value proposition toward cost structure (combo adapters = shorter inference context, faster per-query) rather than fundamental capability advantage. That's a real and useful claim, but it's different from "absorption beats context."