# Hierarchical Compression and Corpus Dependency Structure

**Date:** 2026-05-07
**Headline:** **Hypothesis partially supported but non-monotonic.** Direction of the MI-vs-gap relationship is right (Spearman ρ = −0.50), with WT103 cleanly delivering a *negative* PPL gap (compressed beats baseline by 5.3%). But TS and code are out of MI-rank order: code has lower MI ratio (0.42) yet smaller gap (+9.6%) than TS (MI 0.66, gap +17.3%). The MI ratio captures something but isn't the full story; with n=3 corpora the experiment is underpowered for a clean test, and one likely confound (training-budget interaction with corpus size) plausibly accounts for WT103's dominant negative gap.

The user spec listed 5 corpora; this run uses 3 (TS, Python code from this repo, WT103) per scope-reduction discussed before launching. Each corpus × architecture × 3 seeds = 18 training runs. PubMed and ArXiv weren't included due to data acquisition cost in this session.

## Setup

Architecture per spec (with one deviation):
- d_model 256, n_heads 8, d_ff 1024, **n_layers 6** (per spec), learned pos_emb, dropout 0.1.
- Compression stack: 4× causal Conv1d(k=2, s=2) + LN + GELU = 16× total.
- Both architectures predict every 16th token (matched task).
- ctx 1024, **batch 8** (deviation: spec said 32, but the 16GB VRAM OOMed at 32 with these dims; 8 is the largest that fits).
- 5000 steps, AdamW lr=3e-4 cosine to 3e-5, warmup 200, weight decay 0.01, bf16.
- 3 seeds per condition.

Corpora:
- **ts** (Tiny Shakespeare): 304k GPT-2 BPE tokens; single-novel character-driven prose.
- **code**: Python from this HRS repo, 1.5M GPT-2 BPE tokens, 371 .py files, file-boundary markers.
- **wt103** (WikiText-103): 118M GPT-2 BPE tokens; encyclopedia text, varied topics.

## MI ratio metric

Per spec — empirical MI between (X_t, X_{t+k}) at distances k∈{1, 2, 4, 8, 16, 32, 64} on 200k random sample positions. Plug-in MLE estimator. Reported metric: MI(X_t; X_{t+16}) / MI(X_t; X_{t+1}).

| corpus | MI@1 (nats) | MI@16 (nats) | MI ratio MI@16/MI@1 |
|---|---|---|---|
| **code** | (highest local) | (low at 16) | **0.4152** |
| **ts** | (medium) | (medium) | **0.6612** |
| **wt103** | (lowest local) | (highest 16-relative) | **0.8322** |

Code has the highest MI@1-relative drop-off — local BPE bigrams are extremely informative (function/class boundaries, variable patterns), but at 16 tokens away the dependence weakens fast. WT103 has the *flattest* decay: distant tokens still carry meaningful information. TS is in between.

(Surprise vs my prior intuition: I expected TS to have the steepest local decay because of repetitive prose, but the BPE-tokenized version has more spread-out information than either code or wt103. The MI metric is grounded in the actual data, not in priors.)

## Training results — 3 seeds, per spec

| corpus | MI@16/1 | baseline PPL (mean ± std) | compressed PPL (mean ± std) | PPL gap (mean ± std) | gap % |
|---|---|---|---|---|---|
| ts | 0.6612 | 123.53 ± 3.58 | 144.91 ± 1.91 | +21.37 ± 1.81 | **+17.34% ± 1.92%** |
| code | 0.4152 | 17.51 ± 0.24 | 19.18 ± 0.07 | +1.67 ± 0.28 | **+9.58% ± 1.73%** |
| wt103 | 0.8322 | 403.51 ± 0.71 | 382.15 ± 0.91 | −21.36 ± 1.44 | **−5.29% ± 0.35%** |

Cross-corpus signal is much larger than seed noise:
- TS gap signal-to-noise: 17.34/1.92 ≈ 9
- Code gap S/N: 9.58/1.73 ≈ 5.5
- WT103 gap S/N: 5.29/0.35 ≈ 15

The gaps are real, not artifacts of single-seed variance.

### Coarsened-anchor accuracy (3-seed mean ± std)

| corpus | baseline | compressed |
|---|---|---|
| ts | 0.219 ± 0.056 | 0.219 ± 0.068 |
| code | 0.542 ± 0.033 | 0.568 ± 0.050 |
| wt103 | 0.208 ± 0.086 | 0.240 ± 0.079 |

**Compressed never loses on anchor accuracy** — TS ties, code and WT103 slightly favor compressed. The PPL gap on TS/code does NOT translate into anchor-accuracy degradation; the predicted-token-16-ahead task is unaffected by the architecture difference.

## Correlation analysis

Spearman ρ (MI ratio vs gap %) = **−0.50**. Pearson r = **−0.56**.

The predicted sign is negative (higher MI ratio → smaller gap). The observed sign IS negative. But the relationship is non-monotonic: TS has higher MI than code yet larger gap.

| | rank by MI | rank by gap (smaller = better) |
|---|---|---|
| code | 1 (lowest MI) | 2 |
| ts | 2 | 3 (largest gap) |
| wt103 | 3 (highest MI) | 1 (negative gap, best) |

Concordant pairs: (code, wt103). Discordant pairs: (code, ts), (ts, wt103). 1 concordant of 3 ⇒ Kendall's tau = (1 − 2)/3 = −0.33.

With n=3 corpora, neither correlation reaches significance. The test is underpowered for a strong claim. **Per spec falsification criteria:**
- "Five perplexity gaps showing no monotonic relationship" → we have 3, and they're non-monotonic. Mild evidence against the strict version of the hypothesis.
- "Run-to-run variance exceeding the cross-corpus signal" → not the case; cross-corpus signal is 5-15× the seed noise.
- "A confounding variable explaining the gap pattern better" → see below.

Figure: `figures/gap_vs_mi.png`.

## What the data actually suggest

Three observations the data licenses:

**1. WT103's negative gap is robust and surprising.** Across 3 seeds, compressed beats baseline on WT103 by 5.3% PPL (S/N = 15). This replicates from prior compression_conv work where compressed was within 0.6%; with longer training (5000 vs 2000 steps) the compressed model passes baseline. **At this training budget on a 118M-token corpus**, both models are far from converged. Compressed has slightly more parameters (18.4M vs 17.9M) but operates on T/16 = 64 attention positions, which may be easier to optimize at low compute. **The negative gap is plausibly a training-budget artifact**, not a pure architectural advantage. Repeating with 50k steps would test this.

**2. Code's gap is in the WRONG direction relative to MI.** Code has the lowest MI ratio (most locally regular) but a SMALLER gap than TS. Possible mechanism: the compression layer's strided convolution captures local syntactic patterns very efficiently — Python's tight syntactic structure (def/class headers, variable repetition, indentation) is exactly what 4 levels of conv-pooling can summarize. The MI ratio measures empirical pairwise dependence; what compression preserves is *learnable mixed signal at the kernel's scale*, which isn't the same thing.

**3. Anchor accuracy is unaffected.** Compressed matches or exceeds baseline on the every-16th-token prediction task across all 3 corpora, even where it loses on PPL. This is consistent with compression discarding *fine-grained next-token uncertainty* (which raises PPL) without losing *coarse-grained next-anchor signal* (which is what the architecture is built to predict). For applications where anchor-cadence prediction is the goal — e.g., long-context summarization, coarse generation — the PPL gap may be misleading.

## What this DOESN'T support

The strong version of the spec's hypothesis: **"the perplexity gap is a measurable function of corpus structure, with this specific predicted relationship to mutual information decay."** Three corpora, non-monotonic ordering. The hypothesis isn't refuted, but it isn't validated.

Possible alternative metrics that might capture the gap better:
- Perplexity of an n-gram model at varying n (more directly architecture-relevant than MI).
- A learnt-compression-specific metric: train a 1-layer compressor on the corpus and measure reconstruction error.
- Fact-density (fraction of probes that are local lookups vs synthesis) — the architecture's failure modes by fact type in the prior `k2_crossterms` and `trainable_decoder` experiments suggest compositional facts are most affected by interference of any kind.

## Per-spec deliverables

✓ Mean & std per (corpus, architecture, metric) — table above.
✓ Scatter plot of gap vs MI ratio with corpora labeled — `figures/gap_vs_mi.png`.
✓ Mutual information ratio computation — `mi_metric.py` + per-corpus json files.
✗ Five corpora — only three (TS, code, WT103) due to data-acquisition cost in this session.
✓ Three seeds per condition.

## Honest verdict

**The data show that the gap depends on the corpus, but not cleanly on the MI ratio at 16 alone.** WT103 exceeds prediction (negative gap, possibly training-budget artifact). Code and TS are non-monotonic in MI rank. Spearman ρ = −0.50 in the predicted direction but n=3 leaves the test inconclusive.

The architectural framing should be: **compression's gap depends on the corpus, in ways that the simplest single-distance MI metric only partially captures.** Stronger conclusions need either more corpora or a different structural metric — or both.

## Suggested follow-ups

In order of cost/value:

1. **Add 2 more corpora** (PubMed, ArXiv per spec, OR similar like NaturalQuestions or BookCorpus). Costs an afternoon of data prep + ~10 min training each. Resolves whether the 3-point non-monotonicity is real or a small-n artifact.

2. **Add longer-training runs on WT103** (e.g., 50k steps vs 5k). Tests whether the negative WT103 gap persists at convergence or was a low-budget artifact. This single comparison (one corpus, two budgets) would distinguish "compression genuinely helps WT103" from "compression is easier to train cheaply."

3. **Replace MI@16/MI@1 with a multi-distance MI feature** (e.g., area under MI(k) curve from k=1 to k=64). The single-distance ratio is the simplest summary but throws away information.

4. **Check a different structural metric**: bigram-model perplexity vs full-context perplexity on each corpus. The ratio (local-only PPL) / (long-context PPL) measures how much long-context buys you. Likely more directly architecture-relevant than MI.

If the hypothesis holds across 5+ corpora with the same metric, the architectural claim — that compression's value tracks measurable corpus structure — becomes deployable: you can predict whether compression will preserve quality on a new corpus from a one-pass MI estimate, without a full retraining sweep.

## Files

- `build_code_corpus.py` — Python code corpus builder
- `mi_metric.py` — MI ratio metric per spec
- `train_corpus.py` — single (corpus, variant, seed) training run
- `analyze.py` — aggregate + figure generation
- `data/code_corpus.pt` — cached Python code corpus (1.5M tokens)
- `results/mi_{ts,code,wt103}.json` — per-corpus MI metrics
- `results/train_{corpus}_{variant}_seed{0,1,2}.json` — per-run training summaries
- `results/aggregate.json` — final summary table + correlation
- `figures/gap_vs_mi.png` — scatter plot

Checkpoints (`checkpoints/*.pt`) gitignored.
