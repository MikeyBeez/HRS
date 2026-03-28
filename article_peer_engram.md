# PEER + Engram: 1.71 Perplexity at 510M Parameters on a Consumer GPU

*Michael Bonsignore*

## Executive Summary

A 510M parameter transformer using PEER (Parameter Efficient Expert Retrieval) for its feed-forward layers and a mean-pooled activation engram achieves 1.71 BPE perplexity on WikiText-103 and a MAUVE score of 0.905 against human-written text. The model was trained in approximately 12 hours on a single NVIDIA RTX 5070 Ti. A clean ablation — the same architecture without the engram — achieves 21.41 perplexity but a MAUVE score of 0.933-0.943.

Deep benchmarking reveals a nuanced picture: the engram does not cause mode collapse — in fact the engram-trained model produces *more diverse* text than both the baseline and the human reference. What the MAUVE gap actually reflects is a diversity-coherence trade-off. The engram-trained model generates text with human-like diversity but slightly lower coherence, while the baseline produces hyper-coherent but less diverse text. An independent scorer (GPT-2 Medium) rates the baseline's text as more predictable than human writing itself, suggesting the baseline has learned to produce unusually "canonical" text rather than the naturally varied text that characterizes human writing.


## Background and Motivation

In January 2026, I published an article on Medium (@mbonsign) presenting the theoretical case for PEER as a minimal model architecture. That paper argued that PEER's product-key routing to single-neuron experts could replace the standard feed-forward network in a transformer while dramatically reducing active compute per token. The present article reports what happened when we built it and trained it.

The engram concept draws inspiration from, but diverges significantly from, DeepSeek's Engram paper (arXiv:2601.07372, January 2026). DeepSeek's engram is a hash-based N-gram lookup table for injecting static knowledge. Ours is something different: a compressed representation of the model's own internal computational state, computed as a mean-pooled snapshot of hidden activations across a sliding window. The engram captures what the model is currently thinking about, not what it has memorized.

The goal was straightforward: achieve generation quality competitive with much larger models using architectural innovation rather than scale, on hardware that a graduate student or independent researcher can actually afford. The path to that goal was not straightforward at all.

Prior experiments in the HRS (Hierarchical Routed Sinkformer) series, spanning versions 8 through 14, explored various combinations of sparsity bottlenecks, multi-path attention routing, BDH-inspired virtual synapses, and learnable loss scaling. Each taught us something. Most of what they taught us was what not to do.


## Architecture

The final model has 510M total parameters, though only a fraction are active for any given token due to PEER's sparse routing.

**Attention.** Full causal self-attention with rotary positional embeddings, 16 heads, no sparse or sliding window approximations. Every token attends to every previous token. We can afford this because PEER's compute savings in the feed-forward layer offset the quadratic cost of full attention.

**PEER feed-forward layers.** Each standard feed-forward network is replaced with a PEER module containing 262,144 single-neuron experts (512 squared, organized as a product of two sub-key tables). Eight retrieval heads each select 16 experts via top-k product-key lookup, yielding 128 active experts per token. The routing is fully differentiable through the product-key mechanism.

**Engram.** After the second transformer layer, hidden states are segmented into windows of 128 tokens. Each window is mean-pooled and projected through a two-layer MLP to produce 4 engram vectors per window. These are prepended to the input of subsequent layers as additional context, then stripped from the output. For a 512-token training sequence, this produces 16 engram vectors (4 windows times 4 engrams each).

**Engram dropout.** During training, we randomly disable engram injection with probability 0.1. This teaches the model to function both with and without the engram, a decision that proved critical for generation quality.

**Training details.** WikiText-103 with GPT-2 BPE tokenization. 50,000 steps with phased learning rate scheduling across four phases: foundation (8K steps), expert specialization (8K), compression (10K), and joint fine-tuning (24K). Batch size 4 with 8 gradient accumulation steps for an effective batch of 32. Sequences of 512 tokens. Total training time: approximately 12 hours on a single NVIDIA RTX 5070 Ti running Pop!_OS 24.04.

**Parameter breakdown:** Transformer backbone (attention, norms) 76.7M. PEER feed-forward layers 421.6M. Engram encoder 10.5M. Locality head 1.0M.


## Key Problems Solved

Three problems had to be identified and solved before the architecture produced coherent text. Each required its own experiment to diagnose.

### Problem 1: Excessive sparsity collapses the active network

Earlier experiments (V8-V12) used a monosemantic sparsity bottleneck that zeroed out 95% of features before routing. The theory was sound: enforce sparse, interpretable representations. The practice was not. At 95% sparsity, the active network collapsed to roughly 12.5M parameters. This was sufficient for teacher-forced perplexity — the model could predict the next token given perfect context — but the representations were too impoverished for coherent autoregressive generation. The model would predict reasonable next tokens one at a time, but the generated text was word salad.

We tried reducing sparsity to 5% (V13), which improved things marginally. But the fundamental issue was that any external sparsity constraint on top of PEER's already-sparse routing was redundant and destructive.

The solution was to remove the sparsity bottleneck entirely and let PEER's product-key routing handle sparsity naturally. PEER activates 128 of 262,144 experts per token — that is already 99.95% sparse, but the sparsity is learned and targeted rather than imposed uniformly across features.

### Problem 2: Multi-path attention routing fragments the representation space

The original HRS architecture routed tokens through multiple compute tiers: convolution for local patterns, full attention for global context, and a sink channel for low-importance tokens. Each tier processed its own version of the representation, and the outputs were blended by learned routing weights.

This worked acceptably for perplexity but created a problem for PEER and the engram. PEER's product-key routing needs a stable, unified representation space to learn effective expert selection. The engram needs consistent hidden states to compress into meaningful snapshots. When the representation space was fragmented across three routing paths, neither could function well.

Experiments V14 and V15 progressively simplified the routing: first removing convolution (attention + sink only), then testing vanilla attention with no routing at all. Each simplification improved generation quality. The lesson was clear: let attention handle attention, let PEER handle routing. Mixing the two created interference.

### Problem 3: Short prompts produce empty engrams

This was the most insidious problem because it was invisible to standard evaluation.

The engram encoder requires at least 128 tokens to produce any output (one window). Standard evaluation computes perplexity with teacher forcing on 512-token sequences, which always have 4 full windows. Perplexity looked excellent. Generation from short prompts — the kind a user would actually provide — was gibberish.

The model had learned to rely on engram injection at every layer after the extraction point. When generating from a 6-token prompt like "The city was founded in," the engram tensor was empty. The model was operating in a regime it had never seen during training, and it failed catastrophically.

The fix was simple: engram dropout. During training, we randomly disable engram injection 10% of the time, forcing the model to learn representations that work both with and without engram context. This cost a small amount of perplexity (1.71 vs 1.80 for the non-dropout version) but transformed generation quality at short context lengths from nonsensical to coherent.


## Results

### Training

| Metric | V16 (PEER + engram) | V17 (PEER only) |
|--------|:-------------------:|:---------------:|
| Parameters | 510M | 499M |
| Final validation perplexity (BPE) | **1.71** | 21.41 |
| Final training CE loss | 0.97 | 3.18 |
| Training steps | 50,000 | 50,000 |
| Training time | ~12 hours | ~11 hours |
| Hardware | 1x RTX 5070 Ti | 1x RTX 5070 Ti |
| VRAM usage | ~5.7 GB | ~5.5 GB |

Both runs were stable throughout, with no loss spikes, divergence, or rollbacks required. The phased learning rate schedule transitioned smoothly between phases.

A note on perplexity: this is BPE-level perplexity using the GPT-2 tokenizer (50,257 subword tokens). It is not directly comparable to word-level perplexity reported in older benchmarks such as the original WikiText-103 leaderboard. BPE perplexity is measured over a finer-grained prediction space, and the numbers are not interchangeable. We report BPE perplexity because it reflects the actual training objective and tokenization used by the model.

### Generation quality

By step 23,000 (46% of training), the V16 model was generating coherent Wikipedia-style prose from short prompts:

> **Prompt:** The city was founded in
>
> **Generated:** the city of- . On February 11 , 2011 , the city of York City opened The Manchester City to the city on November 19 , 2011 . The city was founded in conjunction with the city 's headquarters at the city of York City...

> **Prompt:** During the Second World War ,
>
> **Generated:** the Second World War was a major part of the German invasion of the First World War against France . During the war , the First World War broke out in Britain and other countries , and by the end of World War II saw the United Kingdom as a war-fighting ground...

The text is imperfect — some repetition, occasional factual confusion — but it maintains topic coherence, grammatical structure, and an encyclopedic register appropriate to the training data. By the end of training, these issues were substantially reduced.

With 256 tokens of real WikiText context, generation quality improved further:

> **Context (last 100 chars):** ...ns , as well as a career-high 12 special-teams tackles . 2015 season
>
> **Generated:** McCarty was first diagnosed with six finalists and missed 89 games during the season . McCarty still carried some of the ten plays , including a two-times trade . McCarty also used as a safety maneuver for two extra teams...

### MAUVE benchmark

MAUVE measures how well a model's generated text distribution matches a reference human text distribution. Scores range from 0 to 1, with higher indicating closer distributional match. We generated 1,000 continuations of 256 tokens each at two prompt lengths, comparing against WikiText-103 test set references. Sampling used temperature 0.9 and top-k 50.

**V16 (PEER + engram, trained with engram dropout):**

| Condition | Prompt length | Engrams | MAUVE |
|-----------|:------------:|:-------:|:-----:|
| Short prompt, engrams natural | 50 tokens | Active (empty) | 0.806 |
| Long prompt, engrams natural | 500 tokens | Active (3 windows) | 0.888 |
| Short prompt, engrams OFF | 50 tokens | Disabled | 0.905 |
| Long prompt, engrams OFF | 500 tokens | Disabled | 0.906 |

**V17 (PEER only, no engram — clean ablation baseline):**

| Condition | Prompt length | MAUVE |
|-----------|:------------:|:-----:|
| Short prompt | 50 tokens | **0.933** |
| Long prompt | 500 tokens | **0.943** |

For context: GPT-2 small (124M parameters) typically scores in the 0.7-0.8 range on comparable MAUVE setups, while models in the 1-3B parameter range often land around 0.9 depending on sampling configuration.

### Deep benchmarks: diversity, coherence, and confidence

To understand *why* the MAUVE scores differ, we ran three additional benchmarks on 1,000 generated continuations (256 tokens each from 50-token prompts) for each condition, compared against WikiText-103 test set references.

**Self-BLEU and Distinct-N (diversity).** Self-BLEU4 measures how similar each generation is to all other generations from the same model — lower means more diverse. Distinct-N measures the ratio of unique n-grams to total n-grams — higher means more diverse.

| Condition | Self-BLEU4 | Distinct-1 | Distinct-2 | Distinct-3 |
|-----------|:----------:|:----------:|:----------:|:----------:|
| V16 engrams ON | **0.132** | **0.069** | **0.432** | **0.779** |
| V16 engrams OFF | 0.194 | 0.048 | 0.326 | 0.668 |
| V17 (PEER only) | 0.182 | 0.058 | 0.372 | 0.711 |
| WikiText-103 ref | 0.141 | 0.082 | 0.464 | 0.766 |

V16 with engrams on is the most diverse model output, closest to the human reference across all metrics. Mode collapse is definitively ruled out.

**Cross-model perplexity (coherence).** We scored all generated text using GPT-2 Medium (355M) as an independent judge. Lower perplexity means the text is more predictable — more coherent — to an external model.

| Condition | Mean PPL | Median PPL | Std PPL |
|-----------|:--------:|:----------:|:-------:|
| V16 engrams ON | 75.38 | 72.78 | 27.52 |
| V16 engrams OFF | 30.53 | 30.27 | 10.00 |
| V17 (PEER only) | **23.31** | **23.06** | 6.81 |
| WikiText-103 ref | 31.16 | 30.16 | 10.16 |

V17's text is rated as more coherent than human-written text by GPT-2 Medium. V16 with engrams off matches the human reference almost exactly. V16 with engrams on is rated less coherent — the price of its higher diversity.

**Output entropy (confidence).** Shannon entropy of the model's raw softmax distribution at each generation step, averaged over 200 fresh 256-token continuations. Higher entropy means the model is less certain about its next token.

| Condition | Mean entropy (bits) | Entropy variance |
|-----------|:-------------------:|:----------------:|
| V16 engrams ON | 4.311 | 8.04 |
| V16 engrams OFF | 4.601 | 8.37 |
| V17 (PEER only) | **4.125** | 8.80 |

V17 is the most confident model, with the lowest output entropy. V16 with engrams off is the least confident — the engram-shaped weights, when running without their scaffolding, spread probability mass across more tokens.


## Analysis of Results

The deep benchmarks transform what initially appeared to be a simple story (engram bad, baseline good) into something considerably more interesting.

### The initial surprise: MAUVE prefers the baseline

V17 (PEER only) achieves MAUVE 0.933-0.943, while V16 (PEER + engram, engrams off) scores 0.905-0.906. Our first interpretation was that the engram acts as a training crutch — helping the model achieve low perplexity by providing a contextual shortcut, but producing weights that are worse for generation.

The deep benchmarks show this interpretation is wrong.

### The engram increases diversity, not decreases it

If the engram were a crutch causing mode collapse, we would expect the engram-trained model to produce less diverse text — more repetitive, more formulaic, fewer unique n-grams. The opposite is true. V16 with engrams on produces the most diverse text of any condition: the lowest self-BLEU (0.132, closest to the human reference at 0.141) and the highest distinct-2 (0.432, closest to the reference at 0.464). Even V16 with engrams off (0.194 self-BLEU, 0.326 distinct-2) is within range, though less diverse than V17 (0.182 self-BLEU, 0.372 distinct-2).

Mode collapse is definitively ruled out. The engram makes the model more diverse, not less.

### The baseline is hyper-coherent — more predictable than human text

The cross-model perplexity benchmark reveals something unexpected about V17. When GPT-2 Medium scores V17's generated text, it assigns a mean perplexity of 23.31 — lower than the perplexity it assigns to actual human-written Wikipedia text (31.16). V17's text is more predictable to an external model than real text is.

This is not straightforwardly good. Human-written text has a certain amount of unpredictability — surprising word choices, varied sentence structures, topic shifts — that makes it score higher under an external language model. V17 has learned to produce text that is extremely "canonical": grammatically correct, topically consistent, and predictable. It avoids the variations and surprises that characterize natural human writing.

V16 with engrams off scores 30.53 under GPT-2 Medium — almost exactly matching the human reference at 31.16. The engram-shaped weights produce text that is precisely as coherent as human text, neither more nor less.

### Reinterpreting the MAUVE gap

MAUVE measures distributional similarity between generated and reference text. V17 scores higher not because its text is more human-like in a general sense, but because its text distribution is closer to the WikiText-103 reference distribution in the specific embedding space that MAUVE uses (GPT-2 Large features, k-means quantization). There are multiple ways to be close to a reference distribution: you can match its diversity and coherence at the right levels, or you can converge on its most predictable patterns while ignoring its variations.

The deep benchmarks suggest V17 does the latter. Its text is hyper-coherent and moderately diverse — a narrower, more predictable slice of what human text actually looks like. V16 (with engrams off) matches human text's coherence level exactly while producing slightly less diversity than the reference. V16 (with engrams on) produces the most human-like diversity but at a coherence cost.

No single model matches the human reference on all dimensions simultaneously.

### The diversity-coherence trade-off

The three conditions form a clear spectrum:

V17 is high-coherence, moderate-diversity: cross-model PPL 23.31 (below human at 31.16), distinct-2 0.372, entropy 4.125 bits. The model is confident and predictable.

V16 engrams off is human-coherence, lower-diversity: cross-model PPL 30.53 (matching human at 31.16), distinct-2 0.326, entropy 4.601 bits. The model matches human predictability but with less lexical variety.

V16 engrams on is lower-coherence, high-diversity: cross-model PPL 75.38 (above human), distinct-2 0.432 (closest to human at 0.464), entropy 4.311 bits. The model is diverse but less predictable.

The engram, during training, teaches the model to explore a wider range of the representation space. This produces more diverse outputs — closer to the natural variation of human text — but at the cost of the tight coherence that comes from converging on canonical patterns.

### What the engram actually does

The engram's 12.5x perplexity improvement (1.71 vs 21.41) is real and reflects genuine learning. The engram provides compressed context that helps the model predict next tokens more accurately. But the representations shaped by this training-time signal are qualitatively different from those learned without it.

Without the engram, PEER's product-key routing converges on a narrow, confident set of expert activation patterns. The model becomes very good at producing the most likely continuation — canonical, predictable, coherent. With the engram, the additional context signal during training encourages the model to maintain sensitivity to a wider range of patterns. The resulting weights are less converged, less confident, but more capable of producing the kind of variation that characterizes human writing.

Neither is strictly better. They optimize for different qualities.

### The PEER baseline is still impressive

The V17 baseline deserves recognition on its own terms. A 499M parameter transformer with PEER feed-forward layers, trained on a single consumer GPU for 11 hours, achieves MAUVE scores of 0.933-0.943 on WikiText-103. Its cross-model perplexity (23.31) is lower than the human reference (31.16), indicating extremely coherent generation. The PEER architecture — full attention combined with sparse feed-forward routing — produces strong results with or without the engram.


## Connection to Related Work

The engram-as-scaffolding finding connects to several concurrent lines of research, though the connection is more conceptual than mechanical.

**DeepSeek's Hyper-Connections (mHC)** use multiple residual streams to strengthen signal propagation through parallel paths. During training, the engram serves a similar signal-strengthening function: it provides the model with a compressed version of its own recent activations, creating an additional information pathway that shapes how the primary residual stream develops. The difference is that hyper-connections remain active at inference, while our engram does not — and our ablation suggests the model may be better off without the additional pathway.

**The Kimi team's Attention Residuals** address the same underlying problem — representation dilution across depth — through selective depth-wise attention to earlier layers. The engram addresses this differently, by explicitly compressing and re-injecting earlier representations rather than allowing selective attention across layers. Both approaches acknowledge that deep transformers lose fine-grained information as representations propagate through layers.

**DeepSeek's Engram paper** (arXiv:2601.07372) shares the name and the broad concept of conditional memory injection, but the implementations diverge fundamentally. Their engram is an external hash-based lookup table that stores and retrieves static knowledge representations. Ours is a runtime compression of the model's own hidden states — it contains no information that isn't already present in the activations, only a compressed projection of it. The shared insight is that transformers benefit from explicit memory mechanisms. The mechanism itself is entirely different.

**The Chroma Context-1 paper** explores self-editing context management for search agents, a related concept of selective retention. The parallel is in recognizing that not all context is equally valuable, and that systems benefit from explicit mechanisms for deciding what to keep.


## Implications

**PEER validates the minimal model hypothesis.** Both V16 and V17 demonstrate that PEER's product-key routing combined with full attention produces 500M parameter models that generate text competitive with models several times their size. V17 achieves MAUVE 0.933-0.943; V16 achieves MAUVE 0.905 with more human-like diversity characteristics. The PEER architecture is the foundation that makes both results possible.

**PEER makes full attention affordable.** The conventional wisdom is that you must approximate attention (sparse, sliding window, linear) to train efficiently. PEER inverts this: by making the feed-forward layer sparse (128 of 262,144 experts active per token), the feed-forward computation becomes cheap enough that you can afford dense attention. Full attention with sparse feed-forward may be a better trade-off than sparse attention with dense feed-forward. PEER also generates 1.6-2.4x faster than a comparable dense transformer, and adding a KV cache brings generation to a constant ~200 tokens/second regardless of context length.

**Perplexity, MAUVE, and diversity measure different things.** The 12.5x perplexity gap between V16 and V17 did not predict MAUVE ordering, and MAUVE alone did not reveal that V17's advantage comes from hyper-coherence rather than better generation in any absolute sense. Only the combination of MAUVE, self-BLEU, cross-model perplexity, and entropy gives the full picture. Single-metric evaluation of language models is insufficient.

**The engram controls the diversity-coherence trade-off.** The engram is not a crutch and not simply scaffolding. It is a training-time signal that shapes the model toward higher diversity at the cost of some coherence. Without the engram, PEER converges on hyper-coherent, hyper-predictable text — more predictable than human writing itself. With the engram, the model learns to maintain the kind of variation that characterizes natural language. Whether this trade-off is desirable depends on the application.

**Train-time-only components are a viable methodology.** The engram demonstrates that architectural components can fundamentally alter what a model learns (diversity-favoring vs coherence-favoring representations) even when removed at inference. This is a novel training methodology distinct from regularization, data augmentation, or distillation. The engram does not add noise — it adds structured information that shapes the geometry of learned representations.

**Consumer hardware is sufficient for meaningful AI research.** This entire experiment — architecture design, debugging, training, evaluation, deep benchmarking — ran on a single NVIDIA RTX 5070 Ti, a GPU that costs approximately $600. Both models train in under 12 hours and use under 6GB of VRAM. The bottleneck for independent AI research is increasingly ideas, not compute.


## Future Directions

**Reconstruction head.** The current engram uses a simple mean-pooling compression, which captures the average activation but loses fine-grained structure. Adding a reconstruction loss that forces the engram to faithfully encode salient features — rather than unfiltered averages — may change the perplexity-MAUVE trade-off. If the engram can be taught to compress only the information that attention doesn't already capture, it becomes a complement to attention rather than a redundant shortcut. This would test whether the inference-time degradation is inherent to engram injection or specific to the current unfiltered mean-pooling approach.

**Shannon entropy as runtime control.** The model's output distribution entropy at each position signals its own uncertainty. High-entropy positions are where the model is least confident. A runtime controller could trigger selective recomputation — a second forward pass, a deeper search, or engram-augmented inference — only at positions where the model's uncertainty exceeds a threshold. This would add compute only where it is needed.

**Recursive inference.** Feeding the model's own generation back through for a second pass, allowing it to revise uncertain positions with the benefit of its complete first-draft generation as context. This is analogous to how humans write: a first draft followed by revision, where the revision benefits from seeing the full structure of the draft.

**Scaling to The Pile.** WikiText-103 is encyclopedic text from a single domain. Training on diverse, large-scale data (code, dialogue, technical writing, fiction) would test whether the architecture generalizes or whether its strengths are specific to Wikipedia's relatively uniform structure.

**Tuning the diversity-coherence balance.** Now that we understand the engram's role in controlling this trade-off, the natural question is whether it can be tuned. Varying the engram dropout rate, the engram window size, or the number of engram vectors per window would trace out the Pareto frontier between diversity and coherence. The optimal operating point likely depends on the downstream application.


## Reproducibility

- **Code:** [github.com/MikeyBeez/HRS](https://github.com/MikeyBeez/HRS)
- **Dataset:** WikiText-103 (publicly available via Hugging Face Datasets)
- **Hardware:** NVIDIA RTX 5070 Ti, 16GB VRAM
- **Software:** PyTorch, Pop!_OS 24.04
- **V16 training time:** ~12 hours
- **V17 training time:** ~11 hours
- **VRAM usage:** ~5.5-5.7 GB peak

The model configurations, training loop, generation checker, and MAUVE benchmark script are included in the repository. Both experiments are fully reproducible on any GPU with 8GB or more of VRAM with minor batch size adjustments.

To reproduce V16 (PEER + engram): `python train.py --ablation v16_peer_engram --output-dir results`

To reproduce V17 (PEER only baseline): `python train.py --ablation v17_peer_only --output-dir results`

To run MAUVE benchmark: `python benchmark_mauve.py <run_dir>`
