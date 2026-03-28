# PEER + Engram: 1.71 Perplexity at 510M Parameters on a Consumer GPU

*Michael Bonsignore*

## Executive Summary

A 510M parameter transformer using PEER (Parameter Efficient Expert Retrieval) for its feed-forward layers and a mean-pooled activation engram achieves 1.71 BPE perplexity on WikiText-103 and a MAUVE score of 0.905 against human-written text. The model was trained in approximately 12 hours on a single NVIDIA RTX 5070 Ti. A clean ablation — the same architecture without the engram — achieves 21.41 perplexity but a MAUVE score of 0.933-0.943, revealing a striking dissociation between next-token prediction quality and distributional generation quality. The engram dramatically improves perplexity but the baseline PEER model produces text that better matches the human reference distribution. This suggests a broader methodology — train-time-only architectural components — that may generalize beyond this specific architecture, but also raises important questions about what perplexity actually measures.


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


## Analysis of Results

The results reveal two findings we did not expect, and one we did.

### Finding 1: The engram dramatically improves perplexity

This was expected. The engram-trained V16 model achieves 1.71 BPE perplexity versus 21.41 for the V17 baseline — a 12.5x improvement. The engram provides compressed context that helps the model predict next tokens more accurately during teacher-forced evaluation. This is a large, unambiguous effect.

### Finding 2: The engram hurts inference generation quality

This was not expected. With engrams disabled at inference, V16 scores MAUVE 0.905-0.906. With engrams active, scores drop to 0.806-0.888. We expected the engram to help generation, not hurt it. The data showed otherwise.

### Finding 3: The baseline PEER model generates better text than the engram-trained model

This was the most surprising finding. The V17 baseline — trained without any engram — achieves MAUVE 0.933-0.943, substantially higher than even the best V16 score (0.906 with engrams off). A model with 12.5x worse perplexity produces text that is distributionally closer to human writing.

### What does this mean?

The dissociation between perplexity and MAUVE forces us to reconsider what the engram is actually doing.

**Perplexity measures next-token prediction accuracy.** The engram provides a compressed summary of recent context that helps the model predict the specific next token that appeared in the training data. This is a form of memorization-adjacent behavior — the engram provides a shortcut for recalling training-distribution patterns, yielding very low perplexity.

**MAUVE measures distributional match of generated text.** It asks: does the text this model generates, taken as a whole, look like it came from the same distribution as human-written text? This captures qualities like diversity, naturalness of phrasing, and avoidance of degenerate patterns (repetition, mode collapse).

The engram appears to optimize for the first at the expense of the second. By providing a compressed echo of recent activations, it narrows the model's predictions toward high-confidence, low-perplexity outputs. This is excellent for predicting exactly which token comes next in a held-out test sequence. But it makes the model's generation distribution narrower and less diverse than human text.

The V17 baseline, without this compression shortcut, learns more robust representations. It cannot rely on the engram to recall recent context, so it must encode that information more fully in its weights and attention patterns. The result is worse per-token prediction but more human-like generation — the model has learned the distribution rather than the specific sequences.

**The scaffolding metaphor needs revision.** Our initial interpretation — that the engram shapes better representations during training — is contradicted by the V17 ablation. If the engram-shaped representations were genuinely better, then V16 with engrams off should outperform V17 (which never had the engram's shaping influence). It does not. V17 scores 0.933-0.943 versus V16's 0.905-0.906.

A more accurate interpretation: the engram is a training-time crutch. It helps the model achieve low perplexity by providing a contextual shortcut, but the weights learned in the presence of this shortcut are *worse* for generation than weights learned without it. The model learns to lean on the engram rather than developing fully self-sufficient representations.

This does not mean the engram is useless. It means its value lies in a different direction than we assumed. The engram creates a model that is excellent at next-token prediction — which may be the right objective for applications like scoring, ranking, or fill-in-the-blank tasks. For open-ended generation, the simpler PEER-only architecture produces better results.

### The PEER baseline is the real story

Perhaps the most important result is the V17 baseline itself. A 499M parameter transformer with PEER feed-forward layers, trained on a single consumer GPU for 11 hours, achieves MAUVE scores of 0.933-0.943 on WikiText-103. This places it in the range of models several times its size. The PEER architecture — full attention combined with sparse feed-forward routing — is doing the heavy lifting, not the engram.


## Connection to Related Work

The engram-as-scaffolding finding connects to several concurrent lines of research, though the connection is more conceptual than mechanical.

**DeepSeek's Hyper-Connections (mHC)** use multiple residual streams to strengthen signal propagation through parallel paths. During training, the engram serves a similar signal-strengthening function: it provides the model with a compressed version of its own recent activations, creating an additional information pathway that shapes how the primary residual stream develops. The difference is that hyper-connections remain active at inference, while our engram does not — and our ablation suggests the model may be better off without the additional pathway.

**The Kimi team's Attention Residuals** address the same underlying problem — representation dilution across depth — through selective depth-wise attention to earlier layers. The engram addresses this differently, by explicitly compressing and re-injecting earlier representations rather than allowing selective attention across layers. Both approaches acknowledge that deep transformers lose fine-grained information as representations propagate through layers.

**DeepSeek's Engram paper** (arXiv:2601.07372) shares the name and the broad concept of conditional memory injection, but the implementations diverge fundamentally. Their engram is an external hash-based lookup table that stores and retrieves static knowledge representations. Ours is a runtime compression of the model's own hidden states — it contains no information that isn't already present in the activations, only a compressed projection of it. The shared insight is that transformers benefit from explicit memory mechanisms. The mechanism itself is entirely different.

**The Chroma Context-1 paper** explores self-editing context management for search agents, a related concept of selective retention. The parallel is in recognizing that not all context is equally valuable, and that systems benefit from explicit mechanisms for deciding what to keep.


## Implications

**PEER is the headline result, not the engram.** The V17 baseline demonstrates that PEER's product-key routing combined with full attention produces a 499M parameter model that generates text competitive with models several times its size (MAUVE 0.933-0.943). This validates the minimal model hypothesis from the January 2026 PEER paper.

**PEER makes full attention affordable.** The conventional wisdom is that you must approximate attention (sparse, sliding window, linear) to train efficiently. PEER inverts this: by making the feed-forward layer sparse (128 of 262,144 experts active per token), the feed-forward computation becomes cheap enough that you can afford dense attention. Full attention with sparse feed-forward may be a better trade-off than sparse attention with dense feed-forward.

**Perplexity and generation quality can diverge dramatically.** The 12.5x perplexity difference between V16 and V17 did not translate to better generation — it translated to worse generation. This has implications for how we evaluate language models. Perplexity remains useful as a training signal, but it should not be treated as a proxy for generation quality. MAUVE or similar distributional measures provide complementary and sometimes contradictory information.

**Train-time-only components remain an interesting methodology, but with caveats.** The engram demonstrates that architectural components can dramatically alter training dynamics (1.71 vs 21.41 perplexity). Whether this is useful depends on the application. For tasks that benefit from strong next-token prediction (scoring, ranking, classification), the engram-trained model may be superior. For open-ended generation, the baseline is better. The broader pattern — add structure to shape learning, remove it once the weights internalize the constraint — remains promising but requires careful evaluation of what the structure actually teaches the model.

**Consumer hardware is sufficient for meaningful AI research.** This entire experiment — architecture design, debugging, training, evaluation, ablation — ran on a single NVIDIA RTX 5070 Ti, a GPU that costs approximately $600. Both models train in under 12 hours and use under 6GB of VRAM. The bottleneck for independent AI research is increasingly ideas, not compute.


## Future Directions

**Reconstruction head.** The current engram uses a simple mean-pooling compression, which captures the average activation but loses fine-grained structure. Adding a reconstruction loss that forces the engram to faithfully encode salient features — rather than unfiltered averages — may change the perplexity-MAUVE trade-off. If the engram can be taught to compress only the information that attention doesn't already capture, it becomes a complement to attention rather than a redundant shortcut. This would test whether the inference-time degradation is inherent to engram injection or specific to the current unfiltered mean-pooling approach.

**Shannon entropy as runtime control.** The model's output distribution entropy at each position signals its own uncertainty. High-entropy positions are where the model is least confident. A runtime controller could trigger selective recomputation — a second forward pass, a deeper search, or engram-augmented inference — only at positions where the model's uncertainty exceeds a threshold. This would add compute only where it is needed.

**Recursive inference.** Feeding the model's own generation back through for a second pass, allowing it to revise uncertain positions with the benefit of its complete first-draft generation as context. This is analogous to how humans write: a first draft followed by revision, where the revision benefits from seeing the full structure of the draft.

**Scaling to The Pile.** WikiText-103 is encyclopedic text from a single domain. Training on diverse, large-scale data (code, dialogue, technical writing, fiction) would test whether the architecture generalizes or whether its strengths are specific to Wikipedia's relatively uniform structure.

**Understanding the perplexity-MAUVE dissociation.** The most pressing open question is why the engram improves perplexity so dramatically while degrading generation quality. A deeper investigation — examining the entropy of the output distributions, the diversity of generated text, and the degree to which the engram encourages mode collapse — would clarify whether this is a fundamental trade-off or an artifact of the current engram design.


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
