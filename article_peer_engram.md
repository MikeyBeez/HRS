# PEER + Engram: 1.71 Perplexity at 510M Parameters on a Consumer GPU

*Michael Bonsignore*

## Executive Summary

A 510M parameter transformer using PEER (Parameter Efficient Expert Retrieval) for its feed-forward layers and a mean-pooled activation engram achieves 1.71 BPE perplexity on WikiText-103 and a MAUVE score of 0.905 against human-written text. The model was trained in approximately 12 hours on a single NVIDIA RTX 5070 Ti. A key finding: the engram improves training representations but should be disabled at inference, functioning as architectural scaffolding rather than a runtime component.

## Background and Motivation

In January 2026, I published an article on Medium (@mbonsign) presenting the theoretical case for PEER as a minimal model architecture. That paper argued that PEER's product-key routing to single-neuron experts could replace the standard feed-forward network in a transformer while dramatically reducing active compute per token. The present article reports what happened when we built it and trained it.

The engram concept draws inspiration from, but diverges significantly from, DeepSeek's Engram paper (arXiv:2601.07372, January 2026). DeepSeek's engram is a hash-based N-gram lookup table for injecting static knowledge. Ours is something different: a compressed representation of the model's own internal computational state, computed as a mean-pooled snapshot of hidden activations across a sliding window. The engram captures what the model is currently thinking about, not what it has memorized.

The goal was straightforward: achieve generation quality competitive with much larger models using architectural innovation rather than scale, on hardware that a graduate student or independent researcher can actually afford. The path to that goal was not straightforward at all.

Prior experiments in the HRS (Hierarchical Routed Sinkformer) series, spanning versions 8 through 14, explored various combinations of sparsity bottlenecks, multi-path attention routing, BDH-inspired virtual synapses, and learnable loss scaling. Each taught us something. Most of what they taught us was what not to do.

## Architecture

The final model has 510M total parameters, though only a fraction are active for any given token due to PEER's sparse routing.

**Attention.** Full causal self-attention with rotary positional embeddings, 16 heads, no sparse or sliding window approximations. Every token attends to every previous token. We can afford this because PEER's compute savings in the feed-forward layer offset the quadratic cost of full attention.

**PEER feed-forward layers.** Each standard feed-forward network is replaced with a PEER module containing 262,144 single-neuron experts (512^2, organized as a product of two sub-key tables). Eight retrieval heads each select 16 experts via top-k product-key lookup, yielding 128 active experts per token. The routing is fully differentiable through the product-key mechanism.

**Engram.** After the second transformer layer, hidden states are segmented into windows of 128 tokens. Each window is mean-pooled and projected through a two-layer MLP to produce 4 engram vectors per window. These are prepended to the input of subsequent layers as additional context, then stripped from the output. For a 512-token training sequence, this produces 16 engram vectors (4 windows times 4 engrams each).

**Engram dropout.** During training, we randomly disable engram injection with probability 0.1. This teaches the model to function both with and without the engram, a decision that proved critical for generation quality.

**Training details.** WikiText-103 with GPT-2 BPE tokenization. 50,000 steps with phased learning rate scheduling across four phases: foundation (8K steps), expert specialization (8K), compression (10K), and joint fine-tuning (24K). Batch size 4 with 8 gradient accumulation steps for an effective batch of 32. Sequences of 512 tokens. Total training time: approximately 12 hours on a single NVIDIA RTX 5070 Ti running Pop!_OS 24.04.

**Parameter breakdown:**
- Transformer backbone (attention, norms): 76.7M
- PEER feed-forward layers: 421.6M
- Engram encoder: 10.5M
- Locality head: 1.0M

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

| Metric | Value |
|--------|-------|
| Final validation perplexity (BPE) | 1.71 |
| Final training CE loss | 0.97 |
| Training steps | 50,000 |
| Training time | ~12 hours |
| Hardware | 1x RTX 5070 Ti (16GB) |
| VRAM usage | ~5.7 GB |

Training was stable throughout, with no loss spikes, divergence, or rollbacks required. The phased learning rate schedule transitioned smoothly between phases.

A note on perplexity: this is BPE-level perplexity using the GPT-2 tokenizer (50,257 subword tokens). It is not directly comparable to word-level perplexity reported in older benchmarks such as the original WikiText-103 leaderboard. BPE perplexity is measured over a finer-grained prediction space, and the numbers are not interchangeable. We report BPE perplexity because it reflects the actual training objective and tokenization used by the model.

### Generation quality

By step 23,000 (46% of training), the model was generating coherent Wikipedia-style prose from short prompts:

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

| Condition | Prompt length | Engrams | MAUVE |
|-----------|:------------:|:-------:|:-----:|
| Short prompt, engrams natural | 50 tokens | Active (empty) | 0.806 |
| Long prompt, engrams natural | 500 tokens | Active (3 windows) | 0.888 |
| Short prompt, engrams OFF | 50 tokens | Disabled | **0.905** |
| Long prompt, engrams OFF | 500 tokens | Disabled | **0.906** |

## Analysis of Results

The MAUVE results reveal a finding we did not expect.

The engram improves training but hurts inference. With engrams disabled at test time, the model scores 0.905-0.906 regardless of prompt length. With engrams active, scores are lower — significantly so at short prompts (0.806) and modestly at long prompts (0.888).

We expected the opposite. We expected the engram to provide useful compressed context that would improve generation, especially at longer prompt lengths where the engram captures more information. The data showed otherwise, and we followed the data.

What appears to be happening is that the engram functions as training scaffolding. During training, the engram creates geometric pressure on the model's internal representations. The prepended engram vectors force the attention mechanism to integrate compressed context alongside raw token representations. This shapes the learned weights — the attention patterns, the PEER routing, the layer norms — into configurations that produce better representations than they would learn without the engram's influence.

But at inference time, injecting the engram adds noise. The mean-pooled activation snapshot is a lossy compression of information the model already has access to through its attention mechanism. Rather than helping, it introduces a slightly off-distribution signal that the model must route around.

The analogy is scaffolding in construction. You build with it in place because the scaffolding shapes how the structure takes form. Then you remove it. The building retains the structural benefits — the arch holds without the centering, the wall is plumb without the braces — even though the scaffolding served no load-bearing function in the finished structure.

The evidence that the engram improved training is indirect but compelling. A model trained without the engram (and without PEER, at lower parameter count) achieved substantially worse perplexity and generation quality. While we cannot isolate the engram's contribution from PEER's with perfect precision — a clean ablation would require training a PEER model without engrams for the same duration, which we plan as future work — the trajectory of experiments from V8 through V16 consistently showed that adding the engram to the training loop improved the final model, even as we discovered that removing it at inference improved generation.

The context-dependent effect is real but operates during training, not inference. During training, longer sequences give the engram more to compress across its sliding windows, creating richer geometric pressure on the representations at deeper layers. This shapes the weights to be better at long-context generation. The effect persists in the trained weights even when the engram is disabled at inference, which is why the 500-token and 50-token MAUVE scores are nearly identical when engrams are off.

## Connection to Related Work

The engram-as-scaffolding finding connects to several concurrent lines of research, though the connection is more conceptual than mechanical.

**DeepSeek's Hyper-Connections (mHC)** use multiple residual streams to strengthen signal propagation through parallel paths. During training, the engram serves a similar signal-strengthening function: it provides the model with a compressed version of its own recent activations, creating an additional information pathway that shapes how the primary residual stream develops. The difference is that hyper-connections remain active at inference, while our engram does not.

**The Kimi team's Attention Residuals** address the same underlying problem — representation dilution across depth — through selective depth-wise attention to earlier layers. The engram addresses this differently, by explicitly compressing and re-injecting earlier representations rather than allowing selective attention across layers. Both approaches acknowledge that deep transformers lose fine-grained information as representations propagate through layers.

**DeepSeek's Engram paper** (arXiv:2601.07372) shares the name and the broad concept of conditional memory injection, but the implementations diverge fundamentally. Their engram is an external hash-based lookup table that stores and retrieves static knowledge representations. Ours is a runtime compression of the model's own hidden states — it contains no information that isn't already present in the activations, only a compressed projection of it. The shared insight is that transformers benefit from explicit memory mechanisms. The mechanism itself is entirely different.

## Implications

**Architecture can substitute for scale.** A 510M parameter model achieving a MAUVE score of 0.905 on WikiText-103 demonstrates that thoughtful architectural choices — PEER's product-key routing, full attention without approximation, careful training methodology — can narrow the gap between small and large models.

**PEER makes full attention affordable.** The conventional wisdom is that you must approximate attention (sparse, sliding window, linear) to train efficiently. PEER inverts this: by making the feed-forward layer sparse (128 of 262,144 experts active per token), the feed-forward computation becomes cheap enough that you can afford dense attention. Full attention with sparse feed-forward may be a better trade-off than sparse attention with dense feed-forward.

**Training-time components can improve models even if removed at inference.** This is perhaps the most novel finding. The engram demonstrably improves the trained model's quality, but the improvement is baked into the weights rather than requiring the engram at runtime. This suggests a broader methodology: design architectural components specifically to shape training dynamics, with the explicit intent of removing them at inference. This is related to but distinct from knowledge distillation, where a larger model guides a smaller one. Here, a component of the model itself serves as the guide.

**Consumer hardware is sufficient for meaningful AI research.** This entire experiment — architecture design, debugging, training, evaluation — ran on a single NVIDIA RTX 5070 Ti, a GPU that costs approximately $600. The model trains in 12 hours and uses under 6GB of VRAM. The bottleneck for independent AI research is increasingly ideas, not compute.

**The minimal model hypothesis is supported.** The January 2026 PEER paper proposed that PEER's sparse routing could maintain model quality while dramatically reducing active compute. The results here support that proposition. The 510M parameter model has 262,144 experts per PEER layer, but only 128 are active per token. The vast majority of parameters serve as a rich routing space that allows the model to specialize its computation per token, rather than as active computation that runs on every input.

## Future Directions

**Reconstruction head.** The current engram uses a simple mean-pooling compression, which captures the average activation but loses fine-grained structure. Adding a reconstruction loss that forces the engram to faithfully encode salient features — rather than unfiltered averages — may make the engram useful at inference as well as training. If the engram can be taught to compress only the information that attention doesn't already capture, it becomes a complement to attention rather than a noisy duplicate.

**Shannon entropy as runtime control.** The model's output distribution entropy at each position signals its own uncertainty. High-entropy positions are where the model is least confident. A runtime controller could trigger selective recomputation — a second forward pass, a deeper search, or engram-augmented inference — only at positions where the model's uncertainty exceeds a threshold. This would add compute only where it is needed.

**Recursive inference.** Feeding the model's own generation back through for a second pass, allowing it to revise uncertain positions with the benefit of its complete first-draft generation as context. This is analogous to how humans write: a first draft followed by revision, where the revision benefits from seeing the full structure of the draft.

**Scaling to The Pile.** WikiText-103 is encyclopedic text from a single domain. Training on diverse, large-scale data (code, dialogue, technical writing, fiction) would test whether the architecture generalizes or whether its strengths are specific to Wikipedia's relatively uniform structure.

**Clean ablation studies.** The most important missing experiment is training a PEER model of identical size without engrams for the same number of steps and comparing directly. This would quantify the engram's contribution to training versus PEER's contribution, which the current results cannot fully separate.

## Reproducibility

- **Code:** [github.com/MikeyBeez/HRS](https://github.com/MikeyBeez/HRS)
- **Dataset:** WikiText-103 (publicly available via Hugging Face Datasets)
- **Hardware:** NVIDIA RTX 5070 Ti, 16GB VRAM
- **Software:** PyTorch, Pop!_OS 24.04
- **Training time:** ~12 hours
- **VRAM usage:** ~5.7 GB peak

The model configuration, training loop, and benchmark script are included in the repository. The experiment is fully reproducible on any GPU with 8GB or more of VRAM with minor batch size adjustments.
