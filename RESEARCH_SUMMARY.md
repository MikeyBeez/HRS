# HRS Research Summary

A memory-aid document covering the experimental program toward a foundation-level model that runs on consumer hardware. Each section grounds claims in specific result files and articles. Numbers come from the artifacts directly.

## The Goal

A foundation-level model designed to train and run on a single consumer GPU (RTX 5070 Ti class, ~$600). The motivating constraint is that a useful model has to fit on hardware an independent researcher can afford. Dense transformers of useful capacity don't fit. The research program has been about finding the architectural ingredients that let a small-active-parameter model behave more capably than its size class would suggest.

The research has been done by one person with limited compute. The goal is not to build the foundation model, but to articulate the architecture clearly enough that a team with cluster-scale resources (DeepSeek being the obvious candidate) could implement it.

## The Architecture, in Order of Importance

### Component 1: PEER — the parameter-efficiency backbone

Source: `peer.py`, `article_peer_engram.md`, V16/V17 results.

PEER (Parameter Efficient Expert Retrieval) replaces the standard feed-forward network in every transformer layer with product-key retrieval over 262,144 single-neuron experts. Eight retrieval heads each select 16 experts via top-k product-key lookup, yielding 128 active experts per token out of 262K. The routing is fully differentiable. Reference: Lample et al., "Large Memory Layers with Product Keys" (2019).

This is the load-bearing component. **Without PEER, none of the rest matters because the model can't fit on the target hardware.** A 510M-parameter model with PEER feed-forward layers trains in approximately 12 hours on a single RTX 5070 Ti at ~5.7 GB VRAM. The same parameter budget in a dense transformer does not.

**Headline result (V17, PEER-only baseline):** 499M total parameters, 21.41 BPE perplexity on WikiText-103, MAUVE 0.933-0.943. Generation quality preferred unanimously over PEER+engram variant in blind A/B human evaluation across five domains (paleontology, naval engineering, television criticism, professional wrestling, military history). Cross-model perplexity 23.31 vs 31.16 for human reference text. Generates 1.6-2.4× faster than a comparable dense transformer; with KV cache, ~200 tokens/second constant regardless of context length.

**Why this matters:** PEER inverts the conventional efficiency trade-off. The usual move is to approximate attention (sparse, sliding window, linear) to make training affordable. PEER instead makes the feed-forward layer sparse via learned routing, which lets you afford full dense attention. **Full attention with sparse FFN may be a better trade-off than sparse attention with dense FFN.** This is the architectural insight that makes the consumer-hardware target reachable.

### Component 2: The Bonsignore Kernel — alternative attention mechanism

Source: `results/v20_bonsignore`, `results/learned_kernel_*`, `paper_bonsignore_kernel.md`.

Replaces dot-product attention with an exponential-distance kernel that includes per-head learnable temperatures and per-head MLPs. The motivation is that the dot product is a historical artifact from pre-positional-encoding RNN attention, not the load-bearing operation. What matters is that attention produces a Cartesian-product-like structure of pairwise scalars that gradient descent can shape — the specific scalar function is interchangeable.

**Headline result:** 6.19-point validation perplexity improvement on WikiText-103 (23.26 → 17.07), with per-head temperature specialization accounting for 5.81 of those points. Scaffolded co-evolution (fix kernel first, unfreeze later) preserves geometric structure.

**Why this matters:** Layered onto the PEER backbone, the Bonsignore kernel buys additional perplexity at the same parameter count. It's a drop-in replacement for the attention mechanism, not a structural change.

### Component 3: The Engram Effect — within-forward-pass signal-to-noise

Source: V18-V23 architectures, the published Engram Effect paper from February 2026.

Within a single forward pass, intermediate hidden states get mean-pooled into engram vectors that are reinjected into the residual stream at later layers. The mechanism is principal-direction extraction: mean-pooling produces a centroid aligned with the dominant directions of the hidden-state distribution, which when reinjected produces dual benefits — feature amplification through the MLP pathway and attention sharpening through the QK pathway. The two effects compound through depth.

**Headline result:** 55-61% perplexity reduction on WikiText-103 with engrams added to a small dense baseline. The engram component is decisive in ablation — without it, routing actively harms the model. This work is solid and published.

**Important caveat from the V16/V17 work:** When PEER is the substrate, the within-forward-pass engram becomes a training-time crutch that hurts inference-time generation. V16 (PEER + engram) reaches 1.71 perplexity but human evaluators unanimously preferred V17 (PEER only) on generation quality. The engram drives perplexity down by providing redundant context the attention layer already has access to; the resulting weights are less self-sufficient at inference. **The Engram Effect's positive results in V18-V23 may not transfer cleanly to a PEER backbone.** This is an open question for the integrated architecture.

### Component 4: The HRS Adapter Library — addressable memory

Source: `results/identity_ae` phases (most importantly Phases 21, 47), `experiments/per_passage_dickens`.

The architecture stores absorbed content as per-passage LoRA adapters with a frozen base model. Engrams serve as routing keys that select which adapter to load. Once selected, inference runs with the adapter active and the prompt — the engram's role is purely addressing.

**Headline result:** 100% routing accuracy, 93% retrieval (substring match) on a 50-passage Dickens library with rank-128 LoRA adapters and L0-mean-to-L5-engram routing via InfoNCE-trained projection.

**Why this matters:** This solves the addressable-memory problem for the foundation model — content that doesn't fit in the model's parameters can be absorbed as adapters, and the engram-routed addressing scheme retrieves the relevant adapter at inference.

## Phase-by-Phase, the HRS Adapter Library Arc

The phase work in `results/identity_ae` traces the development of this component. Read these as the experimental arc within the larger architectural picture.

### Phase 0-9: Identity autoencoder groundwork
Building a baseline GPT-2-V22 model on Dickens, training autoencoders for hidden-state reconstruction, OOD detection scaffolding. Foundational work.

### Phase 10: The "0/20 → 20/20" reference experiment
Source: `results/identity_ae/phase10/summary.json`.

Often miscited. What it actually shows: with no context, baseline retrieval is 0%. With full text in context, retrieval is 18% (across passage types, mixed). With weights-based test-time training on the full passage set (full_50), retrieval reaches 66% overall — 80% numeric, 90% entity, 70% technical, 10% fact. At full_100, 80% overall with 95% numeric. **Engrams are not tested in this phase.** This is a no-context-vs-full-context-vs-TTT comparison.

### Phase 21: First per-passage adapter library
Source: `results/identity_ae/phase21/library.json`.

20-passage library with per-passage adapters. Routing accuracy 60%, retrieval 60% — uneven across types (entity 100%, technical 60%, fact 80%, numeric 0%). The numeric type fails because all five numeric passages share the prompt template "What is the system access code for the X facility?" — the engrams cannot distinguish them on routing. This identifies the substrate problem early.

### Phase 32-33: Engram-as-context-replacement (K=1)
Sources: `results/identity_ae/phase33/engram_context_ppl.json`.

50 WikiText passages, 256-token passages, 200 token context, K=1 mean-pooled L5 engram. Recovery as fraction of full_context-to-no_context gap: engram alone 18%, engram+tokens (101 positions) 92%, two_engrams 21%. **The 18% is the source of the K=1 mean-pool number.** Engram-then-tokens at 92% recovery is the key positive result — with even a few raw tokens after the engram, recovery is near-complete.

### Phase 37: Compression sweep — the steep curve
Source: `results/identity_ae/phase37/compression_sweep.json`.

50 passages, 384-token context, 64-token continuation, L5 engrams replace various fractions of the context. Recovery curve as engram fraction increases: 0% engram 100% gap closed, 25% engram 97%, 50% 95%, 75% 84%, 87.5% 76%, 93.75% 64%, 96.875% 55%, 98.4% 46%, 100% engram 18%.

**The architectural finding that matters most for context economics.** A few raw tokens preserved alongside engrams produce near-full recovery at meaningful compression ratios. The 18% floor is engram-only replacement; everything above is engram-plus-some-raw-tokens.

### Phase 47: The headline routing result
Source: `results/identity_ae/phase47/l0_to_l5_projection.json`.

20-passage library with InfoNCE-trained W projection mapping L0 mean-pool keys to L5 engram space. Routing 60/60, retrieval 58/60. Per-type: numeric 15/20, entity 15/15, technical 14/15, fact 14/15. The numeric type that failed in Phase 21 now mostly works because the W projection learns the small differences in templated prompts.

### Phase 55: Multi-token engram (K sweep, attention pool)
Source: `results/identity_ae/phase55/multi_token.json`.

Attention-pooled engrams replacing all 200 context tokens (no raw tokens). K=1: 26.6% recovery. K=2: 28.7%. K=5: 29.8%. K=10: 28.96%. **The "30% ceiling" finding.** Recovery saturates; the principal-direction subspace contributes ~30% of full-context predictive signal regardless of K.

### Phase 60: Resolvability curve
Source: `results/identity_ae/phase60/resolvability_curve.json`.

5 passages, rank-128 LoRA on layers 4-5, training step counts 0/15/38/75/113/150/225/300. Retrieval mean: 0.0 / 0.8 / 1.0 / 1.0 / 1.0 / 1.0 / 1.0 / 1.0. **Adapters absorb passages quickly — 38 steps is enough for 100% retrieval.**

### Phase 62: K ceiling rank sweep
Source: `results/identity_ae/phase62/k_ceiling_rank_sweep.json`.

Tests adapter composition at varying rank. K=1: 100% retrieval at every rank tested (16, 32, 64, 128). K=2: partial retrieval (mean fraction 0.4-0.5). K=4: 0% retrieval at every rank. **Composition ceiling at K=2 is operator-level interference, not address-space anisotropy.**

### Phase 66: Routing scaling
Source: `results/identity_ae/phase66/scaling.json`.

Routing accuracy at various library sizes: 20 passages 50%, 50 passages 80%, 100 passages 40%, 200 passages 20%, 500 passages 8%. **Routing degrades with scale.** Mean pairwise cosine across the bank is 0.92-0.99 throughout. (Note: this is mean-pool engrams, not the InfoNCE-trained projection from Phase 47.)

### Phase 67: Engram vs summary
Source: `results/identity_ae/phase67/engram_vs_summary.json`.

Full context 100% gap closed (144 tokens). Extractive last-100 93% (100 tokens). Engram+recent 81% (45 tokens). Model-generated summary -10.8% (100 tokens). Extractive first-100 38% (100 tokens). **Engram+recent at 45 tokens reaches 81% of full context performance with 1/3 the token budget.**

### Phase 68: Adaptive router
Source: `results/identity_ae/phase68/adaptive_router.json`.

Position correlation 0.53 — earlier chunks compress better than later ones. Attention correlation 0.25, entropy correlation 0.20. **Position is the strongest proxy for compressibility — older context is more compressible than recent context, validating engram-old-keep-recent.**

## Named Experiments (Beyond the Phase Sequence)

### per_passage_dickens
Source: `experiments/per_passage_dickens/results/RESULT.md`.

The 50-passage Dickens experiment. Per-passage rank-128 LoRA adapters on layers 4-5, 150-500 training steps each. L0-mean-to-L5-engram routing via Phase-47 W projection. **Headline: 100% routing accuracy, 93% retrieval (substring match).**

### Recent failure series (multi_engram, prompt_vs_response_engrams, k2_engrams)
These tested whether engrams alone could do retrieval over a library without adapters. Cleanly failed — F1 below recent-only-truncation in every case. The architecture without adapters cannot do synthesis-style retrieval on naturalistic content. The geometric explanation: engrams are by construction aligned with shared principal directions of in-domain content, which makes them good routing keys (when paired with adapters) but poor discriminative content carriers (when used alone).

### engram_inference_role and engram_injection_dickens (running)
Tests whether engrams contribute at inference beyond routing, and whether adding engram injection beats the 93% Dickens baseline. Results not yet aggregated.

## Key Findings, Organized by What They Tell Us

### What works (demonstrated)
- **PEER as the FFN substrate.** 510M parameters trainable in 12 hours on a $600 GPU. MAUVE 0.933-0.943, generation preferred over engram-augmented variant in unanimous human eval.
- **Bonsignore kernel attention.** 6.19 PPL improvement on WikiText-103 over dot-product attention.
- **Engram Effect within forward passes (in dense baseline).** 55-61% perplexity reduction. Caveat: may not transfer to PEER backbone — V16 vs V17 evidence is concerning.
- **Per-passage LoRA adapters with frozen base.** No catastrophic forgetting, 100% retrieval after 38-150 training steps (Phase 60).
- **L0-to-L5 InfoNCE-trained routing.** 100% routing accuracy on 50 passages of natural prose (per_passage_dickens).
- **Engram + recent tokens for context compression.** 81% recovery at 1/3 token budget vs full context (Phase 67).
- **Adapter composition at K=2.** Two adapters loaded simultaneously work; K=4 fails to operator-level interference (Phase 62).

### What doesn't work (demonstrated)
- **Within-forward-pass engram on top of PEER substrate.** Drives perplexity down (1.71 vs 21.41) but degrades generation. V17 unanimously preferred over V16 in blind eval.
- **Engrams alone replacing all context.** 18% recovery for K=1 mean-pool, 30% saturation at K=5+ for attention-pooled engrams.
- **Engram routing without adapters on naturalistic content.** Multi-engram, prompt_vs_response, k2_engrams all fail to beat recent-only truncation.
- **Routing at scale with raw mean-pool engrams.** 80% at 50 passages, 8% at 500 passages (Phase 66). May be fixable with InfoNCE projection (Phase 47), not yet tested at large scale.
- **Self-judging LLMs for synthesis evaluation.** Mistral-as-judge rated random_engrams above recent_only.

### What's open
- **Whether the within-forward-pass engram (Engram Effect) integrates with PEER.** V16/V17 evidence suggests no with the current implementation. A reconstruction-based engram that captures information outside the attention window might fix this. Untested.
- **Routing at 200, 500, 1000+ passages with the Phase 47 projection.** Phase 47 used 20 passages, per_passage_dickens used 50. Larger scales untested.
- **K=2-to-K=4 composition recovery.** Whether reduced rank, sequential generation, or other tricks push past the K=2 ceiling.
- **Bonsignore kernel + PEER integration.** The kernel result is on a different substrate. Combined with PEER untested.
- **Engram injection beating the 93% Dickens baseline.** Currently running.
- **Cross-model engram transfer.** Phase 59 shows alignment patterns; deeper integration untested.
- **The supervision question — what to save from a stream of unlabeled content.** This is the actually hard problem the architecture doesn't solve.

## What This Adds Up To

A modular architecture for a foundation-level model on consumer hardware:

1. **PEER feed-forward layers** as the parameter-efficiency substrate that makes the whole target reachable — 510M parameters, 12 hours on a $600 GPU, full attention affordable because FFN is sparse.
2. **Bonsignore-kernel attention** as the attention mechanism, layered onto the PEER substrate (~6 PPL improvement, untested in combination with PEER).
3. **Engram Effect within forward passes** as a candidate signal-to-noise mechanism, with the caveat that the V16/V17 evidence suggests the current implementation doesn't transfer to PEER cleanly.
4. **HRS adapter library** as addressable memory: per-passage LoRA adapters addressed by InfoNCE-routed engrams, with engram+recent-token compression for in-context content (100% routing on 50 passages, 81-93% gap closed at meaningful compression ratios).

Each component has been demonstrated independently. The integration of Components 1 and 4 is the natural next test — does the adapter library work on a PEER backbone? Components 2 and 3 are layered improvements that need testing in combination with PEER.

## Methodological Notes

The recent experimental arc (multi_engram, prompt_vs_response_engrams, k2_engrams) tested whether the HRS architecture could be simplified by removing the adapters. This failed cleanly. The deleted Medium piece "We Solved Part of the Memory Problem" oversold this finding by describing the engram as content replacement rather than as a routing key. The architectural finding that survives is stronger than what was published: the engram is purely a routing key in the working architecture; adapters do the content storage; engram-plus-recent-tokens does the context compression; PEER does the parameter efficiency that makes everything else fit on consumer hardware.

The V16/V17 result also has implications I want to be honest about. The within-forward-pass engram (Component 3) is the headline result of the published Engram Effect paper. On a PEER substrate, V16 vs V17 shows that the same mechanism degrades generation quality despite improving perplexity. This is an awkward finding for the integrated architecture story. It's possible the engram needs to be reconceived for the PEER setting — perhaps as a cross-document memory rather than within-pass signal-to-noise — but that's an open question, not a solved problem.

## Document Status

This summary is grounded in result files and articles read directly from the repository. Specific numerical claims cite the file they came from. Where I'm uncertain or extrapolating, I've said so.

PEER added as Component 1 (the load-bearing component) per Mikey's correction on 2026-05-02. Original draft put PEER nowhere — a serious omission given that PEER is what makes the consumer-hardware target reachable in the first place.

Last verified against artifacts: 2026-05-02.
