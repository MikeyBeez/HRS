# Hierarchical Routed Sinkformer

## Geometry-Shaped Representations for Compute-Adaptive Language Modeling

### Abstract

Transformer language models process context as a flat token buffer and apply global attention uniformly. This design couples computation and memory: every token participates equally in every layer. As context grows, this causes quadratic compute, cross-topic interference, and degraded reasoning.

We argue that these failures arise from **missing routing and memory structure**, not insufficient parameter count.

We introduce the **Hierarchical Routed Sinkformer (HRS)**, a transformer architecture organized around a core principle: computation should be proportional to relevance. (The "Sink" refers to the sink memory channel, not Sinkhorn normalization — our empirical work demonstrated that Sinkhorn-based optimal transport is the wrong formalism for dual-objective routing; see Section 5.) The core architecture provides:

* dual-head geometry shaping to prevent representational collapse
* learned routing via Nash equilibrium between competing objectives
* tiered compute operators (convolution → experts → global attention)
* a sink channel for interference reduction

Two optional extensions address long-horizon memory:

* a Temporal Routing Cache (TRC) that amortizes routing across coherent spans
* compressed thread "engrams" for long-term memory with bounded growth

A phased training protocol with differential learning rates prevents gradient monopolization across HRS's heterogeneous components — addressing a failure mode we have empirically demonstrated in prior multi-component architectures.

This design decouples memory from compute: tokens are cheap to store but expensive to attend. By controlling attention participation rather than deleting history, HRS achieves better reasoning and lower compute. We hypothesize that a well-routed 1B HRS model can match or exceed dense 7B models at equivalent FLOPs on reasoning-heavy tasks through superior context curation.

---

# Part I: Core Architecture

---

# 1. Motivation

Current systems from organizations such as **Google** and **Anthropic** scale primarily by increasing parameters and context length.

However:

* storing tokens in KV cache is cheap (O(d))
* attending to all tokens is expensive (O(n·d))

Thus the bottleneck is not memory capacity but **indiscriminate attention**.

Flat attention leads to:

* context pollution across threads
* wasted FLOPs on irrelevant tokens
* degraded long-horizon reasoning

We propose that performance is limited more by **routing quality** than by parameter count.

---

# 2. Design Principle

**Computation should be proportional to relevance.**

Most tokens require only local processing.
Few tokens require global reasoning.

Therefore, the model should:

1. cheaply process everything locally
2. escalate only important tokens to expensive compute
3. quarantine irrelevant information
4. compress old signal into summaries

This yields a hierarchical compute and memory system rather than a monolithic transformer.

The analogy is deliberate. In *The Context Window Is a Computer* (Bee 2026) and *The Protocol Harness: An Operating System Written in Protocols* (Bee 2025), we argued that the context window is not a document but RAM, and that without an operating system — mechanisms for managing what is in active memory, secondary storage, and the index — it is useless for complex tasks. HRS is that operating system realized as architecture:

* the router is a **scheduler**
* the sink is **swap space**
* the hot/warm/cold KV tiers are a **cache hierarchy**
* global attention is a **system call** — privileged and expensive
* engrams are **checkpoints**

This is not a metaphor applied after the fact. The architecture was designed from this framing.

---

# 3. Architecture Overview

HRS is organized as a **core** plus **extensions**.

### Core (Sections 4–8)

1. Dual-Head backbone (geometry shaping)
2. Learned routing with conservation constraints
3. Tiered compute operators (conv → experts → attention)
4. Sink memory channel
5. KV cache as memory hierarchy

The core is self-contained. It delivers routing-based sparse attention, interference reduction, and tiered compute without requiring long-horizon memory.

### Extensions (Sections 9–10)

6. Temporal Routing Cache (TRC)
7. Recursive thread engrams

These add long-context memory management. They require the core to be functioning but are independently valuable — either can be adopted without the other.

### Training (Section 11)

A phased protocol that applies to both core and extensions.

---

# 4. Dual-Head Geometry Shaping

### Head A — Generative

Standard next-token cross-entropy.

### Head B — Locality / task objective

Predicts weak supervision signals:

* local window membership
* contrastive positives/negatives
* tool or code boundaries
* optional thread labels

### Purpose: Representational Tension, Not Classification

Single-loss transformers exhibit embedding anisotropy and low effective rank. We have measured this directly: in *Generative Auxiliary Training as Anti-Collapse Regularization* (Bee 2026), single-objective fine-tuning collapsed effective rank by 98.7% — from 984 to 12.6 dimensions on AG News. Only 3 dimensions captured 95% of variance. The model compressed its representations down to barely enough to separate classes, destroying the distributional richness built during pre-training.

Adding the generative head restored effective rank to 1097, *exceeding* the pre-trained baseline. On AG News 4-class classification, accuracy improved from 90.76% to 95.54% — a 4.78 percentage point gain attributable entirely to representational preservation.

The mechanism is not additive but tensional. In *MCR2 as Language Model Regularizer* (Bee 2026), we tested expansion and compression objectives independently:

* expansion alone: validation perplexity 335.5 (worse than baseline 315.2)
* compression alone: 317.2 (worse than baseline)
* both together: 306.4 (2.8% improvement)

Neither objective helps alone. Each alone makes things worse. The improvement comes from the irreconcilable conflict: the generative head demands distributed representations for next-token prediction; the locality head demands clustered representations for routing. This tension prevents collapse while preserving the fine-grained structure that downstream components need.

---

# 5. Learned Routing

Each token/state receives a lightweight relevance score from a small router network.

The router decides:

* local processing
* specialist expert
* global attention
* sink

Similarity is learned (MLP/cosine), not fixed L2, avoiding brittle geometry assumptions.

Routing is soft and probabilistic rather than hard pruning.

### Conservation Constraints

Routing tokens between tiers involves mixing operations. Unconstrained mixing amplifies signal magnitude exponentially with depth.

This is the same failure mode identified in DeepSeek's mHC architecture and analyzed in *Symmetry Constraints and Scaling Stability: A Noether-Inspired Framework* (Bee 2026) and *The Four-Lane Highway: Making Sense of DeepSeek's mHC Paper* (Bee 2026). At 27B parameters, unconstrained mixing amplified signals by up to 3000x, making training unstable.

Routing weights must satisfy conservation properties:

* all entries non-negative
* every row sums to 1 (each token's routing weights form a probability distribution)
* global tier utilization approximately uniform (no tier collapse)

Our initial design enforced full doubly stochastic constraints via the Sinkhorn-Knopp algorithm. This failed for two reasons — one practical, one theoretical.

**The practical failure:** Sinkhorn normalization on the actual routing matrix shape (B·T × n_tiers, where B·T >> n_tiers) produces per-entry values of approximately 1/(B·T). With typical batch sizes (B=24, T=512), this yields routing weights of ~0.00008 — effectively zeroing out all tier outputs during evaluation and producing random routing during training where Gumbel noise dominates the near-zero logits.

**The theoretical failure:** Sinkhorn-Knopp solves entropy-regularized optimal transport — finding the minimum-cost assignment under a single objective. But HRS is a dual-objective system. The generative head (Head A) demands distributed representations for next-token prediction; the locality head (Head B) demands clustered representations for routing. These are competing objectives with no single cost function to minimize. The routing weights do not emerge from a transport plan — they emerge from a **Nash equilibrium** between the two heads. Optimal transport is the wrong formalism; game theory is the right one.

This distinction is not cosmetic. Optimal transport has a unique solution for a given cost matrix. A Nash equilibrium depends on the dynamics of the competing players — which is exactly why phased training (Section 11) matters. The order in which objectives are introduced determines which equilibrium the system converges to.

### Formal Objective

The dual-head routing objective can be expressed as a constrained optimization with a dynamic Lagrange multiplier:

```
min_{P1, P2}  Cost(P1, P2)  +  λ(θ) · D_KL(P1 || P2)
```

where P1 is the routing distribution induced by the generative head, P2 is the routing distribution induced by the locality head, θ are the learned model weights, and λ(θ) is a learned scalar controlling the tension between the two objectives.

This formulation captures three essential properties:

1. **Two distributions, not one.** Optimal transport minimizes cost under a single distribution. Here, P1 and P2 are jointly optimized but irreducibly distinct — the generative head and locality head induce different preferences over routing, and the system must balance them.

2. **The tension is learned, not fixed.** λ(θ) depends on the model's weights, making it adaptive. Early in training, when representations are unstable, the model may learn a small λ (tolerating large divergence between the heads). As training progresses and the heads converge on a shared geometry, λ can increase (penalizing divergence to stabilize the equilibrium). This connects directly to the self-adaptive loss weighting in Section 11, where sigmoid-activated scalar parameters serve as learned λ values for each loss component.

3. **KL divergence measures productive disagreement.** D_KL(P1 || P2) quantifies how much the two heads disagree about routing. Some disagreement is essential — it is the source of the representational tension that prevents collapse (Section 4). Too much disagreement means the heads are fighting rather than composing. The learned λ finds the productive balance.

In practice, this objective is realized through the combined loss function where each head's gradient competes through the shared backbone, and the routing weights emerge from the resulting equilibrium.

We implement the game-theoretic routing using three complementary mechanisms:

1. **Softmax normalization** ensures per-token routing weights are non-negative and sum to 1 (row constraint)
2. **Balance loss** (squared-mean formulation: n·Σp_i² where p_i is mean tier utilization) penalizes deviation from uniform global distribution (soft column constraint), acting as a regularizer on the equilibrium
3. **Gumbel-softmax** with temperature annealing provides differentiable discrete routing during training — the temperature schedule controls the sharpness of the equilibrium, analogous to annealing in game-theoretic learning

This formulation lets routing emerge from gradient competition between objectives rather than from solving a transport problem. A learnable per-dimension output gate (initialized at 0.1) further controls the magnitude of tier outputs relative to the residual stream, preventing any single tier from dominating the equilibrium.

Additionally, per-token **entropy regularization** prevents premature routing commitment. Without it, the router collapses to a fixed assignment within the first few thousand steps (routing entropy < 0.01), before the tiers have had time to specialize. With entropy regularization (weight 0.01), routing entropy decreases gradually from ~1.0 to ~0.1 over 40K steps — the system explores the equilibrium landscape before committing.

These conservation constraints are essential for scaling HRS beyond small models.

### Router Stability

During training, the router will oscillate — assigning the same token to different tiers across consecutive steps. Not all oscillation is pathological.

*Binary Node Oscillation in Neural Networks* (Bee 2025) distinguished two types:

**Type I (pathological):** The router receives contradictory gradient signals. The same input is routed to experts in one batch and to sink in the next, not because the input is ambiguous but because the training signal is inconsistent. Symptom: high routing entropy that does not decrease with training.

**Type II (meaningful):** The input is genuinely ambiguous — a token that is relevant in some contexts and irrelevant in others. The router correctly oscillates because the optimal routing depends on context. Symptom: routing entropy decreases for most tokens but remains high for a learnable subset.

**Diagnostic:** Track per-token routing entropy across training. Healthy training shows global routing entropy decreasing, a long tail of high-entropy tokens for genuinely ambiguous inputs, and routing decisions for unambiguous tokens stabilizing within Phase 2 of training (see Section 11).

---

# 6. Tiered Compute

Instead of full attention everywhere, each layer provides multiple operators:

### Tier 1 — Local mixing (default)

Depthwise convolution or small local attention.

* O(n) cost
* handles most tokens
* captures short-range structure

### Tier 2 — Specialists

Parameter-efficient expert modules for domain skills.

Only activated when routed.

### Tier 3 — Global attention (rare)

Full softmax attention for high-importance tokens.

Used sparingly.

This makes attention an **exception path**, not the default.

### Empirical Support: The Two-Stage Hypothesis

This tiered design is supported by *Attention Builds Maps, Then Reads Them: The Two Stages of Transformer Intelligence* (Bee 2025), which demonstrated that hybrid models — replacing later-layer attention with static functions — lost only 1.4% quality while gaining 58% speed. Early layers build geometric structure via attention; later layers navigate established geometry and can use cheaper operators.

Further evidence comes from *Static Functions Can Approximate Deep Attention Layers* (Bee 2025), where per-head MLP approximations explained up to 92.6% of variance in deeper-layer attention head outputs. Deeper layers compute increasingly predictable functions — they do not need the full dynamic attention mechanism.

Together, these findings justify the tier structure: global attention in early layers where geometry is forming, cheap operators in later layers where geometry has stabilized.

---

# 7. Sink Memory Channel

Transformers lack a forgetting mechanism; irrelevant tokens continue to participate in attention and interfere with reasoning.

We introduce a sink pathway:

Low-relevance tokens are routed toward:

* sink tokens
* or a sink subspace
* or low-priority KV tier

They remain stored but minimally attended.

This:

* prevents interference
* avoids brittle deletion
* allows recovery if later needed

Memory is preserved while compute is reduced.

### Salience Criteria

What determines which tokens are sunk? *Memory, Salience, and Why Your LLM Forgets* (Bee 2025) proposed measurable dimensions of token persistence:

* **Information density:** High-density tokens (numbers, rare words, IDs) recalled at ~89%; low-uniqueness tokens at ~34%
* **Position:** U-shaped recall curve — early and late tokens persist, middle tokens are most vulnerable
* **Structure:** Tokens wrapped in delimiters or markers show higher persistence
* **Geometry:** Tokens with high projection magnitude in the residual stream survive compression better

The sink router should learn these dimensions implicitly through the dual-head training signal. The locality head's contrastive objective naturally creates separation between high-salience and low-salience tokens — the sink channel exploits this learned separation rather than requiring hand-engineered rules.

---

# 8. KV Cache as a Memory Hierarchy

Maintaining KV cache is cheap compared to attention.

Therefore we keep all tokens but control **participation** rather than **existence**.

We maintain tiers:

* hot: active thread
* warm: specialists
* cold: sink

All live in KV; only routing controls which are attended.

This decouples memory size from compute cost.

Current approaches treat compression as reactive — summarize after overflow. The correct approach is proactive hierarchical memory with tiered access. HRS's learned routing serves as a memory controller, deciding what lives in active working memory (hot), what is available for specialist retrieval (warm), and what is stored but not attended (cold).

This reframes the scaling problem: we do not need longer context windows, we need **better context management**.

### Core Architecture Summary

The core HRS — dual-head geometry, router, tiered compute, sink, KV hierarchy — is a complete system. It delivers:

* geometry-aware sparse attention
* interference reduction via the sink
* mostly O(n) compute with rare O(n²) escalation
* all tokens preserved, participation controlled

What it does not address is **long-horizon memory**. Conversations still grow linearly. The extensions in Part II solve this.

---

# Part II: Extensions

---

# 9. Temporal Routing Cache (TRC)

Routing decisions exhibit strong temporal correlation (e.g., code blocks, math derivations).

We cache routing indices across spans and reuse them for multiple tokens.

Routing is recomputed only on distribution shifts.

This amortizes routing overhead and reduces compute significantly in coherent segments.

### Simplification Note

A simpler alternative — low-pass filtering router logits across adjacent tokens — may capture most of the TRC's benefit without explicit caching or statefulness. We recommend benchmarking low-pass filtering as a baseline before adopting the full TRC. If the gap is marginal, the simpler approach is preferable. TRC adds statefulness across tokens, routing inertia, and harder credit assignment; these costs are justified only by proportional gains.

---

# 10. Recursive Thread Engrams (Long-Term Compression)

Even with routing and sinking, long conversations grow linearly.

We introduce **thread engrams**:

Periodically, the locality head extracts a compressed summary of the active thread:

1. pool hidden states from the model's middle layers
2. compress via small encoder
3. produce fixed-size engram vector(s)
4. append back to context as memory tokens

Older raw tokens can then be down-tiered to sink.

This provides:

* bounded memory growth
* persistent long-term knowledge
* cheaper reasoning over history

The model learns to treat engrams as high-signal summaries.

### Prior Empirical Results

Engrams are not speculative. In *Engrams: Learned Semantic Compression for Transformers* (Bee 2026), we demonstrated that middle-layer hidden state extraction, compressed to 32 engram vectors from 8,192 source tokens (256x compression), achieved 96% factual accuracy versus 80% for full-text RAG retrieval with 64x fewer tokens.

The critical design choice is extraction layer. A layer sweep confirmed that the middle third of the model is optimal — deep enough to encode semantic relationships, not so deep that representations have specialized for next-token prediction. In a 7B model, layer 16 (of 32) produced the best engrams.

Three mechanisms explain why compressed engrams outperform full-text retrieval:

* **Signal concentration:** Mean-pooling blends information from multiple tokens, increasing density
* **Noise reduction:** Compression strips formatting, tangential text, and syntactic scaffolding
* **Representation alignment:** Injecting at the embedding layer in the model's native representational format avoids the lossy text→token→embedding round-trip that RAG requires

Further investigation in *Engrams Don't Inject Information — They Retrieve It* (Bee 2026) showed that engrams function as retrieval cues for information already stored in the model's weights, not as information containers. *What We Got Wrong About Engram Steering* (Bee 2026) and *Golden Ratio Engram: A Corrected Account* (Bee 2026) documented iterative corrections to the extraction and injection procedure — the kind of empirical refinement that distinguishes a tested technique from a theoretical proposal.

### Engram Extraction as Energy Minimization

The engram is not ad-hoc compression. *Recursive Refinement as Approximate Hopfield Dynamics* (Bee 2026) showed that a single attention pass is mathematically equivalent to a Hopfield update rule, and that recursive passes converge toward stored attractor patterns. Small recursive models achieved 99.2% improvement over single-pass baselines on word-level language modeling.

Critically, the bottleneck compression in engrams filters noise *geometrically* — by excluding low-variance dimensions — rather than by attenuation. This is strictly more effective than temperature scaling. The engram is a compressed representation of an attractor basin's neighborhood in activation space: the essential structure of a thread, with the noise excluded by geometry.

### When to Extract

Engram extraction should not follow a fixed schedule. *Differentiable Time: When Neural Networks Learn They Are Finished* (Bee 2026) proposed using Jacobian sensitivity via Hutchinson's stochastic trace estimator to detect when layer computations have converged. When a thread's representations stabilize — sensitivity drops below a learned threshold — the representations are ready for compression.

This gives the model a self-assessed signal for when a thread has been "understood" well enough to compress. The same mechanism serves double duty: it triggers engram extraction at inference time and provides the Phase 3→4 transition criterion during training (Section 11).

### Engram Refinement

Standard training will underutilize engrams. *Iterative Refinement: Breaking Through Convergence Plateaus* (Bee 2025) demonstrated that freezing converged representations and retraining the classifier head improved validation loss by 24-28% — with no additional parameters or data. The mechanism: jointly trained classifiers reach a local optimum that leaves extractable information on the table.

The same principle applies to engram consumers. After the full system converges:

1. Freeze the engram encoder
2. Reinitialize the connection weights from engram inputs to downstream layers
3. Retrain those downstream layers

The retrained layers will extract more information from the same compressed representations. If the 24-28% improvement transfers to engram consumption — and the Hopfield dynamics analysis suggests it should, since the engram representations form the same kind of stable attractors that iterative refinement exploits — skipping this step wastes a quarter of the engram's informational capacity.

### The Memory Triangle

Routing handles **selection**. The sink handles **forgetting**. Engrams handle **compression**.

That is the full memory triangle. Each addresses a distinct failure mode of flat attention:

* without selection, compute is wasted on irrelevant tokens
* without forgetting, irrelevant tokens interfere with reasoning
* without compression, history grows without bound

The core (Part I) delivers the first two. Engrams complete the third.

---

# Part III: Training

---

# 11. Training Strategy

## The Problem: Five Gradients, One Backbone

HRS combines five component types in a shared representational space. Each produces its own loss gradient. All gradients flow through the same backbone. Naive end-to-end training will fail.

This is not speculation. In *We Stacked 3 AI Upgrades. The Combined System Was Worse Than Using Just 1* (Bee 2026), we demonstrated that combining three complementary architectural improvements produced worse performance than any single improvement alone. The failure mode was gradient monopolization: the fastest-learning component captured representational real estate before slower components could specialize, and subsequent components optimized around an already-distorted geometry.

HRS is more complex than that three-component system. Without a principled training strategy, the same failure mode will occur at larger scale.

## Principle: Phased Specialization with Differential Learning Rates

We adopt the phased training approach from *Phased Specialization: Unlocking Hybrid Sequence Models* (Bee 2026), adapted for HRS's component architecture.

Heterogeneous components learn at different natural speeds. Convolutions converge fast. Expert routing converges slowly. Attention is somewhere between. If all components train at the same learning rate, the fast learners monopolize the gradient signal before the slow learners have established useful representations.

The fix is not sequential freezing (which creates brittle handoff points) but **differential learning rates with soft phases**. No component is ever fully frozen (except the engram encoder in early phases). Minimal learning rates (1e-5 to 1e-6) maintain compatibility with the evolving backbone without allowing distortion. This was the key finding in the stacking experiments: fully frozen components fall out of sync and cause instability when later unfrozen.

### Phase 1: Foundation (Backbone + Generative Head + Conv)

**Goal:** Establish stable representations and local structure.

| Component | Learning Rate | Status |
|-----------|--------------|--------|
| Backbone | full LR | lead |
| Generative head (Head A) | full LR | lead |
| Conv operators | full LR | lead |
| Locality head (Head B) | 0.1x LR | warming |
| Router | 0.01x LR | minimal |
| Expert modules | 0.01x LR | minimal |
| Sink channel | 0.01x LR | minimal |
| Engram encoder | frozen | off |

The backbone learns basic language structure. Convolutions establish local mixing patterns. The generative head provides the primary training signal.

**Duration:** Until backbone validation loss plateaus.

### Phase 2: Geometry Shaping (Add Locality Head + Router)

**Goal:** Shape representational geometry for routing via dual-head tension.

| Component | Learning Rate | Status |
|-----------|--------------|--------|
| Backbone | 0.5x LR | supporting |
| Generative head (Head A) | full LR | lead |
| Conv operators | 0.5x LR | supporting |
| Locality head (Head B) | full LR | **lead** |
| Router | 0.5x LR | **ramping** |
| Expert modules | 0.1x LR | warming |
| Sink channel | 0.1x LR | warming |
| Engram encoder | frozen | off |

The locality head comes online as co-lead. Its weak supervision signal shapes the backbone geometry to form the clusters that routing will exploit. The router begins learning at 0.5x LR, exploiting the emerging cluster structure. It does not need the geometry to be perfect — it needs it to be forming.

**Transition criterion:** Monitor effective rank via SVD on backbone hidden states. When effective rank stabilizes at or above the Phase 1 baseline, geometry shaping is working. If effective rank drops below 50% of baseline, reduce locality head LR — the auxiliary objective is too aggressive.

### Phase 3: Specialization (Experts + Sink)

**Goal:** Activate specialist computation and forgetting.

| Component | Learning Rate | Status |
|-----------|--------------|--------|
| Backbone | 0.3x LR | supporting |
| Generative head (Head A) | full LR | lead |
| Locality head (Head B) | 0.5x LR | supporting |
| Conv operators | 0.3x LR | supporting |
| Router | full LR | **lead** |
| Expert modules | full LR | **lead** |
| Sink channel | full LR | **lead** |
| Engram encoder | frozen | off |

The router, experts, and sink come online together. The router has had two phases of warmup and a forming cluster geometry to work with.

**Diagnostic:** Monitor output magnitudes per component. If one component's output norm exceeds 3x the others', it is monopolizing. Reduce its LR by 50%. This was the primary failure signal in the stacking experiments.

**Note:** For core-only HRS (without extensions), training ends here. The core delivers geometry-aware sparse attention, tiered compute, and interference reduction. Phases 4 and 5 apply only when engrams are included.

### Phase 4: Compression (Engram Encoder)

**Goal:** Learn long-term memory compression. *Extension phase — requires engram module.*

| Component | Learning Rate | Status |
|-----------|--------------|--------|
| Backbone | 0.1x LR | supporting |
| Generative head (Head A) | 0.5x LR | supporting |
| Locality head (Head B) | 0.3x LR | supporting |
| Conv operators | 0.1x LR | supporting |
| Router | 0.5x LR | supporting |
| Expert modules | 0.5x LR | supporting |
| Sink channel | 0.5x LR | supporting |
| Engram encoder | full LR | **lead** |

The engram encoder trains last because it depends on all other components being stable. Training engrams on unstable representations produces unstable compressions.

**Transition criterion from Phase 3:** Jacobian sensitivity (Hutchinson's estimator) across routing and expert layers drops below learned threshold — the same convergence signal described in Section 10.

### Phase 5: Engram Refinement *(Optional — see Section 13, Finding 3)*

**Goal:** Maximize information extraction from compressed representations. *Extension phase.*

Freeze the engram encoder. Reinitialize downstream connection weights. Retrain the layers that consume engrams. (See Section 10, Engram Refinement.)

**Duration:** Fixed — equivalent to Phase 4 duration.

**Empirical note:** Phase 5 consistently degrades performance at the 50M-parameter scale tested (Section 13). All three engram variants show best perplexity at the end of Phase 4, with Phase 5 adding +0.62 to +1.40 PPL regression. We recommend treating this phase as optional pending validation at larger scale.

## Self-Adaptive Loss Weighting

Manual tuning of loss weights across four or more objectives is fragile. We adopt the approach from *Self-Adaptive Loss Weighting Through Network Node Designation* (Bee 2025): designate specific network nodes whose sigmoid-activated outputs serve as learnable loss scaling factors.

For each loss component L_i:

```
w_i = sigmoid(node_i) + epsilon
L_total = sum(w_i * L_i)
```

where node_i is a dedicated scalar parameter updated by backpropagation, and epsilon (1e-4) prevents any objective from being fully suppressed.

Phased differential LRs provide the coarse schedule. Adaptive loss weights provide fine-grained balancing within each phase. The two mechanisms are complementary.

## Conservation Constraints on Routing

Routing tokens between tiers involves mixing operations. Unconstrained mixing amplifies signal magnitude exponentially with depth (see Section 5 for the DeepSeek mHC precedent).

Routing conservation is enforced through three complementary mechanisms:

1. **Per-token softmax** ensures routing weights are non-negative and sum to 1
2. **Balance loss** (n·Σp_i²) penalizes global tier imbalance, preventing routing collapse
3. **Entropy regularization** (negative mean per-token entropy) prevents premature commitment

During training, Gumbel-softmax with annealed temperature (τ: 1.0 → 0.3 over 40K steps) provides differentiable discrete routing. At evaluation, deterministic softmax with minimum temperature produces stable assignments.

A learnable per-dimension **tier output gate** (initialized at 0.1) controls the magnitude of tier outputs relative to the residual stream. Without this gate, tier outputs (magnitude ~1.0) dominate the residual addition (baseline MLP outputs ~0.007 after GPT-2-style output scaling), causing effective rank collapse. The gate learns to scale tier contributions to match the residual stream's magnitude profile.

**Empirical note:** Our initial implementation used Sinkhorn-Knopp normalization for full doubly stochastic constraints. This failed both practically (non-square matrix pathology) and theoretically — Sinkhorn solves optimal transport under a single cost function, but the dual-head system is a two-player game whose routing equilibrium emerges from competing gradients, not from a transport plan. See Section 5 for the full analysis and Section 13 for empirical validation.

## Router Stability Diagnostics

During training, the router will oscillate — assigning the same token to different tiers across consecutive steps. Not all oscillation is pathological.

*Binary Node Oscillation in Neural Networks* (Bee 2025) distinguished two types:

**Type I (pathological):** The router receives contradictory gradient signals. The same input is routed to experts in one batch and to sink in the next, not because the input is ambiguous but because the training signal is inconsistent. Symptom: high routing entropy that does not decrease with training.

**Type II (meaningful):** The input is genuinely ambiguous — a token that is relevant in some contexts and irrelevant in others. The router correctly oscillates because the optimal routing depends on context. Symptom: routing entropy decreases for most tokens but remains high for a learnable subset.

**Diagnostic:** Track per-token routing entropy across training. Healthy training shows:

* global routing entropy decreasing
* a long tail of high-entropy tokens (genuinely ambiguous inputs)
* routing decisions for unambiguous tokens stabilizing within Phase 2

If global routing entropy fails to decrease by Phase 3, the router is receiving contradictory gradients. Reduce the locality head LR — the geometry-shaping signal is confusing the router rather than helping it.

## The Hopfield Justification

The phased strategy has a theoretical basis beyond empirical convenience. Each phase corresponds to establishing a new level of attractor structure in the model's energy landscape:

* Phase 1 creates the base energy surface (language structure)
* Phase 2 sculpts basins (semantic clusters via dual-head tension)
* Phase 3 creates routing channels between basins (expert specialization)
* Phase 4 creates compressed summaries of basin neighborhoods (engrams)
* Phase 5 refines the readout of those summaries

Each phase must wait for the previous landscape to stabilize. Training all phases simultaneously is equivalent to sculpting valleys in a landscape that is still forming — the valleys shift before they can be used.

## Training Cost

Total training cost is approximately 1.4-1.6x a standard single-phase run of equivalent total steps (1.2x for core-only without Phases 4-5). The overhead comes from reduced LR utilization in early phases. The alternative — naive end-to-end training — produces the stacking failure: components that fight rather than compose.

## Summary

| Phase | Duration | Lead Components | Key Metric |
|-------|----------|----------------|------------|
| 1: Foundation | Until val loss plateaus | Backbone, GenHead, Conv | Validation loss |
| 2: Geometry | Until effective rank stabilizes | LocalityHead, Router (ramping) | Effective rank (SVD) |
| 3: Specialization | Until routing entropy stabilizes | Router, Experts, Sink | Routing entropy, output magnitudes |
| 4: Compression* | Until engram reconstruction converges | Engram encoder | Reconstruction quality |
| 5: Refinement* | Fixed (= Phase 4 duration) | Downstream engram consumers | Val loss with engram context |

*Phases 4-5 apply only when engram extension is included.

---

# 12. Training Data

Precise thread labels are unnecessary.

Weak supervision for Head B suffices:

* local window positives
* distant negatives
* structural cues (tools/code)

This allows training without proprietary chat logs and avoids brittle labeling.

---

# Part IV: Evaluation

---

# 13. Experimental Validation

## Setup

We validated HRS on WikiText-103 (118M tokens) using a 4-layer, 8-head, d_model=512 transformer (38M parameters for the dense baseline, up to 50M with all HRS components). All configurations were trained for 50,000 steps with effective batch size 48, learning rate 3e-4 with cosine decay, on a single NVIDIA RTX 5070 Ti (16GB VRAM). BF16 mixed precision was used throughout.

Seven ablation configurations were trained, each adding one component. Two additional configurations were run after correcting an implementation error in the original config 7 (see note below):

1. **dense_baseline** — Standard transformer, cross-entropy only (38M params)
2. **dual_head** — + locality head with InfoNCE contrastive loss
3. **dual_head_router** — + learned router with tiered compute (conv/expert/attn/sink)
4. **dual_head_router_sink** — + sink channel enabled
5. **full_core** — + phased training protocol (Phases 1-3)
6. **full_hrs** — + engram encoder (Phases 1-5, 50M params)
7. **full_hrs_refined** — + engram refinement (freeze encoder, reinitialize injector at P5)
8. **full_hrs_trc** — full_hrs + Temporal Routing Cache (causal moving average, window=8)

## Results

| # | Configuration | Val PPL | Best PPL (Phase) | Δ vs Baseline | Eff. Rank | Routing Entropy | Tier Dist. (c/e/a/s) |
|---|---------------|--------:|-----------------:|--------------:|----------:|----------------:|----------------------|
| 1 | dense_baseline | **23.83** | — | — | 391.5 | — | — |
| 2 | dual_head | **24.08** | — | +1.0% | 90.1 | — | — |
| 3 | dual_head_router | **31.63** | — | +32.7% | 13.3 | 0.026 | 21 / 29 / 32 / 18 |
| 4 | dual_head_router_sink | **31.90** | — | +33.9% | 12.3 | 0.035 | 20 / 32 / 29 / 19 |
| 5 | full_core | **37.84** | — | +58.8% | 19.1 | 0.055 | 24 / 31 / 25 / 20 |
| 6 | full_hrs | **12.04** | 10.64 (P4) | -49.5% | 43.1 | 0.101 | 26 / 30 / 32 / 12 |
| 7 | full_hrs_refined | **10.51** | **9.19** (P4) | **-55.9%** | — | — | — |
| 8 | full_hrs_trc | **10.05** | **9.43** (P4) | **-57.8%** | — | — | — |

**Note on config 7:** The original ablation run labeled "full_hrs_refined" contained an implementation error — no actual engram refinement was performed (the Phase 5 freeze/reinitialize protocol was not triggered). That run (val_ppl 10.04) was effectively a second full_hrs run, confirming substantial run-to-run variance across engram configurations (range: 10.0–12.0). The results above reflect the corrected implementation where the engram encoder is frozen and the injector type embedding is reinitialized at the Phase 5 transition.

**Best PPL column:** For engram configurations (6-8), the best validation perplexity occurs at the end of Phase 4, before Phase 5 degrades performance. This is discussed in Finding 3 below.

## Analysis

### Finding 1: Engrams are the decisive component

The clearest result is the discontinuity between configurations 5 and 6. Adding routing without engrams (configs 3-5) consistently *increases* perplexity by 33-59% — the "router tax" exceeds the routing benefit. Adding engrams (config 6) produces a 49.5% *decrease* — from 23.83 to 12.04 at end of training (and to 10.64 at best). The best-of-run result across all engram configurations reaches 9.19 (config 7 at end of Phase 4), a 61.4% reduction from baseline. This is not an incremental gain from one more component. The engram system transforms what the architecture can do.

The mechanism is visible in effective rank. Routing alone collapses effective rank from 391.5 (baseline) to 12-19 — the router, even when healthy, concentrates representations into a low-dimensional routing-optimal subspace. Engrams restore effective rank to 43-44, recovering representational diversity through the compression bottleneck. The engram's lossy compression acts as a geometric regularizer, preserving the distributional structure that routing alone discards.

### Finding 2: The router tax is real but not fatal

Configurations 3-5 demonstrate that routing machinery — the router MLP, tier operators, balance/entropy losses — imposes meaningful overhead at this model scale. The 38M-parameter model does not have enough capacity for the router to find savings that offset the additional parameters and optimization complexity.

However, all routed configurations show healthy routing dynamics: all four tiers remain active throughout training (no tier collapse), routing entropy decreases smoothly from ~1.0 to ~0.03, and the tier distribution stabilizes by step 10K. The routing machinery works correctly — it simply does not help perplexity without the engram system to exploit the structured representations it creates.

This suggests the router's value is infrastructural: it creates the organized representational geometry that engrams need to compress effectively. Routing without engrams is scaffolding without a building.

### Finding 3: Phased training dynamics and Phase 5 regression

The `full_hrs` trajectory reveals how the phased protocol shapes training:

| Phase | Steps | Best Val PPL | Tier Distribution | Observation |
|-------|------:|-------------:|-------------------|-------------|
| P1 (Foundation) | 1-8K | 27.0 | 68/10/20/3 | Conv-dominated bootstrap |
| P2 (Geometry) | 8-16K | 15.1 | 36/28/25/11 | Tiers rebalance, PPL drops sharply |
| P3 (Specialization) | 16-26K | 14.8 | 27/28/31/14 | Modest improvement, routing stabilizes |
| P4 (Compression) | 26-38K | 11.1 | 27/30/31/12 | Engrams activate — PPL drops 25% |
| P5 (Refinement) | 38-50K | 12.0 | 26/30/32/12 | Regression: +0.9 PPL |

Phase 1 reveals an unexpected pattern: the router initially sends 68% of tokens to the conv tier. This is rational — during foundation training, the conv tier is the only tier with meaningful gradients (experts and attention start with near-random weights). The router learns to route around untrained components.

Phase 4 produces the largest single-phase improvement (14.8 → 11.1, a 25% drop). This is the engram activation phase, confirming that the compressed memory system is the primary driver of HRS performance.

**Phase 5 consistently degrades performance across all engram configurations.** This pattern holds for all three variants tested:

| Configuration | Best PPL (end P4) | Final PPL (end P5) | P5 Regression |
|--------------|------------------:|-------------------:|--------------:|
| full_hrs (v4) | 10.64 @ 38K | 12.04 | +1.40 |
| full_hrs_refined (v5) | 9.19 @ 33K | 10.51 | +1.32 |
| full_hrs_trc (v5) | 9.43 @ 36K | 10.05 | +0.62 |

The refinement protocol — freezing the engram encoder and retraining downstream consumers — does not help: `full_hrs_refined` regresses by +1.32, comparable to the unrefined `full_hrs` (+1.40). TRC shows the smallest regression (+0.62), suggesting temporal routing smoothing provides some stability during the phase transition.

This is a clear negative result for Phase 5 at this model scale. Possible explanations: (a) the 24-28% iterative refinement improvement observed in prior work requires larger models or longer training to manifest; (b) freezing the encoder disrupts the joint optimization dynamic — the engram encoder, router, and backbone form a coupled system, and freezing one part shifts the equilibrium in ways that require more than 12K steps to recover; (c) the Phase 5 learning rate schedule (backbone at 0.3x, heads at 0.5x) is too aggressive for stable refinement. We recommend that future work either eliminate Phase 5 or investigate gentler transitions (e.g., gradual encoder LR decay rather than hard freeze).

### Finding 4: Causal convolution matters

During development, we discovered that symmetric convolution padding (seeing 3 future tokens with kernel_size=7) allowed the router to route 99%+ of tokens to the conv tier, achieving val_ppl of 1.1 — the model was cheating by reading the future. After correcting to causal (left-only) padding, conv tier usage dropped to a healthy 20-25%. This confirms that the router is sensitive to tier capabilities and will exploit any information advantage. Causal padding is essential for autoregressive HRS.

### Finding 5: Routing is Nash equilibrium, not optimal transport

The most theoretically significant finding was the Sinkhorn failure. Our initial implementation used Sinkhorn-Knopp normalization to enforce doubly stochastic routing constraints, following the standard optimal transport formalism used in MoE routing literature. This failed catastrophically — but not only for the practical reason (non-square matrix pathology). The deeper issue is that **Sinkhorn solves the wrong problem**.

Sinkhorn-Knopp finds the entropy-regularized optimal transport plan — the minimum-cost assignment under a single objective. But HRS is a dual-objective system. The generative head pulls representations toward distributed geometry; the locality head pulls toward clustered geometry. The router's assignment emerges from the equilibrium of these competing gradients, not from minimizing any single transport cost. This is a Nash equilibrium, not an optimal transport solution.

The distinction has practical consequences:

1. **No unique optimum.** Optimal transport has a unique solution for a given cost matrix. A Nash equilibrium depends on the dynamics of competition — which is why phased training matters (the order of objective introduction determines which equilibrium the system finds).

2. **Sinkhorn normalization destroys the game.** On routing matrices of shape (B·T, n_tiers) where B·T=12,288 and n_tiers=4, Sinkhorn produces per-entry values of ~0.00008. But even with correct dimensions, Sinkhorn would impose a single-objective constraint on a multi-objective system, suppressing the competitive dynamics that produce useful routing.

3. **Game-theoretic formulation works.** Softmax (per-player normalization) + balance loss (equilibrium regularizer) + entropy regularization (exploration) + Gumbel-softmax (annealed commitment) — these are game-theoretic mechanisms, not transport mechanisms. They let the equilibrium emerge rather than computing it.

Two additional implementation details proved critical:

* **Tier output scaling.** Without a learnable output gate, tier outputs (~1.0 magnitude) overwhelm the residual stream (baseline MLP outputs ~0.007 after GPT-2 scaling). A per-dimension gate initialized at 0.1 allows the model to learn appropriate scaling. Without it, effective rank collapses from 64 to 11 within 5K steps.

* **Entropy regularization prevents premature commitment.** Without the entropy loss term (weight 0.01), routing entropy drops below 0.01 within 3K steps, freezing the router into a fixed assignment before tiers have specialized. With entropy regularization, entropy decreases gradually (1.0 → 0.1 over 40K steps), allowing the router to explore the equilibrium landscape throughout training.

### Finding 6: TRC provides modest routing stability

The Temporal Routing Cache (config 8) applies a causal moving average over router logits with window size 8, smoothing routing decisions across adjacent tokens. Compared to the identical architecture without TRC (config 6), the TRC variant achieves a better best-of-run PPL (9.43 vs 10.64) and a smaller Phase 5 regression (+0.62 vs +1.40).

However, cross-run variance is substantial: the three full_hrs variants span 10.05–12.04 at end of training and 9.19–10.64 at best. The TRC improvement is within this variance band. A definitive assessment requires multiple seeds per configuration. What TRC clearly demonstrates is that temporal smoothing does not *hurt* — the low-pass filter imposes a mild inductive bias (adjacent tokens should route similarly) that is compatible with language's temporal coherence structure. At longer sequence lengths where routing volatility may increase, TRC's stabilizing effect is likely more pronounced.

## Summary

The ablation validates the HRS thesis with an important caveat: **routing alone is insufficient; routing plus compression is transformative.** The best HRS configuration achieves a best-of-run perplexity of 9.19 (61.4% below the dense baseline) at the end of Phase 4, with final end-of-training perplexity of 10.05–10.51 across engram variants (55.9–57.8% below baseline). The engram system is the decisive component, not the router — but the router creates the structured representations that make effective compression possible.

Phase 5 (engram refinement) consistently degrades performance and should be considered optional or experimental at this model scale. The best practical configuration is a 4-phase protocol (P1–P4) with optional TRC.

---

# 14. Expected Benefits

Based on the experimental results above, we can now distinguish validated from projected benefits:

### Validated (Section 13)

* representational diversity preserved through routing + compression (effective rank 43 vs 12.3 routing-only)
* 55.9–61.4% perplexity reduction over dense baseline at equivalent training budget (end-of-training 10.05–10.51 vs 23.83; best-of-run 9.19 vs 23.83)
* all four compute tiers utilized without collapse
* phased training successfully sequences component activation (Phases 1–4)
* Phase 5 (engram refinement) degrades performance and should be considered optional

### Projected (require larger-scale validation)

* reduced cross-thread interference at longer context lengths
* better long-context reasoning on downstream tasks
* mostly O(n) compute at scale (routing overhead amortized)
* performance improvement via routing and compression rather than parameter growth

---

# 15. Minimal Model Hypothesis

We hypothesize:

A well-routed 1B HRS model can match or exceed dense 7B transformers **at equivalent FLOPs** on reasoning-heavy tasks.

This is a claim about compute efficiency, not parameter equivalence. Routing allows a smaller model to concentrate its capacity where it matters rather than spreading it uniformly. The claim is falsifiable: if a 1B HRS model cannot outperform a 1B dense model at equivalent FLOPs, routing overhead exceeds routing benefit and the architecture fails its own test.

The WikiText-103 ablation (Section 13) provides preliminary evidence: the full HRS system (50M parameters) achieves best-of-run val_ppl 9.19 at end of Phase 4 versus 23.83 for the dense baseline (38M parameters) — a 61.4% improvement with only 32% more parameters. End-of-training perplexity ranges from 10.05 to 10.51 across engram variants (55.9–57.8% improvement). However, we note that the router-only configurations (configs 3-5) show the opposite pattern: more parameters, worse perplexity. The hypothesis depends on the engram system scaling to larger models, which remains to be validated.

This would demonstrate that **structured compute allocation plus learned compression can substitute for scale**.

---

# 16. Ablation Order and Results

The ablation was designed to test each component's independent contribution. Results (Section 13) reveal a different pattern than predicted:

| Step | Configuration | Predicted | Actual (Val PPL) | Outcome |
|------|--------------|-----------|:----------------:|---------|
| 1 | Dense baseline | — | 23.83 | Reference |
| 2 | + Dual-head (locality) | Measurable gain | 24.08 (+1.0%) | Neutral — geometry shaping alone adds overhead without benefit at this scale |
| 3 | + Router + tiered compute | Measurable gain | 31.63 (+32.7%) | **Wrong** — router tax exceeds routing benefit without engrams |
| 4 | + Sink channel | Measurable gain | 31.90 (+33.9%) | Neutral — sink adds no measurable benefit at 512-token sequences |
| 5 | + Phased training (core only) | Additional gain | 37.84 (+58.8%) | **Wrong** — phased training without engrams hurts; phase transitions destabilize |
| 6 | + Engrams (Phases 1-5) | Additional gain | 12.04 (-49.5%) | **Dramatic gain** — engrams transform the architecture |
| 7 | + Engram refinement (P5) | Additional gain | 10.51 (-55.9%) | Gain vs baseline, but P5 itself regresses from best P4 PPL of 9.19 |
| 8 | + TRC (window=8) | — | 10.05 (-57.8%) | Modest smoothing benefit; smallest P5 regression (+0.62) |

Our prediction that "the core architecture should already demonstrate the thesis before extensions are added" was wrong. The core architecture (steps 2-5) consistently underperforms the dense baseline. The thesis is validated only when engrams are included. This has an important implication: **routing is necessary infrastructure for compression, not an independent source of improvement.** The value chain is routing → structured geometry → effective compression → better perplexity, and removing the final link (compression) leaves only the cost of the preceding links.

The best practical protocol is a 4-phase training schedule (P1–P4), achieving best-of-run val_ppl 9.19 (61.4% below baseline). Phase 5 consistently degrades performance across all engram variants and should be considered optional pending further investigation at larger scale.

Future work should test: (a) whether the router tax decreases at larger model scales where routing can save proportionally more compute; (b) whether TRC provides greater benefit at longer sequence lengths where temporal coherence spans are larger; (c) whether Phase 5 benefits from gentler encoder freezing (gradual LR decay rather than hard freeze) or longer training; (d) multi-seed runs to bound the substantial cross-run variance observed in engram configurations.

---

# 17. Contributions

### Core Architecture

1. Dual-head supervision for geometry shaping, with empirical evidence that single-objective training collapses 98.7% of representational capacity
2. Learned routing with softmax + Gumbel-softmax conservation constraints, balance loss, and entropy regularization — replacing Sinkhorn-Knopp, which we empirically demonstrated fails on non-square routing matrices
3. Convolution-first tiered operators with causal padding, supported by evidence that deeper attention heads compute near-static functions (92.6% variance explained by MLP approximations)
4. Sink channel for interference reduction, informed by measured salience dimensions (89% vs 34% recall by information density)
5. KV memory hierarchy treating the context window as a managed computer, not a passive buffer
6. Learnable tier output gate for magnitude-controlled residual integration, preventing effective rank collapse from tier output mismatch

### Extensions

7. Temporal Routing Cache for amortized routing across coherent spans — empirically tested with causal moving average (window=8), showing modest stability benefit and the smallest Phase 5 regression among engram variants
8. Recursive thread engrams grounded in Hopfield energy minimization, with demonstrated 96% accuracy at 256x compression versus 80% for full-text RAG — and now validated as the decisive HRS component (up to 61.4% perplexity reduction vs dense baseline at best-of-run)
9. Convergence-triggered engram extraction via Jacobian sensitivity, replacing fixed schedules with learned signals
10. Iterative engram refinement exploiting the 24-28% untapped representational capacity in frozen representations — empirically demonstrated to regress at small model scale across all variants tested (see Section 13, Finding 3), establishing that Phase 5 should be considered optional

### Training

11. Phased training protocol with differential learning rates and self-adaptive loss weighting, addressing the empirically demonstrated stacking failure mode
12. Empirical demonstration that routing without compression increases perplexity (the "router tax"), establishing that routing is infrastructure for compression rather than an independent source of improvement

### Empirical Findings

13. Demonstration that dual-objective routing is a Nash equilibrium, not optimal transport — Sinkhorn-Knopp solves the wrong formalism for multi-head architectures, an important negative result for MoE routing research
14. Eight-configuration ablation on WikiText-103 demonstrating up to 61.4% perplexity reduction (23.83 → 9.19 best-of-run) with the full HRS system, including corrected engram refinement and TRC variants
15. Phase dynamics analysis showing conv-dominated bootstrap (68% P1), engram-driven improvement (25% PPL drop in P4), and consistent P5 regression across all three engram variants — establishing that the optimal practical protocol is 4 phases (P1–P4)

---

# References (Author's Prior Work)

* Bee, M. (2025). "Attention Builds Maps, Then Reads Them: The Two Stages of Transformer Intelligence."
* Bee, M. (2025). "Binary Node Oscillation in Neural Networks: A Framework for Distinguishing Learning Failure from Active Computation."
* Bee, M. (2025). "Iterative Refinement: Breaking Through Convergence Plateaus in Neural Language Models."
* Bee, M. (2025). "Memory, Salience, and Why Your LLM Forgets: An Investigation We Actually Need to Run."
* Bee, M. (2025). "Self-Adaptive Loss Weighting Through Network Node Designation."
* Bee, M. (2025). "Static Functions Can Approximate Deep Attention Layers: Evidence from Per-Head Analysis."
* Bee, M. (2025). "The Context Window Is a Computer."
* Bee, M. (2025). "The Protocol Harness: An Operating System Written in Protocols."
* Bee, M. (2026). "Differentiable Time: When Neural Networks Learn They Are Finished."
* Bee, M. (2026). "Engrams: Learned Semantic Compression for Transformers."
* Bee, M. (2026). "Engrams Don't Inject Information — They Retrieve It."
* Bee, M. (2026). "Generative Auxiliary Training as Anti-Collapse Regularization in Large Language Models."
* Bee, M. (2026). "Golden Ratio Engram: A Corrected Account."
* Bee, M. (2026). "MCR2 as Language Model Regularizer: What Works, What Doesn't, and Why the Theory Oversells It."
* Bee, M. (2026). "Phased Specialization: Unlocking Hybrid Sequence Models via Optimization-Aware Training."
* Bee, M. (2026). "Recursive Refinement as Approximate Hopfield Dynamics."
* Bee, M. (2026). "Symmetry Constraints and Scaling Stability: A Noether-Inspired Framework for Neural Architecture Design."
* Bee, M. (2026). "The Four-Lane Highway: Making Sense of DeepSeek's mHC Paper."
* Bee, M. (2026). "We Stacked 3 AI Upgrades. The Combined System Was Worse Than Using Just 1."
* Bee, M. (2026). "What We Got Wrong About Engram Steering (And What We Actually Found)."
