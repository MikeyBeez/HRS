# The Engram Is an Address: Test-Time Memory via Stable Geometric Routing Spaces

Anonymous Authors

## Abstract

We present evidence that mean-pooled transformer hidden states function primarily as addresses into a geometric routing space rather than as content summaries. The same engram retrieves 0/20 passkeys before test-time training and 20/20 after, with no change to the vector — only the model's learned content at that address changes. Random vectors of identical norm route at 0/10 (chance: 10%) and achieve K-space alignment of 0.525 versus the real engram's 0.856, confirming the engram's utility resides in its specific direction, not in non-zero activation of the residual stream. The key empirical observation is a phase transition in resolvability: retrieval jumps from 0% to 100% in 38 training steps while K-space alignment remains constant (0.84-0.87) throughout, cleanly separating the geometric routing structure (which exists before training) from the content that training provides. Cross-model transfer experiments show this routing geometry is stable under training perturbation: engrams transfer between models fine-tuned from a shared checkpoint with only 2.5% K-space alignment loss. All core findings — including the K/V asymmetry, information recovery floor, cross-model transfer, and adapter routing — replicate on a standard softmax dot-product transformer, confirming these are consistent properties of the Q/K/V attention decomposition across the two distinct transformer implementations tested.

From this decomposition, three applications emerge. (1) A per-passage adapter library achieves 100% routing correctness and 97% held-out retrieval via a learned L0-to-L5 projection, with saliency-weighted absorption enabling 2x rank compression and 2x faster convergence. (2) Engram-compressed KV caches close 92% of the full-context NLL gap at 2x compression, with learned attention pooling lifting the information recovery floor from 18% to 30%. (3) Activation-level block-stacking composes multiple adapters without the bilinear cross-term interference of weight merging, achieving 4/5 compositional retrieval at K=2.

Seven independent experiments attempting to improve V-space information preservation through reconstruction losses, gated residuals, metadata enrichment, and bilinear attention reveal a consistent tradeoff: interventions that improve V-space organization degrade language modeling quality, and vice versa. Under all architectures tested that preserve standard next-token prediction objectives and unconstrained residual streams, V-space lossiness appears load-bearing, with an observed information recovery limit of 30%. Validated across 62 experimental phases on a single consumer GPU.

## 1. Introduction

Mean-pooled hidden states are ubiquitous in NLP as passage representations. We argue this usage understates their nature. A mean-pooled hidden-state vector primarily functions as an address into the model's representation geometry; its utility depends more on the model's learned content at that address than on the information encoded in the vector itself.

The distinction is operational. We construct 20 passages containing unique facts the base model cannot guess. The layer-5 mean-pooled engram of each passage, injected as a single hidden-state prefix in front of a retrieval prompt, retrieves 0/20 facts before any test-time training. After per-passage LoRA absorption (150 gradient steps, 1.4 seconds per passage on consumer hardware): 20/20. The vector did not change. The model's ability to interpret it did.

**The key empirical observation** is a phase transition in resolvability (Figure 1). Training 5 passages at step counts from 0 to 300, retrieval jumps from 0/5 at 0 steps to 4/5 at 15 steps to 5/5 at 38 steps. K-space cosine alignment is constant throughout — 0.84 to 0.87 at every step count. The geometric routing structure exists before any training. Training does not deepen the basin in K-space; it fills the basin in weight space with the content that the address points to. The address is always valid. The content becomes available through training.

This separation of routing geometry from learned content predicts behavior across three deployment surfaces. Per-passage adapter libraries (Section 4) use the engram as a routing key — possible because routing requires address fidelity, which K-space provides at 0.82-0.89 cosine. KV cache compression (Section 5) uses the engram as a content proxy — limited to 30% information recovery under all conditions tested because V-space is lossy at 0.65-0.70 cosine. Compositional retrieval (Section 6) sums engram-selected adapters at the activation level — limited to K=2 by operator interference in the residual stream, not by representation overlap or total parameter budget.

The experimental arc comprises 62 phases across four days on a single RTX 5070 Ti. Of these, 7 are positive results that define the architecture, 12 are measurements that characterize the routing space, and 43 are negative results that constrain the design space. We report all of them.

## 2. Related Work

**Test-time training.** TTT methods update model parameters at inference to adapt to new inputs (Sun et al., 2020; Gandelsman et al., 2022). These approaches collapse memory into parameters, causing catastrophic forgetting when multiple passages are absorbed sequentially — a failure we reproduce (Phase 12: +473% perplexity drift after 20 passages) and circumvent through per-passage adapter isolation.

**Retrieval-augmented generation.** RAG (Lewis et al., 2020; Borgeaud et al., 2022) collapses memory into context, paying quadratic attention cost. Our adapter library separates the indexing function (engram cosine, O(1) per query) from the content function (adapter loading, O(1) per passage), avoiding the context-length scaling that RAG requires.

**LoRA and parameter-efficient fine-tuning.** Hu et al. (2022) introduced low-rank adaptation. We extend LoRA from a fine-tuning technique to a memory primitive: each adapter is a learned deformation of the model's manifold at a specific address, indexed by the engram of its training prompt and loaded on demand.

**Knowledge editing.** Methods that surgically modify model weights for specific facts (Meng et al., 2022; Mitchell et al., 2022) collapse memory into mutation. Our decomposition keeps the base model frozen and stores knowledge in modular, composable adapter pages.

**KV cache compression.** Prior work includes quantization (Hooper et al., 2024), token eviction (Zhang et al., 2024), and learned compression (Nawrot et al., 2024). Our engram-as-cache approach operates at a different granularity — replacing contiguous token spans with single mean-pooled vectors — and characterizes the information-theoretic tradeoffs of this compression.

**Mixture of experts.** Our compositional retrieval (Section 6) is structurally a content-addressable MoE where experts are per-passage adapters, the router is the engram cosine, and expert combination is activation-level block-stacking. Unlike standard MoE (Shazeer et al., 2017; Fedus et al., 2022), our routing requires no learned router parameters and adding a new expert is O(1).

## 3. The Routing Space

### 3.1 Centroid theory of context

Each attention head computes a convex combination of value vectors. We hypothesize that the mean-pooled hidden state — a first-order approximation of this centroid — lands in the same K-space basin as the full passage. For in-distribution content, the model's learned projections map the centroid to key vectors that are structurally indistinguishable from the passage's mean key vectors.

We define a **basin** as a set of initial hidden states whose forward trajectories, under the fixed model dynamics, converge to similar final-layer representations. Basin validation (Section 7.3) makes this operational: after LoRA training, the cosine similarity between engram-injection and full-passage hidden states grows from 0.03 at layer 0 to 0.45 at layer 5, demonstrating literal trajectory convergence through the layer stack. The adapter lifts final-layer convergence by 14 points over the no-adapter condition.

### 3.2 K-space and V-space alignment

We extract two engrams per passage — the L5 mean (hidden states after all blocks) and the L0 mean (input embeddings before any block) — inject each as a single hidden-state position, and measure cosine similarity between the engram's K/V projections and the passage's mean K/V at every layer. Fifty WikiText passages, 128 tokens each.

**K-space alignment.** The L5 engram achieves K cosine 0.82-0.89 at layers 1-5, with a bootstrap failure at layer 0 (L5 vectors are not what W_K at layer 0 was trained on). The L0 engram achieves 0.988 at layer 0, decaying to 0.65 at layer 5. Each engram is well-aligned at its extraction layer. Random baseline is approximately 0.

**V-space alignment.** K consistently exceeds V (0.82-0.89 vs 0.65-0.70 for L5). This asymmetry reflects different computational roles: K determines which positions attention selects (routing, requiring fidelity), V determines what flows back (content, where compression is beneficial). The asymmetry predicts the performance profile of all three applications.

**V-space effective rank.** SVD analysis of the V matrices at each layer reveals effective rank approximately 48. The first singular vector captures only 11-24% of variance. SVD-optimal engrams recover negative information (-12.5%), worse than no context — the maximum-variance direction is dominated by magnitude patterns, not semantic content.

### 3.3 The L0/L5 duality

L0 embeddings are pre-attention, high-entropy, and preserve discriminative variance that L5 compresses. L5 embeddings capture processed understanding but collapse surface-form variation. This duality determines the architectural choice: L0 for routing (Section 4), L5 for content when the engram is the sole information source (Section 5).

The duality motivates the projection bridge: a 1024x1024 linear map W trained with InfoNCE contrastive loss, learning the L5-discriminative directions and rotating L0 queries onto them. This gives L5's discriminative power at L0's inference cost (one embedding lookup plus one matmul, no forward pass).

### 3.4 The phase transition in resolvability

We absorb 5 passages at varying fractions of the standard 150-step training budget: 0, 15, 38, 75, 113, 150, 225, and 300 steps.

Retrieval is a step function: 0/5 at 0 steps, 4/5 at 15 steps, 5/5 at 38 steps, 5/5 at all subsequent step counts. K-space cosine alignment is constant throughout — 0.84 to 0.87 at every step count, including step 0.

This is the cleanest evidence for the address/content decomposition. The routing geometry exists before training. The architecture creates a stable K-space basin for each passage, and this basin does not deepen or sharpen with training. What training provides is the content: the LoRA adapter that produces the right output when the routing geometry directs attention to the right region. Resolvability is a phase transition in weight space, not a gradual process in representation space.

### 3.5 Stability under training perturbation

If the routing geometry is determined primarily by the architecture and training distribution, it should be stable under weight perturbation. We test this by fine-tuning the base model for 5,000 steps with a different random seed, creating Model B with meaningfully different weights.

K-space alignment of Model A's engrams in Model B: 0.842 (mean over 5 passages). In Model A: 0.864. Degradation: 2.5%. Model B's own engrams in Model B: 0.879.

The routing geometry is largely invariant within a training basin and appears induced by architectural constraints — the projection matrices impose structure that is determined by the architecture and training distribution more than by specific weight values. What differs between models is the content stored at each address. Validating full independence across training runs from different random initializations remains future work; the current result establishes stability under moderate perturbation, not universality.

### 3.6 Random vector ablation

A natural alternative explanation is that the engram functions as a generic semantic embedding — any semantically meaningful vector would work equally well. We test this by injecting random vectors of identical norm into the same pipeline. Random vectors (matched to the engram's L2 norm) achieve K-space cosine alignment of 0.525 versus the real engram's 0.856, and route to the correct adapter 0/10 times versus 10/10 for the real engram (chance baseline: 10%). A zero vector achieves intermediate K-cos of 0.602 — better than random because it does not send attention to the wrong region, but worse than the real engram because it carries no directional information. The engram's utility is entirely in its specific direction within representation space, not in its magnitude or in non-zero activation of the residual stream.

This does not fully rule out a semantic interpretation — the engram's direction likely correlates with semantic content. But it establishes that the relationship is geometric (direction-sensitive, not magnitude-sensitive) and that arbitrary activation of the residual stream does not suffice. The engram must point to the right region of K-space for routing to succeed, which is what we mean by "address."

### 3.7 Cross-architecture validation

All core findings replicate on a standard 127M-parameter softmax dot-product transformer with dense MLP feed-forward (no PEER, no learned kernel, no Sinkhorn — see Appendix B for details). The K/V asymmetry holds and is wider (K at 0.85-0.94, V at 0.56-0.70). Mean-pooling information recovery is 16.4% (vs 18.0%). Cross-model transfer degradation is -2.1% (routing geometry is fully stable). Same-prompt adapter retrieval achieves 18/20. The phase transition in resolvability is present but slower on the smaller model (3/5 at 150 steps vs 5/5 at 38 steps on the larger model), consistent with a capacity difference rather than a qualitative change.

These results confirm the findings are consistent across two distinct transformer implementations — one with learned exponential kernels and PEER expert-retrieval FFN (512M parameters), one with standard softmax attention and dense MLP (127M parameters). The routing geometry, the K/V asymmetry, the information recovery floor, and the cross-model stability are properties of the Q/K/V attention decomposition as observed in these architectures, not artifacts of any specific attention kernel or feed-forward mechanism.

## 4. Application 1: Per-Passage Adapter Library

### 4.1 Architecture

A frozen 510M-parameter base model (6 transformer blocks, d_model=1024, 8 attention heads with per-head learned exponential kernels, PEER expert-retrieval FFN). LoRA adapters (Hu et al., 2022) on layers 4-5, rank 128, alpha 256, targeting 8 projection matrices (Q, K, V, output for each layer). Each adapter: 2.6M parameters. Training: 150 steps of next-token prediction loss with learning rate 3e-4 decaying to 1e-4, on the passage plus 3 paraphrased prompt-answer pairs.

### 4.2 Routing

The routing key is the L0 mean of the query. Library keys are L5 means of training prompts, computed under the base model with LoRA weights zeroed. The query's L0 embedding is projected through W (InfoNCE-trained 1024x1024 projection) and compared by cosine to all stored L5 keys. Threshold: 0.50.

Routing performance: 60/60 correct on held-out paraphrases (100% routing correctness). Retrieval: 58/60 (97%). The 2 failures are decoding errors (correct adapter loaded, greedy decoding fails to produce the passkey). A cosine-threshold gate achieves 99% recall and 98% specificity. An identity-autoencoder novelty gate tested at 48% specificity (worse than random); combining it with cosine reduced recall by 21 points.

**Leakage check.** Routing is a cosine comparison between precomputed CPU vectors, not a forward pass — no attention, no token sequence, no path for information leakage. L5 keys and L0 queries are computed with LoRA weights zeroed. The matched adapter loads only after the routing decision is recorded. Manual inspection of all 60 held-out generations confirmed zero substring false positives.

### 4.3 Saliency-weighted absorption

A learned attention-pooling encoder (3M parameters, trained in 30 seconds on 2,000 WikiText passages to minimize continuation NLL) identifies informative token positions. Weighting the NTP loss by these saliency scores during absorption concentrates gradient on content-bearing tokens.

Results: rank-64 with saliency weighting achieves 20/20 same-prompt retrieval, matching rank-128 with uniform loss. Same retrieval at 75 steps instead of 150. This represents 2x rank compression and 2x convergence speedup. Saliency scores are computed once per passage before absorption and are not updated during training.

### 4.4 Storage and deployment

Rank sweep: rank 128 matches rank 512 on all tests. Rank 256 scores 19/20 (interference, not capacity). Rank 16 achieves 8/8 on same-prompt retrieval — the per-adapter floor for exact-match queries is much lower than for paraphrases. Int8 quantization: 4x additional compression with zero quality loss. Combined: 32x compression at rank-64 + int8. A 1,000-passage library fits in 1.3 GB. Absorption latency: 1.4 seconds per passage.

### 4.5 Adapter clustering is impossible

For all 5 closest passage pairs in the benchmark (L0 cosine 0.96-0.97), continuing training on passage B fully erased passage A (0/5 survived in either training order). Adapters are local deformations of the model's manifold; two passages require two incompatible deformations. The one-adapter-per-passage architecture is necessary even for near-identical passages.

## 5. Application 2: Engram-Compressed KV Cache

### 5.1 Compression curve

Six prefix conditions on 50 WikiText passages (200-token context, 56-token continuation). Engram replacing the distant half + recent tokens kept: 92% gap closed at 2x compression. Replacing 75%: 84% gap closed at 4x. The curve is smooth from 0 to 384x with no knee. Each compression doubling costs approximately 0.5 PPL points.

**Recency asymmetry.** Compressing distant context closes 92% of the gap; compressing recent context closes only 32%. Distant context needs only the right neighborhood; recent context requires token-level precision.

### 5.2 Learned engram encoders

Mean pooling recovers 18% of context information. Attention pooling (single learned query, 3M parameters, 30 seconds training): 28% (+10 points). Five learned queries (K=5): 30% at 40x compression. K=10 does not improve over K=5.

### 5.3 The V-space tradeoff

Seven independent experiments attempted to push past 30%, revealing a consistent pattern: interventions that improve V-space organization degrade language modeling quality under standard transformer architectures with unconstrained residual streams.

**Phase 48:** KL-divergence regularization during LoRA absorption. Trades retrieval for drift reduction at a ratio the architecture doesn't need.

**Phase 49/49b:** Metadata-enriched embeddings. Bolt-on: NLL explodes 2.87 to 8.87. From scratch: 4.4% PPL gain with metadata, but V-space unchanged and metadata-free path degrades.

**Phase 54:** Saliency-weighted continued training. PPL drifts +22% in 2,000 steps.

**Phase 56:** V-space reconstruction loss (auxiliary decoder at each layer forcing W_V to preserve input). V-space cosine improves 0.71 to 0.73, PPL degrades 21 to 43, recovery drops 18% to 14%. Extra information in V amplifies across layers and destabilizes training.

**Phase 57:** Reconstruction loss with sigmoid-gated residual connections (convex combinations instead of unconstrained addition). V-space alignment holds at 0.805 — 10 points above baseline — throughout training. But constrained signal flow caps PPL at 46 and recovery at 7%. This is the sharpest demonstration: the constraint that preserves V-space organization is the same constraint that limits learning capacity. The unconstrained residual stream is what makes transformers powerful, and it is what makes V-space lossy.

**Phase 58:** Full bilinear attention (q^T W k, d_head x d_head matrices per head, identity-initialized). PPL matches standard model (44 vs 44). Recovery gains 3 points (15.7% vs 12.6%). Bilinear weights move far from identity (||W-I|| approx 2.9, full rank, sigma_1 at 1.6%). The model uses all d^2 cross-dimensional interactions extensively, but the gain does not break the observed limit.

**Summary.** Under all architectures tested that preserve standard next-token prediction and unconstrained residual streams, V-space lossiness and language modeling quality are in tension. We have not tested fundamentally different architectures (e.g., linear attention, state-space models) and the observed 30% limit may be addressable through approaches outside the design space explored here. Within the tested space, the evidence is consistent: downstream layers require abstracted, compressed representations, and the K/V asymmetry (K at 0.82-0.89, V at 0.65-0.70) reflects the different requirements of the routing channel (fidelity) and the content channel (abstraction).

## 6. Application 3: Compositional Retrieval

### 6.1 Weight merging fails

Averaging two LoRA adapters' weights: 0/5 BOTH. The cross terms (A1+A2)/2 . (B1+B2)/2 → x.A1.B2 + x.A2.B1 project through untrained basis combinations. Result is rank-independent.

### 6.2 Activation-level block-stacking

Concatenating rank dimensions — A_stacked = [A1 | A2], B_stacked = [B1 ; B2] — computes x.A1.B1 + x.A2.B2 exactly. Combined with clause-split routing: 4/5 BOTH with 10/10 routing correctness.

### 6.3 The K=2 ceiling is K-limited

K-capacity sweep across K in {1, 2, 4, 8} and rank in {128, 64, 32, 16}:

At equal total budget (K x rank = 128): K=1 retrieves 100%, K=2 retrieves 40-50%, K=4 retrieves 0%. At equal budget 256: K=2 retrieves 50%, K=4 retrieves 0%. The pattern holds at every budget level and every rank tested. K=4 is always 0%.

The ceiling is about the number of simultaneous operators on the residual stream, not their size or total parameter count. L5-contrastive orthogonalization during absorption reduces representation overlap by 32% but K-capacity regresses (K=2: 90% to 70%), confirming the ceiling is a property of the LoRA matrices as linear operators, not of the representations they produce.

**Why K=2?** Our hypothesis is that the residual stream has limited capacity for concurrent additive perturbations. Each rank-R LoRA adapter adds a rank-R perturbation to the hidden state at each layer. At K=2, the two perturbations occupy disjoint rank subspaces and their sum is representable at rank-2R. At K=4, the four perturbations compete for the residual stream's effective bandwidth — the model must simultaneously maintain four distinct perturbation directions, and the attention mechanism (which sees the summed hidden state, not the individual components) can no longer resolve which perturbation to amplify at which position. The transition from K=2 to K=4 is sharp rather than gradual, suggesting a capacity threshold rather than smooth degradation. We do not have a formal proof that K=2 is the theoretical limit; it may be possible to push to K=3 with architectural modifications (e.g., per-adapter gating on the residual contribution) that we have not tested.

## 7. Validation Experiments

### 7.1 Cross-model address transfer (Phase 59)

Described in Section 3.5. K-space alignment degrades by only 2.5% under model transfer (5,000-step fine-tuning with different seed). Establishes routing geometry stability under training perturbation.

### 7.2 Resolvability curve (Phase 60)

Described in Section 3.4. Step-function retrieval with constant K-space alignment establishes that training fills content at pre-existing addresses.

### 7.3 Basin validation (Phase 61)

Forward pass with full passage vs. forward pass with engram + prompt. Hidden-state cosine at each layer:

No adapter: 0.03 (L0) to 0.31 (L5). Adapter loaded: 0.03 (L0) to 0.45 (L5). Adapter lifts final-layer convergence by 14 points. Monotonic growth L0 to L5 demonstrates trajectory convergence. Conditions "base" and "trained-but-adapter-zeroed" are identical, confirming no information leaks through training alone.

### 7.4 K-ceiling rank sweep (Phase 62)

Described in Section 6.3. K-limitation confirmed across all ranks 128 to 16 and all budget levels.

## 8. Negative Results

We report 43 negative experimental phases. Key categories:

**Shared-adapter approaches** (Phases 12, 19, 20, 53): Full-model TTT drifts +473%. Shared LoRA: 50-75%, chaotic. Rank 1024 worse than 512. Clustering impossible at cosine 0.97.

**Routing approaches** (Phases 21-25): Passage-derived keys: 60%. Single-key: 50% on paraphrases. Six L5 pooling strategies. Stopword filtering: correct idea, wrong layer.

**V-space interventions** (Phases 48, 49, 49b, 54, 56, 57, 58): Seven approaches, all confirming the V-space tradeoff (Section 5.3).

**Composition interventions** (Phase 46): L5-contrastive absorption reduces overlap but worsens K-capacity, establishing the ceiling as operator-level.

## 9. Discussion

### Stable geometric routing spaces

The phase transition result (Section 3.4) is our strongest evidence for the address/content decomposition. K-space alignment is constant from 0 to 300 training steps while retrieval undergoes a phase transition at 15-38 steps. Cross-model transfer (Section 3.5) shows this alignment is stable under training perturbation. The random vector ablation (Section 3.6) confirms the engram's utility is direction-specific, not a consequence of generic residual-stream activation. Together these suggest that the embedding space defines the map — a geometric routing structure — and training fills in the territory.

The routing geometry's stability is partly a consequence of shared embeddings: models trained on the same data with the same tokenizer share the same input geometry, and the projection matrices preserve much of that structure. This does not diminish the finding — it sharpens it. The address space is embedding-induced geometry, and attention's K-space projections preserve it with high fidelity. The content at each address is weight-specific and must be learned.

If this property holds more broadly — across fully independent training runs, larger scales, and more diverse architectures — it would provide a mechanistic account of why transfer learning works (the routing space transfers; only content needs adaptation), why embeddings generalize across tasks (they address the same geometry), and why different models trained on similar data produce alignable representations (they share the same routing space). The cross-architecture validation (Section 3.7) provides initial evidence in this direction but is limited to two architectures at similar scale.

### V-space lossiness as a design principle

The most unexpected finding is that V-space lossiness and language modeling quality are consistently in tension under the architectures tested. Phase 57 is the sharpest demonstration: gated residuals hold V-space alignment 10 points above baseline throughout training, but the same constraint that preserves V-space organization caps learning capacity at PPL 46 vs baseline 21.

This suggests a design principle for the practical applications: rather than trying to make V-space less lossy, engineer the reading and writing operations to work within the lossy channel. Attention pooling (18% to 28% recovery), multi-token engrams (28% to 30%), and saliency-weighted absorption (2x compression, 2x speed) all follow this principle.

### Three points in a tradeoff space

The three applications operate at different points on the fidelity-compression tradeoff. Application 1 operates in K-space (high fidelity, 0.82-0.89 cosine). Application 2 operates in V-space (lossy, 0.65-0.70 cosine, 30% recovery limit under tested conditions). Application 3 operates in operator space (K=2 ceiling from residual-stream interference). Each deployment surface is constrained by a different property of the same underlying geometry.

## 10. Limitations

The benchmark uses 20 passages with template paraphrases — small and synthetic. Routing is clean at 20 passages; behavior at 100,000 is unmeasured. Application 2 is tested on WikiText prose only; code, math, and structured data are untested. The K=2 ceiling on Application 3 is the binding constraint; operator-level fixes (sqrt(K) scaling, sequential generation, per-adapter gating) are untested. The cross-model transfer experiment uses a fine-tuned variant from a shared checkpoint, not a fully independently trained model; the 2.5% result establishes stability under perturbation, not universality across arbitrary training runs. The cross-architecture validation covers two implementations at similar scale; larger-scale and more diverse architectures (linear attention, state-space models) are untested. The 30% information recovery limit is observed under architectures that preserve standard NTP objectives and unconstrained residual streams; fundamentally different designs may not share this tradeoff. The routing geometry's stability is partly a consequence of shared embeddings; the extent to which the geometry is embedding-induced versus architecture-induced is not fully disentangled. The random vector ablation confirms direction-sensitivity but does not rule out semantic explanations entirely — the engram's effective direction likely correlates with semantic content, and separating "address" from "semantics" at a philosophical level remains open. The Phase 47 projection W is library-specific. Prepend-as-token engram injection is a lower bound; direct K/V cache injection would likely improve Application 2.

## Reproducibility

All code at [repository] under experiments/identity_ae/. 67 standalone deterministic scripts with full per-trial JSON results. Core experimental phases run in approximately 20 minutes on a consumer GPU (RTX 5070 Ti, 16 GB VRAM). Pre-training ablation phases (48-58) require approximately 60 GPU-hours additional.

## Acknowledgments

Developed in collaboration with Claude (Anthropic) across all 62 phases. The centroid-theory framing, the per-passage adapter architecture, the block-stacking linearization, the L0 routing finding, and the V-space tradeoff characterization emerged through iterative human-AI collaboration. The validation experiments were suggested by an external review that correctly identified the claims needing stronger evidence. Experimental design decisions and the determination of which negative results were diagnostic were the human author's. Implementation and iteration were collaborative.

## References

Borgeaud, S., et al. (2022). Improving language models by retrieving from trillions of tokens. ICML.

Fedus, W., et al. (2022). Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. JMLR.

Gandelsman, Y., et al. (2022). Test-time training with masked autoencoders. NeurIPS.

Hooper, C., et al. (2024). KVQuant: Towards 10 million context length LLM inference with KV cache quantization. NeurIPS.

Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR.

Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.

Meng, K., et al. (2022). Locating and editing factual associations in GPT. NeurIPS.

Mitchell, E., et al. (2022). Fast model editing at scale. ICLR.

Nawrot, P., et al. (2024). Dynamic memory compression: Retrofitting LLMs for accelerated inference. ICML.

Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. ICLR.

Sun, Y., et al. (2020). Test-time training with self-supervision for generalization under distribution shifts. ICML.

Zhang, Z., et al. (2024). H2O: Heavy-hitter oracle for efficient generative inference of large language models. NeurIPS.

## Appendix A: Complete Phase Listing

Phase 0: Identity autoencoder training on V22 hidden states.
Phase 1: OOD detection via reconstruction error.
Phases 3-4: Engram extraction and integrated gating.
Phase 5: Simple test-time training baseline.
Phase 7: LoRA test-time training.
Phases 9-13: Dual gate, passkey benchmark, LR scheduling, forgetting measurement.
Phases 14-17: L4-5 LoRA sweep, forgetting analysis, stratified evaluation.
Phases 18-20: Gate replay, shared-LoRA rehearsal (negative), rank-1024 interference (negative).
Phase 21: First per-passage adapter library (passage keys, 60% routing — negative).
Phase 22: Engram key source ablation (L3/L5, mean/last, L2/cosine).
Phases 23-24: Prompt-derived keys (100% routing), 150-step adapters (100/100 same-prompt).
Phases 25-26: Paraphrase failure (50%), multi-key + multi-paraphrase fix (100/100).
Phase 27: Held-out paraphrase generalization (63% with L5 mean).
Phase 28: Staged absorption (zero-blocking deployment).
Phase 29/29b: Compositional weight merging (0/5 BOTH, rank-independent — negative).
Phase 30/30b: Int8 quantization (4x, zero quality loss).
Phase 31: Entity-weighted pooling (63% to 77% with nonstop_mean).
Phase 32/32b: K-space alignment measurement (K: 0.82-0.89, V: 0.65-0.70) and L0 dual (0.988).
Phase 33/33b: Engram-as-cache compression curve, L0 variant.
Phase 35: Centroid theory test (0/20 before TTT, 20/20 after).
Phase 37: Full compression sweep (0 to 384x).
Phase 38/38b: Rank sweep (floor at 128), held-out at rank 128 (100/100/77).
Phase 39: Gate ablation (cosine 99/98, gate 48% specificity — negative for gate).
Phase 40: Sparse magnitude pruning (29x total with held-out tradeoff).
Phase 41: Activation-level block-stacking (4/5 BOTH oracle).
Phase 42/42b: Clause-split routing (4/5 BOTH, 10/10 routing), L0 variant.
Phase 43: K-capacity sweep (K=2 holds, K=4 collapses to 0/5).
Phase 44: L0 mean routing (100/100/90 held-out, +15 over L5).
Phase 45: L0+L5 pair (does not dominate either alone — negative).
Phase 46: L5-contrastive absorption (K-capacity regresses — negative).
Phase 47/47b: L0-to-L5 projection (100/100/97), manual generation inspection.
Phase 48: KL-MLP regularizer (negative).
Phase 49/49b: Metadata enrichment bolt-on and from-scratch (negative).
Phase 50: V-space SVD analysis (effective rank ~48, SVD engrams negative).
Phase 51: Attention-pooling encoder (18% to 28% recovery).
Phase 52: Saliency-weighted absorption (2x rank, 2x speed).
Phase 53: Adapter clustering (impossible even at cosine 0.97 — negative).
Phase 54: Saliency-weighted continued training (PPL +22% in 2K steps — negative).
Phase 55: Multi-token engrams (K=5 at 30%, K=10 no improvement).
Phase 56: V-space reconstruction pre-training (V-cos 0.73, PPL 43 — negative).
Phase 57/57b: Gated residuals + reconstruction (V-cos 0.805, PPL 46 — negative).
Phase 58: Bilinear attention full scale (PPL matched, +3 points recovery — marginal).
Phase 59: Cross-model address transfer (2.5% K-space loss — routing space is stable).
Phase 60: Resolvability curve (phase transition at 15-38 steps, constant K-space).
Phase 61: Basin validation (trajectory convergence L0 to L5, cosine 0.45 with adapter).
Phase 62: K-ceiling rank sweep (K-limited, not budget-limited, across all ranks).
Phase 63: Standard softmax transformer pre-training (127M params, PPL 20.15).
Phase 64: Cross-architecture validation on softmax model (all five core findings replicate).
Phase 65: Random vector ablation (random 0/10 routing, real 10/10; K-cos 0.525 vs 0.856).
