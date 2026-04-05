# From Dot Product to Learned Kernel: A 6-Point Perplexity Improvement by Changing What Attention Computes

*Michael Bonsignore and Claude (Anthropic)*

---

We replaced dot-product attention with a learned exponential kernel in a 510M parameter transformer and improved validation perplexity from 23.26 to 17.07 on WikiText-103 — a 6.19-point improvement from changing the attention scoring function and letting each head specialize. An LLM judge rated the learned kernel's generation quality higher than the dot product baseline in 58% of blind comparisons.

The path from V18 (dot product, 23.26) to V22 (learned kernel, 17.07) took five model versions, 23 experiments, and a series of findings that each built on the last. This paper reports the complete arc.

## The starting point

V18 is a 510M parameter transformer with PEER feed-forward layers (262,144 single-neuron experts), cross-attention engram injection at alternating layers, and a topic categorization head. It uses standard scaled dot-product attention. Trained on WikiText-103 for 43,000 steps on a single RTX 5070 Ti, it achieves 23.26 BPE validation perplexity and a MAUVE score of 0.946 against human-written reference text.

V18 was already a strong model. The PEER architecture makes full attention affordable by moving sparsity to the feed-forward layer, and the cross-attention engram provides corpus-level context without the causal attention leakage that plagued earlier designs. The question was whether the attention scoring function — the last piece we hadn't touched — could do better.

## The kernel experiments

### Five scoring functions on Shakespeare

We trained five identical 10.8M parameter character-level transformers on Tiny Shakespeare, changing only the attention scoring function. Every function trained to comparable validation loss. But the hidden-state representations told a different story.

At play level (256-character segments with real topical structure), topic classification accuracy ranged from 64.4% for dot product to 84.2% for exponential kernel. The dot product — the function the entire field uses — produced the worst topic separation of anything we tested. A frozen random projection beat it by 12 percentage points.

| Kernel | Val Loss | Topic Accuracy |
|--------|----------|---------------|
| Exponential | 1.623 | 84.2% |
| Learned MLP | 1.629 | 79.4% |
| Random projection | 1.640 | 76.8% |
| Soft rank | 1.634 | 68.8% |
| Dot product | 1.644 | 64.4% |

The exponential kernel works by computing negative squared Euclidean distance between projected queries and keys, divided by a learnable temperature: scores = -||q-k||²/τ. After softmax, attention weights decay exponentially with distance rather than scaling linearly with alignment. The gradient is proportional to the function value — learning reinforces itself on the relationships the model has already discovered.

### ALiBi changes the picture

A collaborator noted that RoPE (rotary positional embeddings) was designed for angular dot-product similarity, not Euclidean distance. We tested ALiBi (static linear position bias) as an alternative.

ALiBi transformed the dot product. Dot product + ALiBi achieved 84.8% topic accuracy — higher than exponential + RoPE (84.2%). The positional encoding was the real bottleneck, not the scoring function. But exponential + ALiBi achieved 86.9%, the highest of any combination. The improvements are additive.

### Per-head kernel specialization

We trained a model where each attention head has its own small MLP that refines the exponential scores. When trained from scratch on Shakespeare, different heads learned different kernel shapes. Head-level correlations with pure exponential ranged from r = 0.12 (a nearly novel function) to r = 0.77 (exponential with corrections). No single fixed kernel is optimal — the ideal scoring function varies by attention head.

A scaffolded approach (exponential Phase 1, then learned Phase 2) achieved 86.1% topic separation — the highest of any single model on Shakespeare. The exponential scaffolding gives the projection matrices a geometric head start that the learned kernel builds on.

## Scaling to WikiText-103

### V19: Fixed exponential kernel (23.15)

Replacing dot product with exponential kernel in the full 510M architecture improved validation perplexity from 23.26 to 23.15. Small margin, same direction as Shakespeare. An LLM judge rated the exponential kernel's generation a full point higher on human-likeness (4.72 vs 3.72 on a 10-point scale) across 50 blind comparisons.

### V20: Per-head temperatures (17.34)

Adding a learnable temperature per head produced a dramatic jump. Six heads learned sharp attention (τ ≈ 13-15) while two learned broad attention (τ ≈ 20-21). Validation perplexity dropped from 23.15 to 17.34 — a 5.81-point improvement from per-head temperature alone.

The cross-attention gates became highly selective: layer 3's gate dropped to 0.024 (nearly closed) while layer 5 remained open at 0.223. The model discovered it doesn't need engram injection at the middle depth.

### V21: The alpha reset mistake (18.93)

We tried to unfreeze per-head kernel MLPs by resetting their interpolation weight (alpha) from 0.73 to 0.5. This destroyed the geometry V20 had established. The model never recovered. Validation perplexity plateaued at 18.93 — worse than V20.

The useful finding: per-head output scalars revealed a hierarchy. The most important head contributed 46% more than the least important. And layer 3's cross-attention scalar went to effectively zero, confirming it should be removed.

### V22: Learned kernels from the mature checkpoint (17.07)

The lesson from V21: don't reset when co-evolving. V22 continued from V20's final checkpoint (step 43K), calibrated each head's MLP to match its current exponential behavior, then unfroze for 20,000 steps of co-evolution. Alpha values were preserved at V20's learned 0.72.

This worked. Validation perplexity improved from 17.34 to 17.15 in the first 10,000 steps and to 17.07 after 20,000 steps of extension. The per-head MLPs contributed 20-26% of each head's scoring (alpha range: 0.74-0.80). The broad-temperature heads (τ ≈ 20) received the highest output scalars, meaning the model values broad contextual attention more than sharp local attention.

### V23/V23b: The categorization weight experiments

A Nash bargaining experiment on Shakespeare suggested the categorization objective should be weighted at 0.5, not 0.1. We tested this at WikiText scale.

V23 (fixed 0.5) cost 10 PPL points — the categorization gradient overwhelmed language modeling. V23b made the weight learnable, starting at 0.5. The model drove it to 0.0001 — effectively zero. Given the choice, the model eliminates categorization entirely. The TF-IDF cluster labels used for categorization are too noisy to help language modeling at WikiText scale.

V22 with fixed 0.1 remains the best. The mild categorization regularization doesn't cost much and provides some topic structure in the representations.

## The final model

V22 is a 510M parameter transformer with:

- Bonsignore kernel attention: exponential distance scoring with per-head learnable temperatures and per-head MLPs that co-evolve from a calibrated exponential initialization
- 8 attention heads: 6 sharp (τ ≈ 12-15) and 2 broad (τ ≈ 20)
- Per-head output scalars: broad heads contribute more (0.82) than sharp heads (0.69)
- PEER feed-forward: 262,144 single-neuron experts with product-key routing
- Cross-attention engram injection at layers 1 and 5 only (layer 3 removed)
- Categorization head with fixed 0.1 loss weight

Trained on WikiText-103 for 63,000 steps (~16 hours) on a single NVIDIA RTX 5070 Ti.

### Results

| Metric | V18 (dot product) | V22 (Bonsignore kernel) |
|--------|-------------------|------------------------|
| Validation perplexity | 23.26 | **17.07** |
| LLM judge (vs V18) | — | **29-21 (58% win rate)** |
| Categorization loss | 1.2 | 0.16 |
| Per-head specialization | None | 6 sharp + 2 broad |
| Training time | 11.4 hours | 16 hours |

The 6.19-point perplexity improvement comes from three sources: the exponential kernel geometry (0.11 points, V18→V19), per-head temperature specialization (5.81 points, V19→V20), and learned kernel refinement (0.27 points, V20→V22). Per-head temperature is the dominant factor.

## What we learned along the way

### The dot product is scaffolding

Five different scoring functions produced five different representation geometries from the same architecture and data. A frozen random projection beat the dot product. The learning happens in the projection matrices. The scoring function is the differentiable bottleneck that shapes the gradient landscape, which determines how the projections organize representation space. But the function itself isn't computing semantic alignment. It's scaffolding.

### The exponential kernel's gradient signal is self-reinforcing

The derivative of exp(-d²/τ) with respect to the query is proportional to the function value. The projections receive the strongest update signal where the model has already discovered the strongest relationships. This produces sharper, more specialized representations and implicit regularization against overfitting.

### Different heads want different kernels

On Shakespeare, per-head correlations with exponential ranged from 0.12 to 0.77. At WikiText scale, per-head alphas ranged from 0.74 to 0.80. No single fixed kernel is optimal. The ideal scoring geometry varies by head, and giving each head freedom to specialize produces measurably better representations.

### Temperature is the dominant hyperparameter

The 5.81-point jump from V19 to V20 came entirely from making the exponential kernel's temperature learnable per head. The learned MLP refinement (V22) added only 0.27 more points. The sharpness of attention matters more than the shape of the kernel. Six heads converge to sharp attention (τ ≈ 13) while two converge to broad (τ ≈ 20). This 6+2 split appears consistently across experiments.

### Scaffolded co-evolution works, resetting doesn't

V21 proved that resetting the learned kernel's parameters destroys the geometry established during scaffolding. V22 proved that calibrating the MLP to match current behavior, then allowing gradual drift, preserves the geometry while enabling refinement. The principle: initialize to reproduce, then let gradient descent decide what to change.

### Nash bargaining on objectives doesn't transfer across scales

The model's preferred categorization weight was 50/50 on Shakespeare (binary topic structure) but 0.0001 on WikiText (50-cluster TF-IDF labels). The natural equilibrium depends on the label quality and the complexity of the task. Fixed weights work better than dynamic weights when the auxiliary labels are noisy.

### Broad attention heads are more valuable than sharp ones

Across V20, V21, and V22, the per-head output scalars consistently rank the two broad-temperature heads (τ ≈ 20) above the six sharp heads (τ ≈ 13). The model uses sharp attention for local pattern matching but values broad contextual attention more for generation quality.

### Layer 3 cross-attention is useless

Both V20 and V21 independently drove layer 3's cross-attention gate to near zero. V22 removes it entirely. The engram provides useful context at the first layer (initial conditioning) and the last cross-attention layer (final refinement) but not in the middle.

## Reproducibility

Code: github.com/MikeyBeez/HRS

| Step | Command | Time |
|------|---------|------|
| V18 (baseline) | `python train.py --ablation v18_cross_attn` | 11.4 hours |
| V19 (exponential) | `python train.py --ablation v19_exp_kernel` | 14.3 hours |
| V20 (per-head temps) | `python train_v20.py` | 11.3 hours |
| V22 (learned kernel) | `python train_v22.py` | 4 hours (from V20) |
| Kernel sweep | `python exp_kernel_attention.py --kernels all` | 2 hours |
| ALiBi ablation | `python exp_kernel_attention.py --kernels dot_product_alibi,exponential_alibi` | 40 min |
| Nash objectives | `python exp_nash_objectives.py --conditions A,B,C` | 30 min |

Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM. Total compute for the complete experimental arc: approximately 80 GPU-hours.

## References

Bonsignore, M. (2026). PEER + Engram: 1.71 Perplexity at 510M Parameters on a Consumer GPU. Medium.

Choromanski, K., et al. (2020). Rethinking Attention with Performers.

Press, O., Smith, N., & Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization. ICLR 2022.

Tay, Y., et al. (2021). Synthesizer: Rethinking Self-Attention for Transformer Models.

Vaswani, A., et al. (2017). Attention Is All You Need.
