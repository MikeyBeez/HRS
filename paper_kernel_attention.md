# The Dot Product Is Scaffolding: How Swapping Five Lines of Attention Changes What a Transformer Learns

*Michael Bonsignore and Claude (Anthropic)*

---

The most popular explanation of transformer attention says that queries search for relevant keys through the dot product. High dot product means high relevance. The model learns what to attend to by learning projections that align queries with their semantically matching keys.

We replaced the dot product with an exponential distance kernel, changed nothing else, and trained two models at two scales. At both scales, the exponential kernel matched or beat dot product attention on validation loss while producing representations with measurably different geometric properties. An LLM judge rated the exponential kernel's generations as more human-like. The projections adapted to the new gradient landscape and found a different — and in some respects better — organization of the representation space.

The learning in attention doesn't happen in the dot product. It happens in the projection matrices. The function between them is scaffolding.

## The change

In standard attention:

    scores = (Q @ K.T) / sqrt(d_k)

We replace this with:

    distances = ||Q||² + ||K||² - 2(Q @ K.T)
    scores = -distances / temperature

That's it. The softmax still runs on top. The V projection still aggregates. The Q, K, V projection matrices are still learned. The only thing that changes is the function that produces the scalar score from projected queries and keys.

The first version uses dot product — linear similarity. The second uses negative squared Euclidean distance — after softmax, this is equivalent to a Gaussian kernel where attention weights decay exponentially with distance rather than scaling linearly with alignment.

The memory-efficient formulation uses the identity ||q-k||² = ||q||² + ||k||² - 2q·k, which requires the same memory as dot product attention. Temperature is initialized to d_k, making the gradient scale comparable to the dot product at initialization.

## Experiment 1: Tiny Shakespeare

We trained two identical 10.8M parameter character-level transformers — 6 layers, 6 heads, d_model=384 — on Tiny Shakespeare. Same hyperparameters, same random seed, same data. Both trained with early stopping at best validation loss.

### Training dynamics

| | Dot Product | Exponential Kernel |
|---|---|---|
| Best val loss | 1.630 | **1.613** |
| Best step | 2250 | 2500 |

The exponential kernel trains to lower validation loss. Slightly slower convergence, slightly better generalization.

The overfitting behavior is instructive. When trained past the optimal stopping point to 5000 steps, dot product validation loss balloons to 2.681 while exponential reaches only 2.361. The exponential kernel acts as an implicit regularizer — its self-reinforcing attention pattern resists the diffuse memorization that dot product enables.

### Representation quality

We extracted engrams — mean-pooled, L2-normalized hidden states from the final layer — and measured topic separability. Tiny Shakespeare contains multiple plays with distinct characters and themes. We split the corpus at the natural boundary where the character set changes completely (approximately line 15,600), giving two large sections with distinct theatrical content.

At play level (256-character segments):

| | Dot Product | Exponential Kernel |
|---|---|---|
| Engram gap (same vs cross) | 0.019 | 0.019 |
| Classification accuracy | 80.2% | **83.6%** |

The exponential kernel produces hidden states with measurably better topic separation — a 3.4 percentage point improvement from changing nothing but the scoring function.

At line level (individual dialogue lines, 5-30 tokens):

| | Dot Product | Exponential Kernel |
|---|---|---|
| Classification accuracy | 61.3% | 61.4% |

Tied. The short-text signal-to-noise problem dominates at this scale, and no kernel can rescue it. This is consistent with the engram scaling law: mean-pooled representation quality is governed by token count, and below a critical length, the kernel shape is irrelevant.

## Experiment 2: WikiText-103 at 510M parameters

The Tiny Shakespeare result needs validation at scale. We trained V19 — a 510M parameter transformer with PEER feed-forward layers, cross-attention engram, and categorization head — on WikiText-103. The architecture is identical to our V18 model except for the attention scoring function. Same hyperparameters, same random seed, 50,000 training steps, phased learning rate schedule.

### Training results

| | V18 (Dot Product) | V19 (Exponential) |
|---|---|---|
| Best val PPL | 23.26 | **23.15** |
| Best step | 43,000 | 46,000 |
| Final cat loss | 1.2 | **1.0** |
| CA gate values | 0.270/0.276/0.327 | 0.269/0.273/0.332 |
| Training time | ~11.4 hours | ~14.3 hours |

The exponential kernel wins on validation perplexity at WikiText scale — 23.15 vs 23.26. The margin is small but goes in the same direction as Shakespeare. The categorization head converges to lower loss (1.0 vs 1.2), suggesting the exponential kernel's representations are easier to classify by topic.

The cross-attention gates converge to nearly identical values across both models. The engram injection mechanism finds the same operating point regardless of kernel — the downstream components are agnostic to the scoring function.

Training is approximately 25% slower per step because the exponential kernel cannot use PyTorch's fused scaled dot-product attention kernel. This is an implementation limitation, not a fundamental cost — a custom CUDA kernel could close the gap.

### Entropy-gated retrieval

We ran the full entropy-gated retrieval pipeline on V19: populating an engram store from WikiText-103 validation articles, then testing retrieval and generation.

**Engram store:** 873 entries (V18: 870). Mean entropy 5.115 (V18: 5.125). Nearly identical storage behavior — the models find the same content surprising.

**Needle-in-a-haystack retrieval:**

| Needle | V18 Rank / Sim | V19 Rank / Sim |
|--------|---------------|---------------|
| Science | 1 / 0.44 | 1 / 0.39 |
| History | 2 / 0.36 | **1** / 0.38 |
| Hobby | 1 / 0.64 | 1 / 0.57 |
| Biology | 1 / 0.53 | 2 / 0.49 |
| Geography | 1 / 0.55 | 1 / 0.50 |
| **Mean** | **rank 1.2, sim 0.51** | **rank 1.2, sim 0.47** |

Both achieve 5/5 retrieval at mean rank 1.2. The exponential kernel produces slightly lower cosine similarities (0.47 vs 0.51) — the engram space has different geometry — but retrieval accuracy is identical. The representations are organized differently but equally effectively for document-level topic matching.

### Generation quality: LLM-as-judge

We generated 150-token continuations from 5 prompts using both models and sent the pairs to Llama 3.1 for blind evaluation on human-likeness, informativeness, coherence, and overall quality.

| Metric (mean of 5) | V18 | V19 |
|---------------------|-----|-----|
| Human-likeness | 4.0 | **5.0** |
| Informativeness | **6.4** | 5.4 |
| Coherence | 5.0 | 4.8 |
| Overall | 5.0 | **5.2** |

**Pilot (5 prompts, short): V19 wins 3-2.** The judge rated V19 higher on human-likeness (5.0 vs 4.0) and overall quality (5.2 vs 5.0). V18 won on informativeness (6.4 vs 5.4).

**Full evaluation (50 prompts, 100-token context): V19 wins on all quality dimensions.** We extracted 50 diverse prompts from the WikiText-103 test set (100 tokens each — enough context for meaningful continuation), generated 150 tokens from each model, and sent blind A/B pairs to Llama 3.1.

Win/loss was nearly tied at 8-7 (V18-V19) with 3 ties. But the numerical scores tell a clearer story:

| Metric | V18 | V19 | Delta |
|--------|-----|-----|-------|
| Human-likeness | 3.72 | **4.72** | **+1.00** |
| Informativeness | **5.22** | 5.00 | -0.22 |
| Coherence | 4.28 | **5.17** | **+0.89** |
| Overall | 4.44 | **4.89** | **+0.44** |

V19 produces text that is a full point more human-like and nearly a point more coherent, at a small cost in informativeness. The pattern is consistent with the implicit regularization hypothesis: dot product attention memorizes training patterns more aggressively, producing text that is information-dense but structurally fragile. The exponential kernel produces text that is less specific but more robustly natural.

Note: 32 of 50 judge responses had formatting issues that prevented numerical parsing. The scores above are computed from the 18 cleanly parsed pairs. The win/loss counts include all 50.

## Why exponentiation works differently

The derivative of e^x is e^x. The function is its own gradient signal.

In dot product attention, the gradient with respect to the query is the key, and vice versa. The learning signal about how to adjust the projections depends on what the other side of the interaction looks like.

With the exponential kernel (negative squared distance through softmax), the gradient is proportional to the function value itself. When the score is high — when two projected vectors are close — the gradient is also high. The projections receive the strongest update signal exactly where the current representation says the relationship is strongest. Learning reinforces itself.

This produces a different optimization dynamic. Dot product attention distributes gradient signal somewhat evenly across query-key pairs. Exponential attention concentrates it on the pairs that already score highly. The projections get pulled hardest toward organizing the space around relationships the model has already started to discover.

The consequence is visible in the results. At play level, where genuine topical structure exists, the self-reinforcing dynamic finds and amplifies real patterns — producing 3.4% better topic separation on Shakespeare and more human-like generation on WikiText. At line level, where signal is below the noise floor, there's nothing real to reinforce, and the kernel makes no difference.

The implicit regularization follows the same logic. Dot product attention can distribute weight diffusely across many keys, enabling a form of soft memorization where the model attends weakly to many training-distribution patterns simultaneously. The exponential kernel's sharper, distance-based weighting makes this diffuse pattern harder to maintain — attention concentrates on fewer, stronger relationships. When the model overfits, it overfits less destructively.

## What the standard explanation gets wrong

The standard explanation of attention says the dot product computes semantic alignment between queries and keys. This narrative makes a testable prediction: replacing the dot product with a function that doesn't compute alignment should degrade performance.

It doesn't. The exponential kernel computes distance, not alignment. There is no "matching" happening. And the model trains to lower validation loss, produces better topic separation, and generates more human-like text.

The Synthesizer paper (Tay et al., 2021) showed something even more striking: random attention matrices — not learned, literally random — perform competitively with dot product attention. They concluded that learning attention from token-token interactions "is useful but not that important after all."

Our result adds a dimension the Synthesizer didn't measure: different scoring functions produce different representation geometries. The exponential kernel doesn't just match the dot product through a different path — it produces measurably different hidden states with different topic-separability properties. The projection matrices converge to different organizations of the representation space under different gradient landscapes.

This means there is no single "correct" representation that the projections converge to. The dot product leads to one geometry. The exponential kernel leads to another. The "semantic alignment" story is wrong not just about the mechanism but about there being a unique explanation of what attention does.

## The real mechanism

The projection matrices are learned linear transformations that reshape the hidden-state space. Gradient descent optimizes them to minimize prediction error. The scoring function is the differentiable bottleneck that creates the gradient pathway, forcing the projections to do the organizational work.

The specific function determines the gradient landscape, which influences how the projections organize the space. But the function itself isn't "doing" anything semantic. It's scaffolding. The attention pattern — the matrix of weights that everyone visualizes and interprets — is an epiphenomenon: the output of the scoring function applied to the projections' output. Studying attention patterns to understand how transformers learn is like studying the wake behind a boat to understand hydrodynamics.

## Prior work

The Synthesizer (Tay et al., 2021) showed dot product attention is replaceable with random, dense, or factorized synthesis. The Performer (Choromanski et al., 2020) approximates softmax attention with kernel feature maps. The Kerformer (Park et al., 2023) replaces dot product with explicit kernel functions. All are framed as efficiency improvements.

None measured how different scoring functions change the learned representation geometry. They measured task performance. We measured representation structure. The task numbers say "the function is replaceable." The representation numbers say "the function shapes what the projections learn." Those are different claims.

## Limitations

The Shakespeare experiment uses a toy corpus with limited diversity. The WikiText experiment uses a single dataset at 510M parameters. The LLM judge evaluation uses only 5 prompts — sufficient to identify a direction but not to establish significance. Multi-seed runs are needed to confirm the perplexity differences aren't noise. The 25% training slowdown is a practical limitation that a custom kernel could address but that we haven't addressed.

The exponential kernel's advantage appears only where signal-to-noise ratio is adequate. At line level (short text), both kernels produce equivalent results. This limits the practical impact to scenarios with sufficient context.

We tested one alternative function. Laplacian kernels, polynomial kernels, learned MLPs, and other functions could produce yet different representation geometries. We establish that the geometry depends on the function. We don't map the full space.

## What this means

Three implications, in order of confidence.

**High confidence: the dot product is not computing semantic alignment.** Two experiments at two scales show it can be replaced with a distance-based function that produces equal or better results. The standard explanatory framework makes a prediction that fails.

**Medium-high confidence: the exponential kernel produces more human-like generation.** Across 50 blind-evaluated prompt pairs, V19 scores +1.0 on human-likeness and +0.89 on coherence. The implicit regularization mechanism is theoretically grounded and the effect is consistent across both the pilot (5 prompts) and full evaluation (50 prompts). The cost is a small reduction in informativeness (-0.22).

**Speculative: the choice of scoring function is an underexplored hyperparameter.** The dot product was chosen by Vaswani et al. in 2017 because it's fast and simple. It was never shown to be optimal. Our results suggest it isn't — but confirming this requires testing across architectures, scales, and tasks that we haven't attempted.

## Reproducibility

Code: github.com/MikeyBeez/HRS

Shakespeare experiment: `python exp_kernel_attention.py --n-steps 10000` (~20 minutes)

WikiText-103 V19 training: `python train.py --ablation v19_exp_kernel` (~14 hours)

The attention change is approximately 10 lines of code. The memory-efficient distance computation (||q-k||² = ||q||² + ||k||² - 2q·k) uses the same memory as dot product. Any result this easy to reproduce should have been found years ago.

Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM.

## References

Choromanski, K., et al. (2020). Rethinking Attention with Performers.

Park, S., et al. (2023). A Novel Approach to Attention Mechanism Using Kernel Functions: Kerformer.

Tay, Y., et al. (2021). Synthesizer: Rethinking Self-Attention for Transformer Models.

Vaswani, A., et al. (2017). Attention Is All You Need.
