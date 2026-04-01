# Vaswani et al. Don't Understand Attention

*Michael Bonsignore and Claude (Anthropic)*

---

The most popular explanation of transformer attention goes something like this: queries search for relevant keys through geometric alignment. The dot product measures how well they match. High dot product means high relevance. The model learns what to attend to by learning projections that align queries with their semantically matching keys.

This explanation is wrong. Or at best, it's a fiction that happens to be consistent with the math while having nothing to do with what the system is actually doing.

Here's how we know: we replaced the dot product with an exponential distance kernel, changed nothing else, and got better results. If the dot product were computing meaningful semantic alignment, removing it should hurt. It didn't. The projections adapted, learned a different geometry, and produced representations with better topic separability — all without the dot product "matching" anything.

The learning in attention doesn't happen in the dot product. It happens in the projection matrices. The Q, K, and V weight matrices — which contain the vast majority of the attention block's parameters — are what reshape the representation space during training. The function between them is just a differentiable bottleneck that creates a gradient pathway. Swap it for any other differentiable function that produces a scalar, and the projections still learn. They just learn differently.

## The experiment

We trained two identical transformers on Tiny Shakespeare — same architecture, same hyperparameters, same data, same random seed. Six layers, six heads, character-level vocabulary. The only difference: the attention scoring function.

Model A uses the standard scaled dot product. Queries times keys, divided by the square root of the head dimension, fed through softmax.

Model B uses a negative squared Euclidean distance, scaled by a temperature parameter, fed through the same softmax. This is mathematically equivalent to a Gaussian (RBF) kernel — after softmax, the attention weights follow an exponential decay with distance rather than scaling linearly with alignment.

We trained both to convergence with early stopping at best validation loss.

## Training dynamics

The exponential kernel trains to slightly lower validation loss: 1.613 versus 1.630 for dot product. It reaches its best checkpoint at step 2500 versus step 2250. Slightly slower convergence, slightly better generalization.

If the dot product were essential to the learning process — if "queries finding matching keys" were the mechanism driving learning — the exponential kernel should train to worse loss. It doesn't. The projections find a way to minimize prediction error through a completely different scoring geometry. The optimization landscape is different, but the projections navigate it to a comparable destination.

The overfitting behavior is instructive. When trained past the optimal stopping point, dot product attention overfits much harder. At 5000 steps, dot product validation loss balloons to 2.681 while exponential only reaches 2.361. The exponential kernel acts as an implicit regularizer — its sharper, self-reinforcing attention pattern prevents the diffuse memorization that dot product enables.

## Representation quality

We extracted engrams — mean-pooled, L2-normalized hidden states from the final layer — and measured topic separability at two scales.

At play level (256-character segments), the exponential kernel produces hidden states with measurably better topic separation. Accuracy at distinguishing same-play from different-play segments: 83.6% for exponential, 80.2% for dot product. A 3.4 percentage point improvement from changing nothing but the scoring function.

At line level (individual dialogue lines, 5 to 30 tokens), both models are essentially tied: 61.4% for exponential, 61.3% for dot product. The short-text problem dominates — neither kernel can rescue the semantic signal when there aren't enough tokens for mean-pooling to denoise.

This pattern is revealing. The kernel shape matters where the signal-to-noise ratio is adequate. Where signal is too weak, the kernel is irrelevant — both produce noise. The effect is multiplicative: you need sufficient token count to produce signal, and then the kernel shape determines how well the projections organize that signal into geometrically separable representations.

## Why exponentiation

The choice of exponential kernel wasn't arbitrary. The derivative of e to the x is e to the x. The function is its own gradient signal.

In dot product attention, the gradient with respect to the query is the key, and vice versa. The learning signal about how to adjust the projections depends on what the other side of the interaction looks like. The gradient and the function are different mathematical objects with different shapes.

With exponentiation, the gradient is proportional to the function value itself. When the score is high — when two projected vectors are close in the distance space — the gradient is also high. The projections receive the strongest update signal exactly where the current representation says the relationship is strongest. Learning reinforces itself.

This produces a different optimization dynamic. Dot product attention spreads gradient signal somewhat evenly across all query-key pairs. Exponential attention concentrates it on the pairs that already score highly. The projections get pulled hardest toward organizing the space around relationships the model has already started to discover.

At play level, where there's genuine topical structure to discover, this self-reinforcing dynamic finds and amplifies real patterns. At line level, where the signal is below the noise floor, there's nothing real to reinforce, so the dynamic makes no difference.

## What the standard explanation gets wrong

The standard explanation of attention — the one you see in textbooks, blog posts, and beautifully animated YouTube videos — says the dot product computes semantic alignment between queries and keys. Queries "search for" relevant keys. The model "learns what to attend to" through the dot product similarity.

This narrative makes a testable prediction: replacing the dot product with a function that doesn't compute alignment should degrade performance. It doesn't. The exponential kernel computes distance, not alignment. There's no "matching" happening. And the representations are better.

The Synthesizer paper (Tay et al., 2021) showed something even more striking: random attention matrices — not learned, literally random — perform competitively with dot product attention. A Random Synthesizer was 60% faster and improved perplexity by 3.5%. They concluded that learning attention from token-token interactions "is useful but not that important after all."

Our result adds a new dimension: it's not just that the dot product can be replaced — it's that different replacements produce different representation geometries. The exponential kernel doesn't just match the dot product's representations through a different path. It produces measurably different hidden states with different topic-separability properties. The geometry of the learned space depends on the scoring function.

This means the projection matrices are not converging to a single "correct" organization of the representation space. They're finding different organizations under different gradient landscapes. The dot product leads to one geometry. The exponential kernel leads to another. Neither is canonical. The "semantic alignment" story isn't just wrong about the dot product — it's wrong about there being a single correct explanation of what attention does.

## The real mechanism

If the dot product isn't computing semantic alignment, what is actually happening in attention?

The projection matrices are learned linear transformations that reshape the hidden-state space. Gradient descent optimizes them to minimize prediction error. Over billions of updates, this reshaping organizes the space so that abstract structure — topic, meaning, relationships between concepts — becomes geometrically explicit.

The scoring function — dot product, exponential distance, or apparently even random noise — is the differentiable bottleneck that creates the gradient pathway. It forces the projections to do the organizational work. The specific function determines the gradient landscape, which influences how the projections organize the space, which is why different functions produce different geometries. But the function itself isn't "doing" anything semantic. It's scaffolding.

The attention pattern — the matrix of weights that people visualize and interpret — is an epiphenomenon. It's the output of the scoring function applied to the projections' output. It's what you can draw a picture of, which is why everyone studies it. But studying attention patterns to understand how transformers learn is like studying the wake behind a boat to understand hydrodynamics. The wake is real. It's caused by the boat. But it's not the thing doing the work.

## Prior work

The Synthesizer paper (Tay et al., 2021) is the closest prior work. They showed that dot product attention can be replaced with random, dense, or factorized attention synthesis. Their conclusion — that token-token interaction is "useful but not that important" — supports our claim about the dot product being replaceable.

The Performer architecture (Choromanski et al., 2020) approximates softmax attention using kernel feature maps, motivated by computational efficiency. The Kerformer (Park et al., 2023) explicitly replaces dot product with kernel functions. Both are framed as efficiency improvements — reducing quadratic complexity — rather than as investigations of what the dot product is doing.

What none of these papers did is measure how different scoring functions change the learned representation geometry. They measured task performance (perplexity, BLEU, accuracy). We measured representation structure (engram separability). The task performance numbers say "the function is replaceable." The representation structure numbers say "the function shapes what the projections learn." Those are different claims, and the second one is deeper.

## What this means for the field

Three implications.

First, the explanatory framework that dominates how people learn about transformers — queries searching for keys through dot product alignment — is not just simplified but wrong. It makes a testable prediction (replacing the dot product should hurt) that fails empirically. The field needs a better explanation of why attention works, and that explanation will center on the projection matrices, not the scoring function.

Second, the choice of scoring function is an underexplored hyperparameter. The dot product was chosen by Vaswani et al. in 2017 because it's fast and simple. It was never shown to be optimal. Our results suggest the exponential kernel produces better-organized representations at adequate signal-to-noise ratios, with the added benefit of implicit regularization against overfitting. This is a small-scale result on Tiny Shakespeare — it needs validation at transformer scale — but the direction is clear enough to warrant investigation.

Third, the implicit regularization property of exponential attention may matter more than the representation quality difference. In practice, perfectly identifying the early stopping point is difficult. A scoring function that degrades gracefully under overfitting — maintaining representation quality even when trained past the optimum — has practical value independent of its peak performance.

## Limitations

This is a small-scale experiment on a toy corpus. Tiny Shakespeare has a single domain (plays), a small vocabulary (65 characters), and limited topical diversity. The 3.4 percentage point improvement at play level is consistent and reproducible but modest in absolute terms. Scaling to a full-sized transformer on a diverse corpus (WikiText-103, The Pile) is necessary before making strong claims about the exponential kernel's advantages.

The line-level null result means this doesn't solve the short-text representation problem. At very short sequence lengths, the signal-to-noise ratio is too low for any kernel shape to help. This is consistent with the engram scaling law we described in our earlier work on topic-routed context assembly — mean-pooled representation quality is governed by token count, and below a critical length, the kernel is irrelevant.

We tested one alternative function. The space of possible differentiable scoring functions is large — Laplacian kernels, polynomial kernels, learned MLPs, and other exotic functions could produce yet different representation geometries. Our experiment establishes that the geometry depends on the function. It doesn't map the full space of possibilities.

## Reproducibility

Code: github.com/MikeyBeez/HRS (exp_kernel_attention.py)

Model: 6-layer, 6-head character-level transformer, d_model=384, d_k=64, context length 256, trained on Tiny Shakespeare. Exponential kernel temperature initialized to d_k.

Hardware: NVIDIA RTX 5070 Ti. Both models train in under 20 minutes.

The experiment requires changing approximately 5 lines of code in the attention computation and rerunning the training script. Any result that's this easy to reproduce and this surprising should have been found years ago.

## References

Choromanski, K., et al. (2020). Rethinking Attention with Performers.

Park, S., et al. (2023). A Novel Approach to Attention Mechanism Using Kernel Functions: Kerformer.

Tay, Y., et al. (2021). Synthesizer: Rethinking Self-Attention for Transformer Models.

Vaswani, A., et al. (2017). Attention Is All You Need.
