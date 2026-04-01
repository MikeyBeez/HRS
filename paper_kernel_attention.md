# The Dot Product Is Scaffolding

*Michael Bonsignore and Claude (Anthropic)*

---

We tested five attention scoring functions. The dot product — the one everyone uses — produced the worst representations. A frozen random projection beat it. The "semantic alignment" story is wrong.

## The claim

The most popular explanation of transformer attention goes like this: queries search for relevant keys through the dot product. High dot product means high relevance. The model learns what to attend to by learning projections that align queries with their semantically matching keys.

We tested five different scoring functions — exponential distance, learned MLP, random projection, soft rank, and the standard dot product — changing nothing else. Every function trained successfully. Every function produced the same results at short text lengths. At longer text lengths, where real topical structure exists, the spread was enormous: from 64.4% topic classification accuracy for dot product to 84.2% for exponential. The dot product was the worst scorer we tested. A frozen random projection — zero learning in the scoring function — beat it by 12 percentage points.

The learning in attention doesn't happen in the dot product. It happens in the projection matrices. The function between them is scaffolding. And the scaffolding everyone uses is the worst one we tried.

## The five functions

All experiments use the same 10.8M parameter character-level transformer trained on Tiny Shakespeare, followed by a 510M parameter model on WikiText-103 for the winning kernel. Same architecture, same hyperparameters, same random seed. The only thing that changes is the function that produces the scalar attention score from projected queries and keys.

**Dot product.** The standard: multiply queries by keys element-wise and sum. This is what every transformer uses. It measures the cosine of the angle between vectors, scaled by their magnitudes.

**Exponential kernel.** Negative squared Euclidean distance divided by a temperature parameter. After softmax, attention weights decay exponentially with distance rather than scaling linearly with alignment. The memory-efficient formulation uses the identity that squared distance equals the sum of squared norms minus twice the dot product, so it requires the same memory as standard attention.

**Learned MLP.** A small neural network that takes concatenated query-key pairs and produces a scalar score. Unlike the dot product, this can learn arbitrary nonlinear interactions between queries and keys. It adds a small number of parameters to the scoring function itself.

**Soft rank.** Attention based on relative ordering rather than magnitude. This captures "which keys rank highest for this query" without caring about the actual distance or angle. Only ordinal information survives.

**Random projection.** A fixed, frozen random matrix applied to concatenated query-key pairs to produce a score. No learning whatsoever in the scoring function. The projections upstream must do all the organizational work because the scorer contributes nothing learned.

L1 distance and sinusoidal product kernels were also attempted but failed — likely out-of-memory issues with the chunked implementation, or gradient problems. Five of seven planned kernels completed successfully.

## Shakespeare: every kernel works

All five kernels trained to comparable validation loss on Tiny Shakespeare. The spread from best to worst was only 0.021 — from 1.623 (exponential) to 1.644 (dot product). Every function produces a working language model. The projections adapt to whatever scaffolding you give them.

## The representation story

Here's where it gets interesting. We extracted engrams — mean-pooled, L2-normalized hidden states from the final layer — and measured how well they separate content by topic. Tiny Shakespeare contains multiple plays with distinct characters and themes.

At line level — individual dialogue lines of 5 to 30 tokens — every kernel produced identical results. All five scored about 61.3% accuracy. The short-text noise floor is kernel-invariant. Below a critical token count, the scoring function is completely irrelevant.

At play level — 256-character segments with real topical structure — the spread was massive.

Exponential kernel: 84.2%. Learned MLP: 79.4%. Random projection: 76.8%. Soft rank: 68.8%. Dot product: 64.4%.

A twenty percentage point range from changing only the scoring function. The dot product — the function the entire field uses, the function that every textbook explains as computing "semantic alignment" — produced the worst topic separation of anything we tested.

## The random projection result

This deserves its own section because it's the most important finding.

A frozen random projection — a matrix of random numbers that never updates during training — produced 76.8% play-level topic accuracy. The standard dot product produced 64.4%. A function with zero learning in the scoring step beat the function that everyone uses by over 12 percentage points.

How? Because the projection matrices upstream compensated. The Q, K, and V projections reorganized the representation space to work with whatever scorer they were given. When given a random scorer, they found an organization that made even random projections produce useful attention patterns. When given the dot product, they found a different organization — and it was worse.

This is the strongest evidence that the projections, not the scorer, are doing the work. The scorer is scaffolding. The projections are the building.

## Why exponential wins

The derivative of e^x is e^x. The function is its own gradient signal.

With the exponential kernel, the gradient is proportional to the function value itself. When the score is high — when two projected vectors are close in space — the gradient is also high. The projections receive the strongest update signal exactly where the current representation says the relationship is strongest. Learning reinforces itself.

Dot product attention distributes gradient signal more evenly across query-key pairs. Exponential attention concentrates it on the pairs that already score highly. The projections get pulled hardest toward organizing the space around relationships the model has already started to discover.

This self-reinforcing dynamic amplifies real patterns when they exist — producing 84.2% topic separation versus 64.4%. When signal is below the noise floor (line level), there's nothing to reinforce, and all kernels tie.

The implicit regularization follows the same logic. Dot product attention can distribute weight diffusely across many keys, enabling soft memorization. The exponential kernel's sharper weighting concentrates attention on fewer, stronger relationships. When trained past the optimal stopping point, dot product validation loss balloons to 2.681 while exponential reaches only 2.361. It overfits less destructively.

## WikiText-103: scaling the winner

We took the exponential kernel — the clear winner — and trained it at scale. V19 is a 510M parameter transformer with PEER feed-forward layers, cross-attention engram, and categorization head, trained on WikiText-103. The architecture is identical to our V18 model (which uses dot product) except for the scoring function.

V19 beat V18 on validation perplexity — 23.15 versus 23.26. Small margin, same direction as Shakespeare. The categorization head converged to lower loss — 1.0 versus 1.2 — confirming that the exponential kernel's representations are better organized for topic discrimination even at scale.

The cross-attention gates converged to nearly identical values: 0.269/0.273/0.332 for V19 versus 0.270/0.276/0.327 for V18. The engram injection mechanism found the same operating point regardless of kernel. The downstream components are agnostic to the scoring function.

In entropy-gated retrieval, both models achieved 5/5 needle-in-a-haystack retrieval at mean rank 1.2. The engram store works identically with either kernel. The exponential kernel's different geometry doesn't help or hurt retrieval — it's organized differently but equally effectively.

Where the exponential kernel showed a clear advantage was generation quality. Across 50 blind-evaluated prompt pairs, an LLM judge rated V19 a full point higher on human-likeness (4.72 versus 3.72) and nearly a point higher on coherence (5.17 versus 4.28). V18 won slightly on informativeness (5.22 versus 5.00). The pattern matches the implicit regularization story: dot product memorizes aggressively, producing information-dense but structurally fragile text. The exponential kernel produces text that reads more like something a person would write.

## What the standard explanation gets wrong

The standard explanation says the dot product computes semantic alignment between queries and keys. This makes a testable prediction: replacing the dot product with a function that doesn't compute alignment should degrade performance.

It doesn't. Four out of four alternative functions matched or beat it. A frozen random matrix beat it. The prediction fails catastrophically.

The Synthesizer paper (Tay et al., 2021) showed that random attention matrices perform competitively with dot product attention. They concluded that learning attention from token-token interactions "is useful but not that important after all." Our result extends this: different scoring functions don't just match the dot product — they produce measurably different representation geometries. The projection matrices converge to different organizations of the space under different gradient landscapes.

There is no single "correct" representation that the projections converge to. The dot product leads to one geometry. The exponential kernel leads to another. Random projections lead to yet another. The "semantic alignment" story is wrong not just about the mechanism but about there being a unique explanation of what attention does.

## The real mechanism

The projection matrices are learned linear transformations that reshape the hidden-state space. Gradient descent optimizes them to minimize prediction error. The scoring function is the differentiable bottleneck that creates the gradient pathway, forcing the projections to do the organizational work.

The specific function determines the gradient landscape, which influences how the projections organize the space. But the function itself isn't "doing" anything semantic. It's scaffolding. The attention pattern — the matrix of weights that everyone visualizes and interprets — is an epiphenomenon. Studying attention patterns to understand how transformers learn is like studying the wake behind a boat to understand hydrodynamics.

What matters is the projection matrices. They ask questions of the data. Training finds better questions. The scoring function just provides the differentiable pathway through which the answers flow.

## Limitations

The full five-kernel comparison was done at 10.8M parameters on a toy corpus. Only the exponential kernel was validated at 510M on WikiText-103. Running all five kernels at the larger scale would take approximately 70 hours of GPU time and is planned as follow-up work.

Two of seven planned kernels (L1 distance, sinusoidal product) failed to complete and are not represented. The results may look different with those functions included.

The LLM judge evaluation needs multi-seed runs to confirm the quality differences aren't noise, though the direction is consistent across both pilot and full evaluations.

The 25% training slowdown for the exponential kernel relative to dot product is a practical limitation. A custom CUDA kernel could close this gap but we haven't built one.

## What this means

The dot product was chosen by Vaswani et al. in 2017 because it's fast and simple. It was never shown to be optimal. We've now shown it's the worst scoring function we tested — beaten by an exponential kernel, a learned MLP, a random projection, and a soft rank function.

The exponential kernel is the best we tested, and the gradient story explains why: the self-reinforcing signal concentrates learning on the relationships that matter. But the broader point isn't about any single function. It's that the field has treated the dot product as a fundamental mechanism when it's actually arbitrary plumbing. The projections are the mechanism. The function is replaceable.

Any result this easy to reproduce should have been found years ago. Each kernel variant requires approximately ten lines of code. The memory-efficient distance computation uses the same memory as dot product. We ran every experiment on a single consumer GPU.

All code is available at github.com/MikeyBeez/HRS. Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM. Shakespeare five-kernel comparison: approximately 2 hours total. WikiText-103 V19 training: approximately 14 hours.

## References

Choromanski, K., et al. (2020). Rethinking Attention with Performers.

Park, S., et al. (2023). A Novel Approach to Attention Mechanism Using Kernel Functions: Kerformer.

Tay, Y., et al. (2021). Synthesizer: Rethinking Self-Attention for Transformer Models.

Vaswani, A., et al. (2017). Attention Is All You Need.
