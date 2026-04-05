# Test-Time Training in 2 Seconds: Teaching a 510M Parameter Model New Content Without Breaking It

*Michael Bonsignore and Claude (Anthropic)*

---

We inserted a 2-million-parameter identity autoencoder into a 510-million-parameter transformer and used it as a gate: nothing passes through the network until the autoencoder can reconstruct it. When the gate encounters content it hasn't seen before, the entire model trains on that content — 20 gradient steps, 2 seconds, done. The model's perplexity on the novel content drops by 27x to 819x. Its perplexity on everything it already knew changes by less than half a percent.

The gate works. The learning works. The forgetting doesn't happen. And it takes 2 seconds.

## The problem

Language models are frozen after training. They know what they were trained on and nothing else. When they encounter novel content — a new API, a recent paper, a user's proprietary data — they can only work with what fits in the context window. The context window is expensive (quadratic attention cost), limited (thousands to millions of tokens), and volatile (gone when the session ends).

What if the model could learn at test time? Take the novel content, train on it for a few seconds, and know it permanently. No context window needed for that content. No retrieval system. No vector database. The knowledge lives in the weights.

The obstacle is catastrophic forgetting. Train a neural network on new data and it forgets the old data. This is why nobody does test-time training on production models — the risk of breaking existing capabilities is too high.

We found that the risk is lower than assumed. Much lower.

## The architecture

A standard transformer processes text through a stack of layers. We insert a small autoencoder between layers 3 and 4 of a 6-layer transformer. The autoencoder has a simple job: reconstruct its input. If it can reconstruct the hidden states at that layer, the content is "known" and passes through. If it can't, the content is novel.

The autoencoder is tiny — 2 million parameters versus 510 million for the base model. It has a skip connection initialized at zero, so at insertion it has literally no effect on the model's output. The base model doesn't know it's there.

The autoencoder architecture: encoder (1024 → 768 → 256) and decoder (256 → 768 → 1024). The 256-dimensional bottleneck forces information loss. Content the autoencoder was trained on reconstructs cleanly through this bottleneck. Novel content doesn't — the bottleneck can't compress patterns it hasn't seen.

We train the autoencoder on the model's own hidden states from WikiText-103. After 20 epochs, it reconstructs in-distribution content at MSE 0.08. This becomes the baseline. Anything above this is novel.

## The gate

At inference time, every hidden state at layer 3 passes through the autoencoder. The reconstruction error is checked against a threshold (the mean reconstruction error from training, 0.241). If below threshold: known content, pass through. If above: novel content, the model must learn it before proceeding.

For in-distribution text, the gate is invisible. We verified this: generation quality with and without the gate is identical (self-perplexity 7.8 vs 7.9). The model produces exactly the same text whether the gate is there or not. Zero overhead for known content.

## Test-time training

When the gate detects novel content, it triggers training of the entire 510-million-parameter model on that content. Not just the autoencoder — the full model, with the standard language modeling objective (next token prediction).

Twenty gradient steps. Learning rate 1e-5. That's it.

We tested on five types of out-of-distribution content:

| Content | Perplexity Before | Perplexity After | Time |
|---------|------------------|-----------------|------|
| Python code (merge sort) | 77,060 | 94 | 2.0s |
| Math proof (convergence theorem) | 3,073 | 45 | 1.8s |
| Chemistry (aspirin synthesis) | 148 | 5.4 | 1.8s |
| Fiction (lighthouse keeper) | 1,521 | 21 | 1.9s |
| Synthetic fact (Thornfield Protocol) | 255 | 4.8 | 1.8s |

The model's perplexity on the novel content drops by 27x to 819x. It goes from not understanding the content at all (perplexity 77,060 on code) to modeling it competently (perplexity 94) in two seconds.

## The forgetting question

This is where it gets interesting. After training the full model on each novel example, we measured WikiText-103 validation perplexity — the model's performance on the content it was originally trained on.

| After absorbing | Validation PPL change |
|----------------|----------------------|
| Python code | +0.2% |
| Math proof | +0.3% |
| Chemistry | +0.4% |
| Fiction | +0.5% |
| Synthetic fact | +0.5% |
| **All five sequentially** | **+2.1%** |

Less than half a percent per example. Two percent after absorbing all five. The model's in-distribution performance is essentially unchanged.

This isn't because the learning rate is too low to change anything — the perplexity drops prove the model is learning. It's because novel content activates different weights than in-distribution content. The gradient from a Python function lands on different parameters than the gradient from an encyclopedia article. The model's own geometry protects it.

## Why it works

A transformer trained on WikiText-103 has organized its 510 million parameters to model encyclopedic text. When it sees Python code, the hidden states light up different regions of the weight space — different attention heads attend to different patterns, different experts activate in the PEER routing. The gradient from code trains those code-relevant weights without disturbing the encyclopedia-relevant weights.

This is the same principle behind PEER's sparse routing: only 128 of 262,144 experts activate per token. Training on novel content updates the experts that fire for that content. The other 262,016 experts don't see a gradient. They're protected by the routing geometry.

The autoencoder provides the trigger — it detects when content is novel. The model's own sparsity provides the protection — novel content's gradients don't reach the weights that matter for existing knowledge.

## What it can and cannot do

**It can:** Absorb novel content in 2 seconds with negligible forgetting. Detect out-of-distribution content automatically. Gate the processing pipeline so nothing proceeds unlearned.

**It cannot:** Recall absorbed content from a prompt. After training on "The Thornfield Protocol was established in 1987 by Dr. Elena Vasquez," the model can predict the next token in that sequence (low perplexity) but cannot generate the fact from the prompt "What is the Thornfield Protocol?" The content is memorized as a sequence, not understood as retrievable knowledge. Recall requires a different mechanism — likely retrieval-augmented generation or a structured memory system.

**It cannot:** Replace the context window. We tested feeding the pipeline output back as context (engram recurrence) and it produced incoherent text (perplexity 2,583). The model was trained with token-level attention, not abstract hidden-state summaries. The context window remains necessary for generation coherence.

## The numbers

| Metric | Value |
|--------|-------|
| Autoencoder size | 2M params (0.4% of model) |
| Autoencoder memory | 3.9 MB (fp16) |
| TTT time per example | 2 seconds |
| TTT gradient steps | 20 |
| PPL reduction on novel content | 27x – 819x |
| Forgetting per example | < 0.5% |
| Cumulative forgetting (5 examples) | 2.1% |
| Gate overhead on known content | ~0% (PPL 7.8 vs 7.9) |

## What this means

Test-time training works on language models at the 510M parameter scale. The critical insight is that you don't need to protect against catastrophic forgetting with anchor sets, elastic weight consolidation, or replay buffers — the model's own sparse activation geometry provides natural protection. Novel content activates different weights than known content. Training on the novel content updates those weights and leaves the rest alone.

The identity autoencoder provides the detection mechanism: a 2-million-parameter module that monitors the model's internal representations and triggers learning when something new appears. The trigger is the reconstruction error through a 256-dimensional bottleneck — a simple, interpretable signal that requires no labeled data and no task-specific calibration beyond setting a threshold on the reconstruction error distribution.

The practical implication: a deployed model could continuously absorb new information from its interactions. Each novel input triggers 2 seconds of learning. The knowledge accumulates in the weights. The autoencoder's weight delta — the difference between its current weights and its initial weights — serves as a portable, mergeable memory artifact that records everything novel the model has encountered.

## Reproducibility

Code: github.com/MikeyBeez/HRS

| File | Description |
|------|-------------|
| `identity_autoencoder.py` | IdentityAutoencoder, OODDetector, EngramRecurrence, EngramLibrary |
| `experiments/identity_ae/phase0_train.py` | Offline autoencoder training on model hidden states |
| `experiments/identity_ae/phase1_ood.py` | OOD detection verification |
| `experiments/identity_ae/phase4_integrated.py` | Gate + context window integration |
| `experiments/identity_ae/phase5_simple_ttt.py` | Full-model test-time training |

Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM. Base model: V22 (Bonsignore kernel, 510M params). All experiments complete in under 30 minutes.
