# Magnitude Pruning Ablation — Specification

## Hypothesis

Transformer MLP weights contain substantial redundancy. Magnitude-based pruning at 50–90% sparsity, followed by short fine-tuning, should preserve LM perplexity and passkey retrieval accuracy. The rate at which each capability degrades as sparsity increases will tell us how much fat is in the current architecture and how compressible the model is *after* the diagonal-attention route failed.

Passkey is the canary: specific retrieval circuits are more pruning-sensitive than general language modeling, so passkey will fall before perplexity does.

## Setup

Start from the **MHA variant** from `experiments/diagonal_attention/`. This is the only variant from that experiment that learned passkey, so it's the only meaningful baseline for pruning.

Train MHA to a stronger passkey baseline first — 20K steps instead of 10K, targeting 60%+ passkey accuracy — then checkpoint. All pruning experiments operate on this checkpoint.

File: `experiments/pruning/checkpoints/mha_baseline_20k.pt`

## Variants to Compare

For each sparsity level in {0%, 30%, 50%, 70%, 80%, 90%, 95%}, run four variants:

1. **One-shot, no fine-tune**: prune, evaluate immediately. Measures raw pruning robustness.
2. **One-shot + short fine-tune (500 steps)**: prune, then fine-tune at lr=1e-4 for 500 steps. Measures quick recovery.
3. **One-shot + long fine-tune (2000 steps)**: same but 2000 steps. Measures full recovery.
4. **Iterative pruning**: prune 20% → fine-tune 1000 steps → prune 20% more → fine-tune 1000 steps, repeating until target sparsity. Measures whether iterative beats one-shot at high sparsity.

## Pruning Scope

Three scopes, run independently to isolate effects:

- **MLP-only**: prune only the `W_in` and `W_out` matrices in FFN blocks. Largest param share, most relevant to the PEER composition story.
- **Attention-only**: prune only `W_Q`, `W_K`, `W_V`, `W_O` projections. Tests how sensitive attention circuits are.
- **All weights**: prune everything except embeddings, layer norms, and biases.

Start with MLP-only since it's the cleanest story and the largest fraction of parameters. Extend to the others if MLP-only results are promising.

## Pruning Method

**Per-layer magnitude pruning**. For each eligible weight matrix separately:

1. Compute |w| for all weights in the matrix.
2. Find the threshold at the target percentile (e.g., 80% sparsity → threshold at 80th percentile of |w|).
3. Zero out weights below the threshold.
4. Store a binary mask so the zeros don't regenerate during fine-tuning.

Per-layer (not global) because different layers have different weight distributions and a global threshold would over-prune some layers and under-prune others.

## Training Protocol

- Fine-tuning uses the same data, optimizer, and batch size as original training.
- Fine-tuning lr: 1e-4 (lower than original 3e-4 to avoid overwriting the structure that survived pruning).
- Masks are applied every forward pass: pruned weights stay at zero even if gradients would push them nonzero.

## Metrics

For each (sparsity, fine-tune strategy, scope) combination, record:

- **Validation perplexity** on Tiny Shakespeare held-out set.
- **Passkey exact-match accuracy** (averaged over positions 0-255).
- **Passkey digit accuracy** (partial credit, catches graceful degradation).
- **Actual sparsity achieved** per layer (sanity check on mask application).
- **Effective param count** (total nonzero weights).

## Implementation note (deviation from the literal spec)

The diagonal_attention scaffold trains LM and passkey with different vocabularies
(65 Shakespeare chars vs 13 digit tokens), so "one checkpoint" measured on both
metrics is not directly possible without joint training. This experiment keeps
things simple by training **two task-specific MHA baselines** (same architecture,
same hyperparameters) and applying **the same pruning recipe** to each. Pairs of
metrics per config come from pairing the pruned LM model's PPL with the pruned
passkey model's accuracy. The hypothesis is about architectural redundancy
under pruning, which this setup tests directly.
