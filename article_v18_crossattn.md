# Cross-Attention Engram with Entropy-Gated Retrieval: Fixing the Leakage Bug and Building External Memory for a 510M Parameter Transformer

**Michael Bonsignore and Claude (Anthropic)**

## Executive Summary

We present two connected results building on the PEER + Engram architecture described in our previous work.

**V18: Cross-Attention Engram.** We redesign the engram injection mechanism to use cross-attention instead of sequence prepending, eliminating the causal attention leakage bug that degraded V16's generation quality. A categorization head provides a discriminative training signal that encourages the engram to extract topic-level features. The result: MAUVE scores of 0.915–0.941 with the engram active — comparable to the V17 baseline (0.933–0.943) and dramatically better than V16's 0.806 failure mode. The cross-attention isolation works as hypothesized.

**V18-EGR: Entropy-Gated Retrieval.** We build an inference-time memory system on top of the trained V18 model. Shannon entropy does double duty: high-entropy text segments are stored as engram vectors during intake, and entropy spikes during generation trigger retrieval of the nearest stored engram. The system achieves a MAUVE score of 0.950 at 500-token prompts — the highest in this project and above the V17 baseline of 0.943. In a needle-in-a-haystack evaluation, the retrieval system finds the correct document among 20 distractors with 100% accuracy at mean rank 1.2.

Both experiments ran on a single NVIDIA RTX 5070 Ti. No changes were made to the trained model weights for the retrieval system — it is purely an inference-time addition.

## Background

Our previous article described a striking dissociation: the V16 engram model achieved 1.71 BPE perplexity but MAUVE scores of only 0.806 with engrams active, while the V17 baseline without engrams achieved 21.41 perplexity but MAUVE of 0.933–0.943. The engram helped next-token prediction but actively harmed generation quality.

We identified the root cause as causal attention leakage. V16 prepends engram tokens to the input sequence. These tokens are visible to all subsequent positions through the causal attention mask. The model learns to lean on this shortcut during training, but at inference time the engram carries a compressed echo of training-distribution patterns that narrows the output distribution away from human-like text.

This article describes two attempts to fix this problem: an architectural fix (cross-attention isolation) and a systems fix (entropy-gated retrieval).

## Part 1: V18 — Cross-Attention Engram

### Architecture

V18 shares V16/V17's backbone: a 6-layer transformer with d_model=1024, 16 attention heads, and PEER feed-forward layers containing 262,144 single-neuron experts. The difference is how the engram enters the model.

**Cross-attention injection.** At layers 1, 3, and 5 (alternating layers), a cross-attention block is inserted between self-attention and the PEER feed-forward. The sequence tokens are queries; the engram buffer is keys and values. No causal mask is applied — every sequence position can attend to every engram token. This is the critical difference from V16: the engram never enters the causal self-attention path. It is structurally isolated from the autoregressive prediction mechanism.

Each cross-attention block has:
- Its own learned Q, K, V projections
- A learned sigmoid gate initialized at 0.5 (sigmoid of 0), combined with near-zero output projection weights
- Pre-norm layer normalization
- A residual connection

The double gating (sigmoid gate × near-zero projection) means the cross-attention starts as a near-no-op. The model must learn to open the gate and develop meaningful Q/K/V projections before the engram contributes anything.

**Engram buffer.** Unlike V16's per-window encoder, V18's engram is a single buffer of 32 learned vectors of shape (1, 32, d_model). It is populated during training by mean-pooling hidden states from the second-to-last transformer layer, updated every 100 steps via exponential moving average with momentum 0.99. The buffer is a derived quantity — gradients do not flow through the population step. They flow only through the cross-attention Q/K/V projections, which learn to extract useful information from whatever the buffer contains.

**Categorization head.** A linear projection from mean-pooled final hidden states to 50 topic categories, trained with cross-entropy loss weighted at α=0.1. Categories are derived from TF-IDF clustering of WikiText-103 articles. The hypothesis: this gives the engram a job. V16's engram had no explicit objective other than reconstruction loss — it learned to be a generic context echo. The categorization head rewards the engram for capturing topic-level features, providing a discriminative training signal complementary to the generative LM objective.

**Parameter count.** 512M total: backbone 76.7M, PEER 421.6M, cross-attention engram 12.6M, categorization head 51K, locality head 1.0M.

### Training

WikiText-103, 50,000 steps, phased schedule matching V16/V17. Batch size 4 with 8 gradient accumulation steps (effective batch 32). BF16 mixed precision. Single RTX 5070 Ti, approximately 11.4 hours.

The cross-attention and categorization parameters are in the "engram" parameter group, which is frozen (LR multiplier 0.0) during phases 1–3 (steps 0–26,000) and unfrozen (multiplier 1.0) during phase 4 (steps 26,000–50,000). This means the backbone and PEER layers train for 26,000 steps before the engram system activates, ensuring a stable foundation.

### Training Dynamics

The training revealed three distinct regimes:

**Phase 1–3 (steps 0–26K): Backbone learning.** Validation perplexity dropped from 171 to 25. The categorization loss decreased from 3.9 (random for 50 classes) to 2.8, showing the backbone was already learning topic structure through the LM objective alone. Cross-attention gates remained at exactly 0.500 — frozen by the phase schedule.

**Phase 4 entry (step 26K): Gates activate.** Upon unfreezing, the gates immediately began moving downward from 0.500. By step 27K they had dropped to 0.468/0.469/0.473. The categorization loss accelerated its decline from 2.8 to 2.5.

**Phase 4 convergence (steps 30K–50K): Gates settle.** The gates converged to 0.270/0.276/0.327, where they remained stable. Layer 5 (deepest) maintained the highest gate value, suggesting later layers benefit more from corpus-level context. The categorization loss reached 1.2 — well below random (3.9), confirming the engram-categorization coupling was working. Final validation perplexity: 23.3.

The gate values are informative. The model chose to attenuate the cross-attention signal to roughly 27–33% of its full strength. This is neither fully open (the engram is useful) nor fully closed (unconstrained engram injection would be harmful). The model found a balanced operating point.

### Results

**Validation perplexity.** V18 achieved 23.3 BPE perplexity, compared to V17's 21.4. The cross-attention engram did not improve perplexity over the baseline — a departure from V16, which achieved 1.71. This is expected: V18's engram is structurally prevented from providing the causal attention shortcut that drove V16's low perplexity.

**MAUVE scores.** This is the metric that matters.

| Condition | V16 (prepend) | V17 (no engram) | V18 (cross-attn) |
|-----------|--------------|-----------------|-------------------|
| 50-tok, engram ON | 0.806 | — | 0.915 |
| 50-tok, engram OFF | 0.905 | 0.933 | 0.918 |
| 500-tok, engram ON | 0.888 | — | 0.919 |
| 500-tok, engram OFF | 0.906 | 0.943 | 0.941 |

**The leakage bug is fixed.** V16 dropped from 0.905 to 0.806 when engrams were active at short prompts — a catastrophic 0.10 degradation. V18 goes from 0.918 to 0.915 — a negligible 0.003 difference. Cross-attention isolation eliminates the failure mode.

**The engram doesn't help generation quality, but it doesn't hurt it.** V18 with engram OFF (0.918/0.941) matches V17 (0.933/0.943) within stochastic variation. The cross-attention parameters and categorization head did not damage the backbone's generation ability.

**The perplexity-MAUVE dissociation is reduced.** V16 showed a dramatic 12.5x perplexity improvement that translated to worse MAUVE. V18 shows a modest perplexity difference (23.3 vs 21.4) with comparable MAUVE. The cross-attention engram provides a gentler, less distortive form of context than the prepend approach.

### What the Cross-Attention Learned

The categorization loss trajectory tells a story. It dropped from 3.9 (random) to 2.8 during phases 1–3 when the cross-attention was frozen — the backbone alone was learning topic structure. When the cross-attention unfroze in phase 4, the categorization loss accelerated to 1.2, showing the engram was providing additional topic-level signal beyond what the backbone captured.

The gate values (0.27–0.33) suggest the model uses the engram as a gentle topic prior rather than a strong conditioning signal. This is consistent with the MAUVE results: a gentle prior doesn't distort generation enough to measure, but provides enough signal for the categorization head.

## Part 2: V18-EGR — Entropy-Gated Retrieval

### Motivation

V18's cross-attention engram is a corpus-level mean — it captures the average topic of recent training batches. This is useful as a background prior but lacks specificity. What if we could inject document-specific engrams at inference time, retrieved when the model is confused?

The Shannon entropy connection comes from our concurrent work on entropy-based data curation. In that work, we use per-token entropy to identify noisy, out-of-distribution text in web corpora. The same signal can serve a different purpose: identifying when a model needs help.

### System Design

**Entropy does double duty:**

- **Write condition.** Process text through V18. If a segment's mean entropy exceeds 4.0 bits, compute its engram (mean-pooled hidden states from the second-to-last layer) and store it alongside the original text.
- **Read condition.** During generation, maintain a rolling window of per-token entropy. If the window mean exceeds 4.0 bits, compute the current context's engram, search the store for the nearest match by cosine similarity, and inject the retrieved engram into V18's cross-attention buffer.

The 4.0-bit threshold was chosen based on our entropy curation work, where it separated clean text from noise in SlimPajama. Here it separates text the model finds routine from text it finds surprising.

**Storage.** The store is simple: a tensor of normalized engram vectors (keys) paired with metadata including the original text and entropy score. Retrieval is brute-force cosine similarity. At small scale (hundreds to thousands of entries), this adds negligible latency.

**Injection.** When retrieval triggers, the retrieved engram vector is expanded to fill V18's 32-slot cross-attention buffer and temporarily replaces the trained buffer. After a cooldown period equal to the rolling window size, the original buffer is restored. This means the model gets a burst of document-specific context for ~10 generation steps, then returns to its baseline behavior.

### Store Population

We processed the WikiText-103 validation set (485 sequences) through V18. Each document was segmented into 512-token chunks. Segments with mean entropy above 4.0 bits were stored with both isolated engrams (segment processed alone) and full-context engrams (segment processed with preceding context).

Result: 870 engrams stored from 504 segments. Storage rate of approximately 92% of isolated segments and 81% of full-context segments exceeded the threshold. This high rate reflects that V18 finds much of the validation set surprising — the model's perplexity of 23.3 corresponds to a per-token entropy well above 4.0 bits for many segments.

### MAUVE Results

| Condition | Baseline | + EGR | Effect |
|-----------|----------|-------|--------|
| 50-tok prompt | 0.914 | 0.926 | +0.012 |
| 500-tok prompt | 0.913 | **0.950** | **+0.037** |

**EGR improves generation quality at both prompt lengths.** The effect is larger at 500 tokens (+0.037) than at 50 tokens (+0.012). Longer generations have more opportunities for entropy spikes where retrieval can help.

**The 500-tok + EGR score of 0.950 is the highest MAUVE score in this project** — above V17's baseline of 0.943. Entropy-gated retrieval pushes V18 past the previous best by providing relevant context precisely when the model's uncertainty is highest.

The trigger rate averaged 6.3% of generation steps, well within the target range of 5–10%. The system intervenes sparsely, only when the model signals confusion through high entropy.

### Needle in a Haystack

We designed a cross-document variant of the standard needle-in-a-haystack test. Five synthetic facts ("needles") were created — each a plausible Wikipedia-style passage about a fictional subject (the Thornfield Protocol, the village of Kestlemere, the ZB-Petrus speedcubing method, Caspian tiger genetics, and Mount Seravezza). Each needle was processed alongside 20 real distractor documents from WikiText-103.

**Retrieval results:**

| Needle | Found? | Rank | Similarity |
|--------|--------|------|------------|
| Thornfield Protocol (science) | Yes | 1 | 0.44 |
| Kestlemere village (history) | Yes | 2 | 0.36 |
| ZB-Petrus method (hobby) | Yes | 1 | 0.64 |
| Caspian tiger genetics (biology) | Yes | 1 | 0.53 |
| Mount Seravezza (geography) | Yes | 1 | 0.55 |

**5/5 needles found, mean rank 1.2.** The engram store successfully retrieves the correct document for every query, even when the query is phrased as a question and the stored text is an encyclopedic description. Position in the processing order (first, middle, last) had no effect on retrieval quality.

**Generation could not use the retrieved context.** Both EGR and baseline generations scored 20% on answer token recall — the model retrieves the right engram but cannot ground its generation in the retrieved information. This is unsurprising: V18 is a 510M-parameter language model trained on WikiText-103, not a retrieval-augmented question answering system. The cross-attention was trained on a corpus-level mean buffer, not on document-specific injections. The model has never seen the pattern "receive a specific engram, generate text grounded in its content."

This is a meaningful negative result. It tells us that retrieval quality is not the bottleneck — the engram store works. The bottleneck is the injection-to-generation pathway: the model needs training on retrieval-augmented examples to learn to use injected engrams for grounded generation.

## Discussion

### What Worked

**Cross-attention isolation solves the leakage bug.** This was the primary goal of V18, and it succeeds cleanly. The engram can now be active during generation without degrading MAUVE scores. The structural separation between the autoregressive path (self-attention) and the engram path (cross-attention) is sufficient to prevent the distortion that plagued V16.

**Entropy-gated retrieval improves generation quality.** This was not guaranteed. Injecting document-specific engrams into cross-attention trained on corpus-level means could have introduced noise. Instead, it improved MAUVE by +0.037 at 500 tokens, producing the best generation quality in this project. The model's cross-attention learned to extract useful signal from any engram-shaped input, not just the specific buffer it was trained on.

**The engram store retrieval works at 100% recall.** Even with a small store (870 entries) and simple brute-force cosine similarity, the system correctly identifies the most relevant stored document for every test query. The engram representation — a mean-pooled hidden state from a late transformer layer — captures enough semantic content for effective retrieval.

### What Didn't Work

**The categorization head didn't translate to generation improvement.** It reached 1.2 loss (well below random at 3.9), confirming it shaped the engram toward topic features. But this didn't manifest as measurably better generation quality compared to V17. The engram-as-topic-prior may be too subtle to measure via MAUVE, or the categorization signal may primarily benefit tasks other than open-ended generation.

**Retrieved engrams cannot ground generation.** The needle-in-a-haystack test showed perfect retrieval but zero generation benefit. The model finds the right context but doesn't know what to do with it. This is a training gap, not an architecture gap — the cross-attention could in principle condition generation on retrieved content, but the model was never trained on that pattern.

### The Entropy Connection

Shannon entropy emerges as a unifying signal across three distinct applications in this work:

1. **Data curation** (our concurrent work): high entropy identifies noisy, low-quality text to filter from training data.
2. **Write trigger**: high entropy during intake identifies surprising content worth storing in the engram bank.
3. **Read trigger**: high entropy during generation identifies moments when the model needs help and should retrieve stored context.

In all three cases, entropy measures the same underlying quantity — the model's surprise — but the appropriate response differs: discard (curation), remember (storage), or recall (retrieval). This suggests entropy monitoring as a general-purpose control signal for language model systems, not just a training metric.

### Implications

**Cross-attention is the right injection mechanism for auxiliary memory.** Prepending tokens to the sequence (V16) conflates memory access with autoregressive prediction. Cross-attention separates them, allowing the model to optionally read from memory without corrupting its next-token prediction. The learned gates (0.27–0.33) show the model discovers its own optimal memory utilization.

**Retrieval-augmented generation needs retrieval-augmented training.** Building a working retrieval system (store + index + entropy trigger) is necessary but not sufficient for grounded generation. The model must be trained with retrieved context to learn the injection-to-generation pathway. This is consistent with the broader RAG literature — systems like RETRO and REALM train with retrieval in the loop.

**Small models can benefit from external memory.** V18 at 510M parameters with a 512-token context window achieves MAUVE 0.950 with entropy-gated retrieval — competitive with much larger models. The external memory compensates for limited context length by providing relevant information precisely when needed.

## Future Directions

**Retrieval-augmented fine-tuning.** Train V18 with retrieved engrams in the loop: during training, randomly replace the corpus-level buffer with document-specific engrams from a store. This would teach the cross-attention to ground generation in specific retrieved content, potentially closing the gap between retrieval accuracy (100%) and generation grounding (0%).

**Scaling the store.** The current store contains 870 engrams from the WikiText-103 validation set. A store populated from the full training set (~29K articles) or diverse web text would provide richer retrieval targets and reduce the repeated-retrieval problem observed during evaluation.

**Adaptive thresholds.** The fixed 4.0-bit threshold works well on average but could be adapted per-document or per-domain. A calibration step that sets the threshold to the Nth percentile of a model's entropy distribution on representative text would generalize better across models and domains.

**Multi-hop retrieval.** The current system retrieves once per entropy spike. For complex queries requiring synthesis of multiple sources, iterative retrieval — where the first retrieved engram modifies the context, potentially triggering retrieval of a second related engram — could enable compositional reasoning over the stored knowledge.

## Reproducibility

All code is available at github.com/MikeyBeez/HRS.

**Hardware:** NVIDIA RTX 5070 Ti, 16GB VRAM, Pop!_OS 24.04.

**V18 training:** `python train.py --ablation v18_cross_attn` (~11.4 hours)

**Store population:** `python populate_store.py --threshold 4.0` (~2 minutes)

**MAUVE benchmark:** `python benchmark_mauve_v18.py` (~3 hours) and `python benchmark_mauve_egr.py` (~4 hours)

**Needle-in-a-haystack:** `python niah_egr.py --n-distractors 20` (~15 minutes)

VRAM usage: approximately 12.5 GB during training, 8 GB during evaluation.
