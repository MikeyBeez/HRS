# Your Transformer Already Knows What It's Talking About — It Just Can't Prove It in One Sentence

*Michael Bonsignore and Claude (Anthropic)*

---

Every transformer has a context window, and every context window has the same problem: it fills up with whatever came most recently. If you're having a conversation that wanders through Roman history, baking, and quantum physics, the model's working memory ends up as a jumble of all three. When you ask about the fall of Rome, the model sifts through cupcake recipes and entangled particles to find the relevant bits.

The standard fix is retrieval-augmented generation — RAG. Build an external index, use a separate embedding model, maintain a vector database and retrieval pipeline. It works, but it's a lot of infrastructure bolted onto the side of your language model.

We built something simpler. We use the language model's own internal representations to organize context by topic. No external models. No retraining. No additional learned parameters. Just a few hundred lines of Python that let the model tell us what the conversation is about.

It works beautifully at scale. It fails at the scale that matters most. And when we thought it was helping anyway, a deeper evaluation revealed we were measuring the wrong thing.

## The core idea

Deep inside a transformer, every layer produces hidden states — internal representations of the text being processed. Take the hidden states from a late layer, average them across all token positions in a passage, normalize the result, and you get a single vector that captures the semantic gist of that passage. We call this vector an engram.

Engrams from documents about similar topics cluster together in vector space. Roman history documents end up near other Roman history documents. Baking clusters near baking. This isn't something we trained. It's an emergent property of next-token prediction — documents about similar subjects require similar computational pathways, which produce similar hidden states.

Cosine similarity between engrams measures topical relatedness. The question is: how reliably?

## Where it works: article-length text

We computed engrams for segments from 2,000 WikiText-103 articles, each segment about 512 tokens long, and built 50,000 pairs — half from the same article, half from different articles.

The results are clean. Same-article pairs average 0.755 similarity. Different-article pairs average 0.135. That's a gap of 0.62 between the two distributions. A simple cosine threshold at 0.45 achieves 97.3% accuracy at distinguishing same-topic from different-topic pairs.

We also trained a small binary classifier — a two-layer neural network with 1,377 parameters — on eight features derived from each pair. It achieved 97.3% accuracy. Identical to the threshold. The space is linearly separable at this scale. There's no hidden non-linear structure a classifier can exploit.

We confirmed this with a needle-in-a-haystack test: five synthetic facts on different topics, each processed alongside 20 real distractor documents. Engram similarity found the correct document every time — 100% retrieval accuracy, mean rank 1.2 out of 21.

At 512 tokens, this is essentially a solved problem. No complex infrastructure needed.

## Where it fails: single sentences

Conversations don't happen in 512-token blocks. They happen one sentence at a time. So we ran the same analysis on individual sentences — 8 to 40 words each — extracted from 3,000 articles.

The picture collapses. Same-article sentence pairs average 0.429 similarity. Different-article pairs average 0.291. The gap shrinks from 0.62 to 0.14. Worse, the distributions heavily overlap: the 75th percentile of different-article pairs (0.369) exceeds the 25th percentile of same-article pairs (0.340).

The best threshold achieves 70.9% accuracy. The trained classifier gets 71.5% — a marginal improvement that confirms there's essentially no non-linear structure to exploit here either.

This is not a threshold problem. It is not a classifier problem. It is a signal-to-noise problem in the engram computation itself, and it follows a scaling law. Mean-pooling is not just aggregation — it is denoising. Over 512 tokens, individual token-level noise averages out and the dominant semantic theme emerges. Over 15 tokens, the noise survives — the representation gets dominated by whichever tokens happen to produce the largest hidden-state magnitudes, not by the overall topic.

The implication is that engram quality scales with token count in the same way that measurement precision scales with sample size. There's a phase transition somewhere between sentence length and article length where the semantic signal crosses above the noise floor and topic separability appears. Our data puts the crossover well below 512 tokens but well above 15.

## Seven stress tests

To map exactly where and how the system breaks under realistic conditions, we ran seven targeted benchmarks on single-sentence prompts.

**Semantic overlap gradient.** We created five levels of topic overlap, from trivially separable (Roman history versus JavaScript) to near-identical (Caesar's military campaigns versus Caesar's political career). Ten single-sentence prompts per level, interleaved. At the default threshold of 0.4, every single level — including the trivially different one — collapses into a single cluster. Roman history and JavaScript programming, topics with zero semantic overlap, end up in the same bucket. That's the short-prompt problem laid bare: single sentences produce engrams within 0.4 of everything.

**Threshold sweep.** We tested thresholds from 0.4 to 0.9 across all benchmarks simultaneously. No single value resolves the fundamental tradeoff. At 0.4, you preserve semantic coherence but can't separate any topics. At 0.7, you get clean separation for obviously different topics but adversarial accuracy drops from 100% to 60% and the system creates too many clusters. At 0.9, every prompt creates its own cluster. The sweet spot is around 0.6 for well-separated topics, but "sweet spot" is generous — you still get three to seven clusters for ten prompts that should form two.

**Centroid drift bomb.** Ten prompts designed to gradually walk a cluster's centroid from "Roman Empire" to "Neural Networks," each close enough to the previous centroid to join the cluster. At threshold 0.4, it works — the centroid drifts 0.41 in cosine distance over ten steps, and a subsequent probe about Rome fails to match its own cluster. The title drifts to "Networks / Roman / Medieval / Infrastructure." But at threshold 0.5, the drift bomb fails immediately: the first off-topic prompt can't meet the join threshold and starts a new cluster.

This tells us something important. Centroid drift is not a flaw in the clustering algorithm — it is a symptom of insufficient semantic signal. When engram quality is low (short prompts, low threshold), everything looks similar enough to merge, and clusters absorb noise until their centroids mean nothing. When signal is adequate (higher threshold or longer text), structure emerges naturally and drift can't take hold. The instability isn't in the clustering. It's in the representations feeding it.

**Topic fork.** Starting with "machine learning" prompts, then interleaving "computer vision" and "NLP" prompts. The fork never splits at any threshold from 0.4 to 0.7. The shared ML vocabulary keeps all subtopics within threshold. At 0.8 it splits, but into three or more noisy clusters, not the two clean subtopics you'd want. Engram similarity cannot detect subtopic divergence within a parent domain. Mean-pooled representations capture the dominant theme but lose fine-grained distinctions between fields that share vocabulary.

**Rapid topic switching.** Alternating between topics every prompt for 50 turns, in both regular and irregular patterns. At appropriate thresholds (0.5-0.6), the system handles regular alternation between clearly distinct topics well. Irregular patterns with runs of the same topic cause more issues, as the centroid updates during a run can shift the cluster enough to cause misrouting when switching back.

**Scale stress.** Increasing the number of topics from 2 to 50. Performance holds up to about 10 well-separated topics, then degrades as the fixed threshold can't accommodate the fact that some topic pairs are naturally closer together than others.

## The result that surprised us

**Adversarial vocabulary.** Five prompt pairs where the metaphorical version uses vocabulary from an entirely different domain. "The layers of a cake are like the layers of a neural network" (baking words, ML topic). "Napoleon's defeat was a catastrophic loss function for the French Empire" (ML words, history topic). "The Roman Senate operated like a distributed system with no single point of failure" (CS words, history topic).

At threshold 0.4, every single metaphorical prompt clustered with its literal counterpart. One hundred percent accuracy. The Napoleon/loss-function prompt matched its literal version at 0.834 similarity — higher than most same-topic pairs in the sentence-level analysis.

This is the system's strongest result and it tells us something important about what hidden states actually encode. The representation tracks what the sentence means, not what it says. "Catastrophic loss function" doesn't pull the engram toward machine learning. "Distributed system with no single point of failure" doesn't pull it toward computer science. The hidden states encode the referent — Napoleon's military defeat, the Roman political structure — not the vocabulary used to describe it. That's genuine semantic understanding, encoded in the geometry of the representation space as an emergent byproduct of learning to predict the next token.

At threshold 0.7, accuracy drops to 60% — the two lowest-similarity pairs (0.629 and 0.661) split off. The semantic signal is real but not infinitely strong.

## A negative result on trained classification

Our model includes a categorization head — a linear layer explicitly trained to predict topic labels derived from TF-IDF clustering of WikiText-103. Training loss reached 1.2, well below the random baseline of 3.9.

Evaluation on held-out documents: 3.3% accuracy. Random for 50 categories is 2.0%.

The failure comes from label noise. During training, each 512-token sequence gets a single category label: the most common topic among its tokens. But sequences carved from a continuous text stream can span article boundaries — half Roman Empire, half cupcake recipe, labeled as whichever topic has more tokens. The head learns to minimize loss on these noisy labels without learning generalizable classification.

The takeaway: low training loss does not mean good classification when labels are noisy. And the model's raw hidden-state similarity — never explicitly trained for topic detection — outperforms a classifier that was.

## The punchline: MAUVE says it helps, perplexity says it doesn't

Here's where the story gets complicated.

We populated the topic context manager from 60 WikiText-103 validation articles, producing 34 topic clusters. Then we ran two kinds of evaluation.

**MAUVE says yes.** We generated 256-token continuations from test set prompts and measured MAUVE score — a distributional metric comparing generated text against reference text. At 50-token prompts, MAUVE rose from 0.915 (baseline) to 0.951 (with topic routing). At 500-token prompts, it rose from 0.919 to 0.962 — the highest score in the project.

**The evaluation suite says no.** We then ran four additional metrics designed to test whether the MAUVE improvement reflects genuine generation quality.

Held-out perplexity — does topic context help the model predict the real continuation? No. Perplexity with topic context (38.1) was worse than with no context (35.8) and worse than with random context (36.1). The topic-routed context actively increases the model's uncertainty about the true continuation — it doesn't just fail to help, it interferes with prediction.

Topic coherence — does the generated text stay on-topic with the prompt? No. Prompt-continuation engram similarity was 0.489 with topic routing versus 0.522 without. Topic-routed generations actually drift further from the prompt's topic.

Routing confidence correlation — does better routing predict better results? No. The Pearson correlation between routing confidence and perplexity improvement was 0.09 — essentially zero. Well-routed prompts don't benefit more than poorly-routed ones.

Repetition and copying — is the model degenerating? No. Repetition rates were unchanged (0.103 vs 0.106), and context overlap was only 4%. The model isn't parroting context.

To put the numbers side by side: without any context, perplexity sits at 35.8 and MAUVE at 0.919. With random context, perplexity rises slightly to 36.1. With topic-routed context, perplexity rises further to 38.1 — but MAUVE jumps to 0.962. Topic coherence drops from 0.522 to 0.489. The routing confidence correlation is 0.09. Every metric that measures whether the context actually helps says no. The one metric that measures distributional similarity says yes.

## What's actually happening

The MAUVE improvement is real but misleading. Here's why.

MAUVE measures distributional similarity between generated text and reference text in aggregate. The topic-routed context comes from WikiText-103 validation articles — encyclopedic text that is distributionally similar to the WikiText-103 test set references. When this context enters the model's window, it biases generation toward encyclopedic style and vocabulary, making the output look more like the reference distribution. MAUVE detects this and scores it higher.

But the model isn't getting better at answering the prompt. It's getting better at sounding like Wikipedia.

The perplexity metric tests whether the context helps the model predict what actually comes next in the test set — and it says no. Worse than no: the topic-routed context actively increases the model's uncertainty about the true continuation. Perplexity rises from 35.8 to 38.1 — the context doesn't just fail to help, it interferes. The topic context is distributionally helpful but conditionally harmful.

We call this distributional contamination: injecting in-distribution text into the context window improves distributional metrics without improving — and potentially degrading — conditional generation quality.

This is the V16 dissociation in reverse. V16 showed better perplexity but worse MAUVE — the engram helped token prediction but narrowed the distribution. Topic routing shows better MAUVE but worse perplexity — distributional contamination broadens the output distribution toward the reference corpus while actively harming the model's ability to predict specific continuations.

The near-zero routing correlation (0.09) is the nail in the coffin for the relevance claim. If topic matching were driving the improvement, better-matched prompts should show larger benefits. They don't. The improvement is coming from distributional contamination, not from topical relevance.

An accuracy evaluation confirms this from two additional angles. We generated 50 paired continuations (baseline vs. topic-routed) and measured both automated and LLM-judged quality.

Automated: baseline wins 29 to 18 on semantic similarity to the real continuation (0.592 vs 0.557 engram cosine), with token overlap essentially tied.

LLM-as-judge: we sent each pair to Llama 3.1 (running on a separate Mac Mini) for blind evaluation on coherence, relevance, and fluency. Baseline wins 28 to 22 — a narrower margin than the automated metrics, but the same direction. The LLM finds topic-routed generations competitive but not better.

Four independent metrics now agree: perplexity (35.8 vs 38.1), semantic similarity (29-18), LLM judge (28-22), and routing correlation (r=0.09). MAUVE (0.962 vs 0.919) is the sole dissenter — measuring distributional contamination rather than generation quality.

## What we actually learned

The honest version of the result is this: injecting encyclopedic text into the context window makes a language model generate more encyclopedic-sounding text, which scores well against encyclopedic reference text. That's not topic routing working. That's distributional contamination — context-induced distributional bias — being measured by a distributional metric.

This is exactly the kind of false positive the evaluation suite was designed to catch. MAUVE alone would have told a convincing story — 0.962, best in the project, consistent across prompt lengths. The perplexity metric reveals that the story is wrong.

The finding has value, but not the value we hoped for. It demonstrates three things.

First, MAUVE is sensitive to distributional contamination. Any system that injects reference-distribution text into the context window will improve MAUVE without necessarily improving generation quality. This is a methodological warning for the RAG and context engineering communities.

Second, the evaluation suite works. It caught a false positive that a single metric would have missed. The combination of distributional (MAUVE) and conditional (perplexity) metrics is necessary, not redundant.

Third, the topic routing mechanism behaves as designed — engram extraction, clustering, context assembly all function correctly — but its outputs are not aligned with the prediction task in this setting. The system fills the context with validation-set article text, which is distributionally similar to the test set but not conditionally relevant to specific prompts. In a conversational setting where the stored context is the user's own prior turns (not external articles), the relevance signal would be fundamentally different.

For perspective on the full trajectory: V16 scored 0.806 MAUVE with a system that helped perplexity but hurt MAUVE. Topic routing scores 0.962 MAUVE with a system that helps MAUVE but hurts perplexity. The elusive combination — help both — remains an open problem.

## How the system works in practice

The practical system has four pieces.

Engram extraction runs during the normal forward pass — no additional computation beyond one mean-pooling and one normalization step.

Online clustering happens as prompts arrive with no predefined topic taxonomy. Each new prompt's engram gets compared against existing cluster centroids. If it's similar enough, it joins. Otherwise, it starts a new cluster. Centroids evolve as running means, so a cluster that begins with "The Roman Empire fell in 476 AD" and later absorbs "The Byzantine economy relied on trade through Constantinople" develops a centroid representing the broader topic rather than just the initial prompt. Clusters that drift close enough together auto-merge.

An active buffer (configurable size, default two) tracks which clusters are currently in play. When a prompt matches a cluster, that cluster moves to the front. If the buffer is full, the least-recently-used cluster gets evicted — but only from the active set. It persists in storage and reactivates if a future prompt matches it. You can discuss Rome, switch to baking, switch back to Rome, and the historical context comes back intact.

Context assembly fills the token budget from active clusters using exponential decay: the most-recently-used cluster gets half the budget, the next gets a quarter, and so on. Users can see and toggle clusters — disable baking when you're deep in history, re-enable it when planning a themed party.

## What this means

Three claims, revised in light of the evaluation suite. All three point to a single mismatch: representation quality, context usefulness, and evaluation metrics can move independently. Good representations do not guarantee useful context, and useful-looking metrics do not guarantee good generations.

The methodological claim: MAUVE alone is insufficient for evaluating context engineering systems. A distributional metric can be inflated by distributional bias in the injected context. Any system that fills the context window with reference-distribution text will improve MAUVE without necessarily improving generation quality. Conditional metrics (held-out perplexity) and mechanism-validation metrics (routing confidence correlation) are necessary complements. This applies to RAG evaluation, memory systems, and any method that modifies context.

The empirical claim: engram quality follows a signal-to-noise scaling law governed by token count. Mean-pooling is denoising. Below a critical sequence length, semantic signal drowns in token-level noise and topic separability vanishes. Above it, clean linear separability emerges (97.3% at 512 tokens, 71.5% at sentence scale). This scaling law likely applies to any mean-pooled representation from any transformer.

The theoretical claim: hidden states encode semantics, not vocabulary, and this encoding is more informative than explicit supervision. The adversarial benchmark proves the first part — the representation tracks what the sentence means, not what it says. The categorization head failure proves the second — raw geometric similarity outperforms a trained classifier by a factor of thirty. The model knows more about topics in its geometry than it can express through a supervised projection.

## What we'd try next

The evaluation suite changed what "next" means. The routing mechanism works at article scale. The short-text problem is real but secondary. The primary open problem is distributional contamination: routed context doesn't help the model predict specific continuations — it only biases the output distribution toward the reference corpus.

Route the model's own conversation, not external articles. The current evaluation populates clusters from WikiText validation articles — external text that is distributionally similar to the test set but not conditionally relevant to specific prompts. In a real conversational system, the stored context would be the user's own prior turns. This is a fundamentally different regime: the context is both distributionally and conditionally relevant. The perplexity result might reverse.

Contrastive context evaluation. For each prompt, compute perplexity under correct-topic context versus deliberately mismatched context. If correct is lower than mismatched, the model is using the topic signal. If they're equal, context is being ignored. This directly tests whether the routing mechanism provides value beyond distributional bias.

Accumulate before routing. Don't try to classify a single sentence. Buffer two or three prompts, concatenate, then compute the engram. This trades latency for accuracy by pushing the effective input length toward the regime where engrams work well.

Attention-weighted pooling. Replace uniform mean-pooling with attention-weighted pooling, using the model's own attention scores to weight token contributions. Topically informative tokens should receive higher weight than function words, producing more discriminative engrams from short text.

Human preference evaluation. Fifty prompt pairs, blind evaluation across three dimensions (coherence, fluency, informativeness), stratified by routing confidence. This is the gold standard that anchors the automatic metrics. Budget 2-3 hours.

## Reproducibility

Code: github.com/MikeyBeez/HRS

The core implementation is in topic_context.py (the TopicContextManager class). Supporting code handles the vector store, entropy monitoring, pair analysis and classifier training for both article-length and short prompts, the seven stress tests, MAUVE benchmarking, the four-metric evaluation suite (eval_topic_routing.py), and the categorization head evaluation.

Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM. Everything runs in minutes except model training, which takes about 11 hours.

## References

Beltagy, I., Peters, M. E., and Cohan, A. (2020). Longformer: The Long-Document Transformer.

Borgeaud, S., et al. (2022). Improving Language Models by Retrieving from Trillions of Tokens (RETRO).

Guu, K., et al. (2020). REALM: Retrieval-Augmented Language Model Pre-Training.

Mu, J., et al. (2023). Learning to Compress Prompts with Gisting.

Pillutla, K., et al. (2021). MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers.

Rae, J. W., et al. (2020). Compressive Transformers for Long-Range Sequence Modelling.

Zaheer, M., et al. (2020). Big Bird: Transformers for Longer Sequences.
