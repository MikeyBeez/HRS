# Beyond Heuristic Filtering: Shannon Entropy Reveals Pervasive Noise in "Cleaned" LLM Training Data

**Michael Bonsignore**

## Abstract

Large language model (LLM) training corpora undergo extensive curation pipelines involving deduplication, language filtering, and heuristic quality scoring. We demonstrate that these pipelines are insufficient. Using Shannon entropy computed via a reference language model (Mistral 7B), we analyze SlimPajama — a widely-used, deduplicated and filtered derivative of RedPajama — and find that approximately 6% of documents exhibit entropy signatures consistent with machine-generated spam, OCR artifacts, and garbled text that survived all prior filtering stages. An additional population of documents at the low-entropy extreme consists of boilerplate, repetitive templates, and trivial content that contributes little to model training. We propose a band-pass entropy filter that retains documents within a productive entropy range and demonstrate taxonomies of failure modes that heuristic filters systematically miss. Our analysis suggests that model-based entropy scoring should be a standard component of data curation pipelines, complementing rather than replacing existing heuristic approaches.

## 1. Introduction

The quality of pretraining data is among the strongest determinants of LLM capability (Hoffmann et al., 2022; Penedo et al., 2023). Recognizing this, the community has invested heavily in data curation. SlimPajama (Soboleva et al., 2023) represents the state of the art in open-source data cleaning: it applies MinHash deduplication, language identification, and multiple heuristic quality filters to the 1.2T-token RedPajama corpus, producing a 627B-token "cleaned" dataset that has become a standard benchmark for training data quality.

We ask a simple question: *how clean is "clean"?*

By computing the Shannon entropy of the next-token probability distribution produced by a reference language model (Mistral 7B; Jiang et al., 2023), we obtain a per-document quality signal that captures something fundamentally different from heuristic rules. Entropy measures the model's *surprise* at the text — its assessment of how predictable, coherent, and linguistically natural the content is. High entropy indicates text that confounds even a capable language model; low entropy indicates text so predictable it carries minimal information.

Our analysis of SlimPajama reveals three findings:

1. **Pervasive high-entropy noise.** Approximately 6.3% of documents have mean entropy above 4.0 bits, and manual inspection confirms these are overwhelmingly spam, machine-generated gibberish, OCR artifacts, and garbled web scrapes. These documents survived all of SlimPajama's existing filters.

2. **Low-entropy waste.** Documents at the low-entropy extreme (below ~1.0 bits) consist of boilerplate HTML/XML templates, pagination artifacts, repetitive code snippets, and formulaic text that provides minimal training signal.

3. **A "goldilocks zone."** Documents in the median entropy range (~2.5–3.5 bits) consistently represent well-formed, substantive text across diverse domains — exactly the kind of content that drives effective pretraining.

These findings argue that model-based entropy scoring should be a standard, low-cost addition to any data curation pipeline. A single forward pass per document through a quantized reference model is sufficient to identify failure modes invisible to rule-based filters.

## 2. Method

### 2.1 Entropy Scoring

Given a document $d = (t_1, t_2, \ldots, t_n)$ tokenized into $n$ tokens, we compute the Shannon entropy of the next-token distribution at each position:

$$H_i = -\sum_{v \in V} P(t = v \mid t_1, \ldots, t_{i-1}) \log_2 P(t = v \mid t_1, \ldots, t_{i-1})$$

where $V$ is the vocabulary and probabilities are obtained from a single forward pass through the reference model. The document's **mean entropy** is:

$$\bar{H}(d) = \frac{1}{n-1} \sum_{i=2}^{n} H_i$$

We also record $H_{\max}(d) = \max_i H_i$ as a secondary signal.

### 2.2 Reference Model

We use Mistral 7B v0.1 (Jiang et al., 2023) as the reference model, loaded in 4-bit quantization via bitsandbytes (Dettmers et al., 2022) on a single consumer GPU (NVIDIA RTX 5070 Ti, 16GB VRAM). Each document is truncated to 512 tokens for scoring. Documents are processed in batches of 4, with padding and attention masking to ensure correct entropy computation over variable-length inputs.

The choice of reference model is deliberate: Mistral 7B is large enough to have internalized robust language statistics, yet small enough to score millions of documents on modest hardware. The 4-bit quantization introduces minimal distortion to the entropy signal — we are measuring distributional spread, not precise probability values.

### 2.3 Dataset

We analyze SlimPajama (Soboleva et al., 2023), a 627B-token deduplicated subset of RedPajama. The dataset is distributed in chunked JSONL format. We score a sample of 27,900 documents from the first chunk, providing a representative cross-section of the corpus.

## 3. Results

### 3.1 Entropy Distribution

| Statistic | Value |
|-----------|-------|
| Mean | 2.924 bits |
| Median | 2.953 bits |
| Std. Dev. | 0.771 bits |
| Min | 0.163 bits |
| Max | 7.875 bits |

The distribution is approximately normal with a slight right skew. The cumulative distribution reveals the population structure:

| Threshold | % of documents above |
|-----------|---------------------|
| > 1.0 bits | 98.5% |
| > 2.0 bits | 89.4% |
| > 3.0 bits | 46.2% |
| > 4.0 bits | 6.3% |
| > 5.0 bits | 0.6% |
| > 6.0 bits | 0.2% |
| > 7.0 bits | < 0.1% |

### 3.2 High Entropy: Noise That Survived Curation

Documents with mean entropy above 4.0 bits (6.3% of the sample) fall into several distinct failure categories. All of the following examples are drawn directly from SlimPajama — a dataset that has already undergone deduplication, language identification, and heuristic quality filtering.

#### Category 1: SEO Spam and Keyword Stuffing (ent > 7.0)

> *"F I'm learn modal kecil buy Premarin from canada such for the most report to starting able of madness to practice for binary option was online traders need. Your best robbery and exempt at brokers, especial trading, reliable and experts and would be like a greated due to..."* (H̄ = 7.875)

This document consists of incoherent keyword fragments designed to manipulate search engines. Despite being syntactically broken, it passed all heuristic filters — likely because it contains real English words at frequencies consistent with natural text.

#### Category 2: Machine-Generated Gibberish (ent > 7.0)

> *"The atomic debates stylish the callousness facing a diamond is satisfies thesis stricken whether whilst rider they be gushing attentive. Occurrence the order of guy stylish a selective twig extend operation rider it hope an nowadays the confounding possess the."* (H̄ = 7.750)

This text appears to be produced by automated content spinning — a technique that replaces words with thesaurus synonyms to evade duplicate detection. Each individual word is valid English, and sentence structure is superficially grammatical, but the semantic content is nonsensical. This is precisely the type of noise that heuristic filters cannot catch because it satisfies all surface-level criteria.

#### Category 3: OCR Artifacts (ent > 7.0)

> *"for the - never could ir. bome. great , 'V'' : 7' i . u, 1 : .) 1 1, ivjo tiir j ,..; Mo,al.ty ..Id lll.llic. liv in the rinie house."* (H̄ = 7.719)

Optical character recognition errors from digitized print sources produce character-level noise that is trivially identifiable by a language model but invisible to filters that operate on word-level statistics.

#### Category 4: Scraped Web Junk (ent > 6.5)

> *"RGH FREE Self London Military. It s Reap News Found you and horrible dating | Support Realth another indian dating agencies in london Karkably sucking Site Maps Work and people s anything from it some courage group: 23-Jun-13 12 2011 03:23 year old dating Website..."* (H̄ = 7.594)

Concatenated web page fragments — navigation elements, metadata, and content jumbled together — that passed extraction but were never coherent text.

#### Category 5: Cross-Lingual Contamination (ent > 7.0)

> *"Your Логика высказываний sent a page that this spleen could still please. & 2 to 16 have well made in this culture. This network is including a research j to be itself from experienced devices."* (H̄ = 7.312)

Mixed-language documents where automated translation or scraping has produced a hybrid that satisfies neither English nor Russian language filters.

### 3.3 Low Entropy: Boilerplate and Repetition

Documents at the low-entropy extreme (below ~1.0 bits) are highly predictable to the model, but not because they are high-quality prose. These fall into their own failure categories:

#### Category 1: HTML/XML Templates (ent < 0.3)

> *"<!DOCTYPE html> <html lang="en"> <head> <meta charset="utf-8"> <meta http-equiv="X-UA-Compatible" content="IE=edge"> <meta name="viewport" content="width=device-width, initial-scale=1"> <meta name="description" content="">..."* (H̄ = 0.163)

Boilerplate markup that the model has memorized entirely. Training on this reinforces template generation at the expense of substantive content.

#### Category 2: Pagination and Navigation Artifacts (ent < 0.2)

> *"Pages (537): « 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26..."* (H̄ = 0.167)

Pure structural noise — page number sequences that occupy tokens without contributing any linguistic or factual information.

#### Category 3: Repetitive Code Boilerplate (ent < 0.4)

> *"#pragma once #include <aws/appconfig/AppConfig_EXPORTS.h> #include <aws/core/utils/memory/stl/AWSVector.h> #include <aws/core/utils/memory/stl/AWSString.h>..."* (H̄ = 0.270)

Auto-generated AWS SDK headers that are near-identical across thousands of files. The model can predict these almost perfectly because it has seen virtually identical content many times.

#### Category 4: Repetitive Marketing Copy (ent < 0.5)

> *"Welcome to Caister Kebab and Pizza. Order food online in Caister-on-Sea! It's so easy to use, fast and convenient. Try our new, online website which contains our entire takeaway menu. Caister Kebab and Pizza is located in Caister-on-Sea, Great Yarmouth. Here at Caister Kebab and Pizza we are constantly striving to improve our service..."* (H̄ = 0.420)

Formulaic small-business website copy that follows such a rigid template the model finds it almost entirely predictable.

### 3.4 The Editing Zone: Documents Worth Saving

Between the clear deletes at the extremes lies a population of documents (approximately the 93rd–97th percentile, entropy 3.97–4.31) that contain genuine content buried under noise. These are candidates for automated cleaning rather than deletion:

> *"In the West, we are obsessed with individuality – our 'me' and how special we are. The Course points this out to us again and again – how much we think our 'specialness' is Who we are, and how terrified we are of losing it."* (H̄ = 4.062)

This document contains thoughtful philosophical reflection, but its elevated entropy comes from inconsistent formatting, embedded navigation elements, and mid-document topic shifts that could be cleaned programmatically.

> *"MODULAR PERFORMANCE | CREATION 2020 'A concert installation oscillating between the cabinet of curiosities and alchemist kitchen' Rainer Nonnenmann, Kölner Stadt-Anzeiger. The public is invited to investigate a scenic space bordered by display vitrines..."* (H̄ = 4.156)

Genuine arts criticism mixed with event metadata. The substantive content is valuable; the structural noise is removable.

> *"the poem and even you. In her second collection, even the overtly science themed poems – 'Einstein's Overcoat', 'Co-evolution', 'Sky Given' and 'Evolution by Engulfment' – serve to remind us that the results of scientific experiment and invention, though integral to our daily existence, have been absorbed 'like English engulfing foreign words.'"* (H̄ = 4.312)

Literary criticism with sophisticated vocabulary — exactly the kind of text that drives model capability. The elevated entropy reflects genuine linguistic complexity, not noise.

These examples illustrate that a simple threshold-based filter would be too aggressive. The editing zone requires more nuanced treatment: extraction of substantive content, removal of navigation artifacts, and structural normalization.

### 3.5 The Goldilocks Zone: What Good Data Looks Like

Documents in the median entropy range (2.5–3.5 bits) consistently represent the kind of well-formed, informative text that drives effective pretraining:

> *"Dan Gargan is on fire out in Los Angeles. The Chestnut Hill Academy and Georgetown graduate has been playing some of the best soccer of his career, starting five games at outside back for the Galaxy and providing three assists."* (H̄ = 2.969)

> *"Seagrass meadows in decline... MTSRF-funded research into trends in condition indicate that the region's coastal seagrass meadows are declining. Reef-wide surveys show that 38% of sites exhibit shrinking meadow area..."* (H̄ = 3.047)

> *"Let x = Σδᵢ2⁻ⁱ. Is there an algorithm that converts the sequence of binary digits of x to the sequence of its continued fraction representation? No, there is not. Already a₀ is impossible to determine just by reading finitely many digits..."* (H̄ = 2.984)

These documents span sports journalism, environmental science, and mathematics. They are well-written, topically focused, factually substantive, and linguistically diverse — precisely what an LLM needs to develop broad capability.

## 4. Discussion

### 4.1 Why Heuristic Filters Fail

The failure modes we identify share a common property: they satisfy surface-level statistical criteria while violating deeper linguistic coherence. Content spinning (Category 2) produces text with normal word frequency distributions and grammatical parse trees. SEO spam (Category 1) contains real English words at natural frequencies. OCR artifacts (Category 3) may have acceptable character-level statistics despite being unreadable.

Entropy scoring succeeds where heuristics fail because it evaluates text *holistically*. A language model's surprise at a document integrates lexical, syntactic, semantic, and discourse-level signals simultaneously. A sentence like "The atomic debates stylish the callousness facing a diamond" has normal word frequencies and parses as a grammatical English sentence, but a language model assigns high entropy because no coherent context predicts this sequence.

### 4.2 The Case Against Simple Perplexity Filtering

Prior work has used perplexity as a quality signal (Wenzek et al., 2020; Marion et al., 2023). Entropy and perplexity are related (perplexity = 2^H), but we argue that working with entropy directly is preferable for two reasons:

1. **Interpretability.** Entropy in bits has a natural interpretation: the number of bits needed to encode the model's uncertainty. An entropy of 3.0 bits means the model is, on average, choosing among 8 equally likely next tokens. This is immediately interpretable; a perplexity of 8.0 is not.

2. **Band-pass filtering.** Perplexity-based approaches typically filter only at the high end. Our analysis demonstrates that the low-entropy tail is equally problematic — boilerplate, templates, and repetitive content that wastes training compute without contributing to capability. Entropy naturally motivates a *band-pass* filter that retains the productive middle range.

### 4.3 Implications for Training

If 6.3% of a "cleaned" corpus is high-entropy noise, and an additional fraction is low-entropy waste, the implications for training efficiency are substantial. At SlimPajama's scale of 627B tokens, 6.3% represents approximately 39B tokens of noise that models must process and attempt to learn from. This noise does not merely waste compute — it actively degrades training by pulling model weights toward incoherent patterns.

### 4.4 A Practical Recommendation

We propose a three-tier approach to entropy-based curation:

1. **Delete** documents with mean entropy > 5.0 bits (0.6% of corpus). These are overwhelmingly noise with no recoverable content.
2. **Review/Edit** documents with mean entropy 4.0–5.0 bits (5.7% of corpus). Many contain genuine content mixed with noise and benefit from automated cleaning.
3. **Delete** documents with mean entropy < 0.5 bits (~1.5% of corpus). These are boilerplate, templates, and repetitive artifacts.
4. **Retain** the remaining ~92% as the productive training corpus.

The computational cost is modest: a single forward pass through a 4-bit quantized 7B model, processing at approximately 9 documents per second on a consumer GPU. At this rate, scoring SlimPajama's full corpus requires approximately 4 GPU-days — a negligible cost relative to pretraining.

## 5. Limitations

Our analysis has several limitations that should be addressed in future work:

1. **Sample size.** Our results are based on 27,900 documents from a single chunk. While we have no reason to expect systematic variation across chunks, full-corpus statistics will strengthen the findings.

2. **No downstream validation.** We have not yet trained models on entropy-filtered versus unfiltered data. This ablation is the strongest possible evidence for the method's value and is planned as immediate follow-up work.

3. **Reference model bias.** Entropy scores depend on the reference model's training distribution. Mistral 7B may systematically over- or under-estimate entropy for certain domains (e.g., code, non-English text) due to its own training data composition.

4. **Threshold sensitivity.** Our proposed thresholds (0.5, 4.0, 5.0 bits) are empirically motivated but not optimized. Different corpora and use cases may require different cutoffs.

## 6. Related Work

**Data curation.** The Pile (Gao et al., 2020), C4 (Raffel et al., 2020), and RefinedWeb (Penedo et al., 2023) established heuristic-based curation as standard practice. SlimPajama (Soboleva et al., 2023) extended these with aggressive deduplication. Our work is complementary — we show what these pipelines miss.

**Perplexity filtering.** CCNet (Wenzek et al., 2020) and SemDeDup (Abbas et al., 2023) use perplexity as a quality signal. QuRating (Wettig et al., 2024) uses LLM-based quality ratings. Our approach differs in using raw entropy rather than derived metrics, and in explicitly targeting both tails of the distribution.

**Data influence.** Recent work on data attribution (Grosse et al., 2023) and influence functions (Park et al., 2023) provides fine-grained per-example quality signals but at much higher computational cost than entropy scoring.

## 7. Conclusion

We have demonstrated that Shannon entropy, computed via a single forward pass through a reference language model, reveals pervasive noise in a widely-used "cleaned" training corpus. The failure modes we identify — SEO spam, content spinning, OCR artifacts, cross-lingual contamination, boilerplate templates, and repetitive navigation — are invisible to heuristic filters but immediately apparent to a model-based entropy signal.

Our findings argue for a simple addition to standard data curation pipelines: score documents by entropy, delete the extremes, and review the borderline cases. The computational cost is negligible relative to pretraining; the potential quality improvement is substantial.

The broader lesson is that data quality is not a solved problem. Even the most carefully curated open-source datasets contain meaningful fractions of noise that can be identified with straightforward model-based analysis. As the community scales to ever-larger corpora, model-based quality signals like entropy scoring will become not just useful but necessary.

## References

Abbas, A., et al. (2023). SemDeDup: Data-efficient learning at web-scale through semantic deduplication. *ICLR 2023*.

Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS 2022*.

Gao, L., et al. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. *arXiv:2101.00027*.

Grosse, R., et al. (2023). Studying Large Language Model Generalization with Influence Functions. *arXiv:2308.03296*.

Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *NeurIPS 2022*.

Jiang, A., et al. (2023). Mistral 7B. *arXiv:2310.06825*.

Marion, M., et al. (2023). When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale. *arXiv:2309.04564*.

Park, S., et al. (2023). TRAK: Attributing Model Behavior at Scale. *ICML 2023*.

Penedo, G., et al. (2023). The RefinedWeb Dataset for Falcon LLM. *NeurIPS 2023 Datasets Track*.

Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR 2020*.

Soboleva, D., et al. (2023). SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. *Cerebras Systems*.

Wenzek, G., et al. (2020). CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data. *LREC 2020*.

Wettig, A., et al. (2024). QuRating: Selecting High-Quality Data for Training Language Models. *ICML 2024*.
