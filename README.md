# Hierarchical Routed Sinkformer (HRS)

**Geometry-Shaped Representations for Compute-Adaptive Language Modeling**

HRS is a transformer architecture organized around a core principle: *computation should be proportional to relevance*. Instead of applying global attention uniformly, HRS routes tokens through a hierarchy of compute tiers based on learned relevance scores.

## Headline Results

**The Engram Is an Address: Test-Time Memory via Stable Geometric Routing Spaces.** A 68-phase experimental arc on a single RTX 5070 Ti, four days of work. The central finding: a mean-pooled hidden-state vector (engram) functions primarily as an address into a geometric routing space shared across models of the same architecture, not as a content summary. The same engram retrieves **0/20** passkeys before test-time training and **20/20** after — the vector did not change; the model's ability to interpret it did. Random vectors of identical norm route at 0/10 and achieve K-space alignment of 0.525 vs the real engram's 0.856. The address space is stable under training perturbation (2.5% K-cos degradation across models) and resolvability is a phase transition (0/5 → 5/5 in 38 training steps while K-space alignment remains constant). **All core findings replicate on a standard softmax dot-product transformer** (127M params, PPL 20.15), confirming these are properties of the Q/K/V attention decomposition, not of the HRS architecture specifically.

Three deployable architectures:

- **Application 1: per-passage adapter library** with a learned **L0 → L5 projection router**. **100% routing correctness, 97% held-out paraphrase retrieval, +0.000% drift.** Saliency-weighted absorption enables **2× rank compression** (rank 64) and **2× faster convergence** (75 steps). **32× storage compression** (rank-64 + int8). Routing scales to 500 passages at 100% accuracy.
- **Application 2: engram-compressed KV cache.** **92% gap closed at 2× compression** (distant half replaced by engram). Learned attention pooling lifts the information recovery floor from **18% to 30%** (5-token engram at 40× compression). The 30% ceiling is load-bearing: seven independent V-space interventions confirm that V-space lossiness and language modeling quality are in tension under standard transformer architectures. Fixed recency-based compression is near-optimal (adaptive routing gains only 0.03 NLL over the position heuristic).
- **Application 3: compositional retrieval via activation-level block-stacking.** **4/5 BOTH at K=2 with 10/10 routing correctness.** K=2 is K-limited, not budget-limited (K=4 collapses to 0% at every rank from 128 down to 16 and every budget level tested).

The architectural principle: **L5 for training, L0 for inference**. The routing key at inference is L0 (one embedding lookup, no forward pass). A 1024×1024 InfoNCE-trained projection bridges to L5's discriminative power. See the [NeurIPS paper](paper_neurips.md) and [Medium article](paper_adapter_library.md).

**V18 Cross-Attention Engram.** Fixes the causal attention leakage bug from V16 by isolating the engram via cross-attention. MAUVE 0.915–0.941 with engram active (vs V16's 0.806 failure mode). See the [V18 article](article_v18_crossattn.md).

**Cross-Attention Grounding Arc (tasks 3–11).** A nine-experiment sequence testing whether V18's cross-attention can be pushed past its ungrounded baseline to produce content-specific recall on a 25-needle benchmark with query-echo-stripped answer tokens (the "cleaned" recall metric). Tasks 3–5 built the training apparatus (RAFT — Retrieval-Augmented Fine-Tuning, the expanded NIAH benchmark, query-echo stripping) and established a 0/138-cleaned-recall floor. Tasks 6–8 tried three training objectives (soft contrastive, hard contrastive at 10× weight, on-policy REINFORCE on rollouts) — all four produced no consistent recall above the 1/138 noise floor despite gates opening to `softplus(2.0)` under the heaviest pressure. Tasks 9–10 tried two architectural changes (attention-pooled per-head engrams sharing self-attn projections; same plus learned private `W_k_ph`, `W_v_ph` per head) — both produced attention entropy at exactly `log(32)` with projection norms growing but unused. Task 11 was a read-only diagnostic: inject residual activations of target tokens at each layer of V18 with norm-matched scaling, measure logit lift. The diagnostic confirmed content does survive V18's MLPs (+2.3 to +2.8 lift at L1–L4, modest −0.8 at L5); the cross-attention path is the right layer to fix, but *what* it injects (a pooled summary) is structurally different from the position-specific activations that produce measurable lift. Reports: [task 6/7](results/v18_raft_grounded_w10/REPORT.md) · [task 8](results/v18_raft_sampled/REPORT.md) · [task 9](results/v18_perhead/REPORT.md) + [diagnostics](results/v18_perhead/DIAGNOSTICS.md) · [task 10](results/v18_perhead_proj/REPORT.md) · [task 11](results/ood_decay/REPORT.md).

**Topic-Routed Context Assembly.** Uses the model's own hidden-state representations to organize context by topic instead of recency. Engram cosine similarity achieves 97.3% topic accuracy at article scale (512 tokens) but only 71.5% at sentence scale — a signal-to-noise scaling law. MAUVE improves to 0.962, but a deeper evaluation suite reveals this is distributional contamination, not genuine quality improvement: held-out perplexity worsens (38.1 vs 35.8) and an LLM judge (Llama 3.1) prefers baseline 28-22. See the [context curation paper](paper_context_curation.md).

**Exponential Kernel Attention.** Replacing dot-product attention with an exponential kernel (negative squared Euclidean distance) on Tiny Shakespeare: +3.4% topic separation at play-level (256 chars), tied at line-level. The kernel shapes representations differently where signal is adequate, but can't rescue the short-text noise floor. Exponential kernel also achieves slightly lower best val loss (1.613 vs 1.630).

**Key methodological finding:** MAUVE alone is insufficient for evaluating context engineering systems. A distributional metric can be inflated by distributional contamination — injecting reference-distribution text into the context window. Conditional metrics (held-out perplexity, LLM-as-judge) are necessary complements.

**Previous headline:** V16 achieved **1.71 BPE perplexity** and MAUVE 0.905 with engrams disabled. See the [V16 article](article_peer_engram.md).

**Important caveat:** Perplexity is BPE (subword), not word-level. Published WikiText-103 benchmarks use word-level tokenization. See the [V12 writeup](v12_article.txt) for discussion.

## Architecture

**Core:**
- **Dual-head backbone** — generative (CE) + locality (InfoNCE) heads
- **PEER FFN** — Parameter Efficient Expert Retrieval with 262K single-neuron experts via product keys
- **Phased training** — differential learning rates across 4 phases

**V18 Cross-Attention Engram:**
- **Cross-attention injection** — engram enters via dedicated cross-attention blocks at alternating layers, structurally isolated from the causal self-attention path
- **Learned gates** — sigmoid-gated output (settled at 0.27–0.33) lets the model control engram influence per-layer
- **Categorization head** — topic classification objective gives the engram a discriminative training signal
- **EMA buffer** — corpus-level engram updated every 100 steps via exponential moving average

**V18-EGR (Entropy-Gated Retrieval):**
- **Entropy as write/read trigger** — high-entropy text stored, retrieved when generation entropy spikes
- **100% needle-in-a-haystack retrieval** at mean rank 1.2 among 20 distractors

**Topic-Routed Context Assembly:**
- **Online engram clustering** — prompts clustered by topic in real time, no predefined taxonomy
- **Evolving centroids** — cluster identity drifts as conversation develops
- **Auto-merge** — clusters that converge are automatically combined
- **User-toggleable topics** — named clusters users can enable/disable
- **Configurable active slots** — 2 for simple conversations, 6-7 for multidisciplinary work

## Results

| # | Configuration | Params | Best BPE PPL | MAUVE | Notes |
|---|---------------|-------:|-------------:|:-----:|-------|
| V18+Topic | PEER + cross-attn + topic routing | 512M | 38.1* | 0.962 | *MAUVE inflated by distributional contamination |
| V18+EGR | PEER + cross-attn + entropy retrieval | 512M | 23.3 | 0.950 | Entropy-gated retrieval |
| V18 | PEER + cross-attn engram + categorization | 512M | 23.3 | 0.915–0.941 | Fixes V16 leakage bug |
| V17 | PEER only, no engram (baseline) | 499M | 21.4 | 0.933–0.943 | Clean ablation baseline |
| V16 | PEER + prepend engram | 510M | 1.71 | 0.806–0.906 | Engram as training scaffolding |
| V12 | V9 + 6 layers, no Phase 5 | 250M | 3.32 | — | Extended to 100K steps |

*Topic routing MAUVE of 0.962 is misleading — held-out perplexity worsens and LLM judge prefers baseline 28-22.

## Key Findings

- **Cross-attention fixes the engram leakage bug.** V16 prepend caused MAUVE to drop 0.10. V18 cross-attention changes MAUVE by only 0.003.
- **Engram similarity achieves 97.3% topic accuracy at 512 tokens** — linearly separable, no complex infrastructure needed. At sentence scale: 71.5%.
- **A learned classifier cannot beat a fixed cosine threshold** — confirming the failure is representational (signal-to-noise), not algorithmic.
- **MAUVE can be inflated by distributional contamination.** Topic routing improved MAUVE from 0.919 to 0.962 while worsening perplexity from 35.8 to 38.1. Four independent metrics (perplexity, semantic similarity, LLM judge, routing correlation) agree the "improvement" is illusory.
- **Engrams encode semantics, not vocabulary.** 100% adversarial routing accuracy — metaphorical cross-domain prompts cluster with their literal counterparts.
- **Exponential kernel attention improves topic separation by 3.4%** at play-level on Shakespeare but is tied at line-level. The kernel shapes representations where signal is adequate.
- **The categorization head fails at 3.3% accuracy** despite training loss of 1.2 — sequence-level label noise prevents generalizable classification.
- **Consumer hardware is sufficient.** All experiments on a single RTX 5070 Ti (~$600). Training takes 11-12 hours. VRAM peaks at 12.5 GB.

## Running the Experiments

### Requirements

```bash
pip install torch datasets transformers scikit-learn mauve-text
```

### V18 (PEER + cross-attention engram)

```bash
python train.py --ablation v18_cross_attn --output-dir results   # ~11.4 hours
python benchmark_mauve_v18.py                                     # ~3 hours
```

### Entropy-Gated Retrieval

```bash
python populate_store.py --threshold 4.0 --output engram_store_data   # ~2 min
python benchmark_mauve_egr.py --store engram_store_data               # ~4 hours
python niah_egr.py --n-distractors 20                                 # ~15 min
```

### Topic-Routed Context Assembly

```bash
python benchmark_mauve_topic.py --threshold 0.5                    # ~2 hours
python eval_topic_routing.py --threshold 0.5                       # ~30 min
python eval_accuracy.py --ollama-host 192.168.12.125 --ollama-model llama3.1  # ~20 min
python benchmark_topic_routing.py --threshold 0.4                  # ~1 min
```

### Topic Classifier Training

```bash
python train_topic_classifier.py --n-articles 2000                 # article-scale pairs
python train_topic_classifier_short.py --n-articles 3000           # sentence-scale pairs
```

### Exponential Kernel Attention

```bash
python exp_kernel_attention.py --n-steps 10000                     # ~20 min
```

### Test-Time Memory (68 phases)

67 standalone deterministic scripts under `experiments/identity_ae/`. Core phases run in ~90 minutes; pre-training ablations (48–58) add ~60 GPU-hours; cross-architecture validation (63–64) adds ~4 hours. All load `results/v22_learned_kernel/best.pt` as the frozen base model.

**Application 1 — per-passage adapter library (100% routing, 97% retrieval):**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase47_l0_to_l5_projection.py  # ~5 min, the headline: 100/100/97
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase52_saliency_weighted.py    # ~20 min, 2× rank compression + 2× speed
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase53_adapter_clustering.py   # ~3 min, clustering impossible (negative)
```

**Verification stage (Phases 65 / 69 / 70 — scoping the deployment claim):**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase65_address_space.py        # ~4 min, trained W partially in base; C0b L0-keys gets 93%
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase65_specificity_hardood.py  # ~1 min, near-neighbor OOD breaks routing at 92% FP
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase69_verification.py         # ~1 min, metadata entity-match verification 100% balanced acc (closed-vocab)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase70_verification_head.py    # ~1 min, per-adapter h_mean_L5 head 90% balanced acc (open-vocab; in-sample calibrated)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase70_kscaling.py             # ~2 min, K-scaling: per-pair x-FPR invariant 19.6%, end-to-end 65% (honest calibration)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase71_hard_ood_aware.py       # ~2 min, hard-OOD-aware training drops x-FPR 20.6%->14.7% (partial)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase72_adapter_aware_base.py   # ~1 min, single-adapter aware base training: composite FAIL (regularizer over-rotates, spillover to general)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase72c_frozen_adapter.py      # ~3 min, base learns to use frozen rank-8 adapter: passkey retrieval FAIL→PASS, pk_ce 2.14→0.05 (40x lift on fixed signal); strict composite FAIL but architectural mechanism validated
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase73_generalization.py       # ~2 min, generalization test: 0/4 held-out adapters PASS; mean Δ +0.25 nat. Phase 72c improvement is per-adapter, not a general substrate.
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase74_frozen_heads.py         # ~6 min, schema-compliance test (16 adapters, 4 width-32 bottleneck heads): Phase A 100% PASS, Phase B 100% PASS, Phase C 0% FAIL + Phase A retention collapses to 9% (catastrophic forgetting). Per-adapter compliance, no general schema skill.
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase75_large_adapter.py        # ~5 min, large-content adapter (rank 128, 335-tok passage): calibration found no useful intermediate regime (15% jumps to 70% between 25 and 50 steps); fallback at 80% baseline → 16/20 retrieval unchanged after 400 steps + WikiText PPL +34%. NO_SUCCESS, mechanism doesn't engage without headroom.
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase76_scaled_sweep.py         # ~18 hr, scaled sweep at "real training scale" (200 patient-record adapters, 5 width-32 heads, 20k QA pairs, ~80k main-sweep steps). 3-stage bootstrap (16 NTP adapters → joint base+heads → 200 head-CE adapters), then main sweep with 4:1 WT:OOD ratio and wall-clock checkpoints at 1h/4h/16h. Held-out per-field accuracy is the architectural metric; alive iff it improves monotonically AND reaches >70% by 16h.
```

**Application 2 — engram-compressed KV cache (92% gap closed at 2×):**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase33_engram_context.py       # ~3 min, six prefix conditions
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase51_learned_engram.py       # ~2 min, attention pooling 18→28%
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase55_multi_token_engram.py   # ~4 min, K=5 at 30%
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase67_engram_vs_summary.py    # ~15 min, engram beats summarization
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase68_adaptive_router.py      # ~35 min, fixed recency is near-optimal
```

**Application 3 — compositional retrieval (4/5 BOTH at K=2):**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase42_clause_routing.py       # ~3 min, 4/5 BOTH, 10/10 routing
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase62_k_ceiling_rank_sweep.py # ~25 min, K-limited not budget-limited
```

**Validation experiments (address space characterization):**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase60_resolvability_curve.py  # ~2 min, phase transition at 15-38 steps
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase61_basin_validation.py     # ~1 min, trajectory convergence 0.03→0.45
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase59_cross_model_transfer.py # ~10 min, 2.5% K-cos degradation
# Phase 65 (random vector ablation) is inline — 0/10 routing for random vs 10/10 real
# Phase 66 (scaling) is inline — 100% routing accuracy at 500 passages
```

**V-space analysis and negative results:**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase50_vspace_svd.py           # ~10 min, effective rank ~48, SVD engrams negative
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase48_kl_mlp_regularizer.py   # ~6 min, KL regularization (negative)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase56_vspace_recon_pretraining.py  # ~3.5 hr, recon loss (negative)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase57_recon_gated.py          # ~3.5 hr, gated residuals (negative)
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase58_bilinear_attention.py   # ~7 hr, bilinear attention (marginal +3 pts)
```

**Cross-architecture validation (standard softmax transformer):**

```bash
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase63_softmax_baseline.py     # ~4 hr, pre-train 127M softmax model
PYTHONPATH=. .venv/bin/python experiments/identity_ae/phase64_softmax_validation.py   # ~30 min, all five core findings replicate
```

### Previous Versions

```bash
python train.py --ablation v16_peer_engram --output-dir results    # V16
python train.py --ablation v17_peer_only --output-dir results      # V17
python train.py --ablation v12_247m --output-dir results           # V12
```

### Cross-Attention Grounding Arc (tasks 3–11)

All tasks start from `results/v18_cross_attn/best.pt` (except task 6/7/8 which start from `results/v18_raft/checkpoint_2000.pt`). Backbone frozen throughout; gates at 100× LR, cross-attn+cat at base LR. Eval is NIAH Config B (25 needles × 40 distractors) scored against query-echo-stripped answer tokens, plus MAUVE-500 at n=200. See each task's REPORT.md for the trajectory table and decision.

```bash
# Task 3 — RAFT phase 1 (50/25/25 retrieve/baseline/wrong buffer mix)
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/raft_train.py \
  --max-steps 5000 --out-dir results/v18_raft                              # ~1 hour

# Task 4 — expanded NIAH benchmark (5 → 25 needles)
# needles_v2.json + eval harness under experiments/hrs_loop/niah_benchmark/

# Task 5 — strip query-echo tokens from answers
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/niah_benchmark/clean_tokens.py

# Tasks 6 & 7 — grounding auxiliary loss (contrastive) at weights 0.1 and 1.0
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/raft_grounded_train.py \
  --max-steps 5000 --grounding-weight 0.1 --out-dir results/v18_raft_grounded       # ~1 hour
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/raft_grounded_train.py \
  --max-steps 5000 --grounding-weight 1.0 --out-dir results/v18_raft_grounded_w10   # ~1 hour

# Task 8 — on-policy REINFORCE with K=8 sampled-token rollouts
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/raft_sampled_train.py \
  --max-steps 5000 --grounding-weight 0.3 --out-dir results/v18_raft_sampled        # ~90 min

# Task 9 — per-head attention-pooled engrams (shared projections)
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/build_perhead_cache.py           # ~2 min, writes 912MB cache
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/raft_perhead_train.py \
  --max-steps 5000 --out-dir results/v18_perhead                                     # ~1 hour
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/diagnostics_perhead.py            # ~15 min, seed/disable/entropy

# Task 10 — per-head engrams + learned private W_k_ph, W_v_ph per head (zero-init)
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/raft_perhead_proj_train.py \
  --max-steps 5000 --out-dir results/v18_perhead_proj                                # ~1 hour

# Task 11 — read-only: OOD content decay through V18's MLP stack
PYTHONPATH=. .venv/bin/python experiments/hrs_loop/diagnostic_ood_decay.py           # ~5 min, no training
```

## Files

### Model & Training
| File | Description |
|------|-------------|
| `model.py` | HRS transformer (backbone, tiers, cross-attention engram, categorization head) |
| `peer.py` | PEER expert retrieval (262K single-neuron experts via product keys) |
| `engram.py` | Engram encoder, injectors, cross-attention block, categorization head |
| `config.py` | All configuration dataclasses and ablation presets (V1–V18) |
| `train.py` | Training loop with phased protocol, differential LRs, engram buffer updates |
| `data.py` | WikiText-103 loading with GPT-2 BPE tokenizer and category labels |
| `losses.py` | Combined loss with CE, locality, reconstruction, categorization |
| `router.py` | Learned token router with TRC, balance/entropy/FLOPs losses |
| `tiers.py` | Tiered compute operators (conv, attention, sink) |
| `bdh.py` | Virtual synapse, hub routing loss, sparsity bottleneck |
| `metrics.py` | Effective rank, routing entropy, tier distribution tracking |

### Entropy-Gated Retrieval
| File | Description |
|------|-------------|
| `engram_store.py` | Engram vector store with cosine similarity retrieval |
| `entropy_monitor.py` | Rolling entropy computation and threshold monitoring |
| `retrieval_engine.py` | Entropy-gated engram retrieval engine for V18 inference |
| `populate_store.py` | Pre-populate engram store from WikiText-103 |
| `evaluate_retrieval.py` | Retrieval system evaluation (perplexity, trigger stats) |
| `niah_egr.py` | Needle-in-a-haystack test for entropy-gated retrieval |

### Topic-Routed Context
| File | Description |
|------|-------------|
| `topic_context.py` | TopicContextManager: online clustering, active buffer, user toggles, auto-merge |
| `train_topic_classifier.py` | Article-length pair analysis and classifier training |
| `train_topic_classifier_short.py` | Short-prompt pair analysis and classifier training |
| `benchmark_topic_routing.py` | Seven stress tests (drift, overlap, adversarial, fork) |
| `eval_topic_routing.py` | Four-metric evaluation suite (perplexity, coherence, repetition, routing correlation) |
| `eval_accuracy.py` | Accuracy evaluation with LLM-as-judge (ollama) support |
| `eval_categorization.py` | Categorization head evaluation (negative result) |
| `benchmark_mauve_topic.py` | MAUVE benchmark for topic-routed context |

### Test-Time Memory: Adapter Library + KV Cache Compression + Compositional Retrieval
| File | Description |
|------|-------------|
| `identity_autoencoder.py` | IdentityAutoencoder, OODDetector, EngramRecurrence, EngramLibrary — historical gate primitives, dropped from the deployed architecture after Phase 39 |
| `experiments/identity_ae/dual_gate.py` | DualGate (frozen base gate + trainable novel gate) — used in phases 9, 16–18 before the per-passage architecture |
| `experiments/identity_ae/lora_wrapper.py` | Minimal LoRA wrapper (LoRALayer, apply_lora, get/load/reset state dict) |
| `experiments/identity_ae/phase0_train.py` | Train the identity autoencoder offline on V22 hidden states from WikiText |
| `experiments/identity_ae/phase10_passkey.py` | Passkey benchmark generator (50 deterministic test cases, 4 types) and full-model TTT baseline |
| `experiments/identity_ae/phase11_lr_schedule.py` | Scheduled-LR full-model TTT (40–60 step variants, 100% per-passage at 4s/passage) |
| `experiments/identity_ae/phase12_forgetting.py` | Cumulative forgetting check for fast schedule — +44% per passage, motivates LoRA |
| `experiments/identity_ae/phase14_lora_l45.py` | Layer 4–5 LoRA single-passage absorption sweep — finds rank 512 / 100 steps as the per-passage winner |
| `experiments/identity_ae/phase16_combined.py` | L4-5 LoRA + dual gate, single-passage retention test |
| `experiments/identity_ae/phase17_stratified.py` | Stratified-type version of phase 16 |
| `experiments/identity_ae/phase18_gate_replay.py` | Replay-augmented novel gate (4/20 → 18/20 gate accuracy) |
| `experiments/identity_ae/phase19_rehearsal.py` | Shared-LoRA continual rehearsal — failing approach (50% cumulative) |
| `experiments/identity_ae/phase20_sweep.py` | Three-config sweep — rank 1024 collapses to 20%, the diagnostic negative result |
| `experiments/identity_ae/phase21_per_passage_adapters.py` | First per-passage adapter library (passage keys, 60% routing — diagnoses the prompt/passage mismatch) |
| `experiments/identity_ae/phase22_engram_key.py` | Key source ablation (L3 vs L5, mean vs last, L2 vs cosine) plus two-pass re-ranking |
| `experiments/identity_ae/phase23_prompt_keys.py` | Prompt-derived keys with same-prompt queries — 100% routing, 90% retrieval |
| `experiments/identity_ae/phase24_150steps.py` | 150-step adapters — closes the retrieval gap to 100/100 on same-prompt queries |
| `experiments/identity_ae/phase25_paraphrase.py` | Single-key paraphrase test — drops to 50%, diagnoses style-bias in mean-pooled engrams |
| `experiments/identity_ae/phase26_multikey.py` | Multi-key + multi-paraphrase training — 100/100 on training-distribution paraphrases |
| `experiments/identity_ae/phase27_held_out.py` | Held-out paraphrase generalization with L5 mean — 63% (the original binding constraint) |
| `experiments/identity_ae/phase28_staged.py` | **Staged absorption** — foreground (1.4 s) + background (1.4 s on a LoRA copy). Validates the zero-blocking deployment story. |
| `experiments/identity_ae/phase29_composition.py` | Compositional queries via weight merging — **negative result**: 0/5 pairs across all merge strategies |
| `experiments/identity_ae/phase29b_rank128.py` | Phase 29 retest at rank 128 — still 0/5, confirms the failure is rank-independent |
| `experiments/identity_ae/phase30_quantized.py` | int8 adapter quantization — 4× storage compression with 0% retrieval loss |
| `experiments/identity_ae/phase30b_rank128.py` | Phase 30 retest at rank 128 — 4× compression confirmed, +0.000% drift |
| `experiments/identity_ae/phase31_weighted_pool.py` | Entity-weighted (`nonstop_mean`) pooling — lifts held-out from 63% → 77% at L5 |
| `experiments/identity_ae/phase32_kv_similarity.py` | K-space alignment for L5 mean engram — **cosine 0.82–0.89** at layers 1–5, random ≈ 0 |
| `experiments/identity_ae/phase32b_l0_kv_similarity.py` | K-space alignment for L0 mean engram — **cosine 0.988** at layer 0 (the dual of L5) |
| `experiments/identity_ae/phase33_engram_context.py` | Engram-as-cache continuation perplexity — six prefix conditions on 50 WikiText passages |
| `experiments/identity_ae/phase33b_l0_engram_context.py` | Phase 33 with L0 engram injection — ties L5 at the standard operating point |
| `experiments/identity_ae/phase35_engram_after_ttt.py` | **The centroid theory test** — same engram, same model, **0/20 before TTT and 20/20 after** |
| `experiments/identity_ae/phase37_compression_sweep.py` | Engram-cache compression curve — smooth from 0 to 384× with no knee |
| `experiments/identity_ae/phase38_rank_sweep.py` | LoRA rank sweep — pins the floor at rank 128, rank 256 *worse* than 128 |
| `experiments/identity_ae/phase38b_rank128_heldout.py` | Held-out paraphrase test at rank 128 — 100/100/77 with L5_nonstop_mean baseline |
| `experiments/identity_ae/phase39_gate_value.py` | **Gate ablation** — cosine threshold gets 99/98, the IAE gate is 48% specificity (worse than coin flip), combining hurts. We dropped the gate. |
| `experiments/identity_ae/phase40_sparsity.py` | Sparse magnitude pruning — 1.8× more compression at threshold 5e-3 (29× total with held-out tradeoff) |
| `experiments/identity_ae/phase41_activation_composition.py` | **Activation-level block-stacking** — bridges the Phase 29 negative result, 4/5 BOTH at oracle pair selection |
| `experiments/identity_ae/phase42_clause_routing.py` | **Clause-split routing** — splits compositional queries on connectives, 4/5 BOTH with 10/10 routing |
| `experiments/identity_ae/phase42b_l0_clause_routing.py` | Phase 42 with L0 routing — confirms the K=2 ceiling is set by capacity, not routing |
| `experiments/identity_ae/phase43_k_capacity.py` | K-capacity sweep — K=2 holds, K=4 collapses to 35%, K=8 saturates at 5% |
| `experiments/identity_ae/phase44_l0_engram.py` | **L0 mean routing** — 100/100/90 held-out paraphrase retrieval, +15 points over L5_nonstop_mean, no forward pass |
| `experiments/identity_ae/phase45_l0_l5_pair.py` | Heterogeneous L0+L5 pair (avg cosine routing, two-position prefix injection) — does not dominate either alone |
| `experiments/identity_ae/phase46_l5_contrastive.py` | **L5-contrastive auxiliary loss** during absorption — reduces L5 anchor overlap as designed but K-capacity *regresses*. Rules out representation-orthogonalization as the K=2 fix. |
| `experiments/identity_ae/phase47_l0_to_l5_projection.py` | **The L0 → L5 projection** — 1024×1024 linear map trained with InfoNCE, lifts to **100/100/97** held-out, the routing ceiling. |
| `experiments/identity_ae/phase47b_inspect.py` | Manual inspection of all 60 held-out generations — zero substring false positives, 2 generation-side failures, headline confirmed |
| `experiments/identity_ae/phase48_kl_mlp_regularizer.py` | KL-divergence regularization during absorption — trades retrieval for drift at wrong ratio (negative) |
| `experiments/identity_ae/phase49_enriched_input.py` | Metadata-enriched embeddings bolt-on — NLL explodes (negative) |
| `experiments/identity_ae/phase49b_train_enriched.py` | Pre-train from scratch with metadata on 90% of batches — V-space unchanged (negative) |
| `experiments/identity_ae/phase50_vspace_svd.py` | V-space SVD analysis — effective rank ~48, SVD engrams recover negative information |
| `experiments/identity_ae/phase51_learned_engram.py` | **Attention-pooling encoder** — lifts information recovery from **18% to 28%** with one learned query |
| `experiments/identity_ae/phase52_saliency_weighted.py` | **Saliency-weighted absorption** — **2× rank compression, 2× faster convergence** |
| `experiments/identity_ae/phase53_adapter_clustering.py` | Adapter clustering — impossible even at L0 cosine 0.97 (negative) |
| `experiments/identity_ae/phase54_vspace_continued_training.py` | Saliency-weighted continued training — PPL +22% (negative) |
| `experiments/identity_ae/phase55_multi_token_engram.py` | Multi-token engrams — K=5 at **30%** recovery, 40× compression. K=10 no improvement |
| `experiments/identity_ae/phase56_vspace_recon_pretraining.py` | V-space reconstruction loss from scratch — V-cos 0.73 but PPL 43 (negative) |
| `experiments/identity_ae/phase57_recon_gated.py` | Gated residuals + recon loss — V-cos **0.805** but PPL 46, confirms V-space lossiness is load-bearing |
| `experiments/identity_ae/phase58_bilinear_attention.py` | Bilinear attention q^T W k — PPL matched, +3 pts recovery, W far from identity (marginal) |
| `experiments/identity_ae/phase59_cross_model_transfer.py` | **Cross-model transfer** — 2.5% K-cos degradation, routing geometry is stable |
| `experiments/identity_ae/phase60_resolvability_curve.py` | **Phase transition** — 0→5/5 in 38 steps, K-space constant throughout |
| `experiments/identity_ae/phase61_basin_validation.py` | **Basin validation** — trajectory convergence 0.03→0.45 from L0 to L5 |
| `experiments/identity_ae/phase62_k_ceiling_rank_sweep.py` | K-ceiling rank sweep — K-limited not budget-limited across all ranks 16-128 |
| `experiments/identity_ae/phase63_softmax_baseline.py` | **Standard softmax transformer** pre-training — 127M params, PPL 20.15 |
| `experiments/identity_ae/phase64_softmax_validation.py` | **Cross-architecture validation** — all five core findings replicate on softmax |
| `experiments/identity_ae/phase67_engram_vs_summary.py` | Engram vs summarization at equal compression — engram wins (81% vs -11% gap closed) |
| `experiments/identity_ae/phase68_adaptive_router.py` | Adaptive compression router — fixed recency is near-optimal (+0.03 NLL for oracle) |
| `experiments/identity_ae/phase69_adapter_composition.py` | Rank-dim concat composition of independent adapters — falsifies free composability: ‖Σδ‖_F grows ~√K (orthogonal) yet K=2 retrieval collapses (rank 128: 75%→0%); functional interference, not magnitude, kills it |
| `experiments/d2l/phase01_starcoder_baseline.py` | StarCoder2-3B baseline cloze on Bleak House — Dickens content at floor (0% char/plot top-1, 6% possession), with PII anonymization (`<NAME>` as top-1 for `Mr.`/`Lady`/`Jarndyce and ___`). GREEN to build Phase 02 Perceiver. |
| `experiments/d2l/phase02_perceiver_train.py` | D2L Perceiver (33M, 5K steps KL distillation) + six-condition eval — **manifold-level claim falsified**: C2 RAG 92% on names vs C3 adapter 6%; engram-prefix composition C4 (2%) *hurts* the adapter (NEGATIVE). `<NAME>` suppression is robust to adapter but trivially bypassed by in-context attention. Failure-mode analysis in `results/d2l/phase02/all_misses.md` shows C2's 4 character misses are all BPE artifacts (true content rate ≈99%); C3's 3 character hits are all the same `my → L`-for-Lady-Dedlock bigram. |
| `experiments/d2l/phase03_perceiver_train.py` | 30K rerun + α=16 (fp32 LoRA) + multi-token / `<NAME>`-banned scoring. **Strict top-1 names: 6→12%** (bigram-memorization sharpened, not broken through). **NAME-banned plot content: C3=38%, C5=76% (+28pp over C2)** — the adapter IS installing first-sub-token content; the suppression token was hiding it. Adapter specializes to proper nouns (training-data artifact) → hurts code (32→8) and possessions (6→0). Latency 1.18× amortized / 1.40× cold; Perceiver inference 3.6 ms. |

### Benchmarks & Generation
| File | Description |
|------|-------------|
| `benchmark_mauve.py` | MAUVE benchmark (V16-style) |
| `benchmark_mauve_v18.py` | MAUVE benchmark for V18 cross-attention engram |
| `benchmark_mauve_egr.py` | MAUVE benchmark for V18 + entropy-gated retrieval |
| `exp_kernel_attention.py` | Exponential kernel vs dot product attention experiment |
| `generate_sample.py` | Generation quality checker with WikiText context seeding |
| `eval_word_ppl_v2.py` | BPE and word-level perplexity evaluation |

## Papers

- [The Engram Is an Address (NeurIPS submission)](paper_neurips.md) — Formal paper: stable geometric routing spaces, K/V asymmetry, phase transition in resolvability, cross-architecture validation, 68 phases, random vector ablation, V-space lossiness characterization
- [The Engram Is an Address (Medium article)](paper_adapter_library.md) — Accessible version: three applications, the address/content decomposition, validation experiments, V-space tradeoff
- [Topic-Routed Context Assembly](paper_context_curation.md) — "Your Transformer Already Knows What It's Talking About." Engram-based topic routing, seven stress tests, distributional contamination finding, LLM-as-judge evaluation
- [V18 Cross-Attention Engram + EGR](article_v18_crossattn.md) — Cross-attention fix, entropy-gated retrieval, NIAH evaluation
- [V16 PEER + Engram](article_peer_engram.md) — 1.71 BPE perplexity, MAUVE 0.905, engram-as-scaffolding finding
- [V12 Results](v12_article.txt) — 3.32 BPE perplexity, Phase 5 diagnosis, tokenization discussion
- [BDH and Learnable Loss Scaling](HRS_paper_medium.md) — Brain-derived heuristics with fixed vs learned coefficients
- [Full HRS paper](paper.md) — Original theoretical framework, training protocol, and ablation study

## Author

Michael Bee ([@mbonsign](https://medium.com/@mbonsign))
