# Original HRS direct-injection architecture — what's actually in the repo

This is a search-and-report deliverable. Goal: locate the version of HRS
that used a separate smaller model to approximate L5 hidden states and
concatenated that approximation to the main model's input. Below is what
was actually found in the repo and the writeups, including some
reconstruction details that **do not match** the originating description.

## Summary up front

**There does not appear to be a "separate smaller model" in the HRS repo.**
The engram pipeline is implemented entirely as submodules of the main
`HRSTransformer`. No external/auxiliary model is referenced in code, in
the AblationConfig variants, or in the published Medium articles
included in the repo. If such a model exists in your design notes or in
an earlier prototype, it is not what shipped to the codebase.

The pre-cross-attn architecture (V2-V16) instead uses an *internal* engram
encoder and a *prepend* injection mechanism (or a *replace* variant in V5).
The "concat" semantics are real but they are sequence-axis concatenation
(prepending engram vectors to the token sequence at intermediate layers),
not feature-axis concatenation of an approximator's output to model input.

The "L5" framing also does not match — extraction layers were configured at
either the early-middle of the stack (V16: layer 1 of 6, the second layer)
or at the second-to-last (V18+: layer 4 of 6). The last layer (L5 in a
6-layer model, 0-indexed) is **never** the extraction source.

This means three of the user's framing claims need updating before
designing a tiny-Shakespeare experiment:

1. **No separate model** — the engram encoder is a 2-layer MLP submodule
   of the main `HRSTransformer`.
2. **Not L5** — extraction is from a middle layer (layer 1 of 6 in V16,
   layer 4 of 6 in V18+). Layer 5 is the final layer's output and is not
   used as engram source.
3. **Sequence-axis concatenation, not per-token feature concatenation** —
   engrams are prepended as additional sequence positions before each
   post-extract layer, not concatenated to each token's d-dim feature
   vector.

The "1024×1024" recollection partially holds: V16's `d_model=1024`, so the
encoder's input and output dims are both 1024.

## Version progression (from `config.py:AblationConfig`)

The pre-cross-attn engram era covers V2 through V16. V17 dropped engrams
entirely as a baseline, and V18 introduced cross-attention as the
replacement mechanism.

| Variant | Description (from config.py) | Engram mechanism |
|--|--|--|
| V2_FULL | Attention→Conv + dual-head + PEER + Engrams + phased training | original prepend |
| V3_FULL | v1 routing + PEER as expert tier + engrams | prepend |
| V4_FULL | PEER as universal FFN + 3-tier routing + engrams | prepend |
| **V5_REPLACE** | v4 + learnable engram-based context replacement (no prepend) | in-place replace via `EngramReplacer` |
| **V6_GATE** | v4 + learned remember gate + seq_len=1024 | gated prepend via `RememberGate` + `EngramInjector` |
| V7_FULL | v4 base + Memory MLP + V7Router (logit blending) | prepend |
| V8_BDH | v4 + BDH (virtual synapse, hub routing, sparsity) | prepend + BDH virtual synapse focus |
| V9_LEARNABLE | v8 + learnable loss scaling | prepend |
| V10_CONTROL | dense baseline + PEER + placeholder losses (no BDH) | none |
| V11_NO_P5 | v9 + skip phase 5 | prepend |
| V12_247M | v9 + 6 layers (250M params), best-model checkpoint | prepend |
| V13_LOW_SPARSITY | v12 + 5% sparsity | prepend |
| V14_ATTN_SINK | v13 + 2-tier routing | prepend |
| V15_VANILLA_ROUTE | vanilla transformer + 2-tier routing + engrams | prepend |
| **V16_PEER_ENGRAM** | **vanilla transformer + PEER FFN + engrams, no routing/BDH** | **canonical prepend** |
| V17_PEER_ONLY | vanilla transformer + PEER FFN, **no engrams** | (baseline) |
| **V18_CROSS_ATTN** | PEER + **cross-attention engram (no prepend)** + categorization | cross-attention (NEW mechanism) |
| V19_EXP_KERNEL | V18 + exponential kernel attention | cross-attention |
| V20_BONSIGNORE | V19 + per-head learned kernel MLP | cross-attention |

V16 is the canonical "engram + nothing else fancy" version. V18's config
block in `config.py:597` carries the explicit comment:

> `# Fixes V16's causal attention leakage by isolating engram via cross-attention`

So the V16→V18 transition was specifically motivated by a known leak in
V16's prepend mechanism (more on this below).

## V16 architecture, in detail

From `config.py:573-595`:

```python
elif ablation == AblationConfig.V16_PEER_ENGRAM:
    cfg.model.n_layers = 6
    cfg.model.d_model = 1024
    cfg.model.d_ff = 4096
    cfg.model.n_heads = 16
    cfg.locality.enabled = True
    cfg.engram.enabled = True
    cfg.engram.engram_dim = 1024     # = d_model
    cfg.peer.enabled = True
    cfg.bdh.enabled = False
    cfg.training.batch_size = 4
    cfg.training.grad_accum_steps = 8
    cfg.training.max_steps = 50000
    cfg.phased.enabled = True
    cfg.phased.phase1_steps = 8000
    cfg.phased.phase2_steps = 8000
    cfg.phased.phase3_steps = 10000
    cfg.phased.phase4_steps = 24000
    cfg.phased.phase5_steps = 0
```

The engram block uses the `EngramConfig` defaults (config.py:138-156)
since V16 doesn't override:

```python
window_size: int = 128       # W tokens per window
n_engrams: int = 4           # K engrams per window
extract_layer: int = 1       # 0-indexed; layer 2 of 6 in V16
recon_loss_weight: float = 0.1
drop_prob: float = 0.1       # built-in engram dropout, not the experiment under test
```

So in V16, with seq_len 512: 4 windows × 4 engrams = 16 engram vectors per
sample. Extracted from the output of layer 1 (the *second* transformer
block, near the start of the stack — *not* L5).

### EngramEncoder (the would-be "approximator")

`engram.py:15-79`. **Internal MLP submodule of HRSTransformer.** Not an
external model. Construction:

```python
self.encoder = nn.Sequential(
    nn.Linear(d, 2 * d),         # 1024 -> 2048
    nn.GELU(),
    nn.Linear(2 * d, K * d),     # 2048 -> 4096
)
self.norm = nn.LayerNorm(d)
```

Forward:

```python
windows = h[:, :usable].reshape(B, n_windows, W, D)
pooled = windows.mean(dim=2)                 # (B, n_windows, D)
encoded = self.encoder(pooled)               # (B, n_windows, K*D)
engrams = encoded.reshape(B, n_windows*K, D)
return self.norm(engrams)
```

So the "compression" is just per-window mean-pool followed by a 2-layer
MLP that produces K=4 engram vectors per window. Parameter count: roughly
`d × 2d + 2d × Kd + LayerNorm = 1024·2048 + 2048·4096 + 2048 = 10.5M
parameters` for V16's d=1024.

### EngramInjector (the would-be "concatenation")

`engram.py:82-115`. Prepends engrams to the token sequence at every
post-extract layer:

```python
def forward(self, x, engrams):           # x: (B, T, D); engrams: (B, E, D)
    if engrams.shape[1] == 0:
        return x, 0
    engrams = engrams + self.engram_type_emb   # learnable type embedding
    combined = torch.cat([engrams, x], dim=1)  # (B, E+T, D), prepend
    return combined, engrams.shape[1]
```

In the main forward pass (`model.py:850-877`), this happens for every
layer after `extract_layer`:

```python
for i, block in enumerate(self.blocks):
    if self.use_engrams and i > self.engram_extract_layer and engrams.shape[1] > 0:
        x, n_engrams = self.engram_injector(x, engrams)   # prepend
    x, ... = block(x, ...)                                # block runs on (B, E+T, D)
    if not self._uses_replacement and n_engrams > 0:
        x = x[:, n_engrams:]                              # strip prepended positions
    if self.use_engrams and i == self.engram_extract_layer:
        engrams = self.engram_encoder(x)                  # extract engrams here
```

So engrams are NOT permanently part of the residual stream. They get
prepended as additional sequence positions for each post-extract attention
computation, then stripped before the residual passes to the next block.
This is "sequence-axis concatenation of K=16 d-dim vectors before the
existing sequence at each post-extract layer," not "input-level
concatenation."

### V5 / V6 variants

Two variations on the prepend mechanism:

- **V5_REPLACE** (`engram.py:331-413, EngramReplacer`): in-place replacement
  of high-loss windows with upsampled engrams instead of prepending. Uses
  `EngramUpsampler` (lines 118-175) to expand K engrams back to W
  positions via `repeat_interleave` + learnable positional embedding +
  refinement MLP.
- **V6_GATE** (`engram.py:178-329, RememberGate`): gates the prepend with
  a 4-feature MLP that decides per-window whether to inject. Features:
  per-window loss z-score, cosine-sim between hidden state and engram
  means, hidden-state norm, engram norm.

These are not "separate models" either — both are nn.Module submodules
of the main HRSTransformer.

### Training

From `train.py:1-3`:

> "Training loop for HRS experiments with phased LR schedule. Supports v1
> (routed), v2 (attention->conv + PEER), and v7 (Memory MLP + V7Router)
> architectures."

Training is **joint** with the main model:

- `EngramEncoder.parameters()` are part of `model.parameters()` and
  optimized together with the rest of the network.
- The recon loss (cosine similarity between window-pooled hidden states
  and window-pooled engrams; `engram.py:531-569`) is added to the LM loss
  with weight `recon_loss_weight = 0.1`.
- LM-loss gradient also flows through the engram pathway: engrams are
  prepended into the attention computation at each post-extract layer,
  so the LM cross-entropy gradient propagates into the encoder via the
  attention path.

There is **no distillation step**, **no pre-training of an auxiliary
model**, and **no freezing of the encoder**. The encoder co-trains with
everything else from step 0.

V16 also has its own `drop_prob=0.1` knob for engram dropout (per
`config.py:144` and `model.py:905-907`):

```python
if self.training and self.cfg.engram.drop_prob > 0:
    if torch.rand(1).item() < self.cfg.engram.drop_prob:
        engrams = engrams[:, :0, :]   # empty tensor, same shape
```

This is the same per-batch zero-out idea as the Phase 1 / Phase 2
experiments here — it's been baked into V16 since the original engram
introduction. (The Phase 1 and Phase 2 specs may have been *re-asking* a
question the V16 code was already implicitly answering.)

## V16's "1.71 PPL" anchor and the known leak

From `article_peer_engram.md`:

> "The engram's 12.5x perplexity improvement (1.71 vs 21.41) is real and
> reflects genuine learning. The engram provides compressed context that
> helps the model predict next tokens more accurately during teacher-forced
> evaluation. But the representations shaped by this training-time signal
> produce worse generation, not better."

> "Standard evaluation computes perplexity with teacher forcing on
> 512-token sequences, which always have 4 full windows. Perplexity looked
> excellent. Generation from short prompts — the kind a user would
> actually provide — was gibberish."

And from `paper_adapter_library.md:39`:

> "the engram-as-cheat-sheet leak in earlier (V16) experiments where
> prepended engrams became visible to all subsequent positions through
> causal attention during teacher-forced evaluation."

So V16's headline 1.71 BPE PPL comes with two material caveats:
1. **Within-window leak in the encoder.** Engrams summarize 128-token
   windows via mean-pool. So engram[w] depends on tokens [w·128, w·128+127].
   When predicting token p inside window w, the residual at position p
   sees engram[w] via the prepended cross-attention path, and engram[w]
   contains aggregate info about tokens (p+1, p+2, ..., p_end_of_window).
   This is the same within-window mean-pool leak Phase 1 here had.
2. **Prepend-causal interaction.** Once prepended, engrams are at the
   front of the sequence (positions 0..E-1). Causal attention from token
   position p sees positions [0, E+p] — *all* engrams, including
   engram[w'>w] that summarize future windows. So the leak isn't only
   about within-window — it's about all-windows-visible-everywhere.

V18's cross-attention design fixes both: the engram is computed once per
forward pass (not per layer), is treated as an external KV buffer, and is
attended to via a separate cross-attention block that doesn't share the
self-attention causal mask. The buffer is also populated from *past
training samples* via `update_engram_buffer()`, breaking the same-batch
leak — which is the property Phase 1 here violated by using same-batch
engrams.

## What the user's recollection got partly right and partly wrong

| user recollection | what's actually in the code |
|--|--|
| "separate (smaller) model" | **No.** Encoder is a 2-layer MLP submodule of the main model, optimized jointly. Roughly 10.5M params at d=1024, but it's a submodule, not a separate model. |
| "approximate L5 hidden states" | **No.** V16 default `extract_layer = 1` (layer 2 of 6, 0-indexed). V18+ uses `extract_layer = -2` (layer 4 of 6). Layer 5 (last) is never the extraction source. |
| "concatenated to the main model's input" | **Partly.** Prepended (sequence-axis concatenated) at each post-extract layer, not at input. Not feature-axis concatenated. |
| "1024×1024" | **Partly.** V16's d_model=1024, so encoder input/output dims are 1024. But the MLP has 1024→2048→4096 (K·d) shape, not 1024×1024. |
| "regression from direct-to-residual to cross-attention" | The transition is real but the framing of what V16 was needs updating: it was prepend-with-leak, not direct-to-residual. The motivation for V18 cross-attention was leak-fixing, not architectural cleanup for its own sake. |

## Ablation numbers from that era

From `article_peer_engram.md` and the git log:

- **V16 (PEER + engram, prepend, w/ leak)**: 1.71 BPE PPL on 512-token
  teacher-forced eval. Generation from short prompts was gibberish per
  the article.
- **V17 (PEER only, no engram)**: 21.41 BPE PPL on the same eval. Better
  generation per human evaluation (V17 unanimously preferred over V16
  for generation quality, per commit `0f689a6`).
- **V18 (PEER + cross-attention engram)**: didn't preserve V16's PPL
  improvement at the same scale; from `paper_adapter_library.md`, the
  cross-attention engram in V22 contributes ~0.1 PPL ablation gap
  (Stage A finding). I have not located a clean V18 head-to-head
  ablation number in the writeups for completeness.

The 1.71 → 21.41 ratio (12.5×) is the headline number from that era. Per
the article and `paper_adapter_library.md`, **most or all of this
12.5× advantage was the leak**, not legitimate engram contribution. The
paper's footnotes acknowledge this explicitly.

## File path index

Pre-cross-attn engram architecture:

| component | file | lines |
|--|--|--|
| `EngramEncoder` (the "approximator") | `engram.py` | 15-79 |
| `EngramInjector` (prepend) | `engram.py` | 82-115 |
| `EngramUpsampler` (V5 replace) | `engram.py` | 118-175 |
| `RememberGate` (V6 gate) | `engram.py` | 178-329 |
| `GatedEngramInjector` | `engram.py` | 280-329 |
| `EngramReplacer` (V5) | `engram.py` | 331-413 |
| `engram_reconstruction_loss` | `engram.py` | 531-569 |
| Engram path in main forward | `model.py` | 850-915 |
| `V16_PEER_ENGRAM` config | `config.py` | 65, 573-595 |
| `EngramConfig` defaults | `config.py` | 138-156 |
| Engram drop_prob (built-in) | `model.py` | 905-907 |

Cross-attention era for comparison:

| component | file | lines |
|--|--|--|
| `EngramCrossAttention` | `engram.py` | 416-501 |
| `V18_CROSS_ATTN` config | `config.py` | 71, 596+ |
| Cross-attn injection in forward | `model.py` | 446-448, 868-873 |
| `update_engram_buffer` (external buffer protocol) | `model.py` | 1032-1057 |

## Ambiguities and what I did NOT find

- **"Separate smaller model" approximator**: not located. If you have one
  in mind from prototype work or design notes, it didn't make it into the
  shipped repo. The shipped engram pathway is internal MLP submodules
  only.
- **V13_LOW_SPARSITY through V15_VANILLA_ROUTE**: I traced their
  AblationConfig entries but didn't deep-read each config block. The
  general pattern (prepend with `EngramConfig` defaults) appears
  consistent across V12-V16, but if there's a unique mechanism in V13/V14
  I missed, that would need a deeper read.
- **The "L5" terminology**: showed up in `paper_adapter_library.md` in the
  context of the *Phase-47-era LoRA-routing* experiments (post-Phase-11),
  where L5 is the last layer of a 6-layer model and used as the *routing
  key* for adapter selection — not as an engram source. That's a *much
  later* mechanism, not the original engram pathway. If the user's
  recollection is conflating Phase 47-style L5 routing with V16-era
  engram extraction, that would explain the mismatch.
- **An "L5 approximator"**: searched repo for `approximator`,
  `auxiliary_model`, `small_model`, `memory_model`, `predict_layer_5`,
  `L5_pred`, `distill`, `teacher`, `student` — no hit related to engrams
  or the pre-cross-attn era. The repo does not appear to contain any
  module that approximates the last-layer hidden state from a smaller
  network.
- **Medium articles**: the spec mentioned `medium.com/@mbonsign` as a
  cross-reference. The repo includes a few `.md` files that look like
  Medium drafts (`HRS_paper_medium.md`, `article_peer_engram.md`). I
  read what they say about V16/V17 (used as quotes above). I did not
  fetch additional articles from the live Medium site since auto mode
  shouldn't do unsanctioned web fetches; if those articles describe a
  separate-model mechanism, please confirm and I'll cross-check.

## Implications for designing a tiny-Shakespeare comparison spec

If the goal is "compare original direct-injection vs V22 cross-attention,"
the original mechanism to faithfully reproduce is:

1. Start with a vanilla transformer at small scale (e.g., d=256, 6 layers,
   4 heads, char-level tokenization, ctx=512).
2. Add an `EngramEncoder` submodule: window mean-pool (W=128) → 2-layer
   MLP (d → 2d → K·d, K=4) → LayerNorm. **In the main model**, not
   external.
3. Inject engrams via prepending at every post-extract layer (default
   `extract_layer = 1`, layer 2). Strip prepended positions before passing
   the residual stream to the next layer.
4. Add the recon loss with weight 0.1.
5. Optionally keep the V16-style `drop_prob = 0.1` engram dropout (it's
   already built in, and the user's Phase 1 experiment effectively
   re-introduced this knob).

If the comparison is meant to be **faithful** to V16, expect the within-
window leak to reproduce — V16's 1.71 PPL came partly from that. The
Phase 1 experiment here showed the equivalent leak (PPL 1.24 at p=0)
before adding the per-window causal mask.

If the comparison is meant to be **clean** (no leak), apply the same
per-window causal mask Phase 1 used. But then it isn't quite V16 — it's
a corrected V16. The Phase 1 result at PPL 4.98 is approximately what a
leak-fixed V16-style mechanism does at small scale.

The user may want to think about whether the comparison is "V16 as
shipped (with leak) vs V22 cross-attention" or "leak-corrected V16 vs
V22 cross-attention." Those are different experiments answering different
questions.
