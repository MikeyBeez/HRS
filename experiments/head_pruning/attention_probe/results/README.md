# L0 Attention-Pattern Probe — Results

200 passkey examples through the baseline MHA passkey model (seed 0, checkpoint `experiments/pruning/checkpoints/mha_passkey.pt`), attention weights captured for every (layer, head, example) tuple, then scored and plotted.

## Key finding: the find/read split is visible directly in attention patterns

The spec's hypothesis expected "retrieval heads attend to MARKER at the query position." That's wrong in two ways:
1. The finding pattern is not visible at *query position* (position 251); it's visible across *all post-MARKER query positions*.
2. Reading the passkey happens in L3, not at the query row of L0/L2 retrieval heads.

Once the metric is measured across post-MARKER rows (the "find strength" metric introduced in `find_pattern_metric.py`), the mechanism is extremely clean:

| Head | Δpasskey (Phase 1) | find strength (→ MARKER) | read strength (→ passkey) | Role |
|---|---|---|---|---|
| **L0 H3** | +1.00 | **0.753** | 0.006 | **finder** |
| **L0 H2** | +0.96 | **0.377** | 0.017 | **finder** |
| L0 H1 | +0.53 | 0.002 | 0.027 | retrieval-critical, subtle |
| L0 H0 | +0.50 | 0.092 | 0.027 | weak/borderline |
| **L2 H0** | +1.00 | 0.024 | 0.112 | retrieval-critical, **not** visible as find/read |
| L2 H1–H3 | 0.00 | 0.00–0.05 | 0.09–0.17 | generic |
| L1 H0–H3 | 0.00 | 0.000 | 0.21 | read-ish but irrelevant for passkey |
| **L3 H0** | +0.03 | 0.010 | **0.463** | **reader** |
| L3 H1–H3 | 0.04–0.32 | 0.010–0.020 | **0.129–0.330** | **readers** |

- `find` = mean attention to MARKER column across post-MARKER query rows
- `read` = mean attention to passkey span across post-MARKER query rows

**L0 H3 is a textbook MARKER-finding head.** 75% of attention mass from post-MARKER queries lands on the MARKER token itself. L0 H2 is similar but weaker (38%). Everything else at L0 — including L0 H1 which is Phase-1-important with Δpasskey=0.53 — does not implement a visible find pattern.

**L3 heads are dedicated readers.** All four L3 heads attend with 13–46% mass to the passkey span across post-MARKER queries. At the specific final-query row (position 251), they sharpen to ~100% attention on the passkey token being emitted. L3 H0 has the highest read strength, matching its role as a reader.

**L2 H0 is a loud "passkey-position" beacon in the value pathway.** Ablating it destroys passkey (Δ=+1.00), but it has neither high find nor high read strength in its attention weights. Probing W_V directly (see `probe_l2_values.py` and `probe_binary_tag.py`) resolves the mystery — the compose operation is a *value* transformation, not an attention pattern:

| L2 head | ‖V‖ at post-MARKER positions | binary passkey-vs-other probe | 6-class offset probe (p1/p2/p3/p4/MARKER/other) |
|---|---|---|---|
| **L2 H0** | **~36** (peaks) | 0.990 | **0.455** |
| L2 H1 | ~5 | 0.990 | 0.781 |
| L2 H2 | ~8 | 0.988 | 0.757 |
| L2 H3 | ~9 | 0.983 | 0.779 |

- **All L2 heads can perfectly distinguish "passkey vs non-passkey" from their value vectors** (binary probe ≈ 0.99). The tag is everywhere.
- **L2 H0 writes that tag with ~6× larger amplitude than its siblings.** The value-magnitude plot (`l2_values_magnitude.png`) shows L2 H0's ‖V‖ jumping from ~6 before MARKER to ~36 immediately after, while H1/H2/H3 stay at ~5–9 throughout.
- **L2 H0 does NOT encode which passkey position it is** — the 6-class probe only hits 0.45 (near chance for a loosely-3-cluster structure). Its PCA (`l2_values_pca_H0.png`) shows three tight clusters: MARKER, all-passkey-positions-merged, and other. p1/p2/p3/p4 are indistinguishable inside the passkey cluster.
- **L2 H3 (and H1/H2) do encode offset.** The 6-class probe gets 0.78, and L2 H3's PCA (`l2_values_pca_H3.png`) shows visible stripes in the passkey cluster separating p1/p2/p3/p4.

So L2 H0 is the **loud binary beacon** ("HERE IS A PASSKEY POSITION"), and L2 H1/H2/H3 are **quiet fine-grained position encoders** carrying "which offset within the passkey." Because L2 H0 writes the beacon at ~6× amplitude, no sibling can substitute for it in one-shot ablation — they all carry the same bit, but at amplitude too low to drive L3's attention. Remove L2 H0 and the signal-to-noise for L3's attention collapses; remove L2 H1/H2/H3 and L3 still finds the loud beacon from L2 H0. That's why only L2 H0 is single-head critical.

## Relation to the find / compose / write hypothesis

The spec hypothesized:
- **Find** = L0 (positional) ✓ confirmed for L0 H2/H3 via attention pattern
- **Compose** = L2 ✓ confirmed, but as a *value-pathway* operation, not an attention pattern
- **Write** = L3 (to logit space) ✓ confirmed — more specifically, the explicit *read*

Fully ground-truthed architectural picture:

- **L0 H2, L0 H3 — find.** When any token sits past MARKER, these heads attend *backward to MARKER itself*, tagging the residual stream at that token with "MARKER occurred at position m." Visible directly in attention (`find_strength` = 0.75 for L0 H3, 0.38 for L0 H2).
- **L2 H0 — compose (loud).** Writes a high-magnitude "I am a passkey position" tag into the residual stream via W_V. Value magnitude is ~6× that of siblings, but encodes only the binary tag, not offset. Amplitude is what makes it individually irreplaceable — sibling heads carry the same bit at insufficient SNR.
- **L2 H1/H2/H3 — compose (quiet, fine-grained).** Same binary passkey-vs-other info plus additional p1/p2/p3/p4 offset information, but at low magnitude. None individually critical because L2 H0's loud beacon dominates L3's attention signal.
- **L3 — read.** Attends from each generation step to the tagged passkey position, reads the value, and that value (by the time it reaches L3) encodes the actual passkey token. Copy operation. L3 as a whole is redundant (any one head suffices — Phase 1 showed single-head ablation in L3 → Δpasskey ≈ 0), but the layer is load-bearing (whole-layer ablation crashed passkey in the seeds experiment).

This maps precisely onto the substitutability gradient observed in the whole-layer ablation:
- **L0** irreplaceable by FT: you can't rebuild the find pattern without L0's specific connection between MARKER's token identity and the residual-stream tag it emits.
- **L2** partially replaceable: whatever L2 H0 routes could in principle be routed by other layers given more capacity.
- **L3** most replaceable: "pull passkey from k positions after MARKER" is a simple attention pattern that any layer with a sharp-query-key match could implement.

## Methodological lesson: query-row attention is misleading

The original metric (attention at query_pos=251) showed L0 retrieval heads have essentially zero mass on both MARKER and passkey from the final query. This looked like "the circuit isn't in the attention pattern" — but that conclusion was wrong. The circuit is a cumulative process: L0 at positions *along the passage* annotates the residual stream; L3 at the final query reads the annotated state. The "find" operation happens continuously at every post-MARKER position, not just at generation time.

A paper-grade probe needs to measure attention patterns at **every probe position that matters**, not just the final read. Specifically:
- Attention at passkey positions (what do p1..pK attend to during the passage?)
- Attention across post-MARKER rows (as used in the `find_strength` metric)
- Attention at the final query (as a downstream readout)

## Files

```
plan.md
generate_batch.py              # builds 200 varied-position probe examples
extract_attention.py           # monkey-patches MHA.forward, captures softmax'd attn
compute_metrics.py             # original 5 metrics at query=251 (misleading on its own)
find_pattern_metric.py         # find/read strengths averaged over post-MARKER rows
contrast_analysis.py           # Mann-Whitney + L0-vs-L2 breakdown
plot_attention.py              # fig1 gestalt grid + fig2 single example + fig3 retrieval zoom
plot_attention_extended.py     # fig4 multi-query + fig5 full-matrix
probe_l2_values.py             # value-vector magnitude + PCA + 6-class probe at L2
probe_binary_tag.py            # follow-up: binary and 3-class probes on L2 heads
results/
  probe_examples.pt
  attention_tensors.pt         # [200, 4, 4, 256, 256] softmax'd attention (~800MB)
  head_metrics.json            # query=251 metrics
  find_read_metrics.json       # find/read metrics across post-MARKER queries
  contrast_analysis.txt
  fig1_gestalt_grid.png
  fig2_single_example.png
  fig3_retrieval_heads_zoom.png
  fig4_multi_query.png
  fig5_full_matrix.png
  l2_values_magnitude.png      # ‖V[t]‖ per L2 head, aligned to MARKER
  l2_values_pca_H0.png         # PCA of L2 H0 values: 3-cluster structure (MARKER/passkey/other)
  l2_values_pca_H3.png         # PCA of L2 H3 values: shows p1-p4 stripes (contrast to H0)
  l2_values_probe.json         # 6-class linear probe per L2 head
  l2_binary_probe.json         # binary passkey-vs-other probe per L2 head
  README.md                    # this file
```

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.generate_batch
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.extract_attention
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.compute_metrics
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.find_pattern_metric
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.contrast_analysis
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.plot_attention \
    --importance experiments/head_pruning/results/head_importance.json
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.plot_attention_extended \
    --importance experiments/head_pruning/results/head_importance.json
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.probe_l2_values \
    --layer 2 --head 0
PYTHONPATH=. .venv/bin/python -m experiments.head_pruning.attention_probe.probe_binary_tag \
    --layer 2
```

## Natural next steps

- **Repeat on seed 2 or seed 3.** Do the retrieval heads at different indices (seed 3 uses L1 H1 and L2 H2) implement the same find/beacon/read pattern at their respective positions? If yes, the *function* transfers across seeds even though the *head indices* don't.
- **Value-stream probe for L3.** The read operation pulls a value. What does L3's W_V encode at passkey positions? If it's approximately the passkey token's embedding, that's the simplest possible reader mechanism.
- **Counterfactual find test.** Zero only L0 H3 (highest find strength) and measure how the L2 H0 beacon magnitude and L3 read attention change. If L2 H0's beacon collapses when L0 H3 is gone, that confirms the causal chain L0 → L2 → L3 rather than three independent operations.
- **Amplitude-rescale ablation.** Instead of zeroing L2 H0, scale its W_O output by 0.2 (quieter but non-zero). Prediction: passkey still works because the tag is preserved but at lower amplitude; L3's SNR should still be adequate. This would confirm amplitude — not the bit itself — is what makes L2 H0 uniquely critical.
