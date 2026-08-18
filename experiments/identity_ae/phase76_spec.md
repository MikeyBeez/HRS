# Phase 76 — Scaled training sweep for adapter-aware base

## The question

Does retrieval reliability improve as training scales? If yes by 4 hours, the architectural mechanism is alive. If no by 16 hours, it isn't.

Phases 73-75 disconfirmed "adapter-aware substrate" at small scale. Twelve adapters and a thousand training steps is not training in any meaningful sense — supervised learning needs thousands of examples and tens of thousands of steps. This phase tests at real training scale.

## Setup

### Content and adapters

- 200 structured records. Each is a synthetic patient-record-style document, 500-1000 tokens, containing names, addresses, dates, numeric IDs, medical entities, and free-form notes.
- Each record gets its own frozen rank-128 LoRA adapter.
- 160 records for training (80%), 40 held out for evaluation (20%).
- 100 synthetic QA pairs per record generated against the structured fields. 20,000 QA pairs total, 16,000 in training.

### Extraction heads

Five small heads, width-32 hidden layer + softmax over field vocabulary:

- name_head: softmax over 50 names
- address_head: softmax over 50 addresses
- date_head: softmax over 50 dates
- number_head: softmax over 100 numeric IDs
- entity_head: softmax over ~30 medical conditions/medications

Width-32 is deliberately small so the base does the organizational work, not the heads.

### Base

Phase 63 softmax baseline (127M params).

## Training procedure

### Bootstrap (the chicken-and-egg fix)

The heads need adapter-loaded forward passes to train. The adapters should be trained against head CE, not next-token CE, so they produce schema-readable representations. Neither can start without the other. Solution: bootstrap.

**Bootstrap step 1 (~30 min):** Pretrain 16 bootstrap adapters using next-token CE on their content (the old way). These are throwaway adapters — they exist only to give the heads something to learn from.

**Bootstrap step 2 (~30 min):** Joint train base + 5 heads on the 16 bootstrap adapters until heads converge on training accuracy. Phase 74's Phase A but smaller — 500 steps should be enough.

**Bootstrap step 3 (~60 min):** Freeze the heads. Pretrain all 200 real adapters from scratch, against the now-fixed head CE instead of next-token CE. Each adapter shapes itself to produce representations the frozen heads can read. ~30 sec per adapter, parallel-batched.

Discard the 16 bootstrap adapters after step 3. Move on to the main sweep with the 200 head-trained adapters and a fresh base.

### Main sweep

- Batch composition: 4 WikiText : 1 OOD.
- Each OOD batch: random training record loaded with its head-trained adapter, random QA pair from that record, all 5 heads evaluated.
- Loss: cross-entropy across the 5 heads plus one-sided hinge regularizer on detached cross-entropy (prevents base from absorbing content directly).
- Heads update during the main sweep (the bootstrap freeze was just for pretraining the adapters).
- Adapters stay frozen throughout the main sweep.

### Checkpoints

Three checkpoints during the main sweep:

- 1 hour (~3,000-5,000 steps)
- 4 hours (~12,000-20,000 steps)
- 16 hours (~50,000-80,000 steps)

At each checkpoint, evaluate:

- Per-field accuracy on training records (sanity check)
- Per-field accuracy on held-out records (the architectural metric)
- WikiText perplexity (general competence)
- Adapter-detached cross-entropy on a sample of training content (absorption check)

Commit and push after each checkpoint. Don't wait for the full run.

## Success criteria

**Alive:** held-out per-field accuracy improves monotonically across the three checkpoints AND reaches >70% by 16 hours.

**Dead:** held-out accuracy doesn't improve between checkpoints, OR it improves but stays below chance-corrected baseline.

**Per-adapter capacity readout:** if average retrieval on training adapters lands at, say, 60% with the main sweep converged, that's evidence the content is too large for rank-128 capacity. Useful even if the main architectural question fails.

WikiText preservation is secondary. If WikiText degrades but held-out retrieval climbs to 90%, the trade is probably worth it.

## Predicted outcomes

Wide intervals. Track record is poor on this arc.

- Held-out >70% by 16 hours: 35%
- Held-out improves but plateaus below 70%: 30%
- Held-out flat or degrades: 25%
- Experiment breaks somewhere: 10%

Composite "architectural mechanism is alive in a meaningful sense": ~65%. Higher than recent track record might suggest because prior failures were at scales the literature suggests are too small to test the question.

## Files

- experiments/identity_ae/phase76_scaled_sweep.py
- data/phase76_records.json (the 200 synthetic records)
- data/phase76_qa_pairs.json (the 20,000 QA pairs)
- models/phase76_adapter_NNN.pt (200 head-trained adapters, gitignored if too large)
- models/phase76_heads.pt (the 5 trained heads)
- models/phase76_base_1h.pt, _4h.pt, _16h.pt (base checkpoints, gitignored)
- results/identity_ae/phase76/checkpoint_1h_README.md
- results/identity_ae/phase76/checkpoint_4h_README.md
- results/identity_ae/phase76/checkpoint_16h_README.md
- results/identity_ae/phase76/final_README.md (full trajectory analysis)
- results/identity_ae/phase76/sweep.json (all numbers)
- results/identity_ae/phase76/trajectory.png

## Directive for Code

Synthesize 200 structured patient-record-style documents using a consistent template (the 5 field types per record, 500-1000 tokens each). Generate 100 QA pairs per record covering each field type. Document the template and the field vocabularies in data/phase76_records.json.

Run the bootstrap sequence:

1. Pretrain 16 bootstrap adapters with next-token CE (~30 min).
2. Joint train base + 5 heads on the 16 bootstrap adapters for 500 steps until heads converge (~30 min).
3. Freeze the heads. Pretrain all 200 real adapters from scratch against head CE (~60 min total in parallel-batched form). Save as models/phase76_adapter_NNN.pt.

Discard bootstrap adapters and bootstrap base. Start the main sweep with a fresh Phase 63 base and the 200 head-trained adapters.

Run the main sweep with checkpoints at 1h, 4h, 16h. At each checkpoint:

- Write checkpoint_Nh_README.md with per-field accuracy on training and held-out records, WikiText PPL, detached CE samples
- Save base checkpoint
- Commit and push immediately

After the 16h checkpoint completes, write final_README.md with the full trajectory: how accuracy evolved across checkpoints, which fields are easier or harder, whether held-out tracks training or diverges. Include trajectory.png plotting all metrics across the three checkpoints.

Append the phase to the experiments index in README.md.

## Runtime budget

- Bootstrap: ~2 hours
- Main sweep to 1h checkpoint: 1 hour
- Main sweep to 4h checkpoint: 3 hours additional
- Main sweep to 16h checkpoint: 12 hours additional

Total: ~18 hours wall time. Start in the morning so the 16h checkpoint completes before the next morning.
