# Phase 76 — Checkpoint 1h

Cumulative wall time (main sweep): **1.00 h**  
Sweep steps completed: **61,292**  
Total adapters: 200 (160 train / 40 held-out)


## Headline

| field   | train acc | held-out acc |
|---------|-----------|---------------|
| name    | 100.0%    | 91.1%        |
| address | 100.0%    | 91.7%        |
| date    | 100.0%    | 100.0%        |
| number  | 100.0%    | 94.7%        |
| entity  | 100.0%    | 88.9%        |
| **overall** | **100.0%** | **93.5%** |


WikiText-2 val PPL: **33.594**  (pristine baseline: 21.539, Δ +56.0%)  
Detached mean CE on training-record content: **8.485** (absorption check; higher = less absorbed).


## Setup recap

- 200 synthetic patient records, 5 field types each (name, address, date, number, entity)
- Records tokenize to 406–463 tokens (capped at base max_seq_len=512)
- Rank-128 LoRA adapters on blocks 4-5 (8 target modules)
- Heads: width-32 bottleneck, softmax over field vocab ({'name':50,'address':50,'date':50,'number':100,'entity':30})
- Main sweep: 4 WikiText : 1 OOD; one-sided hinge regularizer on detached CE
- Heads update during sweep; adapters frozen


## Files

- `models/phase76_base_1h.pt` — base checkpoint (gitignored)
- `models/phase76_heads.pt` — current heads (overwritten across checkpoints)
- `models/phase76_adapters/adapter_NNNN.pt` — 200 head-CE adapters
- `results/identity_ae/phase76/sweep.json` — full numerical record
