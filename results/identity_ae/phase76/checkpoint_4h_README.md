# Phase 76 — Checkpoint 4h

Cumulative wall time (main sweep): **4.00 h**  
Sweep steps completed: **245,020**  
Total adapters: 200 (160 train / 40 held-out)


## Headline

| field   | train acc | held-out acc |
|---------|-----------|---------------|
| name    | 100.0%    | 70.0%        |
| address | 100.0%    | 66.7%        |
| date    | 100.0%    | 81.3%        |
| number  | 100.0%    | 80.0%        |
| entity  | 100.0%    | 55.6%        |
| **overall** | **100.0%** | **71.2%** |


WikiText-2 val PPL: **280.838**  (pristine baseline: 21.539, Δ +1203.9%)  
Detached mean CE on training-record content: **10.975** (absorption check; higher = less absorbed).


## Setup recap

- 200 synthetic patient records, 5 field types each (name, address, date, number, entity)
- Records tokenize to 406–463 tokens (capped at base max_seq_len=512)
- Rank-128 LoRA adapters on blocks 4-5 (8 target modules)
- Heads: width-32 bottleneck, softmax over field vocab ({'name':50,'address':50,'date':50,'number':100,'entity':30})
- Main sweep: 4 WikiText : 1 OOD; one-sided hinge regularizer on detached CE
- Heads update during sweep; adapters frozen


## Files

- `models/phase76_base_4h.pt` — base checkpoint (gitignored)
- `models/phase76_heads.pt` — current heads (overwritten across checkpoints)
- `models/phase76_adapters/adapter_NNNN.pt` — 200 head-CE adapters
- `results/identity_ae/phase76/sweep.json` — full numerical record
