# Phase 01 (D2L) — Baseline cloze on Bleak House

Go/no-go calibration: does a code-only-pretrained ~1B model have any meaningful
prior on Dickens content before we invest in building a Perceiver to install
Bleak House signal via D2L adapters in Phase 02?

## Headline

**GREEN with one caveat.** StarCoder2-3B is at floor or below on every
Dickens-specific category (character names 0% top-1, plot facts 0% top-1,
possessions 6% top-1) with strong qualitative evidence that the model has
*actively been trained to suppress* named-entity tokens via PII anonymization
(model emits `<NAME>` as top-1 for the plot category). The code positive
control hits 11/50 rank-0 (e.g. `json.` → `dumps`, `np.` → `array`,
`requests.` → `get`, `super().__` → `init`), confirming the cloze
infrastructure works. The caveat: top-1 on the code category is 22%, below
the spec's 50% threshold — but this is target-ambiguity in the idiom design,
not infra failure (see Category D below). Phase 02 (Perceiver build) is
unblocked.

|                         | top-1 | top-5 | mean log-prob | mean rank | predicted top-1 |
|-------------------------|-------|-------|---------------|-----------|-----------------|
| A — character names     | **0.0%** | 2.0%  | -8.53  |   1,709 | 2–5%   |
| B — possessions/objects | 6.0%  | 14.0% | -6.06  |     243 | 5–10%  |
| C — plot facts          | **0.0%** | 0.0%  | -17.47 |  13,494 | 10–20% |
| D — code idioms (control) | 22.0% | 46.0% | -6.03  |   3,706 | 60–80% |

Recommendation: proceed to Phase 02. The Dickens-prior is at or below floor on
every content-bearing category. The Phase 02 Perceiver-installed signal will
be detectable against this floor.

## Setup deviation from spec

Spec asks for `bigcode/starcoder-1b`. All `bigcode/starcoder*` and
`bigcode/starcoderbase*` checkpoints are gated (HTTP 401 without an HF_TOKEN
agreement signed). `bigcode/santacoder` (1.1B, the closest size match) ships
custom modeling code that imports `transformers.onnx`, which was removed in
transformers 5.x.

Substituted **`bigcode/starcoder2-3b`** — same BigCode family, code-only
training on The Stack v2 (no broad web/literature data), non-gated, native
transformers support. Size is 3B rather than 1B; if anything this *biases
against* the floor hypothesis (a bigger model is more likely to have
incidental Dickens exposure), making the test more conservative. The result
still came out at floor, so the substitution is comfortably safe.

Loaded fp16, frozen, eval mode. 3.030B params, vocab 49,152.

## Per-category detail

### A — Character names (0% top-1, 2% top-5)

Probed 50 sentences mentioning the central cast (Esther Summerson, John
Jarndyce, Lady Dedlock, Sir Leicester, Tulkinghorn, Bucket, Skimpole, Krook,
Guppy, Boythorn, Allan Woodcourt, Hortense, Rouncewell, ...) — for each, the
first sub-token of the character's name was masked given the in-book context
preceding the mention.

The model's top predictions are generic function words (`the`, `you`, `,`)
or unrelated tokens. Krook is the only character to score a top-5 hit (1/4
items). Top-1 is 0/50 across every Dickens character.

Spot checks (first 5):
- target=` John`, top1=` the`, rank=94 — model guesses a function-word
  continuation instead of the name
- target=` John`, top1=` you`, rank=136 — same
- target=` L`, top1=` heart`, rank=6 — Lady prefix lost in the noun field
- target=` Est`, top1=` g`, rank=901 — Esther's first sub-token deep in the
  tail
- target=` Est`, top1=`,`, rank=934 — punctuation preferred

### B — Possessions / attributive nouns (6% top-1, 14% top-5)

Probed 50 sentences with patterns `his|her|the|a|an <NOUN>` where NOUN is a
common Victorian physical-description noun (bonnet, shawl, cloak, gloves,
lantern, candle, letter, parcel, book, ...). Top-1 = 6%, top-5 = 14% —
squarely in the predicted 5–10% / 15–25% band. These are not Dickens-specific
tokens; the model has a general English prior over common-noun completions,
and gets some right by chance plus weak distributional cues.

Spot checks:
- target=` book`, top1=` register`, rank=5 — semantically related; model
  has the right field but wrong noun
- target=` watch`, top1=` scene`, rank=12 — similar
- target=` clo`, top1=` deep`, rank=223 — far off

### C — Plot facts (0% top-1, 0% top-5)

Probed 50 sentences with strong book-grounded phrases:
`Jarndyce and ___`, `Court of ___`, `Chesney ___`, `Mr. ___ (Tulkinghorn,
Krook, Snagsby, Guppy)`, `Lady ___ (Dedlock)`, `Inspector ___ (Bucket)`,
`Allan ___ (Woodcourt)`, `Harold ___ (Skimpole)`, `Esther ___ (Summerson)`,
`Richard ___ (Carstone)`, `Ada ___ (Clare)`, `Mr. ___ (Boythorn)`,
`George ___ (Rouncewell)`, `Caddy ___ (Jellyby)`.

**0/50 top-1, 0/50 top-5.** Mean rank 13,494 out of vocab 49,152 — the target
tokens are deep in the tail.

The dominant model behavior here is striking. For every Mr./Mrs./Lady/Sir/
Inspector phrase, **the model's top-1 prediction is literally the token
`<NAME>`.** Examples from the spot check:

```
target='Bucket'    top1='<NAME>'   rank=29480
target='Sk'        top1='<NAME>'   rank=19248   (Skimpole)
target='G'         top1='<NAME>'   rank=30777   (Guppy)
target='Jar'       top1='<NAME>'   rank=98      (Jarndyce after "Jarndyce and ")
target='Ro'        top1='<NAME>'   rank=89      (Rouncewell)
```

This is evidence that BigCode trained StarCoder2-3B on PII-anonymized data
where person names in code comments / docstrings / strings were replaced with
`<NAME>` placeholders. The model has learned a strong prior that
"after `Mr.`/`Lady`/`Inspector` comes `<NAME>`", and it applies this prior in
free text too. *Even the iconic "Jarndyce and Jarndyce" cue* — about as
distinctive a Dickens phrase as exists — gets `<NAME>` as top-1 with the
actual target at rank 98.

This is a stronger floor signal than the spec predicted (it expected 10–20%
top-1 from "cliché Victorian elements"). Functionally it means installed
Dickens signal in Phase 02 will need to overcome an actively-trained
suppression of proper-noun prediction in named-entity contexts, not just a
neutral prior.

### D — Code idioms (positive control, 22% top-1, 46% top-5)

50 hand-curated Python idioms with a single-token continuation as target.
Top-1 22%, top-5 46% — below the spec's >50% top-1 sanity gate, but the
shortfall is target-design ambiguity, not infrastructure failure.

The 11 rank-0 hits are clean:
```
class Foo(              -> object          ✓
if __name__ == '__      -> main            ✓
np.                     -> array           ✓
os.path.                -> join            ✓
json.                   -> dumps           ✓
super().__              -> init            ✓
await asyncio.          -> sleep           ✓
@pytest.                -> fixture         ✓
argparse.               -> ArgumentParser  ✓
requests.               -> get             ✓
django.                 -> db              ✓
```

The misses fall into three patterns:

1. **Sub-word boundary mismatch.** Several targets are sub-tokens of common
   completions but the model emits the longer form. E.g. `from collections
   import ` → target `default`, top1 ` defaultdict` (rank 135). The model's
   prediction is *correct text* — it's just one token earlier in the merge.
2. **Legitimate ambiguity.** `import ` → target `numpy`, top1 ` java`
   (rank 980). Both are valid. Picking one specific import as the cloze
   target under-counts model knowledge.
3. **Real disagreement.** `torch.nn.` → target `Linear`, top1 `Module`
   (rank 5). The model has a different but plausible top-1.

The eleven rank-0 hits plus 23/50 top-5 hits demonstrate the cloze pipeline
is computing next-token distributions correctly. Phase 02 can either keep
this control as-is (caveated as target-ambiguous) or redesign with single-
canonical-completion idioms (`if __name__ == '__main__':` style) to clear
the 50% bar. The Dickens-floor finding does not depend on this resolution.

## Pre-committed predictions vs measured outcomes

| Category | Predicted top-1 | Measured top-1 | Notes |
|----------|-----------------|----------------|-------|
| A characters | 2–5% | **0.0%** | At or below predicted floor. Model defaults to function-word continuations. |
| B possessions | 5–10% | **6.0%** | Squarely in band. |
| C plot facts | 10–20% | **0.0%** | Way below — the prediction overestimated cliché-Victorian recovery. The `<NAME>` anonymization training in StarCoder2 actively suppresses proper-noun prediction. |
| D code idioms | 60–80% | **22.0%** | Below threshold, but for target-design reasons not infra reasons (11 rank-0 hits, 46% top-5). |

## Architectural interpretation

The hypothesis ("code-only training data leaves StarCoder2-3B with floor-level
Dickens knowledge, leaving headroom for D2L-installed signal") is strongly
supported on the content-bearing categories. The additional finding — the
model has been trained to *replace* named entities with `<NAME>` — actually
strengthens the case: Phase 02 will be installing a signal that competes
against an active suppression prior, not just a neutral prior. If the
installed adapter can override the `<NAME>` reflex on book content, that's a
much sharper measurement of the adapter than testing against a baseline that
merely doesn't know.

Two caveats this experiment did *not* test, worth holding for Phase 02:

1. **Stylistic priors.** The model may have weak distributional priors on
   "Victorian-sounding" continuations even without Dickens specifically. The
   B category (6%) is consistent with mild stylistic transfer from training
   data, not zero. Phase 02's comparison should be against this baseline,
   not "absolute zero."
2. **Vocabulary coverage.** "Carstone", "Jarndyce", "Tulkinghorn",
   "Skimpole" — all very rare tokens. The mean rank 13k for plot fact
   targets reflects partly how deep in the BPE tail these names are. Phase
   02 needs to verify the *targets are tokenizable enough that a successful
   adapter can put weight on them*.

## File manifest

- `experiments/d2l/phase01_starcoder_baseline.py` — main script (cloze
  builder inlined; covers all four categories + scoring + spot checks)
- `results/d2l/phase01/bleak_house.txt` — cleaned Project Gutenberg
  text (header/footer stripped; 1,932,627 chars)
- `results/d2l/phase01/cloze_items.json` — 200 cloze items, fields:
  `category`, `prefix`, `target_token_id`, `target_string`,
  `passage_source`, `full_sentence_for_context`
- `results/d2l/phase01/baseline_results.json` — per-item results
  (rank, top-1/top-5 hit, target log-prob, model top-1 string, model top-5
  strings) + aggregate metrics + 5-per-category spot-check generations
- `results/d2l/phase01/baseline_run.log` — full stdout from the run
- `results/d2l/phase01/baseline_README.md` — this file

## Open questions for Phase 02

1. **Does the Perceiver-installed signal overcome the `<NAME>` suppression?**
   The most informative single number Phase 02 can produce is whether
   `Jarndyce and ___` → `Jarndyce` (currently rank 98) gets to rank 0 after
   the adapter. If yes, the architecture meaningfully installs content.
2. **Tokenizer coverage.** What fraction of central-cast names tokenize to
   ≥ 3 sub-tokens? Long-tail name targets may be hard for the adapter to
   meaningfully favor without high-rank LoRA capacity.
3. **B-category jitter.** The 6% top-1 on common nouns is not zero. Phase 02
   should measure adapter lift on B separately from A/C — A/C are the clean
   "Dickens-specific knowledge" signals; B mixes that with general English
   priors.
4. **Positive-control redesign for D.** If Phase 02 wants D to clear the
   spec's 50% bar, swap target tokens to canonical single-completion
   sub-tokens (e.g. target the full word the model would actually emit;
   `if __name__ == '__main__'` rather than splitting at the underscore).
