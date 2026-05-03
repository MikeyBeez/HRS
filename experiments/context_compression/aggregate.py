"""Plot context-compression results and write RESULT.md."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/context_compression"


def main():
    run = json.loads((EXP / "results/run.json").read_text())
    abl = json.loads((EXP / "results/ablations.json").read_text())

    # Build a unified scheme list with the tiered baseline + ablations
    tiered = {
        "label": "tiered (5+50+200)",
        "n_eval": run["config"]["n_eval"],
        "mean_ce_base": run["summary"]["mean_ce_base"],
        "mean_ce_comp": run["summary"]["mean_ce_comp"],
        "mean_ce_gap":  run["summary"]["mean_ce_gap"],
        "mean_kl":      run["summary"]["mean_kl"],
        "top1_match_rate": run["summary"]["top1_match_rate"],
        "tokens": 255,
    }
    schemes = []
    schemes.append(tiered)
    for a in abl["schemes"]:
        # Determine token count from label
        toks = {"recent_200_only": 200, "recent_255_only": 255,
                "recent_400_only": 400, "full_uniform_255": 255}
        a["tokens"] = toks.get(a["label"], 0)
        schemes.append(a)
    # Sort by tokens for plotting
    schemes_by_tokens = sorted(schemes, key=lambda s: s["tokens"])

    # ---------- Plot ----------
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))

    labels = [s["label"] for s in schemes]
    top1 = [s["top1_match_rate"] for s in schemes]
    ce_gap = [s["mean_ce_gap"] for s in schemes]
    kl = [s["mean_kl"] for s in schemes]
    tokens = [s["tokens"] for s in schemes]
    colors = ["tab:blue" if "tiered" in l else
              ("tab:red" if "uniform" in l else "tab:gray")
              for l in labels]

    ax = axs[0, 0]
    bars = ax.bar(labels, top1, color=colors)
    ax.set_ylabel("top-1 agreement vs baseline (full 1000-token context)")
    ax.set_title("Top-1 prediction agreement")
    ax.set_ylim(0, 1.05)
    for b, v in zip(bars, top1):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=30); plt.setp(ax.get_xticklabels(),
                                                      ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axs[0, 1]
    bars = ax.bar(labels, ce_gap, color=colors)
    ax.set_ylabel("mean CE gap (compressed - baseline) [nats]")
    ax.set_title("Cross-entropy degradation")
    for b, v in zip(bars, ce_gap):
        ax.text(b.get_x() + b.get_width()/2, max(v, 0) + 0.05, f"{v:+.2f}",
                ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=30); plt.setp(ax.get_xticklabels(),
                                                      ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axs[1, 0]
    bars = ax.bar(labels, kl, color=colors)
    ax.set_ylabel("mean KL(baseline || compressed) [nats]")
    ax.set_title("KL divergence between predictions")
    for b, v in zip(bars, kl):
        ax.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=30); plt.setp(ax.get_xticklabels(),
                                                      ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # KL distribution histogram for tiered scheme
    ax = axs[1, 1]
    per_passage = run["per_passage"]
    kls = [r["kl"] for r in per_passage]
    ax.hist(kls, bins=30, color="tab:blue", alpha=0.7)
    ax.set_xlabel("KL(base || comp) [nats]")
    ax.set_ylabel("# of passages")
    ax.set_title(f"Per-passage KL distribution (tiered scheme, N={len(kls)})")
    ax.axvline(np.mean(kls), color="black", linestyle="--",
                label=f"mean={np.mean(kls):.2f}")
    ax.axvline(np.median(kls), color="red", linestyle="--",
                label=f"median={np.median(kls):.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Context window compression: GPT-2 small, frozen base, "
                  "WikiText-103, 100 eval passages")
    fig.tight_layout()
    out_png = EXP / "results/context_compression.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"saved {out_png}")

    # ---------- RESULT.md ----------
    out = []
    out.append("# Context Window Compression Experiment\n")
    out.append("**Question:** does DeepSeek's temporal compression principle "
                "(recent tokens preserved at high resolution, older tokens "
                "compressed) generalize from KV cache to *raw input* "
                "context, via learned linear projections over token "
                "embeddings?")
    out.append("")
    out.append("**Verdict (mixed, mostly negative):** the temporal "
                "asymmetry principle is **strongly confirmed** — uniform "
                "compression of all 1000 tokens to 255 tokens is "
                "catastrophic (top-1 agreement = 0%, KL = 4.6 nats), "
                "while the tiered scheme that preserves the last 200 "
                "tokens at full resolution achieves 79% top-1 agreement "
                "and a CE gap of +0.21 nats. The architecture's claim "
                "that recent context matters more than old is well-"
                "supported.")
    out.append("")
    out.append("**However, the learned compression of older content "
                "doesn't add value over simply dropping it.** Trivially "
                "truncating to the last 255 tokens (no projection at "
                "all) achieves **81% top-1 agreement** and a smaller "
                "KL of 0.14 nats — *better* than the tiered learned "
                "compression scheme at the same context length. The "
                "older-content compression W_4 (50 tokens) and W_128 "
                "(5 tokens) take up 55 tokens of context budget that "
                "would yield better predictions if those slots held "
                "more recent tokens instead.")
    out.append("")

    out.append("## Setup\n")
    out.append("- **Base model:** GPT-2 small (124M params, 12 layers, "
                "768d, learned absolute position embeddings, 1024-token "
                "context). FROZEN throughout — only the projection "
                "matrices are trained.")
    out.append("- **Source text:** WikiText-103 train split, tokenized "
                "into 1100 non-overlapping 1001-token passages "
                "(1000 for context + 1 target token).")
    out.append("- **Train / eval split:** 1000 train + 100 held-out eval.")
    out.append("- **Compression scheme (tiered):**")
    out.append("  - Older 600 tokens (positions 0-599 of context) → "
                "W_128 (5×600) → 5 tokens. Compression ratio 120:1.")
    out.append("  - Middle 200 tokens (positions 600-799) → W_4 (50×200) "
                "→ 50 tokens. Compression ratio 4:1.")
    out.append("  - Recent 200 tokens (positions 800-999) preserved "
                "at 1:1.")
    out.append("  - Compressed total: 5 + 50 + 200 = 255 tokens "
                "(4× compression of the original 1000).")
    out.append("- **Projection initialization:** average pooling — each "
                "compressed token starts as the mean of its contiguous "
                "bucket of input embeddings.")
    out.append("- **Training:** Adam, lr 1e-3, batch 4, 2000 steps, CE "
                "loss on the actual next token at position 1000. Total "
                "training wall: 61s. ~13k trainable parameters total "
                "(W_4=10000 + W_128=3000).")
    out.append("- **Compressed embeddings injected via** "
                "`model(inputs_embeds=...)`. GPT-2 then adds its standard "
                "wpe[0..254] to them — i.e., compressed older tokens "
                "occupy positions 0..4 of the model's positional space, "
                "irrespective of where their original tokens lived.")
    out.append("")

    out.append("## Tiered scheme results (main run)\n")
    s = run["summary"]
    out.append("| metric | value |")
    out.append("|---|---:|")
    out.append(f"| Mean CE — baseline (full 1000-token context) | "
                f"{s['mean_ce_base']:.3f} |")
    out.append(f"| Mean CE — compressed (255-token context)     | "
                f"{s['mean_ce_comp']:.3f} |")
    out.append(f"| Mean CE gap (compressed − baseline)          | "
                f"**{s['mean_ce_gap']:+.3f}** |")
    out.append(f"| Mean KL(baseline ‖ compressed)               | "
                f"**{s['mean_kl']:.3f}** |")
    out.append(f"| Median / p25 / p75 / p90 KL                  | "
                f"{s['median_kl']:.2f} / {s['kl_p25']:.2f} / "
                f"{s['kl_p75']:.2f} / {s['kl_p90']:.2f} |")
    out.append(f"| Top-1 prediction agreement                   | "
                f"**{s['top1_match_rate']*100:.0f}%** (79/100) |")
    out.append("")

    out.append("## Ablations\n")
    out.append("All schemes use the same 100 held-out passages. Baseline "
                "= full 1000-token context.")
    out.append("")
    out.append("| scheme | context tokens | trained projection? | "
                "CE gap | KL | top-1 |")
    out.append("|---|---:|---|---:|---:|---:|")
    rows = [
        ("**tiered (5+50+200)**", 255, "yes (W_4 + W_128)",
         tiered["mean_ce_gap"], tiered["mean_kl"],
         tiered["top1_match_rate"]),
    ]
    for a in abl["schemes"]:
        if a["label"] == "recent_200_only":
            label = "recent 200 only (truncate)"; trained = "no"
        elif a["label"] == "recent_255_only":
            label = "recent 255 only (truncate)"; trained = "no"
        elif a["label"] == "recent_400_only":
            label = "recent 400 only (truncate)"; trained = "no"
        elif a["label"] == "full_uniform_255":
            label = "uniform 1000→255 (single linear)"; trained = "yes (W_uniform)"
        else:
            label = a["label"]; trained = "?"
        rows.append((label, a["tokens"], trained, a["mean_ce_gap"],
                      a["mean_kl"], a["top1_match_rate"]))
    for r in rows:
        out.append(f"| {r[0]} | {r[1]} | {r[2]} | "
                    f"{r[3]:+.3f} | {r[4]:.3f} | {r[5]*100:.0f}% |")
    out.append("")

    out.append("![curves](context_compression.png)\n")

    out.append("## Reading the result\n")
    out.append("**The temporal asymmetry principle is strongly confirmed.** "
                "Uniform compression of all 1000 tokens to 255 (a single "
                "learned linear projection) is catastrophic: top-1 "
                "agreement is 0%, KL is 4.6 nats, the model produces "
                "completely different predictions. Recent tokens at "
                "full resolution are essential.")
    out.append("")
    out.append("**The learned compression of older tokens, however, does "
                "not beat trivial truncation.** Compare four schemes at "
                "or below the 255-token budget:")
    out.append("")
    out.append("| scheme | tokens | top-1 |")
    out.append("|---|---:|---:|")
    out.append(f"| recent 200 only (truncate) | 200 | 74% |")
    out.append(f"| **tiered (5+50+200)** | 255 | 79% |")
    out.append(f"| recent 255 only (truncate) | 255 | **81%** |")
    out.append(f"| recent 400 only (truncate) | 400 | 84% |")
    out.append("")
    out.append("Truncating to the last 255 tokens — no projection, no "
                "training — achieves 81% top-1 vs the tiered scheme's "
                "79%. The 55 tokens of context budget that the tiered "
                "scheme spends on compressed-older information would be "
                "more useful as 55 additional recent tokens.")
    out.append("")
    out.append("**The implication:** at this scale (GPT-2 small, "
                "WikiText-103, 1000-token context, next-token "
                "prediction), the model's prediction at position 1000 "
                "depends almost entirely on the most recent ~200-400 "
                "tokens. The projection W_128 (compressing 600 older "
                "tokens to 5) and W_4 (compressing 200 middle tokens "
                "to 50) extract some signal — the tiered scheme isn't "
                "destroyed — but the signal extracted is less useful "
                "per token than just keeping more recent tokens.")
    out.append("")

    out.append("## Why might the learned compression underperform?\n")
    out.append("Several possibilities, none ruled out:")
    out.append("")
    out.append("1. **Linear projection is too restrictive.** A 5×600 "
                "linear over embeddings can only compute weighted means "
                "of the 600 input embeddings. A small MLP per output "
                "position might extract more useful signal.")
    out.append("2. **Position embedding mismatch.** GPT-2 uses learned "
                "absolute position embeddings; the compressed "
                "older-tokens-at-positions-0..4 are interpreted as "
                "*beginning of context* rather than as *summaries of "
                "earlier content*. The W projection has no way to "
                "signal \"this is a summary of position 100-220\".")
    out.append("3. **Single-target-position training is weak supervision.** "
                "Training optimizes prediction at exactly position "
                "1000. The projection learns to produce embeddings that "
                "help the model predict the *next* token, which mostly "
                "depends on the recent tokens (which are already "
                "preserved at 1:1). Token-level losses across many "
                "positions might pressure the projections to encode "
                "longer-range information.")
    out.append("4. **GPT-2 small's effective context is short.** Even "
                "at the full 1000-token baseline, the model's "
                "prediction probably draws mostly from the last 100-200 "
                "tokens; the older tokens have weak influence. So "
                "compressing them losslessly wouldn't help much; "
                "compressing them lossily certainly won't.")
    out.append("")
    out.append("Possibilities 1 and 2 are testable as follow-ups; 3 and "
                "4 are more fundamental and would require architectural "
                "or training changes.")
    out.append("")

    out.append("## Wall-clock\n")
    out.append("| Stage | Wall |")
    out.append("|---|---:|")
    out.append(f"| Tokenize WT-103 + build 1100 passages | "
                f"~30 s |")
    out.append(f"| Train tiered W_4 + W_128 (2000 steps × batch 4) | "
                f"{run['training_wall_s']:.0f} s |")
    out.append(f"| Eval tiered (100 passages × 2 forwards) | "
                f"{run['eval_wall_s']:.0f} s |")
    out.append(f"| Train + eval all ablations (4 schemes) | "
                f"{abl['wall_total_s']:.0f} s |")
    out.append(f"| **Total** | **~3 min** |")
    out.append("")

    out.append("## Caveats / what was *not* tested\n")
    out.append("1. **Single base model (GPT-2 small).** A larger model "
                "with longer effective attention range might benefit "
                "more from compressed older content. GPT-2 small's "
                "effective context for next-token prediction is short.")
    out.append("2. **Embedding-level compression only.** The spec "
                "specified projecting token embeddings (the input layer "
                "of the model). DeepSeek's KV-cache compression operates "
                "on intermediate representations after some attention "
                "layers — semantically richer. A fairer test of the "
                "principle would compress at, say, layer 4's hidden "
                "states, not at the embedding layer.")
    out.append("3. **Linear projections only.** A small MLP, attention "
                "over the older tokens with a few learned queries, or "
                "any non-linear pooling would be a different test.")
    out.append("4. **Single ratio (4:1 / 128:1).** The spec's optional "
                "ratios (256:1, 512:1, longer contexts) were not run. "
                "The \"recent X tokens only\" ablation gives a cleaner "
                "signal than these would have.")
    out.append("5. **Single training objective.** Next-token CE at "
                "position 1000 only. Multi-position losses might "
                "pressure the projections to encode longer-range "
                "signal.")
    out.append("")

    out.append("## Implications\n")
    out.append("**For the architecture's case against long-context arms "
                "races:** the temporal asymmetry principle (recent > "
                "old) is empirically supported in absolute terms. Any "
                "system that compresses uniformly will fail. Systems "
                "that preserve recent context at high fidelity and "
                "compress older content can work — but the simplest "
                "such system is *truncation*, and learned linear "
                "compression doesn't beat it at this scale.")
    out.append("")
    out.append("**For deployment:** keep the most recent N tokens at "
                "full resolution; truncate the rest. Don't bother with "
                "a learned linear projection over older context unless "
                "you have evidence (different model, different "
                "objective, different compression mechanism) that it "
                "beats truncation. This is the conservative, defensible "
                "version of the temporal compression idea at this "
                "substrate.")
    out.append("")
    out.append("**For follow-up work:** the most informative next "
                "experiment would test whether the negative result "
                "is substrate-specific. Run the same protocol on a "
                "larger base (Llama-3-8B, Mistral-7B) or with hidden-"
                "state-level compression (compress at L4 instead of "
                "the embedding layer). Either change might unlock the "
                "compressed-older-content advantage that DeepSeek "
                "observed for KV cache.")

    out_md = EXP / "results/RESULT.md"
    out_md.write_text("\n".join(out))
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()
