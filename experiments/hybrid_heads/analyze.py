"""Aggregate Experiment 2 results: PPL, anchor accuracy, NIAH, retrieval-head
detection across 4 archs × 2 corpora × 3 seeds = 24 checkpoints."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

REPO = Path("/mnt/data/Code/HRS")
EXP = REPO / "experiments/hybrid_heads"

CORPORA = ["ts", "wt103"]
ARCHS = ["baseline", "pure_compressed", "hybrid_1plus7", "hybrid_2plus6"]
SEEDS = [0, 1, 2]


def load_train(corpus, arch, seed):
    p = EXP / f"results/train_{corpus}_{arch}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_niah(corpus, arch, seed):
    p = EXP / f"results/niah_{corpus}_{arch}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_retrieval(corpus, arch, seed):
    p = EXP / f"results/retrieval_heads_{corpus}_{arch}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    print(f"\n{'='*100}\nEXPERIMENT 2 SUMMARY (3 seeds each)\n{'='*100}\n")

    # Per-(corpus, arch) summary
    summary = {}
    for c in CORPORA:
        for a in ARCHS:
            ppls = []; ancs = []; niahs = []
            for s in SEEDS:
                t = load_train(c, a, s)
                if t:
                    ppls.append(t["final_val_ppl"])
                    ancs.append(t["anchor_acc"])
                n = load_niah(c, a, s)
                if n:
                    niahs.append(n["accuracy"])
            summary[(c, a)] = {
                "ppl_mean": mean(ppls) if ppls else None,
                "ppl_std": stdev(ppls) if len(ppls) > 1 else 0.0,
                "anchor_mean": mean(ancs) if ancs else None,
                "anchor_std": stdev(ancs) if len(ancs) > 1 else 0.0,
                "niah_mean": mean(niahs) if niahs else None,
                "niah_std": stdev(niahs) if len(niahs) > 1 else 0.0,
                "n_seeds": len(ppls),
                "ppls_per_seed": ppls,
            }

    # Print PPL table
    for c in CORPORA:
        print(f"--- corpus: {c} ---")
        print(f"{'arch':>20} {'PPL':>22} {'anchor':>20} {'NIAH':>20}")
        base_mean = summary[(c, "baseline")]["ppl_mean"]
        for a in ARCHS:
            s = summary[(c, a)]
            ppl = f"{s['ppl_mean']:7.2f} ± {s['ppl_std']:5.2f}"
            anc = f"{s['anchor_mean']:.3f} ± {s['anchor_std']:.3f}"
            niah = (f"{s['niah_mean']:.3f} ± {s['niah_std']:.3f}"
                    if s["niah_mean"] is not None else "—")
            tag = ""
            if a != "baseline" and base_mean and s['ppl_mean']:
                gap_pct = (s['ppl_mean'] - base_mean) / base_mean * 100
                tag = f"  ({gap_pct:+5.1f}%)"
            print(f"{a:>20} {ppl}{tag:>10s}   {anc}   {niah}")
        print()

    # Retrieval-head analysis
    print(f"\n{'='*100}\nRETRIEVAL-HEAD ATTENTION TO NEEDLE (mean across seeds)\n{'='*100}\n")
    for c in CORPORA:
        print(f"--- corpus: {c} ---")
        for a in ARCHS:
            # Aggregate retrieval data across seeds
            head_attns = {}   # (layer, head, kind) -> list of mean_attns
            for s in SEEDS:
                r = load_retrieval(c, a, s)
                if r is None:
                    continue
                for row in r["rows"]:
                    key = (row["layer"], row["head"], row["kind"])
                    head_attns.setdefault(key, []).append(
                        row["mean_attention_to_needle"]
                    )
            if not head_attns:
                continue
            # Find top-3 heads by mean attention to needle
            ranked = sorted(head_attns.items(),
                              key=lambda kv: -mean(kv[1]))[:5]
            print(f"  {a}: top-5 heads (across all layers):")
            for (li, hi, kind), attns in ranked:
                m = mean(attns)
                print(f"    L{li} H{hi} {kind:>10s}  mean_attn_to_needle={m:.4f}")
        print()

    # Save aggregate
    out = {
        "summary": {
            f"{c}_{a}": summary[(c, a)]
            for c in CORPORA for a in ARCHS
        },
    }
    out_path = EXP / "results/aggregate.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
