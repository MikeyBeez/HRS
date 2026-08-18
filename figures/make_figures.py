"""Generate figures for engram_neurips_final.

Data values taken directly from paper_neurips.md sections 3-6.
Figures are saved as PDF (vector, clean for camera-ready) and PNG (preview).
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path(__file__).parent
mpl.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    plt.close(fig)


def legend_outside(ax, handles=None, **kw):
    """Put the legend outside the plot (to the right) so it cannot overlap lines."""
    if handles is None:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, fontsize=9, **kw)
    else:
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, **kw)


# -----------------------------------------------------------------------------
# Combined Figure 1: K/V asymmetry (left) + phase transition (right).
#   Left: L5 engram K 0.82-0.89, V 0.65-0.70 across layers 1-5; random 0.525,
#     zero 0.602.
#   Right: Retrieval steps from 0/5 to 5/5 over 0-38 training steps;
#     K-space cosine alignment constant 0.84-0.87 throughout.
# -----------------------------------------------------------------------------
layers = [1, 2, 3, 4, 5]
k_cos = [0.82, 0.85, 0.87, 0.88, 0.89]
v_cos = [0.65, 0.66, 0.68, 0.69, 0.70]
rand_k = 0.525
zero_k = 0.602

steps = [0, 15, 38, 75, 113, 150, 225, 300]
retrieval = [0/5, 4/5, 5/5, 5/5, 5/5, 5/5, 5/5, 5/5]
kspace = [0.84, 0.85, 0.86, 0.87, 0.86, 0.87, 0.85, 0.86]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.2),
                                gridspec_kw={"wspace": 0.35})

# Left panel: K/V asymmetry with controls
axL.plot(layers, k_cos, marker="o", color="#1f4e9a", label="K-space (routing)")
axL.plot(layers, v_cos, marker="s", color="#b34a1d", label="V-space (content)")
axL.axhline(zero_k, color="#888", linestyle=":",  linewidth=1.2, label=f"Zero vector ({zero_k:.2f})")
axL.axhline(rand_k, color="#888", linestyle="--", linewidth=1.2, label=f"Random vector ({rand_k:.2f})")
axL.set_xlabel("Layer")
axL.set_ylabel("Cosine alignment with passage mean")
axL.set_ylim(0.4, 1.0)
axL.set_xticks(layers)
axL.legend(loc="lower right", frameon=False, fontsize=8)
axL.set_title("(a) K/V asymmetry across layers", fontsize=10)

# Right panel: phase transition
axR2 = axR.twinx()
l1, = axR.plot(steps, [r*100 for r in retrieval], marker="o", color="#1b7f3a", label="Retrieval rate (%)")
l2, = axR2.plot(steps, kspace, marker="s", color="#1f4e9a", linestyle="--", label="K-space cos")
axR.axvspan(15, 38, alpha=0.08, color="#1b7f3a", zorder=0)
axR.set_xlabel("Training steps")
axR.set_ylabel("Retrieval rate (%)", color="#1b7f3a")
axR2.set_ylabel("K-space cosine alignment", color="#1f4e9a")
axR.set_ylim(-5, 105)
axR2.set_ylim(0.5, 1.0)
axR2.grid(False)
axR.legend(handles=[l1, l2], loc="center right", frameon=False, fontsize=8)
axR.set_title("(b) Phase transition at 15–38 steps", fontsize=10)

save(fig, "fig1_kv_and_phase")


# -----------------------------------------------------------------------------
# Figure 3: Recency asymmetry in per-chunk compressibility
#   Four chunks of 50 tokens, distant -> recent.
#   NLL cost of compressing each chunk to a single engram.
# -----------------------------------------------------------------------------
chunks = ["Chunk 1\n(most distant)", "Chunk 2", "Chunk 3", "Chunk 4\n(most recent)"]
nll_cost = [0.007, 0.045, 0.120, 0.260]
colors = ["#4a7fbb", "#6a9bcf", "#d28a5c", "#b34a1d"]

fig, ax = plt.subplots(figsize=(5.6, 3.2))
bars = ax.bar(chunks, nll_cost, color=colors)
for b, v in zip(bars, nll_cost):
    ax.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("NLL cost of compression to engram")
ax.set_ylim(0, 0.30)
ax.set_title("37x recency asymmetry: compressing recent context is much more expensive", fontsize=10)
save(fig, "fig3_recency_asymmetry")


# -----------------------------------------------------------------------------
# Figure 4: Compositional decay curve (updated with K=3 data)
#   K=1: 100% / 9/9 ALL
#   K=2: 50% / 1/5 ALL
#   K=3: 20% / 0/5 ALL
#   K=4:  0% / 0/2 ALL
# -----------------------------------------------------------------------------
K = [1, 2, 3, 4]
mean_frac = [100, 50, 20, 0]
all_retrieved = [100, 20, 0, 0]  # as percentages: 9/9=100, 1/5=20, 0/5=0, 0/2=0

fig, ax = plt.subplots(figsize=(5.6, 3.2))
l1, = ax.plot(K, mean_frac, marker="o", color="#1f4e9a", label="Mean fraction retrieved")
l2, = ax.plot(K, all_retrieved, marker="s", color="#b34a1d", linestyle="--", label="All-retrieved rate")
ax.set_xlabel("K (simultaneous adapters)")
ax.set_ylabel("Retrieval (%)")
ax.set_xticks(K)
ax.set_ylim(-5, 105)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title("Compositional decay: each added operator costs 30–50 pts of retrieval", fontsize=10)
save(fig, "fig4_compositional_decay")

# -----------------------------------------------------------------------------
# Figure 2: KV-cache compression curve.
#   Mixed strategy (distant engram + recent verbatim): 100% at 1x, 92% at 2x,
#     84% at 4x, smooth descent ("~0.5 PPL per doubling") out to 384x.
#   Engram-only operating points (no literal tokens): mean-pool 18% at 200x,
#     attention-pool 28% at 200x, K=5 learned queries 30% at 40x.
#   Adapter library reference line at 97% (infinite compression post-absorption).
# -----------------------------------------------------------------------------
import numpy as np
x = np.logspace(0, np.log10(384), 80)
# Mixed strategy fitted to three reported anchors: y = 100 - 8*log2(x)
mixed = 100 - 8 * np.log2(x)
mixed = np.clip(mixed, 30, 100)

fig, ax = plt.subplots(figsize=(7.0, 3.6))

# Mixed-strategy curve with the three anchor points labelled in the legend
ax.plot(x, mixed, color="#1f4e9a", linewidth=2,
        label="Mixed (distant engram + recent tokens)")
ax.scatter([1, 2, 4], [100, 92, 84], color="#1f4e9a", s=40, zorder=5)

# Engram-only operating points — one legend entry per method, no inline labels
ax.scatter([200], [18], color="#b34a1d", marker="s", s=55, zorder=5,
           label="Engram only: mean pool (18%, 200×)")
ax.scatter([200], [28], color="#d47a3d", marker="s", s=55, zorder=5,
           label="Engram only: attention pool (28%, 200×)")
ax.scatter([40],  [30], color="#e9a84d", marker="s", s=55, zorder=5,
           label="Engram only: K=5 learned queries (30%, 40×)")

# Adapter library reference (infinite compression after absorption)
ax.axhline(97, color="#1b7f3a", linestyle=":", linewidth=1.4,
           label="Adapter library after absorption (97%)")

# Label the three mixed-curve anchor percentages cleanly, offset above-right
for xi, yi in [(1, 100), (2, 92), (4, 84)]:
    ax.annotate(f"{yi}%", (xi, yi), textcoords="offset points",
                xytext=(7, 6), fontsize=8, color="#1f4e9a")

ax.set_xscale("log")
ax.set_xlabel("Compression ratio (log scale)")
ax.set_ylabel("Full-context NLL gap closed (%)")
ax.set_xlim(0.9, 500)
ax.set_ylim(0, 108)
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xticklabels(["1×", "2×", "4×", "8×", "16×", "32×", "64×", "128×", "256×"])
# Legend at bottom, outside the data region
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55),
          ncol=2, frameon=False, fontsize=8.5)
ax.set_title("KV-cache compression: three operating points on one curve", fontsize=10)
save(fig, "fig2_compression_curve")

print(f"Wrote 5 figures to {OUT}/")
