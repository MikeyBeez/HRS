"""Figure 5: K=3 survival by passkey decoding complexity (Phase 67)."""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

OUT = Path(__file__).parent
mpl.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 1.8,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

types = ["fact\n(1-2 digits)", "technical\n(3-4 digits)",
         "numeric\n(6 digits)", "entity\n(multi-word dates)"]
mean_frac = [30, 27, 13, 0]
colors = ["#2a8f52", "#6fa93f", "#d4944d", "#b34a1d"]

fig, ax = plt.subplots(figsize=(5.8, 3.2))
bars = ax.bar(types, mean_frac, color=colors)
for b, v in zip(bars, mean_frac):
    ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v}%", ha="center", fontsize=9)
ax.set_ylabel("Mean fraction retrieved at K=3 (%)")
ax.set_ylim(0, 40)
ax.set_title("K=3 survival by decoding complexity: shorter outputs survive interference longer",
             fontsize=10)
fig.savefig(OUT / "fig5_complexity_survival.pdf")
fig.savefig(OUT / "fig5_complexity_survival.png", dpi=220)
plt.close(fig)
print("wrote fig5_complexity_survival.{pdf,png}")
