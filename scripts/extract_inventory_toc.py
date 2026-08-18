"""
Read EXPERIMENT_INVENTORY.md and extract just the structure — H1, H2, H3 headers
plus the file paths. Output is a concise navigation map.
"""
import re

INVENTORY = "/Users/bard/Code/HRS/EXPERIMENT_INVENTORY.md"
OUT = "/Users/bard/Code/HRS/EXPERIMENT_INVENTORY_TOC.md"

out = []
with open(INVENTORY) as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("# ") or line.startswith("## ") or line.startswith("### "):
            out.append(line)
        elif line.startswith("**Path:**"):
            out.append(f"  {line}")

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")

print(f"Wrote {OUT}")
print(f"Lines: {len(out)}")
