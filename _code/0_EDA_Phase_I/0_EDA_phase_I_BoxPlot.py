#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_I_BoxPlot.py

Generates a box-and-whisker plot of annual meteorite counts (Fell + Found),
using the same strict year filtering as all other EDA Phase I scripts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------------------------
INPUT_CSV = Path("Meteorite_Landings.csv")
df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------------------
# 2. STRICT year validation (identical to histogram + QQ scripts)
# ---------------------------------------------------------------------
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(1000, 3000)]
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# ---------------------------------------------------------------------
# 3. Count per year
# ---------------------------------------------------------------------
year_counts = df.groupby("year")["fall"].count().sort_index()
counts = year_counts.values

print(f"YEARS INCLUDED: {len(year_counts)}")
print(f"YEAR RANGE: {year_counts.index.min()} — {year_counts.index.max()}")
print(f"MIN/YEAR COUNT: {counts.min()}, MAX/YEAR COUNT: {counts.max()}")

# ---------------------------------------------------------------------
# 4. Box Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.boxplot(counts, vert=True, patch_artist=True,
            boxprops=dict(facecolor="#80c4ff", color="black"),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(color="black", markeredgecolor="black"))

plt.title("Box-and-Whisker Plot — Raw Meteorite Landings data\nAnnual Meteorite Landings (Fell + Found)")
plt.ylabel("Annual Meteorite Count")
plt.xlabel("Distribution")

# ---------------------------------------------------------------------
# 5. Figure Stub
# ---------------------------------------------------------------------
plt.figtext(
    0.02, -0.03,
    "Figure 4. Box-and-whisker plot of annual meteorite counts.",
    ha="left",
    fontsize=10
)

plt.tight_layout()

OUTPUT_PNG = Path("0_EDA_phase_I_boxplot_fall_counts.png")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Box plot saved to: {OUTPUT_PNG.resolve()}")
