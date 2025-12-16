#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_II_BoxPlot.py

Generates a box-and-whisker plot of the cleaned Phase II dataset:
- Inputs: Meteorite_Landings_Phase_II.csv
- Uses the 'count' column (annual meteorite totals)
- Outputs: 0_EDA_phase_II_boxplot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load Phase II CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_II.csv")
df = pd.read_csv(INPUT)

if "count" not in df.columns:
    raise ValueError("Phase II CSV must contain 'count' column.")

counts = df["count"].values

print(f"YEARS INCLUDED: {len(df)}")
print(f"YEAR RANGE: {df['year'].min()} — {df['year'].max()}")
print(f"COUNT RANGE: min={counts.min()}, max={counts.max()}")


# ---------------------------------------------------------------------
# 2. Create Box Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.boxplot(
    counts,
    vert=True,
    patch_artist=True,
    boxprops=dict(facecolor="#80c4ff", color="black"),
    medianprops=dict(color="red", linewidth=2),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    flierprops=dict(marker='o', color='black', markeredgecolor='black')
)

plt.title("Box-and-Whisker Plot — Phase II Annual Meteorite Counts")
plt.ylabel("Annual Meteorite Count")
plt.xlabel("Distribution")

# ---------------------------------------------------------------------
# 3. Stub under graph (centered)
# ---------------------------------------------------------------------
plt.figtext(
    0.5, -0.05,
    "Graph 7. Box-and-whisker plot for the Phase II annual meteorite count distribution.",
    ha="center",
    fontsize=10
)

plt.tight_layout()

# ---------------------------------------------------------------------
# 4. Save Output
# ---------------------------------------------------------------------
OUTPUT = Path("0_EDA_phase_II_boxplot.png")
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Phase II box plot saved to: {OUTPUT.resolve()}")
