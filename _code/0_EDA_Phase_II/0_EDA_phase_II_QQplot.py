#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_II_QQ_plot.py

Generates a QQ plot of the cleaned Phase II dataset:
- Inputs: Meteorite_Landings_Phase_II.csv
- Uses the 'count' column (annual meteorite totals)
- Saves: 0_EDA_phase_II_QQplot.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import warnings

# ---------------------------------------------------------------------
# 1. Load Phase II CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_II.csv")
df = pd.read_csv(INPUT)

# Ensure expected columns are present
if "count" not in df.columns or "year" not in df.columns:
    raise ValueError("Phase II input CSV must contain 'year' and 'count' columns.")

counts = df["count"].values

print(f"YEARS INCLUDED: {len(df)}")
print(f"YEAR RANGE: {df['year'].min()} — {df['year'].max()}")
print(f"COUNT RANGE: min={counts.min()}, max={counts.max()}")

# ---------------------------------------------------------------------
# 2. QQ Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Initial (temporary) labels
plt.title("QQ Plot — Phase II Annual Meteorite Counts")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# If few samples, avoid regression line
if len(counts) < 10:
    print(f"⚠ Small sample (n={len(counts)}). Plotting quantiles only, no regression.")

    osm, osr = stats.probplot(counts, dist="norm", fit=False)
    plt.scatter(osm, osr, s=40, color="blue", label="Data Quantiles")
    plt.legend()

else:
    # Hide noisy SciPy warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*SmallSampleWarning.*")
        stats.probplot(counts, dist="norm", plot=plt)

# Restore OUR labels/titles (SciPy overwrites them)
plt.title("QQ Plot — Phase II Annual Meteorite Counts")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# ---------------------------------------------------------------------
# 3. Figure Stub (centered)
# ---------------------------------------------------------------------
plt.figtext(
    0.5, -0.05,
    "Figure 4. QQ plot for the cleaned Phase II annual meteorite counts dataset.",
    ha="center",
    fontsize=10
)

plt.tight_layout()

# ---------------------------------------------------------------------
# 4. Save Output
# ---------------------------------------------------------------------
OUTPUT = Path("0_EDA_phase_II_QQplot.png")
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Phase II QQ plot saved to: {OUTPUT.resolve()}")
