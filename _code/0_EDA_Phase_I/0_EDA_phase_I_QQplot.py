#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_I_QQplot.py

Generates a QQ plot of annual meteorite counts (Fell + Found).
Only uses rows where `year` is a valid 4-digit integer.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import warnings

# ---------------------------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------------------------
INPUT_CSV = Path("Meteorite_Landings.csv")
df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------------------
# 2. STRICT year validation
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

# ---------------------------------------------------------------------
# 4. QQ Plot
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# We set dummy labels now, but will override AFTER probplot()
plt.title("QQ Plot — Raw Meteorite Landings data\nAnnual Meteorite Counts (Fell + Found)")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")   # <-- Updated per request

# A. Small sample (<10)
if len(counts) < 10:
    print(f"⚠ Sample too small for regression (n={len(counts)}). Plotting quantiles only.")

    osm, osr = stats.probplot(counts, dist="norm", fit=False)
    plt.scatter(osm, osr, s=40, color="blue", label="Data Quantiles")
    plt.legend()

# B. Normal sample case
else:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*SmallSampleWarning.*")
        stats.probplot(counts, dist="norm", plot=plt)

# ---------------------------------------------------------
# AFTER probplot: restore OUR title + labels (SciPy overrides)
# ---------------------------------------------------------
plt.title("QQ Plot — Raw Meteorite Landings data\nAnnual Meteorite Counts (Fell + Found)")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# ---------------------------------------------------------
# Add Figure Stub (like Table. 1 …)
# ---------------------------------------------------------
plt.figtext(
    0.02, -0.03,
    "Figure 1. QQ plot assessing normality of annual meteorite counts.",
    ha="left",
    fontsize=10
)

plt.tight_layout()

OUTPUT_PNG = Path("0_EDA_phase_I_qqplot_fall_counts.png")
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ QQ plot saved to: {OUTPUT_PNG.resolve()}")
