#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_I_Histogram.py

Generates TWO histograms of annual meteorite counts (Fell + Found):

1. Standard histogram
2. Histogram with a log-scaled y-axis

Both use strict year filtering identical to the QQ-plot script.
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
# 2. STRICT year validation (matches QQ plot)
# ---------------------------------------------------------------------
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(1000, 3000)]
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# ---------------------------------------------------------------------
# 3. Counts per year
# ---------------------------------------------------------------------
year_counts = df.groupby("year")["fall"].count().sort_index()
counts = year_counts.values

print(f"YEARS INCLUDED: {len(year_counts)}")
print(f"YEAR RANGE: {year_counts.index.min()} — {year_counts.index.max()}")
print(f"MIN/YEAR COUNT: {counts.min()}, MAX/YEAR COUNT: {counts.max()}")

# =====================================================================
#  FIRST HISTOGRAM (normal y-axis)
# =====================================================================
plt.figure(figsize=(8, 6))

plt.hist(counts, bins=20, color="#2a6fdb", edgecolor="black", alpha=0.85)
plt.title("Histogram — Annual Meteorite Landings (Fell + Found)")
plt.xlabel("Annual Meteorite Count")
plt.ylabel("Frequency")

# Stub text (Figure label)
plt.figtext(
    0.02, -0.03,
    "Figure 2. Histogram of annual meteorite counts from the Meteorite Landings dataset.",
    ha="left",
    fontsize=10
)

plt.tight_layout()
OUTPUT_STD = Path("0_EDA_phase_I_histogram_standard.png")
plt.savefig(OUTPUT_STD, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Standard histogram saved to: {OUTPUT_STD.resolve()}")


# =====================================================================
#  SECOND HISTOGRAM (log-scaled y-axis)
# =====================================================================
plt.figure(figsize=(8, 6))

plt.hist(counts, bins=20, color="#4caf50", edgecolor="black", alpha=0.85)
plt.yscale("log")   # LOG SCALE

# UPDATED title and y-label
plt.title("Histogram — Annual Meteorite Landings (Log-Scaled Frequencies)")
plt.xlabel("Annual Meteorite Count")
plt.ylabel("Log(Frequency of Years)")

# Stub text (Figure label)
plt.figtext(
    0.02, -0.03,
    "Figure 3. Histogram of annual meteorite counts with log-scaled frequencies.",
    ha="left",
    fontsize=10
)

plt.tight_layout()
OUTPUT_LOG = Path("0_EDA_phase_I_histogram_logscale.png")
plt.savefig(OUTPUT_LOG, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Log-scale histogram saved to: {OUTPUT_LOG.resolve()}")
