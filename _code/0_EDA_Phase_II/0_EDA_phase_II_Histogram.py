#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_II_Histogram.py

Creates TWO histograms for the Phase II cleaned dataset:
1. Standard histogram of annual meteorite counts
2. Log-scaled Y-axis histogram of annual meteorite counts

Inputs:
    Meteorite_Landings_Phase_II.csv

Outputs:
    0_EDA_phase_II_histogram_standard.png
    0_EDA_phase_II_histogram_logscale.png
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


# =====================================================================
#  FIRST HISTOGRAM (Standard Y-axis)
# =====================================================================
plt.figure(figsize=(8, 6))

plt.hist(counts, bins=20, color="#2a6fdb", edgecolor="black", alpha=0.85)
plt.title("Histogram — Phase II Annual Meteorite Counts")
plt.xlabel("Annual Meteorite Count")
plt.ylabel("Frequency")

# Stub text under graph
plt.figtext(
    0.5, -0.05,
    "Figure 5. Standard histogram for Phase II annual meteorite counts.",
    ha="center",
    fontsize=10
)

plt.tight_layout()
OUT_STD = Path("0_EDA_phase_II_histogram_standard.png")
plt.savefig(OUT_STD, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Standard histogram saved to: {OUT_STD.resolve()}")


# =====================================================================
#  SECOND HISTOGRAM (Log-scaled Y-axis)
# =====================================================================
plt.figure(figsize=(8, 6))

plt.hist(counts, bins=20, color="#4caf50", edgecolor="black", alpha=0.85)
plt.yscale("log")

plt.title("Histogram — Phase II Annual Meteorite Counts (Log-Scaled)")
plt.xlabel("Annual Meteorite Count")
plt.ylabel("Log(Frequency of Years)")

# Stub text under graph
plt.figtext(
    0.5, -0.05,
    "Figure 6. Log-scaled histogram for Phase II annual meteorite counts.",
    ha="center",
    fontsize=10
)

plt.tight_layout()
OUT_LOG = Path("0_EDA_phase_II_histogram_logscale.png")
plt.savefig(OUT_LOG, dpi=300, bbox_inches="tight")
plt.close()

print(f"✔ Log-scale histogram saved to: {OUT_LOG.resolve()}")
