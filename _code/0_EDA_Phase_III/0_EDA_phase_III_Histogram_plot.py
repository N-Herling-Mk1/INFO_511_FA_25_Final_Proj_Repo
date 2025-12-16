#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_bin_analysis.py

Performs bin analysis for transformed meteorite counts:
    - count_log
    - count_sqrt

Computes bins using:
    - Freedman–Diaconis rule
    - Scott’s rule
    - Sturges’ rule
    - Fixed bin size = 1

ALL histograms are Z-score normalized BEFORE plotting.

Outputs (per variable):
    {name}_FD.png
    {name}_Scott.png
    {name}_Sturges.png
    {name}_BIN1.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

df["count_log"] = pd.to_numeric(df["count_log"], errors="coerce")
df["count_sqrt"] = pd.to_numeric(df["count_sqrt"], errors="coerce")
df = df.dropna(subset=["count_log", "count_sqrt"])

log_vals = df["count_log"].values
sqrt_vals = df["count_sqrt"].values


# ---------------------------------------------------------------
# Bin-size rules
# ---------------------------------------------------------------
def freedman_diaconis_bins(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    n = len(x)
    h = 2 * iqr / (n ** (1/3))
    if h == 0:
        return 10
    return int(np.ceil((x.max() - x.min()) / h))


def scott_bins(x):
    sigma = np.std(x)
    n = len(x)
    h = 3.5 * sigma / (n ** (1/3))
    if h == 0:
        return 10
    return int(np.ceil((x.max() - x.min()) / h))


def sturges_bins(x):
    n = len(x)
    return int(np.ceil(np.log2(n) + 1))


# ---------------------------------------------------------------
# Z-score normalization
# ---------------------------------------------------------------
def z_score(values):
    mu = np.mean(values)
    sigma = np.std(values)
    if sigma == 0:
        return values - mu
    return (values - mu) / sigma


# ---------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------
def plot_with_bins(values, bins, title, xlabel, filename, stub):
    z_vals = z_score(values)

    plt.figure(figsize=(8, 6))
    plt.hist(z_vals, bins=bins, color="#99ccff", edgecolor="black")

    plt.title(title, fontsize=16)
    plt.xlabel(f"Z-scored {xlabel}", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    plt.figtext(0.05, -0.06, stub, ha="left", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved → {filename}")


# ---------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------
sets = [
    ("count_log", log_vals, "log(count + 1)", "Count Log"),
    ("count_sqrt", sqrt_vals, "sqrt(count)", "Count Sqrt"),
]

for name, vals, xlabel, label in sets:

    print(f"\n=== Optimal bin analysis for {name} ===")

    fd_bins = freedman_diaconis_bins(vals)
    sc_bins = scott_bins(vals)
    st_bins = sturges_bins(vals)
    bin1_bins = int((vals.max() - vals.min()) // 1) or 1

    print(f"Freedman–Diaconis: {fd_bins}")
    print(f"Scott:             {sc_bins}")
    print(f"Sturges:           {st_bins}")
    print(f"Bin size = 1:      {bin1_bins}")

    # ---------- FD ----------
    plot_with_bins(
        vals, fd_bins,
        title=f"Histogram — {label} (Freedman–Diaconis Rule, bins={fd_bins})",
        xlabel=xlabel,
        filename=f"{name}_FD.png",
        stub=f"Figure — {label} histogram using Freedman–Diaconis rule (Z-scored)."
    )

    # ---------- Scott ----------
    plot_with_bins(
        vals, sc_bins,
        title=f"Histogram — {label} (Scott’s Rule, bins={sc_bins})",
        xlabel=xlabel,
        filename=f"{name}_Scott.png",
        stub=f"Figure — {label} histogram using Scott’s rule (Z-scored)."
    )

    # ---------- Sturges ----------
    plot_with_bins(
        vals, st_bins,
        title=f"Histogram — {label} (Sturges’ Rule, bins={st_bins})",
        xlabel=xlabel,
        filename=f"{name}_Sturges.png",
        stub=f"Figure — {label} histogram using Sturges’ rule (Z-scored)."
    )

    # ---------- Bin size = 1 ----------
    plot_with_bins(
        vals, bin1_bins,
        title=f"Histogram — {label} (BIN SIZE = 1)",
        xlabel=xlabel,
        filename=f"{name}_BIN1.png",
        stub=f"Figure — {label} histogram using FIXED BIN SIZE = 1 (Z-scored)."
    )
