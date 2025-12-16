#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_Box_plot.py

Generates separate box plots for transformed fields in:
    Meteorite_Landings_Phase_III.csv

Outputs:
    boxplot_count_log.png
    boxplot_count_sqrt.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load Phase III dataset
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

required_cols = ["count_log", "count_sqrt"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

log_vals = df["count_log"]
sqrt_vals = df["count_sqrt"]

# ---------------------------------------------------------------------
# Helper function for generating a box plot
# ---------------------------------------------------------------------
def make_box_plot(values, title, y_label, filename, stub_text):
    plt.figure(figsize=(7, 6))

    plt.boxplot(values, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#d9e6ff", color="black"),
                medianprops=dict(color="darkred", linewidth=2),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"))

    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=13)

    # Stub under graph
    plt.figtext(
        0.05, -0.05,
        stub_text,
        ha="left",
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved: {filename}")

# ---------------------------------------------------------------------
# 2. Box plot for log(count + 1)
# ---------------------------------------------------------------------
make_box_plot(
    log_vals,
    title="Box Plot — log(count + 1) Transformed Data",
    y_label="log(count + 1)",
    filename="boxplot_count_log.png",
    stub_text="Figure X. Box plot of log-transformed annual meteorite counts."
)

# ---------------------------------------------------------------------
# 3. Box plot for sqrt(count)
# ---------------------------------------------------------------------
make_box_plot(
    sqrt_vals,
    title="Box Plot — sqrt(count) Transformed Data",
    y_label="sqrt(count)",
    filename="boxplot_count_sqrt.png",
    stub_text="Figure Y. Box plot of sqrt-transformed annual meteorite counts."
)
