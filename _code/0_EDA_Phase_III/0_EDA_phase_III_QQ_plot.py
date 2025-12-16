#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_QQ_plot.py

Generates QQ plots for the transformed fields in:
    Meteorite_Landings_Phase_III.csv

Outputs:
    qqplot_count_log.png
    qqplot_count_sqrt.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import warnings

# ---------------------------------------------------------------------
# 1. Load Phase III dataset
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

required_cols = ["count_log", "count_sqrt"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

log_vals = df["count_log"].values
sqrt_vals = df["count_sqrt"].values

# ---------------------------------------------------------------------
# Helper function for producing QQ plots
# ---------------------------------------------------------------------
def make_qq_plot(values, title, filename, stub_text):
    plt.figure(figsize=(8, 6))

    # Produce QQ Plot
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*SmallSampleWarning.*")
        stats.probplot(values, dist="norm", plot=plt)

    # Overwrite SciPy default title
    plt.title(title, fontsize=16)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")

    # Stub under graph (left aligned)
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
# 2. Generate QQ for log(count + 1)
# ---------------------------------------------------------------------
make_qq_plot(
    log_vals,
    title="QQ Plot — log(count + 1) Transformed Data",
    filename="qqplot_count_log.png",
    stub_text="Figure X. QQ plot for log-transformed annual meteorite counts."
)

# ---------------------------------------------------------------------
# 3. Generate QQ for sqrt(count)
# ---------------------------------------------------------------------
make_qq_plot(
    sqrt_vals,
    title="QQ Plot — sqrt(count) Transformed Data",
    filename="qqplot_count_sqrt.png",
    stub_text="Figure Y. QQ plot for sqrt-transformed annual meteorite counts."
)
