#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_IV_only_log_model_graphs.py

Produces ONLY the graphs for:
    log(count+1) ~ year

Outputs:
    - High-quality scatter plot with regression line + annotation
    - Residual plot
    - QQ plot
    - Terminal output: Outlier check for log(count+1) data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpmath import erf as mp_erf, mp

# ----------------------------------------------------------
# High precision math (prevents p-value underflow)
# ----------------------------------------------------------
mp.dps = 80   # 80 digits precision


# ----------------------------------------------------------
# Load Phase III Data
# ----------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

df["year"] = pd.to_numeric(df["year"], errors="ignore")
df["count_log"] = pd.to_numeric(df["count_log"], errors="coerce")
df = df.dropna(subset=["year", "count_log"])

year = df["year"].values
y_log = df["count_log"].values


# ----------------------------------------------------------
# Manual closed-form OLS (simple regression)
# ----------------------------------------------------------
def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    x_mean = x.mean()
    y_mean = y.mean()

    slope = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - slope * x_mean

    y_pred = intercept + slope*x
    residuals = y - y_pred

    RSS = np.sum(residuals**2)
    TSS = np.sum((y - y_mean)**2)

    R2 = 1 - RSS/TSS
    R2_adj = 1 - (1-R2)*(n-1)/(n-2)

    MSE = RSS/n
    RMSE = np.sqrt(MSE)

    # Standard error
    s2 = RSS/(n-2)
    SE_slope = np.sqrt(s2 / np.sum((x-x_mean)**2))

    # 95% CI (not used here but calculated)
    z = 1.96
    ci_slope = (slope - z*SE_slope, slope + z*SE_slope)

    # F-statistic
    F = (R2/(1-R2))*(n-2)

    # High-precision p-value
    t_slope = slope / SE_slope
    t_abs = mp.mpf(abs(t_slope))
    p_value = 2*(1 - (1 + mp_erf(t_abs/mp.sqrt(2))) / 2)

    return {
        "intercept": intercept,
        "slope": slope,
        "R2": R2,
        "R2_adj": R2_adj,
        "RMSE": RMSE,
        "F": F,
        "p_value": float(p_value),
        "y_pred": y_pred,
        "residuals": residuals
    }


results = linear_regression(year, y_log)


# ----------------------------------------------------------
# Formatters
# ----------------------------------------------------------
def fmt_num(x):
    if abs(x) < 0.0001:
        return f"{x:.3e}"
    return f"{x:.4f}"

def fmt_p(x):
    x = mp.mpf(x)
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    x = abs(x)
    exp = int(mp.floor(mp.log10(x)))
    coef = x / (10**exp)
    return f"{sign}{float(coef):.4f}e{exp}"


# ----------------------------------------------------------
# OUTLIER CHECK (1.5 × IQR rule)
# ----------------------------------------------------------
Q1 = np.percentile(y_log, 25)
Q3 = np.percentile(y_log, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["count_log"] < lower_bound) | (df["count_log"] > upper_bound)]

print("\n================ OUTLIER CHECK — log(count+1) ================")
print(f"Q1 = {Q1:.4f}, Q3 = {Q3:.4f}, IQR = {IQR:.4f}")
print(f"Lower bound = {lower_bound:.4f}, Upper bound = {upper_bound:.4f}")
print(f"Number of outliers detected: {len(outliers)}")

if len(outliers) == 0:
    print("✔ No outliers detected in log(count+1).")
else:
    print(outliers[["year", "count_log"]])
print("==============================================================\n")


# ----------------------------------------------------------
# PLOTTING HELPERS
# ----------------------------------------------------------
title_font = {"fontsize": 18, "fontweight": "bold"}
label_font = {"fontsize": 16, "fontweight": "bold"}


# ----------------------------------------------------------
# SCATTER PLOT
# ----------------------------------------------------------
def make_scatter():
    plt.figure(figsize=(9,7))
    plt.scatter(year, y_log, color="#27ae60", edgecolor="black")

    y_pred = results["y_pred"]
    plt.plot(year, y_pred, color="black", linewidth=2)

    eq_symbolic = "y = β₀ + β₁·x"
    eq_numeric = f"y = {fmt_num(results['intercept'])} + {fmt_num(results['slope'])}·x"
    r2 = f"R² = {fmt_num(results['R2'])}"
    pval = f"p = {fmt_p(results['p_value'])}"

    text = f"{eq_symbolic}\n{eq_numeric}\n{r2}\n{pval}"

    plt.annotate(
        text, xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left", va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="black")
    )

    plt.title("log(count+1) — Scatter Plot with Regression Line", **title_font)
    plt.xlabel("Year", **label_font)
    plt.ylabel("log(count+1)", **label_font)
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("log_scatter.png", dpi=300)
    plt.close()
    print("✔ Saved scatter plot → log_scatter.png")


# ----------------------------------------------------------
# RESIDUAL PLOT
# ----------------------------------------------------------
def make_residual_plot():
    residuals = results["residuals"]
    y_pred = results["y_pred"]

    plt.figure(figsize=(9,7))
    plt.scatter(y_pred, residuals, color="#2980b9", edgecolor="black")
    plt.axhline(0, color="red", linestyle="--")

    plt.title("Residual Plot — Log Model", **title_font)
    plt.xlabel("Predicted Values", **label_font)
    plt.ylabel("Residuals", **label_font)

    plt.tight_layout()
    plt.savefig("log_residuals.png", dpi=300)
    plt.close()
    print("✔ Saved residual plot → log_residuals.png")


# ----------------------------------------------------------
# QQ PLOT
# ----------------------------------------------------------
import scipy.stats as stats

def make_qq_plot():
    plt.figure(figsize=(9,7))
    stats.probplot(y_log, dist="norm", plot=plt)

    plt.title("QQ Plot — log(count+1)", **title_font)
    plt.xlabel("Theoretical Quantiles", **label_font)
    plt.ylabel("Sample Quantiles", **label_font)

    plt.tight_layout()
    plt.savefig("log_qqplot.png", dpi=300)
    plt.close()
    print("✔ Saved QQ plot → log_qqplot.png")


# ----------------------------------------------------------
# RUN ALL GRAPHS
# ----------------------------------------------------------
make_scatter()
make_residual_plot()
make_qq_plot()

print("✔ All log(count+1) graphs complete.")
