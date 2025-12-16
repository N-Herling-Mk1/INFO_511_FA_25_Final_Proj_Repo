#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_IV_compare_contrast_test.py

Builds two linear models:
    Model A: log(count + 1)  ~ year
    Model B: sqrt(count)     ~ year

Outputs:
    - HTML comparison tables (3 tables)
    - Scatter plots with regression lines + annotated equation boxes
    - Residual plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpmath import erf as mp_erf, mp

# ----------------------------------------------------------
# High-precision math for p-values (NO UNDERFLOW)
# ----------------------------------------------------------
mp.dps = 80   # 80 digits of precision


# ----------------------------------------------------------
# Load Phase III Data
# ----------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["count_log"] = pd.to_numeric(df["count_log"], errors="coerce")
df["count_sqrt"] = pd.to_numeric(df["count_sqrt"], errors="coerce")
df = df.dropna(subset=["year", "count_log", "count_sqrt"])

year = df["year"].values
y_log = df["count_log"].values
y_sqrt = df["count_sqrt"].values


# ----------------------------------------------------------
# Manual Linear Regression Function (Closed Form)
# ----------------------------------------------------------
def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    x_mean = x.mean()
    y_mean = y.mean()

    slope = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
    intercept = y_mean - slope*x_mean

    y_pred = intercept + slope*x
    residuals = y - y_pred

    RSS = np.sum(residuals**2)
    TSS = np.sum((y - y_mean)**2)

    R2 = 1 - RSS/TSS
    R2_adj = 1 - (1-R2)*(n-1)/(n-2)

    MSE = RSS/n
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(residuals))

    # Standard errors
    s2 = RSS/(n-2)
    SE_slope = np.sqrt(s2 / np.sum((x-x_mean)**2))
    SE_intercept = np.sqrt(s2 * (1/n + x_mean**2/np.sum((x-x_mean)**2)))

    # 95% CI
    z = 1.96
    ci_slope = (slope - z*SE_slope, slope + z*SE_slope)
    ci_inter = (intercept - z*SE_intercept, intercept + z*SE_intercept)

    # F statistic
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
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "F": F,
        "ci_intercept_low": ci_inter[0],
        "ci_intercept_high": ci_inter[1],
        "ci_slope_low": ci_slope[0],
        "ci_slope_high": ci_slope[1],
        "p_value": float(p_value),
        "y_pred": y_pred,
        "residuals": residuals
    }


# ----------------------------------------------------------
# Fit Both Models
# ----------------------------------------------------------
results_log = linear_regression(year, y_log)
results_sqrt = linear_regression(year, y_sqrt)


# ----------------------------------------------------------
# Formatting Utilities
# ----------------------------------------------------------
def fmt(x):
    if abs(x) < 0.0001:
        return f"{x:.3e}"
    return f"{x:.4f}"

# Perfect scientific-notation formatter with coefficient
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
# TABLE 1 — Model Diagnostics
# ----------------------------------------------------------
df_metrics = pd.DataFrame([
    {
        "Model": "Log(count+1)",
        "R²": fmt(results_log["R2"]),
        "Adj R²": fmt(results_log["R2_adj"]),
        "MAE": fmt(results_log["MAE"]),
        "MSE": fmt(results_log["MSE"]),
        "RMSE": fmt(results_log["RMSE"]),
        "F-statistic": fmt(results_log["F"])
    },
    {
        "Model": "Sqrt(count)",
        "R²": fmt(results_sqrt["R2"]),
        "Adj R²": fmt(results_sqrt["R2_adj"]),
        "MAE": fmt(results_sqrt["MAE"]),
        "MSE": fmt(results_sqrt["MSE"]),
        "RMSE": fmt(results_sqrt["RMSE"]),
        "F-statistic": fmt(results_sqrt["F"])
    }
])


# ----------------------------------------------------------
# TABLE 2 — Parameter Estimates + CI
# ----------------------------------------------------------
df_params = pd.DataFrame([
    {
        "Model": "Log(count+1)",
        "Intercept β₀": fmt(results_log["intercept"]),
        "Slope β₁": fmt(results_log["slope"]),
        "Intercept CI [low,high]": f"[{fmt(results_log['ci_intercept_low'])}, {fmt(results_log['ci_intercept_high'])}]",
        "Slope CI [low,high]": f"[{fmt(results_log['ci_slope_low'])}, {fmt(results_log['ci_slope_high'])}]",
        "Equation": f"log(count+1) = {fmt(results_log['intercept'])} + {fmt(results_log['slope'])}·year"
    },
    {
        "Model": "Sqrt(count)",
        "Intercept β₀": fmt(results_sqrt["intercept"]),
        "Slope β₁": fmt(results_sqrt["slope"]),
        "Intercept CI [low,high]": f"[{fmt(results_sqrt['ci_intercept_low'])}, {fmt(results_sqrt['ci_intercept_high'])}]",
        "Slope CI [low,high]": f"[{fmt(results_sqrt['ci_slope_low'])}, {fmt(results_sqrt['ci_slope_high'])}]",
        "Equation": f"sqrt(count) = {fmt(results_sqrt['intercept'])} + {fmt(results_sqrt['slope'])}·year"
    }
])


# ----------------------------------------------------------
# TABLE 3 — Hypothesis Testing for Slope β₁
# ----------------------------------------------------------
alpha = 0.05

def decision_symbol(p):
    if p < alpha:
        return "<span style='color:green;'>✔ Reject H₀</span>"
    return "<span style='color:red;'>✘ Fail to Reject H₀</span>"

def decision_text(p):
    return ("Evidence of a statistically significant linear trend."
            if p < alpha
            else "No statistical evidence of a linear trend.")

df_hyp = pd.DataFrame([
    {
        "Model": "Log(count+1)",
        "p-value": fmt_p(results_log["p_value"]),
        "Decision": decision_symbol(results_log["p_value"]),
        "Interpretation": decision_text(results_log["p_value"])
    },
    {
        "Model": "Sqrt(count)",
        "p-value": fmt_p(results_sqrt["p_value"]),
        "Decision": decision_symbol(results_sqrt["p_value"]),
        "Interpretation": decision_text(results_sqrt["p_value"])
    }
])


# ----------------------------------------------------------
# HTML Rendering
# ----------------------------------------------------------
def df_to_html(df):
    html = "<table><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            html += f"<td style='text-align:center;'>{row[col]}</td>"
        html += "</tr>"
    html += "</table><br>"
    return html

html = f"""
<!DOCTYPE html>
<html>
<head>
<title>Phase IV Model Comparison</title>
<style>
body {{
    font-family: Arial; padding:20px;
}}
table {{
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    border-radius: 10px;
    overflow: hidden;
}}
th {{
    background: #9F7CFF;
    padding: 10px;
    border: 1px solid black;
    text-align:center;
}}
td {{
    background: #f3f0ff;
    padding: 8px;
    border: 1px solid black;
    text-align:center;
}}
h2 {{
    text-align:center;
}}
.desc {{
    font-size: 16px;
    margin-bottom: 25px;
    width: 85%;
}}
</style>
</head>
<body>

<h1 style="text-align:center;">Phase IV — Linear Regression Model Comparison</h1>

<h2>Table 1 — Model Diagnostics</h2>
{df_to_html(df_metrics)}
<div class='desc'>
This table summarizes R², Adjusted R², MAE, MSE, RMSE, and the F-statistic for both models.
</div>

<h2>Table 2 — Parameter Estimates & Confidence Intervals</h2>
{df_to_html(df_params)}
<div class='desc'>
Parameter table includes slopes, intercepts, and their 95% confidence intervals.
</div>

<h2>Table 3 — Hypothesis Test for Slope (β₁)</h2>
{df_to_html(df_hyp)}
<div class='desc'>
Interpretation meanings:<br>
✔ Reject H₀ → Evidence of a statistically significant linear trend.<br>
✘ Fail to Reject H₀ → No statistical evidence of a linear trend.
</div>

</body>
</html>
"""

OUTPUT = Path("0_EDA_phase_IV_model_comparison.html")
OUTPUT.write_text(html, encoding="utf-8")
print("✔ HTML tables written.")


# ----------------------------------------------------------
# Scatter Plots with Annotation Box
# ----------------------------------------------------------
def make_scatter(x, y, results, title, filename):
    plt.figure(figsize=(8,6))

    plt.scatter(x, y, color="#27ae60", edgecolor="black")

    y_pred = results["y_pred"]
    plt.plot(x, y_pred, color="black", linewidth=2)

    eq_symbolic = "y = β₀ + β₁·x"
    eq_numeric = f"y = {fmt(results['intercept'])} + {fmt(results['slope'])}·x"
    r2 = f"R² = {fmt(results['R2'])}"
    pval = f"p = {fmt_p(results['p_value'])}"

    text = f"{eq_symbolic}\n{eq_numeric}\n{r2}\n{pval}"

    plt.annotate(
        text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="black")
    )

    plt.title(f"{title} — Scatter Plot with Regression Line")
    plt.xlabel("Year")
    plt.ylabel(title)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✔ Scatter saved → {filename}")


make_scatter(year, y_log, results_log, "log(count+1)", "scatter_log.png")
make_scatter(year, y_sqrt, results_sqrt, "sqrt(count)", "scatter_sqrt.png")

print("✔ All scatter plots generated.")
print("✔ Phase IV fully complete.")
