#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_IV_analysis.py

Phase IV — Automated Regression Suitability Analysis
Evaluates ALL THREE models:

    1. year → count
    2. year → count_log
    3. year → count_sqrt

Tests INFO 511 regression assumptions:
- Linearity
- Independence (Durbin–Watson)
- Normality of residuals (Shapiro)
- Homoscedasticity
- Outliers / high leverage
- Numeric predictor check

Outputs:
    0_EDA_phase_IV_analysis.html
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ----------------------------------------------------------
# 1. Load Phase III dataset
# ----------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

required_cols = ["count", "count_log", "count_sqrt"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# predictor X
x = df["year"].astype(float).values

# outcome columns evaluated
models = {
    "Raw Count": df["count"].astype(float).values,
    "Log Transform (count_log)": df["count_log"].astype(float).values,
    "Sqrt Transform (count_sqrt)": df["count_sqrt"].astype(float).values
}

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def durbin_watson(resid):
    diff = np.diff(resid)
    return np.sum(diff**2) / np.sum(resid**2)

def evaluate_model(x, y):
    """Runs all regression assumption tests for a single Y."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    # Fit regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    pred = intercept + slope * x_clean
    resid = y_clean - pred

    results = {}

    # 1. Linearity
    r = np.corrcoef(x_clean, y_clean)[0, 1]
    results["Linearity"] = "PASS" if abs(r) > 0.30 else "FAIL"

    # 2. Independence
    dw = durbin_watson(resid)
    results["Independence"] = "PASS" if 1.5 < dw < 2.5 else "FAIL"

    # 3. Normality
    p_shapiro = stats.shapiro(resid)[1]
    results["Normality of Residuals"] = "PASS" if p_shapiro > 0.05 else "FAIL"

    # 4. Homoscedasticity
    hom_corr = np.corrcoef(abs(resid), pred)[0, 1]
    results["Homoscedasticity"] = "PASS" if abs(hom_corr) < 0.20 else "FAIL"

    # 5. Outliers (Std Resid > 3)
    std_resid = (resid - resid.mean()) / resid.std()
    num_outliers = np.sum(abs(std_resid) > 3)
    results["Outliers / High Leverage"] = "PASS" if num_outliers == 0 else "FAIL"

    # 6. Numeric predictor
    results["Numeric Predictor"] = "PASS" if np.issubdtype(x.dtype, np.number) else "FAIL"

    return results


# ----------------------------------------------------------
# Evaluate all three models
# ----------------------------------------------------------
assumptions = [
    "Linearity",
    "Independence",
    "Normality of Residuals",
    "Homoscedasticity",
    "Outliers / High Leverage",
    "Numeric Predictor"
]

meaning = {
    "Linearity": "Relationship between X and Y should be approximately straight-line.",
    "Independence": "Residuals should not show autocorrelation (Durbin–Watson near 2).",
    "Normality of Residuals": "Residuals should follow an approximately normal distribution.",
    "Homoscedasticity": "Residual variance should be constant across fitted values.",
    "Outliers / High Leverage": "No standardized residuals with |value| > 3.",
    "Numeric Predictor": "Predictor variable (year) must be numeric."
}

model_results = {name: evaluate_model(x, y) for name, y in models.items()}

# ----------------------------------------------------------
# Build final combined table
# ----------------------------------------------------------

html_top = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase IV Regression Assumption Analysis</title>

<style>
body {
  font-family: Arial, sans-serif;
  background: #ffffff;
  padding: 20px;
}
.title {
  text-align: center;
  font-size: 32px;
  font-weight: bold;
  margin-bottom: 10px;
}
.subtitle {
  text-align: center;
  font-size: 20px;
  margin-bottom: 20px;
  color: #444444;
}
table.master {
  width: auto;
  margin: 0 auto;
  border-collapse: separate;
  border-spacing: 0;
  border: 3px solid #000;
  border-radius: 12px;
  overflow: hidden;
}
table.master th {
  background: #d7f7d0;
  padding: 10px;
  border: 1px solid #000;
  font-size: 17px;
  text-align: center;
  white-space: nowrap;
}
table.master td {
  background: #f4e8d2;
  padding: 8px 12px;
  border: 1px solid #000;
  text-align: center;
  font-size: 15px;
}
td.meaning {
  text-align: left;
  font-size: 15px;
}
.footer {
  margin-top: 15px;
  text-align: center;
  font-size: 16px;
}
</style>
</head>
<body>

<div class="title">Table 6 — Regression Assumption Evaluation</div>
<div class="subtitle">Comparison Across Raw, Log, and Sqrt Models</div>

<table class="master">
<thead>
<tr>
  <th>Assumption</th>
  <th>Meaning</th>
  <th>Raw Count</th>
  <th>Log Transform</th>
  <th>Sqrt Transform</th>
</tr>
</thead>
<tbody>
"""

html_mid = ""

for a in assumptions:
    html_mid += "<tr>\n"
    html_mid += f"<td><b>{a}</b></td>\n"
    html_mid += f"<td class='meaning'>{meaning[a]}</td>\n"

    for model_name in models:
        status = model_results[model_name][a]
        color = "#a7f3a7" if status == "PASS" else "#f7a7a7"
        html_mid += f"<td style='background:{color}; font-weight:bold;'>{status}</td>\n"

    html_mid += "</tr>\n"

html_bottom = """
</tbody>
</table>

<div class="footer">
<b>Table 6.</b> Automated evaluation of linear regression assumptions for all three Phase III models.
</div>

</body>
</html>
"""

OUTPUT = Path("0_EDA_phase_IV_analysis.html")
OUTPUT.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Phase IV Regression Assumptions Table saved to: {OUTPUT.resolve()}")
