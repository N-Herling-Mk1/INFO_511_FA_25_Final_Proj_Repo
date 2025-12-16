#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_Data_Topology_check.py

Inputs:
    Meteorite_Landings_Phase_II.csv

Outputs:
    0_EDA_Data_Topology_Table.html

This script evaluates the topology of the Phase II dataset:
- Skewness
- Kurtosis
- Multimodality (via KDE peak count)
- Shape classification
- Transformation suggestions
- Elucidation and detailed guidance columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde, skew, kurtosis
import warnings

# ---------------------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_II.csv")
df = pd.read_csv(INPUT)

if "count" not in df.columns:
    raise ValueError("Phase II dataset must contain column 'count'.")

counts = df["count"].astype(float).values

# ---------------------------------------------------------------------
# 2. Compute Skewness and Kurtosis
# ---------------------------------------------------------------------
sk = skew(counts)
kt = kurtosis(counts, fisher=True)

# ---------------------------------------------------------------------
# 3. Multimodality via KDE peak count
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)

kde = gaussian_kde(counts)
xs = np.linspace(min(counts), max(counts), 512)
dens = kde(xs)

peaks = 0
for i in range(1, len(dens) - 1):
    if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
        peaks += 1

# ---------------------------------------------------------------------
# 4. Shape Classification
# ---------------------------------------------------------------------
if abs(sk) < 0.5:
    shape = "Approximately symmetric"
elif sk > 0.5:
    shape = "Right-skewed"
elif sk < -0.5:
    shape = "Left-skewed"
else:
    shape = "Irregular or ambiguous"

if kt > 3:
    shape += " (heavy-tailed)"
elif kt < -1:
    shape += " (light-tailed)"

# ---------------------------------------------------------------------
# 5. Transform Suggestion (no Winsorization)
# ---------------------------------------------------------------------
if sk > 1:
    suggestion = "Apply log(count + 1) or sqrt(count)"
elif sk > 0.5:
    suggestion = "Consider sqrt(count)"
elif abs(sk) < 0.5:
    suggestion = "No transformation required"
else:
    suggestion = "Consider Box–Cox or Yeo–Johnson transform"

if peaks > 1:
    suggestion += "; multimodality detected — consider segmentation"

# ---------------------------------------------------------------------
# 6. Elucidation Column (no Winsorization)
# ---------------------------------------------------------------------
elucidations = {
    "Skewness": (
        "Skewness measures asymmetry. Positive values indicate a right-tail, "
        "common in count data. Strong skew suggests benefit from transformation."
    ),
    "Kurtosis (Fisher)": (
        "Kurtosis measures tail weight. Positive values indicate heavier-than-normal tails, "
        "which affect normality and regression assumptions."
    ),
    "Estimated # of Modes": (
        "A proxy for modality. More than one mode suggests multimodal behavior and "
        "possible underlying subgroups."
    ),
    "Shape Classification": (
        "Combines skewness and kurtosis into a qualitative description of the distribution's shape."
    ),
    "Suggested Transformation": (
        "Transformations help stabilize variance and reduce skewness. "
        "Log and sqrt are typical for right-skewed count data."
    )
}

suggestion_map = {
    "Skewness": "Examine histogram; apply log transform if skew > 1.",
    "Kurtosis (Fisher)": "Interpret tail weight; check QQ plot for deviation.",
    "Estimated # of Modes": "If >1, consider segmenting dataset.",
    "Shape Classification": "Use this to evaluate linear model assumptions.",
    "Suggested Transformation": suggestion
}

# ---------------------------------------------------------------------
# 7. Build Table
# ---------------------------------------------------------------------
table_data = {
    "Metric": [
        "Skewness",
        "Kurtosis (Fisher)",
        "Estimated # of Modes",
        "Shape Classification",
        "Suggested Transformation"
    ],
    "Value": [
        f"{sk:.3f}",
        f"{kt:.3f}",
        peaks,
        shape,
        suggestion
    ],
    "Elucidation": [
        elucidations["Skewness"],
        elucidations["Kurtosis (Fisher)"],
        elucidations["Estimated # of Modes"],
        elucidations["Shape Classification"],
        elucidations["Suggested Transformation"]
    ],
    "Suggestions": [
        suggestion_map["Skewness"],
        suggestion_map["Kurtosis (Fisher)"],
        suggestion_map["Estimated # of Modes"],
        suggestion_map["Shape Classification"],
        suggestion_map["Suggested Transformation"]
    ]
}

df_out = pd.DataFrame(table_data)

# ---------------------------------------------------------------------
# 8. Build HTML (center all text)
# ---------------------------------------------------------------------
html_top = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Data Topology Summary</title>

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
  margin-left: auto;
  margin-right: auto;
  border-collapse: separate;
  border-spacing: 0;
  border: 3px solid #000;
  border-radius: 12px;
  overflow: hidden;
  width: 95%;
}

table.master th {
  background: #cce6ff;
  padding: 10px;
  border: 1px solid #000;
  font-size: 18px;
  font-weight: bold;
  text-align: center;
}

table.master td {
  background: #f4e8d2;
  padding: 10px;
  border: 1px solid #000;
  font-size: 16px;
  text-align: center;
  vertical-align: middle;
}

.footer {
  text-align: center;
  margin-top: 14px;
  font-size: 16px;
}

</style>
</head>

<body>

<div class="title">Table — Data Topology Summary</div>
<div class="subtitle">Phase II Annual Meteorite Count Distribution</div>

<table class="master">
<thead>
<tr>
"""

html_mid = ""
for col in df_out.columns:
    html_mid += f"<th>{col}</th>"

html_mid += "</tr></thead><tbody>\n"

for _, row in df_out.iterrows():
    html_mid += "<tr>\n"
    html_mid += f"<td>{row['Metric']}</td>\n"
    html_mid += f"<td>{row['Value']}</td>\n"
    html_mid += f"<td>{row['Elucidation']}</td>\n"
    html_mid += f"<td>{row['Suggestions']}</td>\n"
    html_mid += "</tr>\n"

html_bottom = """
</tbody>
</table>

<div class="footer">
Table X. Summary of distribution topology and transformation guidance.
</div>

</body>
</html>
"""

# ---------------------------------------------------------------------
# 9. Save Output
# ---------------------------------------------------------------------
OUTPUT = Path("0_EDA_Data_Topology_Table.html")
OUTPUT.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Data topology table saved to: {OUTPUT.resolve()}")
