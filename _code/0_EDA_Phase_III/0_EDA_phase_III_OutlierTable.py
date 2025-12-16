#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_OutlierTable.py

Generates an Outlier Summary Table for ALL numeric columns
in the Phase III dataset.

Input:
    Meteorite_Landings_Phase_III.csv

Output:
    0_EDA_phase_III_OutlierTable.html

For each numeric column, computes:
- total count
- min
- max
- mean
- std dev
- 25th percentile
- 50th percentile (median)
- 75th percentile
- % within IQR
- number of outliers
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

rows, cols = df.shape

# Identify numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    raise ValueError("No numeric columns found in the Phase III dataset.")

# ---------------------------------------------------------------------
# 2. Compute Outlier Stats for Each Numeric Column
# ---------------------------------------------------------------------
records = []

for col in numeric_cols:
    arr = df[col].dropna().values

    total_count = len(arr)
    vmin = np.min(arr)
    vmax = np.max(arr)
    vmean = np.mean(arr)
    vstd = np.std(arr)

    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    iqr = q75 - q25

    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    within_iqr = ((arr >= lower_bound) & (arr <= upper_bound)).sum()
    pct_within = within_iqr / total_count * 100
    outliers = total_count - within_iqr

    records.append({
        "Column": col,
        "Total Count": total_count,
        "Min": vmin,
        "Max": vmax,
        "Mean": f"{vmean:.3f}",
        "Std Dev": f"{vstd:.3f}",
        "25th %ile": f"{q25:.3f}",
        "50th %ile": f"{q50:.3f}",
        "75th %ile": f"{q75:.3f}",
        "% Within IQR": f"{pct_within:.2f}%",
        "Outliers": outliers
    })

df_out = pd.DataFrame(records)

dataset_banner = f"Dataset Size: {rows:,} rows × {cols:,} columns"

# ---------------------------------------------------------------------
# 3. HTML Template
# ---------------------------------------------------------------------
html_top = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase III Outlier Summary Table</title>

<style>

  body {{
    font-family: Arial, sans-serif;
    background: #ffffff;
    padding: 20px;
  }}

  .title {{
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 6px;
  }}

  .subtitle {{
    text-align: center;
    font-size: 20px;
    margin-bottom: 20px;
    color: #444444;
  }}

  table.master {{
    width: auto;
    margin: 0 auto;
    border-collapse: separate;
    border-spacing: 0;
    border: 3px solid #000;
    border-radius: 12px;
    overflow: hidden;
  }}

  /* Violet column headers (Phase III uses #dcd0ff same as Phase II) */
  table.master th {{
    background: #dcd0ff;
    padding: 6px 10px;
    border: 1px solid #000;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    white-space: nowrap;
  }}

  /* Beige data cells */
  table.master td {{
    background: #f4e8d2;
    padding: 6px 10px;
    border: 1px solid #000;
    font-size: 16px;
    text-align: center;
    white-space: nowrap;
  }}

  /* Bold and left column for column names */
  table.master td.colname {{
    font-weight: bold;
    text-align: left;
  }}

  .footer {{
    font-size: 16px;
    margin-top: 14px;
    text-align: center;
  }}

</style>
</head>

<body>

<div class="title">Table 5 — Phase III Outlier Summary</div>
<div class="subtitle">{dataset_banner}</div>

<table class="master">
  <thead>
    <tr>
"""

# Add column headers dynamically
html_mid = ""
for col in df_out.columns:
    html_mid += f"      <th>{col}</th>\n"

html_mid += "    </tr>\n  </thead>\n  <tbody>\n"

# Add each row
for _, row in df_out.iterrows():
    html_mid += "    <tr>\n"
    html_mid += f"      <td class='colname'>{row['Column']}</td>\n"
    html_mid += f"      <td>{row['Total Count']}</td>\n"
    html_mid += f"      <td>{row['Min']}</td>\n"
    html_mid += f"      <td>{row['Max']}</td>\n"
    html_mid += f"      <td>{row['Mean']}</td>\n"
    html_mid += f"      <td>{row['Std Dev']}</td>\n"
    html_mid += f"      <td>{row['25th %ile']}</td>\n"
    html_mid += f"      <td>{row['50th %ile']}</td>\n"
    html_mid += f"      <td>{row['75th %ile']}</td>\n"
    html_mid += f"      <td>{row['% Within IQR']}</td>\n"
    html_mid += f"      <td>{row['Outliers']}</td>\n"
    html_mid += "    </tr>\n"

html_bottom = """
  </tbody>
</table>

<div class="footer"><b>Table 5.</b> Summary of IQR outlier detection and distribution metrics for all numeric Phase III variables.</div>

</body>
</html>
"""

# ---------------------------------------------------------------------
# 4. Save Output
# ---------------------------------------------------------------------
OUTPUT = Path("0_EDA_phase_III_OutlierTable.html")
OUTPUT.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Phase III Outlier Table saved to: {OUTPUT.resolve()}")
