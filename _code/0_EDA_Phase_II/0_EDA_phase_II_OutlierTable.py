#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_II_OutlierTable.py

Generates an Outlier Summary Table for the Phase II dataset.
Inputs: Meteorite_Landings_Phase_II.csv
Outputs: 0_EDA_phase_II_OutlierTable.html

Computes:
- total count (rows)
- min / max
- mean
- std dev
- 25th, 50th, 75th percentiles
- % within IQR
- number of outliers
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load Phase II CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_II.csv")
df = pd.read_csv(INPUT)

if "count" not in df.columns:
    raise ValueError("Phase II CSV must contain 'year' and 'count' columns.")

counts = df["count"].values
rows, cols = df.shape

# ---------------------------------------------------------------------
# 2. Compute Statistics
# ---------------------------------------------------------------------
total_count = len(counts)
vmin        = counts.min()
vmax        = counts.max()
vmean       = counts.mean()
vstd        = counts.std()

q25 = np.percentile(counts, 25)
q50 = np.percentile(counts, 50)
q75 = np.percentile(counts, 75)

iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

within_iqr = ((counts >= lower_bound) & (counts <= upper_bound)).sum()
pct_within = within_iqr / total_count * 100
outliers   = total_count - within_iqr

# Build table content
table_data = {
    "Statistic": [
        "Total Years in Dataset",
        "Min (annual count)",
        "Max (annual count)",
        "Mean",
        "Standard Deviation",
        "25th Percentile",
        "50th Percentile (Median)",
        "75th Percentile",
        "% Within IQR Range",
        "Number of Outliers"
    ],
    "Value": [
        total_count,
        vmin,
        vmax,
        f"{vmean:.3f}",
        f"{vstd:.3f}",
        q25,
        q50,
        q75,
        f"{pct_within:.2f}%",
        outliers
    ]
}

df_out = pd.DataFrame(table_data)

dataset_banner = f"Dataset Size: {rows:,} rows × {cols:,} columns"

# ---------------------------------------------------------------------
# 3. Generate HTML
# ---------------------------------------------------------------------
html_top = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase II Outlier Summary Table</title>

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

  /* Violet header background */
  table.master th {{
    background: #dcd0ff;
    padding: 6px 8px;
    border: 1px solid #000;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    white-space: nowrap;
  }}

  /* Beige data cells */
  table.master td {{
    background: #f4e8d2;
    padding: 6px 8px;
    border: 1px solid #000;
    font-size: 16px;
    text-align: center;
    white-space: nowrap;
  }}

  /* Bold left column */
  table.master td.statcol {{
    font-weight: bold;
  }}

  .footer {{
    font-size: 16px;
    margin-top: 14px;
    text-align: center;
  }}

</style>
</head>

<body>

<div class="title">Table 4 — Phase II Outlier Summary</div>
<div class="subtitle">{dataset_banner}</div>

<table class="master">
  <thead>
    <tr>
"""

# Add column headers
html_mid = ""
for col in df_out.columns:
    html_mid += f"      <th>{col}</th>\n"

html_mid += "    </tr>\n  </thead>\n  <tbody>\n"

# Add table rows
for _, row in df_out.iterrows():
    html_mid += "    <tr>\n"
    html_mid += f"      <td class='statcol'>{row['Statistic']}</td>\n"
    html_mid += f"      <td>{row['Value']}</td>\n"
    html_mid += "    </tr>\n"

# Footer
html_bottom = """
  </tbody>
</table>

<div class="footer"><b>Table 4.</b> Summary of distribution, central tendency, percentile ranges, and IQR-based outliers for the Phase II dataset.</div>

</body>
</html>
"""

# ---------------------------------------------------------------------
# 4. Save HTML File
# ---------------------------------------------------------------------
OUTPUT = Path("0_EDA_phase_II_OutlierTable.html")
OUTPUT.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Phase II Outlier Table saved to: {OUTPUT.resolve()}")
