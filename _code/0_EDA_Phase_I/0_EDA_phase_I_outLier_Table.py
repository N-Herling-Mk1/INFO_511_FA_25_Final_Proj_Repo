#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_I_outLier_Table.py

Creates an Outlier Summary Table (HTML) for annual meteorite counts.
Uses strict year filtering and consistent formatting with Phase I deliverables.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------------------------
INPUT_CSV = Path("Meteorite_Landings.csv")
df = pd.read_csv(INPUT_CSV)

# ---------------------------------------------------------------------
# 2. STRICT year validation (consistent with prior EDA scripts)
# ---------------------------------------------------------------------
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(1000, 3000)]
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# ---------------------------------------------------------------------
# 3. Compute yearly meteorite counts
# ---------------------------------------------------------------------
year_counts = df.groupby("year")["fall"].count().sort_index()
counts = year_counts.values

# ---------------------------------------------------------------------
# 4. Compute statistics for outlier table
# ---------------------------------------------------------------------
total_count = len(counts)
vmin        = counts.min()
vmax        = counts.max()
vmean       = counts.mean()
vstd        = counts.std()

q25         = np.percentile(counts, 25)
q50         = np.percentile(counts, 50)
q75         = np.percentile(counts, 75)

iqr         = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

within_iqr  = ((counts >= lower_bound) & (counts <= upper_bound)).sum()
pct_within  = within_iqr / total_count * 100
outliers    = total_count - within_iqr

# ---------------------------------------------------------------------
# 5. Build table for HTML
# ---------------------------------------------------------------------
table_data = {
    "Statistic": [
        "Total Count (years)",
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

# Dataset size subtitle
rows, cols = df.shape
dataset_banner = f"Dataset Size: {rows:,} rows × {cols:,} columns"

# ---------------------------------------------------------------------
# 6. HTML construction — with violet header & bold left column
# ---------------------------------------------------------------------
html_top = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Outlier Summary — Meteorite Landings</title>

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

  table.master th {{
    background: #dcd0ff;    /* violet header */
    padding: 6px 8px;
    border: 1px solid #000;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    white-space: nowrap;
  }}

  table.master td {{
    background: #f4e8d2;    /* beige cells */
    padding: 6px 8px;
    border: 1px solid #000;
    font-size: 16px;
    text-align: center;
    white-space: nowrap;
  }}

  /* Bold left-column for “Statistic” */
  table.master td.statcol {{
    font-weight: bold;
  }}

  /* Center footer stub */
  .footer {{
    font-size: 16px;
    margin-top: 14px;
    text-align: center;
  }}

</style>
</head>

<body>

<div class="title">Table 2<br>Outlier Summary for Annual Meteorite Counts<br>Raw data Meteorite Landings data set</div>
<div class="subtitle">{dataset_banner}</div>

<table class="master">
  <thead>
    <tr>
"""

# Add header cells
html_mid = ""
for col in df_out.columns:
    html_mid += f"      <th>{col}</th>\n"

html_mid += "    </tr>\n  </thead>\n  <tbody>\n"

# Add body rows: left column bold, right normal
for _, row in df_out.iterrows():
    html_mid += "    <tr>\n"
    html_mid += f"      <td class='statcol'>{row['Statistic']}</td>\n"
    html_mid += f"      <td>{row['Value']}</td>\n"
    html_mid += "    </tr>\n"

html_bottom = """
  </tbody>
</table>

<div class="footer"><b>Table 2.</b> Summary of central tendency, variability, percentiles, and IQR-based outliers.</div>

</body>
</html>
"""

# ---------------------------------------------------------------------
# 7. Save HTML file
# ---------------------------------------------------------------------
OUTPUT_HTML = Path("0_EDA_phase_I_outlier_table.html")
OUTPUT_HTML.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Outlier summary table saved to: {OUTPUT_HTML.resolve()}")
