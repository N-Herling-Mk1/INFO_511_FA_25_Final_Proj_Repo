#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_II_Master_Table.py

Creates a Master Table (HTML) describing the cleaned Phase II dataset.
Inputs: Meteorite_Landings_Phase_II.csv
Outputs: 0_EDA_phase_II_Master_Table.html
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load Phase II CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_II.csv")
df = pd.read_csv(INPUT)

rows, cols = df.shape

# ---------------------------------------------------------------------
# 2. Build Master Table Columns
# ---------------------------------------------------------------------
feature_names = df.columns.tolist()
dtypes = df.dtypes

# Determine categorical/numerical/other
def classify_dtype(dtype):
    if np.issubdtype(dtype, np.number):
        return "Numerical"
    elif dtype == "object":
        return "Categorical"
    else:
        return "Other"

cn_types = [classify_dtype(t) for t in dtypes]

num_unique = [df[col].nunique() for col in df.columns]
pct_missing = [(df[col].isna().sum() / len(df) * 100) for col in df.columns]

# Descriptions for Phase II dataset
descriptions = {
    "year": "4-digit year indicating the recorded meteorite event.",
    "count": "Number of meteorites recorded for that year."
}

table_data = {
    "Feature Name": feature_names,
    "Pandas dType": [str(t) for t in dtypes],
    "Categorical / Numerical": cn_types,
    "# Unique": num_unique,
    "% Missing": [f"{x:.2f}%" for x in pct_missing],
    "Description": [descriptions.get(col, "") for col in df.columns]
}

df_out = pd.DataFrame(table_data)

# ---------------------------------------------------------------------
# 3. Generate HTML
# ---------------------------------------------------------------------
html_top = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase II Master Table</title>

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

  /* GREEN HEADER */
  table.master th {{
    background: #8fe29d;     /* light green header */
    padding: 6px 8px;
    border: 1px solid #000;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
  }}

  table.master td {{
    background: #f4e8d2;     /* beige cells */
    padding: 6px 8px;
    border: 1px solid #000;
    font-size: 16px;
    text-align: center;
  }}

  .footer {{
    font-size: 16px;
    margin-top: 14px;
    text-align: center;
  }}

</style>
</head>

<body>

<div class="title">Table 3 — Phase II Master Table</div>
<div class="subtitle">Dataset Size: {rows:,} rows × {cols:,} columns</div>

<table class="master">
  <thead>
    <tr>
"""

# Add table headers
html_mid = ""
for col in df_out.columns:
    html_mid += f"      <th>{col}</th>\n"

html_mid += "    </tr>\n  </thead>\n  <tbody>\n"

# Add rows
for _, row in df_out.iterrows():
    html_mid += "    <tr>\n"
    for col in df_out.columns:
        html_mid += f"      <td>{row[col]}</td>\n"
    html_mid += "    </tr>\n"

html_bottom = """
  </tbody>
</table>

<div class="footer"><b>Table 3.</b> Summary of features for the Phase II cleaned dataset.</div>

</body>
</html>
"""

# ---------------------------------------------------------------------
# 4. Save HTML file
# ---------------------------------------------------------------------
OUTPUT = Path("0_EDA_phase_II_Master_Table.html")
OUTPUT.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Phase II Master Table saved to: {OUTPUT.resolve()}")
