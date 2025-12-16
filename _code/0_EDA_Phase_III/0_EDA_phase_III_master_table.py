#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_master_table.py

Generates the Phase III Master Table for the transformed dataset:

Input:
    Meteorite_Landings_Phase_III.csv

Output:
    0_EDA_phase_III_master_table.html

Columns contained:
- Feature Name
- Pandas dtype
- Categorical / Numerical / Other
- # Unique
- % Missing
- Description
- Log Transform Summary
- Sqrt Transform Summary

Header color = #8FE29D
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load Phase III CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

rows, cols = df.shape

# ---------------------------------------------------------------------
# 2. Field descriptions
# ---------------------------------------------------------------------
descriptions = {
    "year": "Four-digit calendar year of meteorite record.",
    "count": "Number of meteorites recorded for the given year.",
    "count_log": "Log-transformed annual meteorite count: log(count + 1).",
    "count_sqrt": "Square-root-transformed annual meteorite count."
}

# ---------------------------------------------------------------------
# 3. Helper: classify data type
# ---------------------------------------------------------------------
def classify_dtype(dtype):
    if dtype in ["int64", "float64"]:
        return "Numerical"
    elif dtype == "object":
        return "Categorical"
    else:
        return "Other"

# ---------------------------------------------------------------------
# 4. Transformation summaries
# ---------------------------------------------------------------------
def summarize(series):
    return (
        f"mean={series.mean():.3f}, "
        f"std={series.std():.3f}, "
        f"min={series.min():.3f}, "
        f"max={series.max():.3f}"
    )

log_summary = summarize(df["count_log"])
sqrt_summary = summarize(df["count_sqrt"])

transform_summaries = {
    "count_log": log_summary,
    "count_sqrt": sqrt_summary,
    "count": "Not transformed",
    "year": "Not transformed"
}

# ---------------------------------------------------------------------
# 5. Build table dataframe
# ---------------------------------------------------------------------
records = []

for col in df.columns:

    dtype = str(df[col].dtype)
    category = classify_dtype(dtype)
    nunique = df[col].nunique()
    pct_missing = df[col].isna().mean() * 100
    desc = descriptions.get(col, "No description available.")
    transform_summary = transform_summaries[col]

    records.append({
        "Feature Name": col,
        "Pandas dtype": dtype,
        "Categorical / Numerical": category,
        "# Unique": nunique,
        "% Missing": f"{pct_missing:.2f}%",
        "Description": desc,
        "Transform Summary": transform_summary
    })

df_out = pd.DataFrame(records)

# ---------------------------------------------------------------------
# 6. Build HTML output
# ---------------------------------------------------------------------
html_top = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Phase III Master Table</title>

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
  margin-bottom: 8px;
}}

.subtitle {{
  text-align: center;
  font-size: 20px;
  margin-bottom: 18px;
  color: #333333;
}}

table.master {{
  width: 95%;
  margin-left: auto;
  margin-right: auto;
  border-collapse: separate;
  border-spacing: 0;
  border: 3px solid #000;
  border-radius: 12px;
  overflow: hidden;
  text-align: center;
}}

table.master th {{
  background: #8FE29D;
  padding: 8px;
  border: 1px solid #000;
  font-size: 18px;
  font-weight: bold;
  text-align: center;
}}

table.master td {{
  background: #f4e8d2;
  padding: 8px;
  border: 1px solid #000;
  font-size: 16px;
  text-align: center;
  vertical-align: middle;
}}

.footer {{
  text-align: center;
  margin-top: 14px;
  font-size: 16px;
}}

</style>
</head>

<body>

<div class="title">Phase III — Master Feature Table</div>
<div class="subtitle">Dataset Size: {rows:,} rows × {cols:,} columns</div>

<table class="master">
<thead>
<tr>
"""

# Add header columns
for col in df_out.columns:
    html_top += f"<th>{col}</th>"

html_top += "</tr></thead><tbody>\n"

html_mid = ""
for _, row in df_out.iterrows():
    html_mid += "<tr>\n"
    html_mid += f"<td>{row['Feature Name']}</td>\n"
    html_mid += f"<td>{row['Pandas dtype']}</td>\n"
    html_mid += f"<td>{row['Categorical / Numerical']}</td>\n"
    html_mid += f"<td>{row['# Unique']}</td>\n"
    html_mid += f"<td>{row['% Missing']}</td>\n"
    html_mid += f"<td>{row['Description']}</td>\n"
    html_mid += f"<td>{row['Transform Summary']}</td>\n"
    html_mid += "</tr>\n"

html_bottom = """
</tbody>
</table>

<div class="footer">
Table X. Summary of Phase III features, transformations, and distribution characteristics.
</div>

</body>
</html>
"""

OUTPUT = Path("0_EDA_phase_III_master_table.html")
OUTPUT.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Phase III Master Table saved to: {OUTPUT.resolve()}")
