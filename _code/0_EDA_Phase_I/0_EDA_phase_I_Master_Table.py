#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_I_Master_Table.py

Reads Meteorite_Landings.csv and constructs the Master Table
with formatting and HTML rendering.

Output:
    master_table.html
"""

import pandas as pd
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# 1. INPUT CSV
# ---------------------------------------------------------------------
INPUT_CSV = Path("Meteorite_Landings.csv")
OUTPUT_HTML = Path("master_table.html")

df = pd.read_csv(INPUT_CSV)

# Dataset dimensions
rows, cols = df.shape
dataset_banner = f"Dataset Size: {rows:,} rows × {cols:,} columns"

# ---------------------------------------------------------------------
# 2. DESCRIPTION MAP (clean ASCII-safe text)
# ---------------------------------------------------------------------
DESCRIPTION_MAP = {
    "name": "Name of the meteorite as recorded in the catalog.",
    "id": "Unique numeric identifier assigned to each meteorite record.",
    "nametype": "Indicates valid meteorite names (Valid) or paired/duplicate names (Relict).",
    "recclass": "Classification based on chemical and petrological type.",
    "mass (g)": "Reported mass of the meteorite in grams.",
    "fall": "Indicates whether the meteorite was Fell (observed fall) or Found.",
    "year": "Year the meteorite was found or fell.",
    "reclat": "Latitude of the recovery site.",
    "reclong": "Longitude of the recovery site.",
    "GeoLocation": "Coordinate pair representing the recovery location (latitude, longitude)."
}

# ---------------------------------------------------------------------
# 3. Helper function: determine categorical/numerical/other
# ---------------------------------------------------------------------
def classify_type(dtype):
    if pd.api.types.is_numeric_dtype(dtype):
        return "Numerical"
    elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        return "Categorical"
    else:
        return "Other"

# ---------------------------------------------------------------------
# 4. Build Master Table
# ---------------------------------------------------------------------
table_rows = []

for col in df.columns:
    dtype = df[col].dtype
    cat_num_other = classify_type(dtype)

    unique_count = df[col].nunique(dropna=True)
    missing_pct = df[col].isna().mean() * 100

    description = DESCRIPTION_MAP.get(col, "No description available.")

    table_rows.append({
        "Feature Name": col,
        "Pandas dType": str(dtype),
        "Categorical / Numerical": cat_num_other,
        "# Unique": unique_count,
        "% Missing": f"{missing_pct:.2f}%",
        "Description": description
    })

master_df = pd.DataFrame(table_rows)

# ---------------------------------------------------------------------
# 5. HTML Styling — centered text + dataset banner
# ---------------------------------------------------------------------
html_top = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Master Table — Phase I</title>
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

  .footer {{
    font-size: 16px;
    margin-top: 12px;
  }}

  table.master {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border: 3px solid #000000;
    border-radius: 12px;
    overflow: hidden;
  }}

  table.master th {{
    background: #94e19c;
    padding: 10px;
    border: 1px solid #000;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
  }}

  table.master td {{
    background: #f4e8d2;
    padding: 10px;
    border: 1px solid #000;
    font-size: 16px;
    text-align: center;
  }}

</style>
</head>

<body>

<div class="title">Table 1 — Meteorite Landings Raw Data Table</div>
<div class="subtitle">{dataset_banner}</div>

<table class="master">
  <thead>
    <tr>
"""

# Add headers
html_mid = ""
for col in master_df.columns:
    html_mid += f"      <th>{col}</th>\n"

html_mid += "    </tr>\n  </thead>\n  <tbody>\n"

# Add rows
for _, row in master_df.iterrows():
    html_mid += "    <tr>\n"
    for col in master_df.columns:
        html_mid += f"      <td>{row[col]}</td>\n"
    html_mid += "    </tr>\n"

html_bottom = """
  </tbody>
</table>

<div class="footer"><b>Table 1.</b> A Summary of the raw 'Meteorite Landgings' dataset features.</div>

</body>
</html>
"""

# ---------------------------------------------------------------------
# 6. Write HTML file
# ---------------------------------------------------------------------
OUTPUT_HTML.write_text(html_top + html_mid + html_bottom, encoding="utf-8")

print(f"✔ Master table written to: {OUTPUT_HTML.resolve()}")
