#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_IV_presentation_table.py

Generates an HTML table summarizing Phase III features
with the following columns:

- Feature Name
- Pandas dType
- Categorical / Numerical
- # Unique
- % Missing
- Description
- Regression Role (split into two lines if parentheses exist)

Outputs:
    0_EDA_phase_IV_presentation_table.html
"""

import pandas as pd
from pathlib import Path
import re

# ----------------------------------------------------------
# Load Phase III data
# ----------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_III.csv")
df = pd.read_csv(INPUT)

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def pct_missing(series):
    return f"{series.isna().mean() * 100:.2f}%"

def num_unique(series):
    return series.nunique()

# ----------------------------------------------------------
# Metadata (copied from your PNG table)
# ----------------------------------------------------------
descriptions = {
    "year": "Four-digit calendar year of meteorite record.",
    "count": "Number of meteorites recorded for the given year.",
    "count_log": "Log-transformed annual meteorite count: log(count + 1).",
    "count_sqrt": "Square-root-transformed annual meteorite count."
}

regression_role = {
    "year": "Predictor (X)",
    "count": "Outcome (Y, raw form)",
    "count_log": "Outcome (Y, log model)",
    "count_sqrt": "Outcome (Y, sqrt model)"
}


# ----------------------------------------------------------
# Split the parentheses part into a second line
# ----------------------------------------------------------
def split_role(text):
    """
    Turn 'Outcome (Y, raw form)' → 'Outcome<br>(Y, raw form)'
    Works for ANY 'word (stuff)' pattern.
    """
    match = re.match(r"^(.*)\s*\((.*)\)$", text)
    if match:
        main, paren = match.groups()
        return f"{main}<br>({paren})"
    return text


# ----------------------------------------------------------
# Build the table rows
# ----------------------------------------------------------
rows = []

for col in ["year", "count", "count_log", "count_sqrt"]:
    role = split_role(regression_role[col])

    rows.append({
        "Feature Name": col,
        "Pandas dType": str(df[col].dtype),
        "Categorical/Numerical": "Numerical",
        "# Unique": num_unique(df[col]),
        "% Missing": pct_missing(df[col]),
        "Description": descriptions[col],
        "Regression Role": role
    })

df_table = pd.DataFrame(rows)


# ----------------------------------------------------------
# Convert DataFrame → HTML
# ----------------------------------------------------------
def df_to_html(df):
    html = "<table>\n<tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>\n"

    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            html += f"<td>{row[col]}</td>"
        html += "</tr>\n"
    html += "</table>"
    return html


# ----------------------------------------------------------
# CSS + HTML Structure
# ----------------------------------------------------------
html_output = f"""
<!DOCTYPE html>
<html>
<head>
<title>Phase III — Master Feature Table</title>

<style>
body {{
    font-family: Arial, sans-serif;
    padding: 30px;
}}

h1 {{
    text-align: center;
    margin-bottom: 5px;
}}

.subtitle {{
    text-align:center;
    font-size: 18px;
    margin-bottom: 25px;
}}

/* ───── Table Styling ───── */

table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 3px solid black; /* THICK border */
    font-size: 16px;
}}

th {{
    background: #3CB371;
    border: 1px solid black;
    padding: 10px;
    font-weight: bold;
    text-align: center;
    vertical-align: middle;   /* ← center vertically */
}}

td {{
    background: #F6EFD9;
    border: 1px solid black;
    padding: 10px;
    text-align: center;        /* ← center horizontally */
    vertical-align: middle;    /* ← center vertically */
}}

caption {{
    caption-side: bottom;
    padding: 10px;
    font-style: italic;
}}
</style>

</head>
<body>

<h1>Phase III — Master Feature Table</h1>
<div class="subtitle">Dataset Size: {df.shape[0]} rows × {df.shape[1]} columns</div>

{df_to_html(df_table)}

<br>
<center><i>Table X. Summary of Phase III features, transformations, and distribution characteristics.</i></center>

</body>
</html>
"""

# ----------------------------------------------------------
# Write Output
# ----------------------------------------------------------
OUTPUT = Path("0_EDA_phase_IV_presentation_table.html")
OUTPUT.write_text(html_output, encoding="utf-8")

print("✔ HTML table generated.")
print(f"  → {OUTPUT.resolve()}")
