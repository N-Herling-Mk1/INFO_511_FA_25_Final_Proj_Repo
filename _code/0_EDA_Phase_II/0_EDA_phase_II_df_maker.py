#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_II_df_maker.py

Creates a cleaned, reduced dataframe for Phase II EDA:
- Columns: [year, count]
- Removes invalid/missing year entries
- Removes rows outside valid year range [0, 2013]
- Removes missing fall entries
- Removes duplicate IDs
- Outputs: Meteorite_Landings_Phase_II.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load original CSV
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings.csv")
df = pd.read_csv(INPUT)

# ---------------------------------------------------------------------
# 2. CLEANING STEPS
# ---------------------------------------------------------------------

# -- Remove duplicate IDs
df = df.drop_duplicates(subset="id", keep="first")

# -- coerce year to numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")

# -- remove rows without valid year
df = df.dropna(subset=["year"])

# -- convert to int
df["year"] = df["year"].astype(int)

# -- remove rows outside valid range
df = df[df["year"].between(0, 2013)]

# -- remove rows missing 'fall'
df = df.dropna(subset=["fall"])

# ---------------------------------------------------------------------
# 3. GROUP BY YEAR → compute counts
# ---------------------------------------------------------------------
year_counts = (
    df.groupby("year")["fall"]
    .count()
    .reset_index()
    .rename(columns={"fall": "count"})
)

# ---------------------------------------------------------------------
# 4. OUTPUT CSV
# ---------------------------------------------------------------------
OUTPUT = Path("Meteorite_Landings_Phase_II.csv")
year_counts.to_csv(OUTPUT, index=False)

print("✔ Phase II dataset created:")
print(f"  {OUTPUT.resolve()}")
print("\nPreview:")
print(year_counts.head())
