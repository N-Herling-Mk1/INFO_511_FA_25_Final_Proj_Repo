#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0_EDA_phase_III_data_transform.py

Phase III transformation pipeline:

1. Load Phase II dataset
2. Remove outliers using IQR method
3. Apply recommended transformations:
       - log(count + 1)
       - sqrt(count)
4. Save cleaned + transformed dataset:
       Meteorite_Landings_Phase_III.csv

This matches the INFO 511 workflow and the conclusions from the
Data Topology Summary table (severe right skew, heavy tail).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# 1. Load Phase II
# ---------------------------------------------------------------------
INPUT = Path("Meteorite_Landings_Phase_II.csv")
df = pd.read_csv(INPUT)

if "count" not in df.columns or "year" not in df.columns:
    raise ValueError("Phase II dataset must contain 'year' and 'count' columns.")

counts = df["count"].astype(float)

# ---------------------------------------------------------------------
# 2. Compute IQR + Outlier Removal
# ---------------------------------------------------------------------
Q1 = counts.quantile(0.25)
Q3 = counts.quantile(0.75)
IQR = Q3
