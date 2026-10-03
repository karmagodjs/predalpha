from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features.parquet"
)

df = pd.read_parquet(PATH)

print("=== FEATURE AUDIT ===")
print("Rows:", len(df))
print("Unique mid_price_up:", df["mid_price_up"].nunique())
print("Unique mid_price_change:", df["mid_price_change"].nunique())
print("Unique spread_up:", df["spread_up"].nunique())
print("Unique spread_down:", df["spread_down"].nunique())
print("Unique spread_difference:", df["spread_difference"].nunique())

print("\n=== ZERO VARIANCE CHECK ===")
for col in df.select_dtypes(include=np.number).columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n=== NON-ZERO SPREAD DIFFERENCE ===")
print((df["spread_difference"].abs() > 1e-10).sum())

print("\n=== MID PRICE CHANGE COUNTS ===")
print(df["mid_price_change"].value_counts(dropna=False).sort_index())