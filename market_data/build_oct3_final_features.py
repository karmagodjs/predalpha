from pathlib import Path

import pandas as pd

SOURCE = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features.parquet"
)

OUTPUT = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_final_features.parquet"
)

df = pd.read_parquet(SOURCE)

keep_columns = [
    "timestamp",
    "mid_price_up",
    "spread_up",
    "spread_down",
    "mid_price_change",
]

final_df = df[keep_columns].copy()
final_df.to_parquet(OUTPUT, index=False)

print("Final feature dataset:", final_df.shape)
print("Columns:", final_df.columns.tolist())
print("Missing values:")
print(final_df.isna().sum())
print("Saved:", OUTPUT)
print("Original dataset was not modified.")