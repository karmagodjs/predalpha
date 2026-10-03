import pandas as pd
from pathlib import Path

FILE = Path("data/processed/btc_88000_features_5s.parquet")

df = pd.read_parquet(FILE)

forbidden = {
    "future_mid",
    "future_delta",
    "label_threshold",
    "timestamp_ms",
}

feature_columns = [
    "mid_price",
    "spread",
    "spread_bps",
    "mid_return_1s",
    "mid_return_3s",
    "mid_return_5s",
    "mid_volatility_5s",
    "bid_change_1s",
    "ask_change_1s",
]

print("Forbidden columns present:", forbidden.intersection(feature_columns))
print("Missing feature columns:", set(feature_columns) - set(df.columns))
print("Feature NaN count:", df[feature_columns].isna().sum().sum())
print("Duplicate timestamps:", df.index.duplicated().sum())
print("Rows:", len(df))
print("Label counts:")
print(df["label"].value_counts())

assert not forbidden.intersection(feature_columns)
assert not (set(feature_columns) - set(df.columns))
assert df[feature_columns].isna().sum().sum() == 0
assert not df.index.duplicated().any()

print("\nBasic leakage checks passed.")