import pandas as pd
from pathlib import Path

FILE = Path("data/processed/btc_88000_labeled_5s.parquet")

df = pd.read_parquet(FILE)

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Missing values:\n", df.isna().sum())

print("\nLabel counts:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\nFuture delta statistics:")
print(df["future_delta"].describe())

print("\nThreshold statistics:")
print(df["label_threshold"].describe())

print("\nLabel vs movement check:")
print(
    df.groupby("label")["future_delta"]
    .agg(["count", "mean", "min", "max"])
)