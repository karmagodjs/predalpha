from pathlib import Path

import pandas as pd

SOURCE = Path("data/processed/btc_oct3_event_5m_dataset.parquet")
OUTPUT = Path("data/processed/btc_oct3_unique_target_dataset.parquet")

df = pd.read_parquet(SOURCE)

# For each target, retain the latest feature event available
# before that target's 5-minute cutoff.
df = df[df["timestamp"] < df["target_cutoff"]].copy()
df = df.sort_values("timestamp")

unique = (
    df.groupby("target_time", as_index=False)
    .tail(1)
    .sort_values("timestamp")
    .reset_index(drop=True)
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
unique.to_parquet(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Shape:", unique.shape)
print("Unique targets:", unique["target_time"].nunique())
print("\nLabels:")
print(unique["label_5m"].value_counts())
print("\nTarget duplicates:", unique.duplicated("target_time").sum())
print("\nSample:")
print(
    unique[
        [
            "timestamp",
            "target_cutoff",
            "target_time",
            "actual_horizon_seconds",
            "label_5m",
        ]
    ].head(10).to_string(index=False)
)