import pandas as pd
from pathlib import Path

files = [
    "btc_5m_direction_labels.parquet",
    "btc_oct3_aligned_5m_dataset.parquet",
    "btc_oct3_event_5m_dataset.parquet",
    "btc_oct3_unique_target_dataset.parquet",
    "btc_oct3_resolution_label.csv",
]

for name in files:
    path = Path("data/processed") / name
    print(f"\n### {name}")

    if not path.exists():
        print("MISSING")
        continue

    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst rows:")
    print(df.head(3).to_string(index=False))
    print("\nLast rows:")
    print(df.tail(3).to_string(index=False))

    for col in df.columns:
        if any(word in col.lower() for word in ("label", "direction", "target", "outcome")):
            print(f"\nCounts for {col}:")
            print(df[col].value_counts(dropna=False).to_dict())
