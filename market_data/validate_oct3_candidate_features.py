from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features.parquet"
)

def main():
    df = pd.read_parquet(INPUT_FILE)

    print("=== DATASET CHECK ===")
    print("Shape:", df.shape)
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())

    print("\n=== MISSING VALUES ===")
    print(df.isna().sum().to_string())

    features = [
        c for c in df.columns
        if c != "timestamp"
    ]

    print("\n=== FEATURE DIAGNOSTICS ===")
    for col in features:
        print(
            f"{col}: unique={df[col].nunique(dropna=True)}, "
            f"std={df[col].std()}, "
            f"missing={df[col].isna().sum()}"
        )

    print("\n=== REDUNDANCY CHECK ===")
    if {"spread_up", "spread_down"}.issubset(df.columns):
        same = df["spread_up"].equals(df["spread_down"])
        print("spread_up == spread_down row-by-row:", same)

    print("\n=== TIMESTAMP CHECK ===")
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())

    print("\nValidation complete. Source file was not modified.")

if __name__ == "__main__":
    main()