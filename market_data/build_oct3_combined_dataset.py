from pathlib import Path
import pandas as pd

BASE = Path("data/processed")

INDIVIDUAL = BASE / "btc_oct3_up_down_session_20261002_04_features.parquet"
PAIRED = BASE / "btc_oct3_up_down_session_20261002_04_paired_features.parquet"
OUTPUT = BASE / "btc_oct3_up_down_session_20261002_04_combined.parquet"


def main():
    individual = pd.read_parquet(INDIVIDUAL)
    paired = pd.read_parquet(PAIRED)

    individual = individual.drop(columns=["timestamp_ms", "asset_id"], errors="ignore")

    combined = individual.merge(
        paired,
        on="timestamp",
        how="inner",
        suffixes=("_individual", "_paired"),
    )

    combined = combined.sort_values(["timestamp", "outcome"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT, index=False)

    print("Individual rows:", len(individual))
    print("Paired rows:", len(paired))
    print("Combined rows:", len(combined))
    print("Outcomes:")
    print(combined["outcome"].value_counts())
    print("Missing values:", combined.isna().sum().sum())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()