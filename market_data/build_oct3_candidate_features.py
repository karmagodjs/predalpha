from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_paired_features.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features.parquet"
)


def main():
    df = pd.read_parquet(INPUT_FILE)

    # Keep one representative from complementary price columns.
    # Keep both outcome spreads because their values can differ.
    candidate_columns = [
        "timestamp",
        "mid_price_up",
        "spread_up",
        "spread_down",
    ]

    result = df[candidate_columns].copy()

    # Derived features for later experiments.
    result["mid_price_change"] = result["mid_price_up"].diff()
    result["spread_difference"] = (
        result["spread_up"] - result["spread_down"]
    )

    result.to_parquet(OUTPUT_FILE, index=False)

    print("Candidate dataset shape:", result.shape)
    print("Columns:", result.columns.tolist())
    print("Missing values:")
    print(result.isna().sum().to_string())
    print("Saved:", OUTPUT_FILE)
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()