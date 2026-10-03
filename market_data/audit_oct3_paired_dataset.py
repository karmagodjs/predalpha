from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/btc_oct3_up_down_session_20261002_04_paired_features.parquet"
)


def main():
    df = pd.read_parquet(INPUT_FILE)

    print("=== DATASET ===")
    print("Shape:", df.shape)
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
    print("Missing values:")
    print(df.isna().sum().to_string())

    print("\n=== TIMESTAMP COVERAGE ===")
    ts = pd.to_datetime(df["timestamp"], utc=True).sort_values()
    gaps = ts.diff().dt.total_seconds().dropna()

    print("First:", ts.min())
    print("Last:", ts.max())

    if not gaps.empty:
        print("Median gap (seconds):", gaps.median())
        print("Max gap (seconds):", gaps.max())
        print("Gaps > 30 seconds:", int((gaps > 30).sum()))

    print("\n=== FEATURE CHECKS ===")
    checks = {
        "mid_price_sum": df["mid_price_up"] + df["mid_price_down"],
        "spread_sum": df["spread_up"] + df["spread_down"],
        "price_difference_check": df["mid_price_up"] - df["mid_price_down"],
    }

    for name, values in checks.items():
        print(
            f"{name}: min={values.min():.6f}, "
            f"max={values.max():.6f}, "
            f"unique={values.nunique()}"
        )

    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        if col != "timestamp":
            print(f"{col}: {df[col].nunique()}")

    print("\nAudit complete. Original dataset was not modified.")


if __name__ == "__main__":
    main()