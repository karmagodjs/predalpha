from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "data/processed/btc_oct3_up_down_session_20261002_04_bbo.parquet"
)
OUTPUT = Path(
    "data/processed/btc_oct3_up_down_session_20261002_04_features.parquet"
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "outcome", "bid", "ask", "mid_price", "spread"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(["outcome", "timestamp"]).copy()

    # Calculate separately for UP and DOWN tokens.
    grouped = df.groupby("outcome", group_keys=False)

    df["spread_bps"] = np.where(
        df["mid_price"].ne(0),
        (df["spread"] / df["mid_price"]) * 10_000,
        np.nan,
    )

    df["mid_return"] = grouped["mid_price"].pct_change()
    df["bid_change"] = grouped["bid"].diff()
    df["ask_change"] = grouped["ask"].diff()

    # Time-aware rolling volatility: 60-second window.
    df = df.set_index("timestamp", drop=False)

    df["volatility_60s"] = (
        df.groupby("outcome")["mid_return"]
        .rolling("60s", min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
    )

    df = df.reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def main():
    df = pd.read_parquet(INPUT)
    features = build_features(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT, index=False)

    print("Input rows:", len(df))
    print("Feature rows:", len(features))
    print("Columns:", features.columns.tolist())
    print("Rows by outcome:")
    print(features["outcome"].value_counts())
    print("Missing feature values:")
    print(
        features[
            ["spread_bps", "mid_return", "bid_change",
             "ask_change", "volatility_60s"]
        ].isna().sum()
    )
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()