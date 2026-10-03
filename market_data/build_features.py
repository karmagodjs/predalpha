from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/processed/btc_88000_session_20261002_01_labeled_5s.parquet"
)
OUTPUT = Path(
    "data/processed/btc_88000_session_20261002_01_features_5s.parquet"
)


def main():
    df = pd.read_parquet(INPUT).sort_index()

    # Current market state
    df["mid_return_1s"] = df["mid_price"].pct_change(1)
    df["mid_return_3s"] = df["mid_price"].pct_change(3)
    df["mid_return_5s"] = df["mid_price"].pct_change(5)

    df["spread_bps"] = (
        df["spread"] / df["mid_price"]
    ) * 10000

    df["mid_volatility_5s"] = (
        df["mid_price"].pct_change()
        .rolling(5)
        .std()
    )

    df["bid_change_1s"] = df["bid"].diff()
    df["ask_change_1s"] = df["ask"].diff()

    # Preserve target separately
    target = df["label"]

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

    df = df.dropna(subset=feature_columns).copy()
    df["label"] = target.loc[df.index]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)

    print("Feature rows:", len(df))
    print("Features:", feature_columns)
    print("Label counts:")
    print(df["label"].value_counts())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()