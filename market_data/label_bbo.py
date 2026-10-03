from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/processed/btc_88000_session_20261002_01_bbo_1s.parquet"
)
OUTPUT = Path(
    "data/processed/btc_88000_session_20261002_01_labeled_5s.parquet"
)

HORIZON_SECONDS = 5
MIN_TICK = 0.001


def main():
    df = pd.read_parquet(INPUT).sort_index()

    required = {"mid_price", "spread"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Dataset must have regular 1-second intervals.
    expected = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1s",
        tz="UTC",
    )
    if not df.index.equals(expected):
        raise ValueError("Timestamps are not continuous 1-second intervals.")

    # Mid-price exactly 5 seconds into the future.
    df["future_mid"] = df["mid_price"].shift(-HORIZON_SECONDS)
    df["future_delta"] = df["future_mid"] - df["mid_price"]

    # Ignore movements smaller than half the current spread,
    # but never use a threshold below one price tick.
    df["label_threshold"] = (df["spread"] / 2).clip(lower=MIN_TICK)

    df["label"] = "FLAT"
    df.loc[df["future_delta"] > df["label_threshold"], "label"] = "UP"
    df.loc[df["future_delta"] < -df["label_threshold"], "label"] = "DOWN"

    # Last 5 rows have no known future price: remove them.
    df = df.dropna(subset=["future_mid", "future_delta"]).copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT)

    print("Input rows:", len(pd.read_parquet(INPUT)))
    print("Labeled rows:", len(df))
    print("Horizon:", HORIZON_SECONDS, "seconds")
    print("Minimum threshold:", MIN_TICK)

    print("\nLabel counts:")
    print(df["label"].value_counts())

    print("\nLabel percentages:")
    print((df["label"].value_counts(normalize=True) * 100).round(2))

    print("\nSample:")
    print(
        df[
            [
                "mid_price",
                "future_mid",
                "future_delta",
                "label_threshold",
                "label",
            ]
        ].head(10)
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()