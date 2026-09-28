import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "market_id",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread",
    "bid_depth",
    "ask_depth",
    "imbalance",
    "microprice",
]


def validate_data(path):

    df = pd.read_parquet(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        format="mixed"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if df["mid_price"].isna().any():
        raise ValueError(
            "Missing mid_price values."
        )

    if (df["mid_price"] <= 0).any():
        raise ValueError(
            "Invalid mid_price detected."
        )

    if (df["spread"] < 0).any():
        raise ValueError(
            "Negative spread detected."
        )

    return df