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


def validate_market_data(path):
    df = pd.read_parquet(path)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["mid_price"].isna().any():
        raise ValueError("mid_price contains NaN.")

    if (df["mid_price"] <= 0).any():
        raise ValueError("mid_price contains non-positive values.")

    print("Rows:", len(df))
    print("Markets:", df["market_id"].nunique())
    print("Unique mid prices:", df["mid_price"].nunique())
    print("Unique bid prices:", df["best_bid"].nunique())
    print("Unique ask prices:", df["best_ask"].nunique())

    if df["mid_price"].nunique() <= 1:
        raise ValueError(
            "Mid-price is constant. "
            "Do not generate directional labels."
        )

    return df