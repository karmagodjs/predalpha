import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/polymarket_orderbook.jsonl")
OUTPUT_PATH = Path("data/processed/market_data.parquet")


def main():

    print("=" * 60)
    print("PredAlpha | Building Market Dataset")
    print("=" * 60)

    print("\nLoading raw market data...")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw data file: {RAW_PATH}"
        )

    # --------------------------------------------------
    # Load raw data
    # --------------------------------------------------

    df = pd.read_json(
        RAW_PATH,
        lines=True
    )

    print(f"Raw rows: {len(df)}")

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    if "timestamp" not in df.columns:
        raise ValueError(
            "Required column 'timestamp' not found."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        format="mixed"
    )

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    required_columns = [
        "asset_id",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "bid_depth",
        "ask_depth",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # Numeric columns
    # --------------------------------------------------

    numeric_columns = [
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "bid_depth",
        "ask_depth",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # Sort by timestamp
    # --------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=[
            "timestamp",
            "asset_id",
            "best_bid",
            "best_ask",
            "mid_price",
            "spread",
            "bid_depth",
            "ask_depth",
        ]
    )

    # Valid order book condition
    df = df[
        df["best_bid"] <= df["best_ask"]
    ]

    # Positive prices
    df = df[
        df["mid_price"] > 0
    ]

    # Non-negative depth
    df = df[
        (df["bid_depth"] >= 0)
        & (df["ask_depth"] >= 0)
    ]

    print(
        f"Removed rows: {before - len(df)}"
    )

    # --------------------------------------------------
    # Microstructure features
    # --------------------------------------------------

    total_depth = (
        df["bid_depth"]
        + df["ask_depth"]
    )

    # Order-book imbalance
    df["imbalance"] = (
        (
            df["bid_depth"]
            - df["ask_depth"]
        )
        / total_depth.replace(0, pd.NA)
    )

    # --------------------------------------------------
    # Microprice
    # --------------------------------------------------

    denominator = (
        df["bid_depth"]
        + df["ask_depth"]
    )

    df["microprice"] = (
        (
            df["best_ask"] * df["bid_depth"]
            +
            df["best_bid"] * df["ask_depth"]
        )
        / denominator.replace(0, pd.NA)
    )

    # --------------------------------------------------
    # Price return
    # --------------------------------------------------

    df["return_1"] = (
        df["mid_price"]
        .pct_change()
    )

    # --------------------------------------------------
    # Remove raw event payload
    # --------------------------------------------------

    # The raw event column can contain mixed
    # list/dict structures. It is not required
    # for model training and causes PyArrow
    # serialization errors.
    if "event" in df.columns:
        df = df.drop(
            columns=["event"]
        )

    # --------------------------------------------------
    # Remove other object columns that are not needed
    # --------------------------------------------------

    # Keep asset_id because it identifies the market.
    # All other raw object columns are unnecessary
    # for the numerical ML pipeline.

    allowed_columns = [
        "timestamp",
        "asset_id",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "bid_depth",
        "ask_depth",
        "imbalance",
        "microprice",
        "return_1",
    ]

    existing_columns = [
        column
        for column in allowed_columns
        if column in df.columns
    ]

    df = df[existing_columns]

    # --------------------------------------------------
    # Final cleanup
    # --------------------------------------------------

    df = (
        df
        .replace([float("inf"), float("-inf")], pd.NA)
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print(f"\nOutput:")
    print(OUTPUT_PATH)

    print(f"\nFinal rows:")
    print(len(df))

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nPrice statistics:")
    print(
        f"  Unique mid-price      : {df['mid_price'].nunique()}"
    )
    print(
        f"  Unique microprice     : {df['microprice'].nunique()}"
    )
    print(
        f"  Unique best bid       : {df['best_bid'].nunique()}"
    )
    print(
        f"  Unique best ask       : {df['best_ask'].nunique()}"
    )

    print("\nTime range:")
    print(
        f"  Start: {df['timestamp'].min()}"
    )
    print(
        f"  End  : {df['timestamp'].max()}"
    )

    print("\nDataset preview:")
    print(
        df.head(5).to_string(index=False)
    )

    print("\nDataset saved successfully.")


if __name__ == "__main__":
    main()