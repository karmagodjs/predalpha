import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/raw/polymarket_orderbook.jsonl"
)


def main():

    print("=" * 60)
    print("PREDALPHA DATA VALIDATION")
    print("=" * 60)

    if not INPUT.exists():
        print(f"\nERROR: {INPUT} not found.")
        return

    df = pd.read_json(
        INPUT,
        lines=True
    )

    print(f"\nRows: {len(df)}")

    # --------------------------------------------------
    # Basic columns
    # --------------------------------------------------

    required = [
        "timestamp",
        "asset_id",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "bid_depth",
        "ask_depth",
    ]

    print("\nMissing columns:")

    missing = [
        col for col in required
        if col not in df.columns
    ]

    print(missing if missing else "None")

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\nMissing values:")

    print(
        df[required]
        .isna()
        .sum()
    )

    # --------------------------------------------------
    # Asset count
    # --------------------------------------------------

    print("\nUnique assets:")

    print(
        df["asset_id"].nunique()
    )

    # --------------------------------------------------
    # Price statistics
    # --------------------------------------------------

    print("\nPrice statistics:")

    print(
        df[
            [
                "best_bid",
                "best_ask",
                "mid_price",
                "spread",
            ]
        ].describe()
    )

    # --------------------------------------------------
    # Unique prices
    # --------------------------------------------------

    print("\nUnique values:")

    print(
        "Best bid:",
        df["best_bid"].nunique()
    )

    print(
        "Best ask:",
        df["best_ask"].nunique()
    )

    print(
        "Mid price:",
        df["mid_price"].nunique()
    )

    print(
        "Spread:",
        df["spread"].nunique()
    )

    # --------------------------------------------------
    # Book integrity
    # --------------------------------------------------

    invalid_books = (
        df["best_bid"]
        > df["best_ask"]
    ).sum()

    negative_spreads = (
        df["spread"] < 0
    ).sum()

    print("\nOrder book integrity:")

    print(
        "Invalid bid > ask:",
        invalid_books
    )

    print(
        "Negative spreads:",
        negative_spreads
    )

    # --------------------------------------------------
    # Mid price movement
    # --------------------------------------------------

    df["mid_change"] = (
        df["mid_price"]
        .diff()
    )

    price_changes = (
        df["mid_change"]
        .ne(0)
        .sum()
    )

    print("\nMid-price movement:")

    print(
        "Price changes:",
        price_changes
    )

    if len(df) > 0:

        print(
            "Price-change ratio:",
            f"{price_changes / len(df):.4%}"
        )

    # --------------------------------------------------
    # Return statistics
    # --------------------------------------------------

    df["return"] = (
        df["mid_price"]
        .pct_change()
    )

    print("\nReturn statistics:")

    print(
        df["return"]
        .describe()
    )

    # --------------------------------------------------
    # Depth statistics
    # --------------------------------------------------

    print("\nDepth statistics:")

    print(
        df[
            [
                "bid_depth",
                "ask_depth",
            ]
        ].describe()
    )

    # --------------------------------------------------
    # Final decision
    # --------------------------------------------------

    print("\n" + "=" * 60)

    if invalid_books > 0:

        print(
            "❌ FAIL: Invalid order books detected."
        )

    elif negative_spreads > 0:

        print(
            "❌ FAIL: Negative spreads detected."
        )

    elif price_changes == 0:

        print(
            "⚠️ WARNING: No mid-price movement."
        )

    else:

        print(
            "✅ DATA HAS PRICE MOVEMENT"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()