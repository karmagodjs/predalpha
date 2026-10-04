import json
from pathlib import Path

import pandas as pd


INPUT_FILES = [
    Path("data/raw/btc-updown-5m-1791062400.jsonl"),
    Path("data/raw/btc-updown-5m-1791062700.jsonl"),
]

OUTPUT = Path("data/processed/btc_oct3_new_markets_bbo.parquet")


def get_outcome_map():
    """
    Read asset IDs from the captured files.
    Each file contains exactly two assets.
    The recorder stores no explicit UP/DOWN mapping,
    so mapping must be supplied from market metadata.
    """
    return {}


def main():
    rows = []

    for input_file in INPUT_FILES:
        if not input_file.exists():
            print(f"Skipping missing file: {input_file}")
            continue

        print(f"Reading: {input_file}")

        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                event = record.get("event", {})

                if event.get("event_type") != "book":
                    continue

                asset_id = str(
                    event.get("asset_id", record.get("asset_id", ""))
                )

                bids = event.get("bids", [])
                asks = event.get("asks", [])

                if not bids or not asks:
                    continue

                try:
                    bid = max(float(x["price"]) for x in bids)
                    ask = min(float(x["price"]) for x in asks)
                    timestamp_ms = int(event["timestamp"])
                except (KeyError, TypeError, ValueError):
                    continue

                if bid > ask:
                    continue

                rows.append({
                    "timestamp_ms": timestamp_ms,
                    "asset_id": asset_id,
                    "bid": bid,
                    "ask": ask,
                    "source_file": input_file.name,
                })

    if not rows:
        raise RuntimeError("No valid BBO rows found.")

    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(
        df["timestamp_ms"],
        unit="ms",
        utc=True,
    )

    df = (
        df.sort_values(["asset_id", "timestamp_ms"])
        .drop_duplicates(
            ["asset_id", "timestamp_ms"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print("\nBBO dataset created")
    print("Rows:", len(df))
    print("Unique assets:", df["asset_id"].nunique())
    print("Rows by source:")
    print(df["source_file"].value_counts())
    print("Rows by asset:")
    print(df.groupby("asset_id").size())
    print("Time range:")
    print(df["timestamp"].agg(["min", "max"]))
    print("Unique mid prices:")
    print(df.groupby("asset_id")["mid_price"].nunique())
    print("Saved:", OUTPUT)
    print(df.head())


if __name__ == "__main__":
    main()