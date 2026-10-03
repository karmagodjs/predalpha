import json
from pathlib import Path

import pandas as pd

INPUT = Path("data/raw/btc_oct3_up_down_session_20261002_04.jsonl")
OUTPUT = Path("data/processed/btc_oct3_up_down_session_20261002_04_bbo.parquet")

TOKEN_NAMES = {
    "32664731353322490016663940157951165038326534411865610104571929420891932752044": "UP",
    "83368144530639073518031297526937851295808514385655078316894643930693648829402": "DOWN",
}


def main():
    rows = []

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            event = record.get("event", {})

            if event.get("event_type") != "book":
                continue

            asset_id = str(event.get("asset_id", record.get("asset_id", "")))
            if asset_id not in TOKEN_NAMES:
                continue

            bids = event.get("bids", [])
            asks = event.get("asks", [])

            if not bids or not asks:
                continue

            bid = max(float(x["price"]) for x in bids)
            ask = min(float(x["price"]) for x in asks)

            if bid > ask:
                continue

            rows.append({
                "timestamp_ms": int(event["timestamp"]),
                "asset_id": asset_id,
                "outcome": TOKEN_NAMES[asset_id],
                "bid": bid,
                "ask": ask,
            })

    if not rows:
        raise RuntimeError("No valid BBO rows found.")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(
        df["timestamp_ms"], unit="ms", utc=True
    )

    df = (
        df.sort_values(["asset_id", "timestamp_ms"])
        .drop_duplicates(["asset_id", "timestamp_ms"], keep="last")
        .reset_index(drop=True)
    )

    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["spread"] = df["ask"] - df["bid"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print("Actual BBO rows:", len(df))
    print("Rows by outcome:")
    print(df["outcome"].value_counts())
    print("Unique mid prices by outcome:")
    print(df.groupby("outcome")["mid_price"].nunique())
    print("Time range by outcome:")
    print(df.groupby("outcome")["timestamp"].agg(["min", "max"]))
    print("Saved:", OUTPUT)
    print(df.head())


if __name__ == "__main__":
    main()