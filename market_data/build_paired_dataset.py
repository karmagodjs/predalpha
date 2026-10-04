from pathlib import Path

import pandas as pd


INPUT = Path("data/processed/btc_oct3_new_markets_bbo.parquet")
OUTPUT = Path("data/processed/btc_oct3_new_markets_paired.parquet")

UP_IDS = {
    "3935334638023970449521487164625699030243727246671252017327689520869105178865",
    "53223802987003930157743840797336064433306955525648436966678907114094880700180",
}

DOWN_IDS = {
    "8929977175814223488184521824094266419231719119229288923275745416162192339520",
    "6594558601693281968695410784179822844560789691061915058860046439686803853338",
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT}")

    df = pd.read_parquet(INPUT)
    df["asset_id"] = df["asset_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df["outcome"] = "UNKNOWN"
    df.loc[df["asset_id"].isin(UP_IDS), "outcome"] = "UP"
    df.loc[df["asset_id"].isin(DOWN_IDS), "outcome"] = "DOWN"

    unknown = df.loc[df["outcome"] == "UNKNOWN", "asset_id"].unique()
    if len(unknown):
        raise ValueError(f"Unknown asset IDs found: {unknown.tolist()}")

    paired_markets = []

    for source_file, market in df.groupby("source_file"):
        up = market[market["outcome"] == "UP"].copy()
        down = market[market["outcome"] == "DOWN"].copy()

        if up.empty or down.empty:
            print(f"Skipping {source_file}: UP={len(up)}, DOWN={len(down)}")
            continue

        up = up.sort_values("timestamp")
        down = down.sort_values("timestamp")

        up = up[
            ["timestamp", "timestamp_ms", "bid", "ask", "mid_price", "spread"]
        ].rename(columns={
            "timestamp_ms": "up_timestamp_ms",
            "bid": "up_bid",
            "ask": "up_ask",
            "mid_price": "up_mid_price",
            "spread": "up_spread",
        })

        down = down[
            ["timestamp", "timestamp_ms", "bid", "ask", "mid_price", "spread"]
        ].rename(columns={
            "timestamp": "down_timestamp",
            "timestamp_ms": "down_timestamp_ms",
            "bid": "down_bid",
            "ask": "down_ask",
            "mid_price": "down_mid_price",
            "spread": "down_spread",
        })

        # Pair each UP quote with the latest DOWN quote within 5 seconds.
        paired = pd.merge_asof(
            up,
            down.sort_values("down_timestamp"),
            left_on="timestamp",
            right_on="down_timestamp",
            direction="backward",
            tolerance=pd.Timedelta(seconds=5),
        )

        paired["source_file"] = source_file
        paired["pair_age_ms"] = (
            paired["timestamp"] - paired["down_timestamp"]
        ).dt.total_seconds() * 1000

        paired["up_down_price_diff"] = (
            paired["up_mid_price"] - paired["down_mid_price"]
        )
        paired["combined_mid_price"] = (
            paired["up_mid_price"] + paired["down_mid_price"]
        )
        paired["combined_spread"] = (
            paired["up_spread"] + paired["down_spread"]
        )

        paired = paired.sort_values("timestamp")
        paired["up_mid_change"] = paired["up_mid_price"].diff()
        paired["down_mid_change"] = paired["down_mid_price"].diff()

        paired_markets.append(paired)

        matched = paired["down_mid_price"].notna().sum()
        print(
            f"{source_file}: UP rows={len(up)}, "
            f"matched pairs={matched}, unmatched={len(up) - matched}"
        )

    if not paired_markets:
        raise RuntimeError("No paired rows generated.")

    result = pd.concat(paired_markets, ignore_index=True)
    result = result.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT, index=False)

    print("\nPAIRED DATASET CREATED")
    print(f"File: {OUTPUT}")
    print(f"Rows: {len(result)}")
    print(f"Matched: {result['down_mid_price'].notna().sum()}")
    print(f"Unmatched: {result['down_mid_price'].isna().sum()}")
    print(f"Columns: {result.columns.tolist()}")


if __name__ == "__main__":
    main()