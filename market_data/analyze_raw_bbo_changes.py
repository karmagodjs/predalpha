import json
from pathlib import Path

import pandas as pd

INPUT = Path("data/raw/btc_88000_session_20261002_01.jsonl")
TOKEN_ID = (
    "41204045870451989441006259077692571105773543114590728783061562183667482414716"
)


def main():
    rows = []

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            event = record.get("event", {})

            if event.get("event_type") == "price_change":
                for change in event.get("price_changes", []):
                    if change.get("asset_id") != TOKEN_ID:
                        continue

                    bid = change.get("best_bid")
                    ask = change.get("best_ask")

                    if bid is None or ask is None:
                        continue

                    rows.append((
                        int(event["timestamp"]),
                        float(bid),
                        float(ask),
                    ))

    df = pd.DataFrame(rows, columns=["timestamp_ms", "bid", "ask"])
    df = df.sort_values("timestamp_ms").drop_duplicates(
        "timestamp_ms", keep="last"
    )

    changed = (
        df["bid"].ne(df["bid"].shift())
        | df["ask"].ne(df["ask"].shift())
    )

    print("Unique timestamps:", len(df))
    print("Actual BBO changes:", int(changed.sum()))
    print("Repeated BBO updates:", int((~changed).sum() - 1))
    print("Unique bid/ask pairs:", df[["bid", "ask"]].drop_duplicates().shape[0])
    print("First BBO:", df[["bid", "ask"]].iloc[0].to_dict())
    print("Last BBO:", df[["bid", "ask"]].iloc[-1].to_dict())


if __name__ == "__main__":
    main()