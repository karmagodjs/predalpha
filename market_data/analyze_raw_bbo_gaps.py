import json
from pathlib import Path

import pandas as pd

INPUT = Path("data/raw/btc_88000_session_20261002_01.jsonl")
TOKEN_ID = (
    "41204045870451989441006259077692571105773543114590728783061562183667482414716"
)


def main():
    timestamps = []

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            event = record.get("event", {})

            if event.get("event_type") == "price_change":
                for change in event.get("price_changes", []):
                    if change.get("asset_id") != TOKEN_ID:
                        continue

                    if change.get("best_bid") is None or change.get("best_ask") is None:
                        continue

                    timestamps.append(
                        pd.to_datetime(
                            int(event["timestamp"]),
                            unit="ms",
                            utc=True,
                        )
                    )

            elif (
                event.get("event_type") == "book"
                and event.get("asset_id") == TOKEN_ID
                and event.get("bids")
                and event.get("asks")
            ):
                timestamps.append(
                    pd.to_datetime(
                        int(event["timestamp"]),
                        unit="ms",
                        utc=True,
                    )
                )

    times = pd.Series(timestamps).drop_duplicates().sort_values()
    gaps = times.diff().dt.total_seconds().dropna()

    print("Unique raw BBO update timestamps:", len(times))
    print("Median gap (seconds):", gaps.median())
    print("95th percentile gap (seconds):", gaps.quantile(0.95))
    print("Maximum gap (seconds):", gaps.max())
    print("Gaps >= 5 seconds:", int((gaps >= 5).sum()))
    print("Gaps >= 10 seconds:", int((gaps >= 10).sum()))
    print("Gaps >= 30 seconds:", int((gaps >= 30).sum()))


if __name__ == "__main__":
    main()