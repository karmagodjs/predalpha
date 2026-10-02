import json
from datetime import datetime, timezone
from pathlib import Path


class MarketRecorder:

    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.filename = (
            self.output_dir /
            "polymarket_orderbook.jsonl"
        )

    def record(
        self,
        asset_id,
        snapshot,
        event=None
    ):

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        record = {
            "timestamp": timestamp,
            "asset_id": asset_id,
            "best_bid": snapshot.get(
                "best_bid"
            ),
            "best_ask": snapshot.get(
                "best_ask"
            ),
            "mid_price": snapshot.get(
                "mid_price"
            ),
            "spread": snapshot.get(
                "spread"
            ),
            "bid_depth": snapshot.get(
                "bid_depth"
            ),
            "ask_depth": snapshot.get(
                "ask_depth"
            ),
            "event": event,
        }

        with open(
            self.filename,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(record) + "\n"
            )