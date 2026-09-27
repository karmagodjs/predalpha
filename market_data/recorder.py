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

    def record(self, data):

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        filename = (
            self.output_dir /
            "polymarket_raw.jsonl"
        )

        record = {
            "timestamp": timestamp,
            "data": data
        }

        with open(
            filename,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(record) + "\n"
            )