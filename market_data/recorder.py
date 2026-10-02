import json
from datetime import datetime, timezone
from pathlib import Path


class MarketRecorder:

    def __init__(self, output_dir="data/raw",
                 filename="polymarket_extended.jsonl"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = self.output_dir / filename

    def record(self, asset_id, event):

        record = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "asset_id": str(asset_id),
            "event": event,
        }

        with self.filename.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")