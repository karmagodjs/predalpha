import json
from pathlib import Path

from market_data.normalizer import MarketNormalizer

RAW_PATH = Path("data/raw/polymarket_extended.jsonl")

TARGET_ASSET_ID = (
    "32338220190071351435772801779725302244575775216413325951443816017994629993401"
)

normalizer = MarketNormalizer(TARGET_ASSET_ID)

checked = 0
mismatches = []
snapshots = 0

with RAW_PATH.open("r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, start=1):
        record = json.loads(line)
        raw_event = record.get("event")

        events = raw_event if isinstance(raw_event, list) else [raw_event]

        for event in events:
            if not isinstance(event, dict):
                continue

            snapshot = normalizer.process(event)

            if snapshot is None:
                continue

            snapshots += 1

            if event.get("event_type") != "price_change":
                continue

            for change in event.get("price_changes", []):
                if str(change.get("asset_id")) != TARGET_ASSET_ID:
                    continue

                checked += 1

                reported = (
                      float(change["best_bid"]),
                      float(change["best_ask"]),
                )
                reconstructed = (
                      float(snapshot["best_bid"]),
                      float(snapshot["best_ask"]),
                )
                if reported != reconstructed:
                    mismatches.append(
                        (line_number, reported, reconstructed)
                    )

print("Snapshots:", snapshots)
print("Target changes checked:", checked)
print("BBO mismatches:", len(mismatches))
print("First 10 mismatches:")
for item in mismatches[:10]:
    print(item)