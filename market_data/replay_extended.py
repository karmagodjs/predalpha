import json
from pathlib import Path

from market_data.normalizer import MarketNormalizer
from market_data.parquet_writer import ParquetWriter


RAW_PATH = Path("data/raw/polymarket_multi_10m.jsonl")
OUTPUT_PATH = Path("data/processed/market_data_multi_10m.parquet")

ASSET_IDS = [
    # Pete
    "91031279171981959197361710127213577102576826515163085396576756946418341946256",
    "29006371059590547574472177233842141554614007974283936149779962779213443973844",

    # Pritzker
    "103988345069738240821107916249290759786442703283901164377120414690701074137717",
    "18494065527924073750217237765784380365430319396356251245973222791140100738948",

    # Booker
    "50887272939612765629559172143901565817521391945540156085421963433918821328137",
    "108650988487983371520585351370999192979899647080710024226285750050382033453124",
]


def build_multi_dataset():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    normalizers = {
        asset_id: MarketNormalizer(asset_id)
        for asset_id in ASSET_IDS
    }

    snapshots = []
    record_count = 0
    event_count = 0
    skipped_count = 0

    with RAW_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Invalid JSON at line {line_number}")
                skipped_count += 1
                continue

            event_data = record.get("event")

            if isinstance(event_data, dict):
                events = [event_data]
            elif isinstance(event_data, list):
                events = event_data
            else:
                skipped_count += 1
                continue

            for event in events:
                if not isinstance(event, dict):
                    skipped_count += 1
                    continue

                event_count += 1

                for normalizer in normalizers.values():
                    snapshot = normalizer.process(event)
                    if snapshot is not None:
                        snapshots.append(snapshot)

    print("\n" + "=" * 45)
    print("MULTI-TOKEN REPLAY SUMMARY")
    print("=" * 45)
    print(f"Raw records       : {record_count}")
    print(f"Events read       : {event_count}")
    print(f"Snapshots created : {len(snapshots)}")
    print(f"Skipped items     : {skipped_count}")

    if not snapshots:
        raise RuntimeError("No snapshots created. Check raw data and asset IDs.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = ParquetWriter(str(OUTPUT_PATH))
    writer.write(snapshots)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_multi_dataset()