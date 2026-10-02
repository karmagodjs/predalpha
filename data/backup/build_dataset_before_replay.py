import json
from pathlib import Path

from market_data.normalizer import MarketNormalizer
from market_data.parquet_writer import ParquetWriter


RAW_PATH = Path(
    "data/raw/polymarket_raw.jsonl"
)

OUTPUT_PATH = Path(
    "data/processed/market_data.parquet"
)


def build_dataset(asset_id: str, output_path: Path = OUTPUT_PATH):

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_PATH}"
        )

    normalizer = MarketNormalizer(
        asset_id
    )

    snapshots = []

    total_records = 0
    total_events = 0
    processed_events = 0

    with open(
        RAW_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Skipping invalid JSON at line "
                    f"{total_records}"
                )
                continue

            data = record.get(
                "data"
            )

            # Case 1: one event
            if isinstance(data, dict):

                events = [data]

            # Case 2: multiple events
            elif isinstance(data, list):

                events = data

            else:

                print(
                    f"Skipping unsupported data type: "
                    f"{type(data).__name__}"
                )

                continue

            for event in events:

                if not isinstance(event, dict):
                    continue

                total_events += 1

                snapshot = normalizer.process(
                    event
                )

                if snapshot is not None:

                    snapshots.append(
                        snapshot
                    )

                    processed_events += 1

    print()
    print("=" * 60)
    print("DATASET BUILD")
    print("=" * 60)

    print(
        f"Raw records       : {total_records}"
    )

    print(
        f"Events processed   : {total_events}"
    )

    print(
        f"Snapshots created  : {processed_events}"
    )

    if not snapshots:

        print()
        print(
            "ERROR: No snapshots generated."
        )

        print(
            "Check the asset ID and raw data."
        )

        return

    writer = ParquetWriter(
        str(output_path)
    )

    writer.write(
        snapshots
    )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    print(
        "Dataset created successfully."
    )


if __name__ == "__main__":

    asset_id = input(
        "Enter asset ID: "
    ).strip()

    if not asset_id:
        raise ValueError(
            "Asset ID cannot be empty."
        )

    build_dataset(
        asset_id
    )