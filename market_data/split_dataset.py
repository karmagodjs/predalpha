from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/processed/btc_88000_session_20261002_01_features_5s.parquet"
)
OUT = Path(
    "data/processed/splits/btc_88000_session_20261002_01"
)

PURGE_GAP = 5  # 5 rows = 5 seconds for this 1-second dataset


def main():
    df = pd.read_parquet(INPUT).sort_index()

    n = len(df)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Purge rows at both split boundaries
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end + PURGE_GAP:val_end].copy()
    test = df.iloc[val_end + PURGE_GAP:].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A split is empty. Reduce PURGE_GAP or check dataset size.")

    OUT.mkdir(parents=True, exist_ok=True)

    train.to_parquet(OUT / "train.parquet")
    val.to_parquet(OUT / "validation.parquet")
    test.to_parquet(OUT / "test.parquet")

    for name, part in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(f"\n{name}: {len(part)} rows")
        print(part["label"].value_counts())
        print("Start:", part.index.min())
        print("End:", part.index.max())

    # Verify chronological order
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()

    # Verify at least 5 seconds between split boundaries
    train_val_gap = (
        val.index.min() - train.index.max()
    ).total_seconds()
    val_test_gap = (
        test.index.min() - val.index.max()
    ).total_seconds()

    print("\nTrain → Validation gap:", train_val_gap, "seconds")
    print("Validation → Test gap:", val_test_gap, "seconds")

    assert train_val_gap >= PURGE_GAP
    assert val_test_gap >= PURGE_GAP

    print("\nPurged chronological split passed.")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()