from pathlib import Path

from training.validate_data import validate_data
from training.labels import create_labels
from training.split import temporal_split
from training.preprocessing import FeatureScaler


INPUT_PATH = (
    "data/processed/market_data_test.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/audit"
)


def main():

    print("Loading data...")

    df = validate_data(INPUT_PATH)
    assert df["timestamp"].is_monotonic_increasing, "Data is not chronological!"

    print(f"Raw samples: {len(df)}")

    # Split raw chronological data BEFORE label generation
    train_raw, val_raw, test_raw = temporal_split(df)

    print("\nRaw split:")
    print("Train:", len(train_raw))
    print("Validation:", len(val_raw))
    print("Test:", len(test_raw))

    # Generate labels independently inside each split
    print("\nCreating labels...")

    train = create_labels(
        train_raw,
        horizon=100,
        threshold=3e-6,
    )

    val = create_labels(
        val_raw,
        horizon=100,
        threshold=3e-6,
    )

    test = create_labels(
        test_raw,
        horizon=100,
        threshold=3e-6,
    )

    print("\nLabeled split:")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    print("\nClass distribution:")
    for name, split_df in [
        ("Train", train),
        ("Validation", val),
        ("Test", test),
    ]:
        print(f"\n{name}:")
        print(
            split_df["label"]
            .value_counts()
            .sort_index()
        )

    # Fit scaler only on train
    scaler = FeatureScaler()

    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print("\nSaved datasets to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()