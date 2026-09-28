import pandas as pd
from pathlib import Path

from training.preprocessing import FeatureScaler, FEATURES


DATA_DIR = Path(
    "data/processed"
)


def main():

    print("=" * 60)
    print("PREDALPHA FEATURE SCALING")
    print("=" * 60)

    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "val.parquet"
    test_path = DATA_DIR / "test.parquet"

    train = pd.read_parquet(
        train_path
    )

    val = pd.read_parquet(
        val_path
    )

    test = pd.read_parquet(
        test_path
    )

    print(
        f"\nTrain: {len(train)}"
    )

    print(
        f"Validation: {len(val)}"
    )

    print(
        f"Test: {len(test)}"
    )

    # --------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------

    scaler = FeatureScaler()

    train = scaler.fit_transform(
        train
    )

    val = scaler.transform(
        val
    )

    test = scaler.transform(
        test
    )

    # --------------------------------------------------
    # Verify
    # --------------------------------------------------

    print("\nFeature means from TRAIN:")

    print(
        train[FEATURES]
        .mean()
        .round(4)
    )

    print("\nFeature std from TRAIN:")

    print(
        train[FEATURES]
        .std()
        .round(4)
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    train.to_parquet(
        DATA_DIR / "train_scaled.parquet",
        index=False
    )

    val.to_parquet(
        DATA_DIR / "val_scaled.parquet",
        index=False
    )

    test.to_parquet(
        DATA_DIR / "test_scaled.parquet",
        index=False
    )

    print("\nSaved:")

    print(
        "train_scaled.parquet"
    )

    print(
        "val_scaled.parquet"
    )

    print(
        "test_scaled.parquet"
    )


if __name__ == "__main__":
    main()