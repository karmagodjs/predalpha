import pandas as pd
from pathlib import Path

from training.preprocessing import (
    FeatureScaler,
    FEATURES,
)


DATA_DIR = Path("data/processed")


def main():

    print("=" * 60)
    print("PHASE 4 — LEAKAGE-SAFE FEATURE SCALING")
    print("=" * 60)

    train = pd.read_parquet(
        DATA_DIR / "train_features.parquet"
    )

    val = pd.read_parquet(
        DATA_DIR / "val_features.parquet"
    )

    test = pd.read_parquet(
        DATA_DIR / "test_features.parquet"
    )

    scaler = FeatureScaler()

    train = scaler.fit_transform(train)

    val = scaler.transform(val)

    test = scaler.transform(test)

    train.to_parquet(
        DATA_DIR / "train_features_scaled.parquet",
        index=False,
    )

    val.to_parquet(
        DATA_DIR / "val_features_scaled.parquet",
        index=False,
    )

    test.to_parquet(
        DATA_DIR / "test_features_scaled.parquet",
        index=False,
    )

    print("\nSaved:")
    print("train_features_scaled.parquet")
    print("val_features_scaled.parquet")
    print("test_features_scaled.parquet")

    print("\nFeatures:")
    print(len(FEATURES))


if __name__ == "__main__":
    main()