from pathlib import Path

from training.validate_data import validate_data
from training.labels import create_labels
from training.split import temporal_split
from training.preprocessing import FeatureScaler


INPUT_PATH = (
    "data/processed/market_data.parquet"
)

OUTPUT_DIR = Path(
    "data/processed"
)


def main():

    print("Loading data...")

    df = validate_data(
        INPUT_PATH
    )

    print(
        f"Raw samples: {len(df)}"
    )

    print("Creating labels...")

    df = create_labels(
        df,
        horizon=50,
        threshold=None,
    )

    print(
        f"Labeled samples: {len(df)}"
    )

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    train, val, test = temporal_split(
        df
    )

    print("\nSplit:")

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_parquet(
        OUTPUT_DIR / "train.parquet",
        index=False
    )

    val.to_parquet(
        OUTPUT_DIR / "val.parquet",
        index=False
    )

    test.to_parquet(
        OUTPUT_DIR / "test.parquet",
        index=False
    )

    print("\nSaved:")

    print(
        "data/processed/train.parquet"
    )

    print(
        "data/processed/val.parquet"
    )

    print(
        "data/processed/test.parquet"
    )


if __name__ == "__main__":
    main()