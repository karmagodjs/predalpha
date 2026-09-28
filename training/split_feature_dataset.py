import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/processed/feature_market_data.parquet"
)

OUTPUT = Path(
    "data/processed"
)


def main():

    df = pd.read_parquet(INPUT)

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train = df.iloc[
        :train_end
    ]

    val = df.iloc[
        train_end:val_end
    ]

    test = df.iloc[
        val_end:
    ]

    train.to_parquet(
        OUTPUT / "train_features.parquet",
        index=False,
    )

    val.to_parquet(
        OUTPUT / "val_features.parquet",
        index=False,
    )

    test.to_parquet(
        OUTPUT / "test_features.parquet",
        index=False,
    )

    print("TRAIN:", len(train))
    print("VAL:", len(val))
    print("TEST:", len(test))

    print("\nTime ranges:")

    print(
        "TRAIN:",
        train.timestamp.min(),
        "→",
        train.timestamp.max()
    )

    print(
        "VAL:",
        val.timestamp.min(),
        "→",
        val.timestamp.max()
    )

    print(
        "TEST:",
        test.timestamp.min(),
        "→",
        test.timestamp.max()
    )


if __name__ == "__main__":
    main()