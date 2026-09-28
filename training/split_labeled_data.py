import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/labeled_market_data.parquet"
)

OUTPUT_DIR = Path(
    "data/processed"
)


def temporal_split(
    df,
    train_ratio=0.70,
    val_ratio=0.15,
):

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    n = len(df)

    train_end = int(
        n * train_ratio
    )

    val_end = int(
        n * (train_ratio + val_ratio)
    )

    train = df.iloc[
        :train_end
    ].copy()

    val = df.iloc[
        train_end:val_end
    ].copy()

    test = df.iloc[
        val_end:
    ].copy()

    return train, val, test


def print_split_info(name, df):

    print(f"\n{name}")
    print("-" * 40)

    print(
        "Rows:",
        len(df)
    )

    print(
        "Start:",
        df["timestamp"].min()
    )

    print(
        "End:",
        df["timestamp"].max()
    )

    print(
        "Labels:"
    )

    print(
        df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )


def main():

    print("=" * 60)
    print("PREDALPHA TEMPORAL SPLIT")
    print("=" * 60)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing: {INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"\nTotal rows: {len(df)}"
    )

    train, val, test = temporal_split(
        df
    )

    print_split_info(
        "TRAIN",
        train
    )

    print_split_info(
        "VALIDATION",
        val
    )

    print_split_info(
        "TEST",
        test
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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