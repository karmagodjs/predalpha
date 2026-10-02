import pandas as pd

from training.walk_forward import (
    walk_forward_split,
)


INPUT_PATH = (
    "data/processed/feature_market_data.parquet"
)


def main():

    print("=" * 60)
    print("PREDALPHA WALK-FORWARD VALIDATION")
    print("=" * 60)

    df = pd.read_parquet(INPUT_PATH)

    print("Total rows:", len(df))

    splits = walk_forward_split(
        df,
        n_splits=3,
        initial_train_ratio=0.40,
        validation_ratio=0.10,
        test_ratio=0.10,
        purge_gap=20,
    )

    for split in splits:

        fold = split["fold"]
        train = split["train"]
        val = split["validation"]
        test = split["test"]

        print(f"\n{'=' * 50}")
        print(f"FOLD {fold}")
        print("=" * 50)

        print("Train rows:", len(train))
        print("Validation rows:", len(val))
        print("Test rows:", len(test))

        print("\nTime ranges:")

        print(
            "Train:",
            train["timestamp"].min(),
            "→",
            train["timestamp"].max(),
        )

        print(
            "Validation:",
            val["timestamp"].min(),
            "→",
            val["timestamp"].max(),
        )

        print(
            "Test:",
            test["timestamp"].min(),
            "→",
            test["timestamp"].max(),
        )

        assert (
            train["timestamp"].max()
            < val["timestamp"].min()
        )

        assert (
            val["timestamp"].max()
            < test["timestamp"].min()
        )

    print("\nAll chronological checks passed.")


if __name__ == "__main__":
    main()