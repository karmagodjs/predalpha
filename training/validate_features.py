import pandas as pd

from training.preprocessing import FEATURES


PATH = (
    "data/processed/"
    "feature_market_data.parquet"
)


def main():

    df = pd.read_parquet(PATH)

    print("=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)

    print(
        "\nRows:",
        len(df)
    )

    print(
        "\nFeature count:",
        len(FEATURES)
    )

    print(
        "\nMissing values:"
    )

    print(
        df[FEATURES]
        .isna()
        .sum()
    )

    print(
        "\nFeature statistics:"
    )

    print(
        df[FEATURES]
        .describe()
        .T[
            ["mean", "std", "min", "max"]
        ]
    )

    print(
        "\nUnique values:"
    )

    for feature in FEATURES:

        print(
            f"{feature:25s}",
            df[feature].nunique()
        )


if __name__ == "__main__":
    main()