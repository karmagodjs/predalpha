import pandas as pd
from pathlib import Path

from features.feature_engineering import (
    build_features
)


INPUT_PATH = Path(
    "data/processed/market_data.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/feature_market_data.parquet"
)


def main():

    print("=" * 60)
    print("PREDALPHA FEATURE DATASET")
    print("=" * 60)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing: {INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        "\nInput rows:",
        len(df)
    )

    print(
        "\nBuilding advanced features..."
    )

    df = build_features(df)

    # Remove rows created by rolling windows
    before = len(df)

    df = df.dropna(
        subset=[
            "return_10",
            "volatility_10",
            "volatility_50",
            "momentum_10",
        ]
    ).reset_index(drop=True)

    print(
        "Removed rows:",
        before - len(df)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nFinal rows:",
        len(df)
    )

    print(
        "\nFeature columns:"
    )

    print(
        df.columns.tolist()
    )


if __name__ == "__main__":
    main()