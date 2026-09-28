import pandas as pd
from pathlib import Path

from training.labels import create_labels


INPUT_PATH = Path(
    "data/processed/market_data.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/labeled_market_data.parquet"
)


def main():

    print("=" * 60)
    print("PREDALPHA LABEL GENERATION")
    print("=" * 60)

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing: {INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"\nInput rows: {len(df)}"
    )

    # --------------------------------------------------
    # Create labels
    # --------------------------------------------------

    df = create_labels(
        df,
        horizon=50
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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
        f"\nRows: {len(df)}"
    )


if __name__ == "__main__":
    main()