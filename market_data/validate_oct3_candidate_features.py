from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features.parquet"
)


def main():
    df = pd.read_parquet(INPUT_FILE)

    print("=== DATASET CHECK ===")
    print("Shape:", df.shape)
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\n=== FEATURE STATS ===")
    print(df.describe().to_string())

    print("\n=== TIMESTAMP CHECK ===")
    print("First:", df["timestamp"].min())
    print("Last:", df["timestamp"].max())

    print("\nValidation complete. Source file was not modified.")


if __name__ == "__main__":
    main()