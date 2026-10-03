import pandas as pd

INPUT_FILE = (
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_paired_features.parquet"
)


def main():
    df = pd.read_parquet(INPUT_FILE)
    features = df.select_dtypes(include="number")

    corr = features.corr(method="pearson")

    print("=== HIGH CORRELATION PAIRS (|r| >= 0.95) ===")
    found = False

    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1:]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= 0.95:
                print(f"{col_a} <-> {col_b}: {value:.4f}")
                found = True

    if not found:
        print("No pairs found.")

    print("\n=== FEATURE VARIANCE ===")
    print(features.nunique().sort_values().to_string())

    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()