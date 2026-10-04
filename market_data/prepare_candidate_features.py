from pathlib import Path
import pandas as pd

SOURCE = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features.parquet"
)
OUTPUT = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_candidate_features_clean.parquet"
)

def main():
    df = pd.read_parquet(SOURCE).copy()

    # Drop floating-point noise / non-informative feature
    df = df.drop(columns=["spread_difference"], errors="ignore")

    # First-row price change can be NaN; remove that row
    df = df.dropna(subset=["mid_price_change"]).reset_index(drop=True)

    df.to_parquet(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print("Clean shape:", df.shape)
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Features:", [c for c in df.columns if c != "timestamp"])

if __name__ == "__main__":
    main()
    