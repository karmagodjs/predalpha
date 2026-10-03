from pathlib import Path
import pandas as pd

INPUT = Path(
    "data/processed/btc_oct3_up_down_session_20261002_04_combined.parquet"
)

def main():
    df = pd.read_parquet(INPUT)

    feature_cols = [
        "bid", "ask", "mid_price", "spread", "spread_bps",
        "mid_return", "bid_change", "ask_change", "volatility_60s",
        "bid_down", "bid_up", "ask_down", "ask_up",
        "mid_price_down", "mid_price_up",
        "spread_down", "spread_up",
        "price_difference", "combined_spread", "up_price_deviation",
    ]

    available = [col for col in feature_cols if col in df.columns]
    corr = df[available].corr(method="pearson")

    output = Path("data/processed/btc_oct3_up_down_session_20261002_04_correlations.csv")
    corr.to_csv(output)

    print("Rows:", len(df))
    print("Paired timestamps:", df["timestamp"].nunique())
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())
    print("\nSaved:", output)

if __name__ == "__main__":
    main()