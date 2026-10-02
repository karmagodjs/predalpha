from pathlib import Path
import pandas as pd

from training.preprocessing import FEATURES

INPUT_PATH = Path("data/processed/feature_market_data.parquet")
OUTPUT_DIR = Path("data/processed/ablation")

FEATURE_GROUPS = {
    "all": FEATURES,
    "orderbook": [
        "best_bid", "best_ask", "spread",
        "bid_depth", "ask_depth", "imbalance", "microprice",
    ],
    "returns_momentum": [
        "return_1", "return_5", "return_10",
        "momentum_10", "volatility_10", "volatility_50",
    ],
    "microstructure_derived": [
        "spread_bps", "depth_imbalance", "microprice_deviation",
    ],
}

def main():
    df = pd.read_parquet(INPUT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for group_name, features in FEATURE_GROUPS.items():
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"Skipping {group_name}; missing columns: {missing}")
            continue

        columns = [
            c for c in df.columns
            if c not in FEATURES
        ] + features

        # Remove duplicate column names while preserving order
        columns = list(dict.fromkeys(columns))
        output = df[columns].copy()

        path = OUTPUT_DIR / f"{group_name}.parquet"
        output.to_parquet(path, index=False)

        print(f"{group_name}: {len(features)} features -> {path}")

if __name__ == "__main__":
    main()