from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/btc_oct3_up_down_session_20261002_04_bbo.parquet")
OUTPUT = Path("data/processed/btc_oct3_up_down_session_20261002_04_paired_features.parquet")


def main():
    df = pd.read_parquet(INPUT)

    wide = df.pivot(
        index="timestamp",
        columns="outcome",
        values=["bid", "ask", "mid_price", "spread"],
    )

    wide.columns = [f"{metric}_{outcome.lower()}" for metric, outcome in wide.columns]
    wide = wide.reset_index()

    wide["price_difference"] = wide["mid_price_up"] - wide["mid_price_down"]
    wide["combined_spread"] = wide["spread_up"] + wide["spread_down"]
    wide["up_price_deviation"] = wide["mid_price_up"] - 0.5

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUTPUT, index=False)

    print("Paired rows:", len(wide))
    print("Columns:", wide.columns.tolist())
    print("Missing values:")
    print(wide.isna().sum())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()