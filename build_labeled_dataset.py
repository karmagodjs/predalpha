import pandas as pd
from pathlib import Path

ROOT = Path(".")
candle_path = ROOT / "data/reference/btcusdt_1m_oct3_2120_2130.csv"
paired_path = ROOT / "data/processed/btc_oct3_new_markets_paired.parquet"

candles = pd.read_csv(candle_path)
paired = pd.read_parquet(paired_path)

candles["open_time"] = pd.to_datetime(candles["open_time"], utc=True)
candles["close"] = pd.to_numeric(candles["close"])

markets = [
    {"epoch": 1791062400, "slug": "btc-updown-5m-1791062400"},
    {"epoch": 1791062700, "slug": "btc-updown-5m-1791062700"},
]

labels = []
for m in markets:
    start = pd.to_datetime(m["epoch"], unit="s", utc=True)
    end = start + pd.Timedelta(minutes=5)

    start_candle = candles.loc[candles["open_time"].eq(start)]
    end_candle = candles.loc[
        candles["open_time"].eq(end - pd.Timedelta(minutes=1))
    ]

    if start_candle.empty or end_candle.empty:
        raise ValueError(f"Missing boundary candle for {m['slug']}")

    start_price = float(start_candle.iloc[0]["open"])
    end_price = float(end_candle.iloc[0]["close"])
    change = end_price - start_price

    labels.append({
        "market": m["slug"],
        "market_start": start,
        "market_end": end,
        "btc_start_price": start_price,
        "btc_end_price": end_price,
        "price_change": change,
        "label": "UP" if change > 0 else ("DOWN" if change < 0 else "FLAT"),
        "label_source": "Binance BTCUSDT 1m candles (proxy, not Chainlink settlement)",
    })

label_df = pd.DataFrame(labels)
label_df.to_csv("data/processed/btc_oct3_new_market_labels.csv", index=False)

# Attach market label using the market epoch in source_file.
paired["source_file"] = paired["source_file"].astype(str)
paired["market_epoch"] = pd.NA

for m in markets:
    mask = paired["source_file"].str.contains(str(m["epoch"]), regex=False)
    paired.loc[mask, "market_epoch"] = m["epoch"]

label_map = {m["epoch"]: row for m, row in zip(markets, labels)}
paired["label"] = paired["market_epoch"].map(
    {epoch: row["label"] for epoch, row in label_map.items()}
)
paired["market"] = paired["market_epoch"].map(
    {m["epoch"]: m["slug"] for m in markets}
)

# Leakage / coverage checks
paired["timestamp"] = pd.to_datetime(paired["timestamp"], utc=True)
paired["market_start"] = paired["market_epoch"].map(
    {m["epoch"]: pd.to_datetime(m["epoch"], unit="s", utc=True) for m in markets}
)
paired["market_end"] = paired["market_start"] + pd.Timedelta(minutes=5)

unmapped = int(paired["market_epoch"].isna().sum())
outside = int((
    paired["market_start"].notna()
    & ~paired["timestamp"].between(paired["market_start"], paired["market_end"], inclusive="left")
).sum())
missing_labels = int(paired["label"].isna().sum())

# Exclude label/market metadata from feature list; no BTC future price fields are added.
metadata = {
    "label", "market", "market_epoch", "market_start", "market_end",
    "timestamp", "up_timestamp_ms", "down_timestamp_ms",
    "down_timestamp", "source_file"
}
feature_cols = [c for c in paired.columns if c not in metadata]

print("\nMARKET LABELS")
print(label_df.to_string(index=False))
print("\nCHECKS")
print("Paired rows:", len(paired))
print("Unmapped rows:", unmapped)
print("Rows outside market window:", outside)
print("Rows missing labels:", missing_labels)
print("Feature columns:", feature_cols)

if unmapped or outside or missing_labels:
    raise ValueError("Checks failed; dataset not saved.")

# Keep only rows with valid market mapping and label.
final = paired.drop(columns=["market_epoch", "market_start", "market_end"])
final.to_parquet("data/processed/btc_oct3_new_markets_labeled.parquet", index=False)

print("\nSaved:")
print("data/processed/btc_oct3_new_market_labels.csv")
print("data/processed/btc_oct3_new_markets_labeled.parquet")
print("\nRows per independent market:")
print(final.groupby(["market", "label"]).size().to_string())
print("\nWARNING: quote rows are correlated; only 2 independent market outcomes.")
