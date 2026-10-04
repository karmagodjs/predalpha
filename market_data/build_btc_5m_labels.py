from pathlib import Path

import pandas as pd

SOURCE = Path("data/reference/btcusdt_1m_oct2_oct3_resolution.csv")
OUTPUT = Path("data/processed/btc_5m_direction_labels.parquet")

df = pd.read_csv(SOURCE)
df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
df = df.sort_values("open_time").drop_duplicates("open_time")
df = df.set_index("open_time")

# Target is the close exactly 5 minutes after the current candle's open time.
df["target_time"] = df.index + pd.Timedelta(minutes=5)
df["target_close"] = df["close"].shift(-5)
df["current_close"] = df["close"]

df["price_change_5m"] = df["target_close"] - df["current_close"]
df["label_5m"] = "FLAT"
df.loc[df["price_change_5m"] > 0, "label_5m"] = "UP"
df.loc[df["price_change_5m"] < 0, "label_5m"] = "DOWN"

result = df.reset_index()[
    [
        "open_time",
        "current_close",
        "target_time",
        "target_close",
        "price_change_5m",
        "label_5m",
    ]
].dropna(subset=["target_close"])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_parquet(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Shape:", result.shape)
print("\nLabel counts:")
print(result["label_5m"].value_counts())
print("\nFirst rows:")
print(result.head().to_string(index=False))
print("\nLast row:")
print(result.tail(1).to_string(index=False))