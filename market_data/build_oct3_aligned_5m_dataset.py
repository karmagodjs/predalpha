from pathlib import Path

import pandas as pd

FEATURE_PATH = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_final_features.parquet"
)
BTC_PATH = Path("data/reference/btcusdt_1m_oct2_oct3_resolution.csv")
OUTPUT_PATH = Path(
    "data/processed/"
    "btc_oct3_aligned_5m_dataset.parquet"
)

features = pd.read_parquet(FEATURE_PATH)
btc = pd.read_csv(BTC_PATH)

features["timestamp"] = pd.to_datetime(
    features["timestamp"], utc=True
).astype("datetime64[ns, UTC]")

btc["open_time"] = pd.to_datetime(
    btc["open_time"], utc=True
).astype("datetime64[ns, UTC]")

btc["close_time"] = pd.to_datetime(
    btc["close_time"], utc=True
).astype("datetime64[ns, UTC]")

# Map each feature event to the latest fully closed BTC candle.
aligned = pd.merge_asof(
    features.sort_values("timestamp"),
    btc.sort_values("close_time"),
    left_on="timestamp",
    right_on="close_time",
    direction="backward",
)

# The 5-minute target is measured from the mapped candle's close time.
aligned["target_time"] = aligned["close_time"] + pd.Timedelta(minutes=5)

targets = btc[["close_time", "close"]].rename(
    columns={
        "close_time": "target_time",
        "close": "target_close",
    }
)

aligned = aligned.merge(targets, on="target_time", how="left")
aligned["price_change_5m"] = aligned["target_close"] - aligned["close"]

aligned["label_5m"] = "FLAT"
aligned.loc[aligned["price_change_5m"] > 0, "label_5m"] = "UP"
aligned.loc[aligned["price_change_5m"] < 0, "label_5m"] = "DOWN"

aligned = aligned.dropna(subset=["close", "target_close"]).copy()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
aligned.to_parquet(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Shape:", aligned.shape)
print("Missing target_close:", aligned["target_close"].isna().sum())
print("Label counts:")
print(aligned["label_5m"].value_counts())
print(
    aligned[
        ["timestamp", "close_time", "close", "target_time",
         "target_close", "price_change_5m", "label_5m"]
    ].head(10).to_string(index=False)
)