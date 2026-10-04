from pathlib import Path

import pandas as pd

FEATURE_PATH = Path(
    "data/processed/"
    "btc_oct3_up_down_session_20261002_04_final_features.parquet"
)
BTC_PATH = Path(
    "data/reference/btcusdt_1m_oct2_oct3_resolution.csv"
)
OUTPUT_PATH = Path(
    "data/processed/"
    "btc_oct3_event_5m_dataset.parquet"
)

features = pd.read_parquet(FEATURE_PATH)
btc = pd.read_csv(BTC_PATH)

features["timestamp"] = pd.to_datetime(
    features["timestamp"], utc=True
).astype("datetime64[ns, UTC]")

btc["close_time"] = pd.to_datetime(
    btc["close_time"], utc=True
).astype("datetime64[ns, UTC]")

features = features.sort_values("timestamp").copy()
btc = btc.sort_values("close_time").copy()

# Latest fully closed BTC candle at or before each feature event.
current = pd.merge_asof(
    features,
    btc[["close_time", "close"]].rename(
        columns={
            "close_time": "current_close_time",
            "close": "current_close",
        }
    ),
    left_on="timestamp",
    right_on="current_close_time",
    direction="backward",
)

# First fully closed BTC candle at or after event timestamp + 5 minutes.
current["target_cutoff"] = (
    current["timestamp"] + pd.Timedelta(minutes=5)
)

targets = btc[["close_time", "close"]].rename(
    columns={
        "close_time": "target_time",
        "close": "target_close",
    }
).sort_values("target_time")

aligned = pd.merge_asof(
    current.sort_values("target_cutoff"),
    targets,
    left_on="target_cutoff",
    right_on="target_time",
    direction="forward",
)

aligned["price_change_5m"] = (
    aligned["target_close"] - aligned["current_close"]
)

aligned["label_5m"] = "FLAT"
aligned.loc[aligned["price_change_5m"] > 0, "label_5m"] = "UP"
aligned.loc[aligned["price_change_5m"] < 0, "label_5m"] = "DOWN"

aligned = aligned.dropna(
    subset=["current_close", "target_close", "target_time"]
).copy()

aligned["actual_horizon_seconds"] = (
    aligned["target_time"] - aligned["timestamp"]
).dt.total_seconds()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
aligned.to_parquet(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Shape:", aligned.shape)
print("Unique feature events:", aligned["timestamp"].nunique())
print("Unique target times:", aligned["target_time"].nunique())
print("\nLabels:")
print(aligned["label_5m"].value_counts())
print("\nActual horizon seconds:")
print(aligned["actual_horizon_seconds"].describe())
print("\nSample:")
print(
    aligned[
        [
            "timestamp",
            "current_close_time",
            "current_close",
            "target_cutoff",
            "target_time",
            "target_close",
            "actual_horizon_seconds",
            "label_5m",
        ]
    ].head(10).to_string(index=False)
)