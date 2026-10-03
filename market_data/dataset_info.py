import json
from pathlib import Path
import pandas as pd

FILE = Path("data/processed/btc_88000_features_5s.parquet")
OUTPUT = Path("data/processed/dataset_info.json")

df = pd.read_parquet(FILE)

info = {
    "rows": len(df),
    "start_time": str(df.index.min()),
    "end_time": str(df.index.max()),
    "features": [
        "mid_price",
        "spread",
        "spread_bps",
        "mid_return_1s",
        "mid_return_3s",
        "mid_return_5s",
        "mid_volatility_5s",
        "bid_change_1s",
        "ask_change_1s",
    ],
    "target": "label",
    "horizon_seconds": 5,
    "class_counts": df["label"].value_counts().to_dict(),
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(info, indent=2), encoding="utf-8")

print(json.dumps(info, indent=2))
print("Saved:", OUTPUT)