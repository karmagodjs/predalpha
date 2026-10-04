import pandas as pd
from pathlib import Path

files = [
    "btc_5m_direction_labels.parquet",
    "btc_oct3_aligned_5m_dataset.parquet",
    "btc_oct3_event_5m_dataset.parquet",
    "btc_oct3_unique_target_dataset.parquet",
    "btc_oct3_resolution_label.csv",
]

for name in files:
    p = Path("data/processed") / name
    if not p.exists():
        print(name, "MISSING")
        continue

    d = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    print(f"\n{name} | rows={len(d)}")
    print("columns:", d.columns.tolist())

    time_cols = [c for c in d.columns if "time" in c.lower() or "date" in c.lower()]
    for c in time_cols[:2]:
        print(c, ":", d[c].min(), "to", d[c].max())

    label_cols = [c for c in d.columns if any(x in c.lower() for x in ["label", "direction", "target", "outcome"])]
    for c in label_cols[:2]:
        print(c, ":", d[c].value_counts(dropna=False).head(6).to_dict())
