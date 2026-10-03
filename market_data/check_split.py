from pathlib import Path
import pandas as pd

BASE = Path("data/processed/splits")

for name in ["train", "validation", "test"]:
    df = pd.read_parquet(BASE / f"{name}.parquet")

    print(f"\n{name.upper()}")
    print("Rows:", len(df))
    print("Class counts:")
    print(df["label"].value_counts())
    print("Class percentages:")
    print((df["label"].value_counts(normalize=True) * 100).round(2))