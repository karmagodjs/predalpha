import pandas as pd

files = [
    "data/reference/btcusdt_1m_oct2_oct3.csv",
    "data/reference/btcusdt_1m_oct2_oct3_resolution.csv",
]

for f in files:
    df = pd.read_csv(f)
    print(f"\n{f}")
    print("rows:", len(df))
    print("columns:", list(df.columns))
    print(df.head(2).to_string(index=False))
    print(df.tail(2).to_string(index=False))
