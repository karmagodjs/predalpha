import pandas as pd
from pathlib import Path

FILE = Path("data/processed/btc_88000_bbo_1s.parquet")

df = pd.read_parquet(FILE).sort_index()

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Index start:", df.index.min())
print("Index end:", df.index.max())

# Check missing seconds
expected = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="1s",
    tz="UTC",
)
missing = expected.difference(df.index)

print("Expected seconds:", len(expected))
print("Missing seconds:", len(missing))
print("Duplicate timestamps:", df.index.duplicated().sum())

# Check stale BBO intervals
same_bbo = (df["bid"].diff() == 0) & (df["ask"].diff() == 0)
print("Unchanged BBO seconds:", int(same_bbo.sum()))

# Check invalid values
print("Missing values:\n", df[["bid", "ask", "mid_price", "spread"]].isna().sum())
print("Crossed BBO rows:", int((df["bid"] >= df["ask"]).sum()))

# Check price movement
df["mid_change"] = df["mid_price"].diff()
print("Mid-price changes:", int(df["mid_change"].ne(0).sum()))
print("Positive changes:", int((df["mid_change"] > 0).sum()))
print("Negative changes:", int((df["mid_change"] < 0).sum()))

print("\nMid-price distribution:")
print(df["mid_price"].describe())

print("\nLargest mid-price changes:")
print(df["mid_change"].abs().nlargest(10))