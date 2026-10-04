import urllib.request
import json
import pandas as pd

url = (
    "https://api.binance.com/api/v3/klines"
    "?symbol=BTCUSDT&interval=1m"
    "&startTime=1791062100000"
    "&endTime=1791063060000"
    "&limit=1000"
)

with urllib.request.urlopen(url, timeout=30) as response:
    data = json.loads(response.read().decode())

cols = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]
df = pd.DataFrame(data, columns=cols)

df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
df["close_time"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)

for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col])

out = "data/reference/btcusdt_1m_oct3_2120_2130.csv"
df.to_csv(out, index=False)

print("Saved:", out)
print("Rows:", len(df))
print(df[["open_time", "open", "close", "close_time"]].to_string(index=False))
