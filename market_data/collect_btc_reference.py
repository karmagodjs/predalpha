import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

# One candle before noon ET, which is 16:00 UTC in October.
START = "2026-10-02 15:59:00"
END = "2026-10-03 16:00:00"  # exclusive open-time boundary

OUTPUT = Path("data/reference/btcusdt_1m_oct2_oct3_resolution.csv")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def to_milliseconds(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def fetch_klines(start_ms: int, end_ms: int) -> list:
    rows = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "startTime": current,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        attempt = 0
        while True:
            try:
                response = requests.get(
                    BASE_URL,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                attempt += 1
                wait = min(5 * (2 ** (attempt - 1)), 60)

                print(
                    f"Connection error: {e}\n"
                    f"Retrying in {wait} seconds... (attempt {attempt})"
                )
                time.sleep(wait)
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        rows.extend(batch)
        current = batch[-1][0] + 60_000

    return rows


def rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)

    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


def save_merged(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new
    elif new.empty:
        merged = existing
    else:
        merged = pd.concat([existing, new], ignore_index=True)

    if merged.empty:
        return merged

    merged = (
        merged.drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT, index=False)
    return merged


def main():
    start_ms = to_milliseconds(START)
    end_ms = to_milliseconds(END)

    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT, parse_dates=["open_time", "close_time"])
        existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
        existing["close_time"] = pd.to_datetime(existing["close_time"], utc=True)
        print("Existing candles:", len(existing))
    else:
        existing = pd.DataFrame(columns=COLUMNS)
        print("No existing CSV found. Starting fresh.")

    print("Incremental BTC reference collector running.")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)

            # Fetch only up to the current time and the requested end boundary.
            fetch_end = min(now_ms, end_ms)

            if not existing.empty and (
                pd.Timestamp(existing["open_time"].min()).timestamp() * 1000 <= start_ms
            ):
                last_ms = int(existing["open_time"].max().timestamp() * 1000)
                fetch_start = max(start_ms, last_ms + 60_000)
            else:
                # Backfill from START if the boundary candle is missing.
                fetch_start = start_ms

            if fetch_start < fetch_end:
                rows = fetch_klines(fetch_start, fetch_end)
                new = rows_to_df(rows)

                if not new.empty:
                    # Keep only fully closed candles inside the requested window.
                    now = pd.Timestamp.now(tz="UTC")
                    new = new[
                        (new["open_time"] >= pd.Timestamp(START, tz="UTC"))
                        & (new["open_time"] < pd.Timestamp(END, tz="UTC"))
                        & (new["close_time"] < now)
                    ]

                previous_count = len(existing)
                existing = save_merged(existing, new)

                added = len(existing) - previous_count
                print(
                    f"Added: {max(added, 0)} | Total: {len(existing)} | "
                    f"Latest: {existing['open_time'].max() if not existing.empty else 'N/A'}"
                )

            if now_ms >= end_ms:
                last_expected_open = pd.Timestamp(END, tz="UTC") - pd.Timedelta(minutes=1)
                if not existing.empty and existing["open_time"].max() >= last_expected_open:
                    print("\nRequested window collected.")
                    break

            time.sleep(30)

    except KeyboardInterrupt:
        print("\nCollector stopped by user.")

    print("Total candles:", len(existing))
    if not existing.empty:
        print("First:", existing["open_time"].min())
        print("Last:", existing["open_time"].max())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()