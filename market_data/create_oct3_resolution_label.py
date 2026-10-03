from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/reference/btcusdt_1m_oct2_oct3_resolution.csv"
)

OUTPUT_FILE = Path(
    "data/processed/btc_oct3_resolution_label.csv"
)

START_TIME = "2026-10-02 15:59:00+00:00"
END_TIME = "2026-10-03 15:59:00+00:00"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)

    start = df[df["open_time"] == pd.Timestamp(START_TIME)]
    end = df[df["open_time"] == pd.Timestamp(END_TIME)]

    if len(start) != 1 or len(end) != 1:
        raise ValueError(
            f"Expected exactly one candle at each boundary. "
            f"Start rows={len(start)}, end rows={len(end)}"
        )

    start_row = start.iloc[0]
    end_row = end.iloc[0]

    start_close = float(start_row["close"])
    end_close = float(end_row["close"])

    difference = end_close - start_close

    if difference > 0:
        label = "UP"
    elif difference < 0:
        label = "DOWN"
    else:
        label = "FLAT"

    pct_change = (difference / start_close) * 100

    result = pd.DataFrame([{
        "market": "Bitcoin Up or Down on October 3, 2026",
        "start_candle_open_time": start_row["open_time"],
        "start_candle_close_time": start_row["close_time"],
        "start_close": start_close,
        "end_candle_open_time": end_row["open_time"],
        "end_candle_close_time": end_row["close_time"],
        "end_close": end_close,
        "price_difference": difference,
        "percentage_change": pct_change,
        "resolution_label": label,
    }])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(result.to_string(index=False))
    print(f"\nSaved resolution label to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()