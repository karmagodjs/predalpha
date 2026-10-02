import numpy as np
import pandas as pd


FEATURES = [
    "best_bid",
    "best_ask",
    "mid_price",
    "spread",
    "bid_depth",
    "ask_depth",
    "imbalance",
    "microprice",
    "return_1",
    "return_5",
    "return_10",
    "spread_bps",
    "depth_imbalance",
    "microprice_deviation",
    "volatility_10",
    "volatility_50",
    "momentum_10",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp", "best_bid", "best_ask", "mid_price",
        "spread", "bid_depth", "ask_depth", "imbalance", "microprice",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    mid = pd.to_numeric(df["mid_price"], errors="coerce")
    spread = pd.to_numeric(df["spread"], errors="coerce")
    bid_depth = pd.to_numeric(df["bid_depth"], errors="coerce")
    ask_depth = pd.to_numeric(df["ask_depth"], errors="coerce")
    microprice = pd.to_numeric(df["microprice"], errors="coerce")

    df["return_1"] = mid.pct_change(1)
    df["return_5"] = mid.pct_change(5)
    df["return_10"] = mid.pct_change(10)

    df["spread_bps"] = np.where(
        mid.ne(0),
        (spread / mid) * 10_000,
        np.nan,
    )

    total_depth = bid_depth + ask_depth
    df["depth_imbalance"] = np.where(
        total_depth.ne(0),
        (bid_depth - ask_depth) / total_depth,
        np.nan,
    )

    df["microprice_deviation"] = np.where(
        mid.ne(0),
        (microprice - mid) / mid,
        np.nan,
    )

    df["bid_depth_change"] = bid_depth.diff()
    df["ask_depth_change"] = ask_depth.diff()

    df["volatility_10"] = df["return_1"].rolling(10, min_periods=10).std()
    df["volatility_50"] = df["return_1"].rolling(50, min_periods=50).std()

    df["momentum_10"] = mid - mid.shift(10)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df