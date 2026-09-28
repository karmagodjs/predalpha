import pandas as pd
import numpy as np


def build_features(df):

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # -----------------------------
    # Basic returns
    # -----------------------------

    df["return_1"] = (
        df["mid_price"]
        .pct_change(1)
    )

    df["return_5"] = (
        df["mid_price"]
        .pct_change(5)
    )

    df["return_10"] = (
        df["mid_price"]
        .pct_change(10)
    )

    # -----------------------------
    # Spread
    # -----------------------------

    df["spread_bps"] = (
        df["spread"]
        / df["mid_price"]
    ) * 10_000

    # -----------------------------
    # Depth imbalance
    # -----------------------------

    total_depth = (
        df["bid_depth"]
        + df["ask_depth"]
    )

    df["depth_imbalance"] = (
        (
            df["bid_depth"]
            - df["ask_depth"]
        )
        / total_depth.replace(
            0,
            np.nan
        )
    )

    # -----------------------------
    # Microprice deviation
    # -----------------------------

    df["microprice_deviation"] = (
        (
            df["microprice"]
            - df["mid_price"]
        )
        / df["mid_price"]
    )

    # -----------------------------
    # Depth changes
    # -----------------------------

    df["bid_depth_change"] = (
        df["bid_depth"]
        .diff()
    )

    df["ask_depth_change"] = (
        df["ask_depth"]
        .diff()
    )

    # -----------------------------
    # Short-term volatility
    # -----------------------------

    df["volatility_10"] = (
        df["return_1"]
        .rolling(10)
        .std()
    )

    df["volatility_50"] = (
        df["return_1"]
        .rolling(50)
        .std()
    )

    # -----------------------------
    # Momentum
    # -----------------------------

    df["momentum_10"] = (
        df["mid_price"]
        - df["mid_price"].shift(10)
    )

    # -----------------------------
    # Clean
    # -----------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df