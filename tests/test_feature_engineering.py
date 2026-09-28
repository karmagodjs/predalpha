import pandas as pd

from features.feature_engineering import (
    build_features
)


def test_feature_engineering():

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-01",
            periods=100,
            freq="s"
        ),
        "mid_price": [
            0.5 + i * 0.001
            for i in range(100)
        ],
        "spread": [0.001] * 100,
        "bid_depth": [100 + i for i in range(100)],
        "ask_depth": [100] * 100,
        "microprice": [
            0.5 + i * 0.001
            for i in range(100)
        ],
    })

    result = build_features(df)

    assert "return_1" in result.columns
    assert "spread_bps" in result.columns
    assert "depth_imbalance" in result.columns
    assert "microprice_deviation" in result.columns
    assert "volatility_10" in result.columns
    assert "momentum_10" in result.columns