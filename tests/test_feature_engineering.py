import pandas as pd

from features.feature_engineering import build_features


def test_feature_engineering_creates_expected_columns():
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="s"),
        "best_bid": [0.036] * n,
        "best_ask": [0.037] * n,
        "mid_price": [0.0365 + i * 0.00001 for i in range(n)],
        "spread": [0.001] * n,
        "bid_depth": [100 + i for i in range(n)],
        "ask_depth": [90 + i for i in range(n)],
        "imbalance": [0.05] * n,
        "microprice": [0.0366] * n,
    })

    result = build_features(df)

    for column in [
        "return_1", "return_5", "return_10",
        "spread_bps", "depth_imbalance",
        "microprice_deviation", "volatility_10",
        "volatility_50", "momentum_10",
    ]:
        assert column in result.columns

    assert len(result) == n
    assert result["return_1"].notna().sum() > 0