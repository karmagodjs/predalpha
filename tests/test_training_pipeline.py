import pandas as pd

from training.labels import create_labels
from training.split import temporal_split
from training.preprocessing import (
    FeatureScaler,
    FEATURES
)


def make_data():

    return pd.DataFrame({

        "timestamp": pd.date_range(
            "2026-01-01",
            periods=100,
            freq="s"
        ),

        "mid_price": [
            0.5 + i * 0.0001
            for i in range(100)
        ],

        "best_bid": [
            0.49 + i * 0.0001
            for i in range(100)
        ],

        "best_ask": [
            0.51 + i * 0.0001
            for i in range(100)
        ],

        "spread": [0.02] * 100,

        "bid_depth": [100.0] * 100,

        "ask_depth": [100.0] * 100,

        "imbalance": [0.0] * 100,

        "microprice": [
            0.50 + i * 0.0001
            for i in range(100)
        ],
    })


def test_label_generation():

    df = make_data()

    result = create_labels(
        df,
        horizon=5
    )

    assert "future_return" in result
    assert "label" in result

    assert set(
        result["label"].unique()
    ).issubset({0, 1, 2})


def test_temporal_split():

    df = make_data()

    train, val, test = temporal_split(
        df
    )

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_scaler():

    df = make_data()

    scaler = FeatureScaler()

    transformed = (
        scaler.fit_transform(df)
    )

    assert list(
        transformed[FEATURES].columns
    ) == FEATURES

    assert transformed[
        FEATURES
    ].notna().all().all()