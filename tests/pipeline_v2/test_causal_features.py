"""
Unit tests for Pipeline V2 Causal Feature Engineering Engine.

Covers all Phase 14 requirements:
A. Mid-price calculation: mid = (bid + ask) / 2
B. Spread calculation: spread = ask - bid
C. Spread BPS: spread_bps = (spread / mid) * 10000
D. 1-second return: (mid_t - mid_t-1) / mid_t-1
E. 3-second return: (mid_t - mid_t-3) / mid_t-3
F. 5-second return: (mid_t - mid_t-5) / mid_t-5
G. Rolling volatility: uses only historical/current rows [t-4, ..., t]
H. Bid/ask changes: bid_t - bid_t-1, ask_t - ask_t-1
I. Microprice calculation: (bid*ask_size + ask*bid_size) / (bid_size + ask_size)
J. Depth imbalance range: within [-1, 1]
K. No future-column ingestion: target columns never used
L. No negative shift: verify absence of shift(-N)
M. No centered rolling: verify center=False
N. No bfill: verify absence of backward fill
O. Explicit feature whitelist: SAFE_FEATURE_COLUMNS
P. Chronological output: strictly ascending timestamps
Q. Duplicate timestamp rejection
R. No Inf values
S. Microprice bounds: bid <= microprice <= ask
T. Synthetic causality test: change future row and verify past features are bit-for-bit identical!
"""

import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.features.causal_features import (
    CausalFeatureBuilder,
    FeatureValidationReport,
    FORBIDDEN_INPUT_COLUMNS,
    SAFE_FEATURE_COLUMNS,
)


@pytest.fixture
def builder():
    return CausalFeatureBuilder()


def create_market_df(rows):
    """Helper to create a valid order book observation DataFrame fixture."""
    records = []
    for i, r in enumerate(rows):
        records.append({
            "grid_timestamp_ms": int(r.get("timestamp_ms", 10000 + i * 1000)),
            "source_timestamp_ms": int(r.get("source_ts", 10000 + i * 1000)),
            "source_sequence_id": int(r.get("seq", i + 1)),
            "market_id": str(r.get("market_id", "mkt1")),
            "asset_id": str(r.get("asset_id", "ast1")),
            "bid": float(r["bid"]),
            "ask": float(r["ask"]),
            "bid_size": float(r.get("bid_size", 100.0)),
            "ask_size": float(r.get("ask_size", 100.0)),
        })
    df = pd.DataFrame(records)
    return df.astype({
        "grid_timestamp_ms": "int64",
        "source_timestamp_ms": "int64",
        "source_sequence_id": "int64",
        "market_id": "string",
        "asset_id": "string",
        "bid": "float64",
        "ask": "float64",
        "bid_size": "float64",
        "ask_size": "float64",
    })


# A. Mid-price calculation
def test_mid_price_calculation(builder):
    df = create_market_df([{"bid": 0.40, "ask": 0.60}])
    res, report = builder.compute_features(df)
    assert res.iloc[0]["mid_price"] == pytest.approx(0.50)


# B. Spread calculation
def test_spread_calculation(builder):
    df = create_market_df([{"bid": 0.40, "ask": 0.60}])
    res, report = builder.compute_features(df)
    assert res.iloc[0]["spread"] == pytest.approx(0.20)


# C. Spread BPS
def test_spread_bps_calculation(builder):
    df = create_market_df([{"bid": 0.40, "ask": 0.60}])
    res, report = builder.compute_features(df)
    # spread = 0.20, mid = 0.50 -> spread_bps = (0.20 / 0.50) * 10000 = 4000.0
    assert res.iloc[0]["spread_bps"] == pytest.approx(4000.0)


# D. 1-second return
def test_mid_return_1s_calculation(builder):
    df = create_market_df([
        {"bid": 99.0, "ask": 101.0},  # mid = 100.0
        {"bid": 101.0, "ask": 103.0},  # mid = 102.0 -> return = 2/100 = 0.02
    ])
    res, report = builder.compute_features(df)
    assert pd.isna(res.iloc[0]["mid_return_1s"])  # warm-up row 0
    assert res.iloc[1]["mid_return_1s"] == pytest.approx(0.02)


# E. 3-second return
def test_mid_return_3s_calculation(builder):
    df = create_market_df([
        {"bid": 99.0, "ask": 101.0},  # t=0, mid=100.0
        {"bid": 99.5, "ask": 101.5},  # t=1, mid=100.5
        {"bid": 100.0, "ask": 102.0},  # t=2, mid=101.0
        {"bid": 102.0, "ask": 104.0},  # t=3, mid=103.0 -> return = (103 - 100) / 100 = 0.03
    ])
    res, report = builder.compute_features(df)
    assert pd.isna(res.iloc[0]["mid_return_3s"])
    assert pd.isna(res.iloc[1]["mid_return_3s"])
    assert pd.isna(res.iloc[2]["mid_return_3s"])
    assert res.iloc[3]["mid_return_3s"] == pytest.approx(0.03)


# F. 5-second return
def test_mid_return_5s_calculation(builder):
    df = create_market_df([
        {"bid": 99.0, "ask": 101.0},  # t=0, mid=100.0
        {"bid": 99.0, "ask": 101.0},  # t=1
        {"bid": 99.0, "ask": 101.0},  # t=2
        {"bid": 99.0, "ask": 101.0},  # t=3
        {"bid": 99.0, "ask": 101.0},  # t=4
        {"bid": 104.0, "ask": 106.0},  # t=5, mid=105.0 -> return = (105 - 100) / 100 = 0.05
    ])
    res, report = builder.compute_features(df)
    assert res.iloc[:5]["mid_return_5s"].isna().all()
    assert res.iloc[5]["mid_return_5s"] == pytest.approx(0.05)


# G. Rolling volatility uses ONLY historical/current rows
def test_rolling_volatility_causal_window(builder):
    # Constant returns produce 0.0 std
    df = create_market_df([
        {"bid": 9.0, "ask": 11.0} for _ in range(10)
    ])
    res, report = builder.compute_features(df)
    # First 5 rows are warm-up for 5-sample volatility
    assert res.iloc[:5]["mid_volatility_5s"].isna().all()
    # Rows 5..9 have 5 return observations (all 0.0) -> std = 0.0
    assert (res.iloc[5:]["mid_volatility_5s"] == 0.0).all()


# H. Bid/ask changes
def test_bid_ask_changes(builder):
    df = create_market_df([
        {"bid": 10.0, "ask": 12.0},
        {"bid": 10.5, "ask": 11.8},
    ])
    res, report = builder.compute_features(df)
    assert pd.isna(res.iloc[0]["bid_change_1s"])
    assert pd.isna(res.iloc[0]["ask_change_1s"])
    assert res.iloc[1]["bid_change_1s"] == pytest.approx(0.5)
    assert res.iloc[1]["ask_change_1s"] == pytest.approx(-0.2)


# I. Microprice calculation
def test_microprice_calculation(builder):
    df = create_market_df([
        {"bid": 10.0, "ask": 12.0, "bid_size": 100.0, "ask_size": 300.0},
    ])
    res, report = builder.compute_features(df)
    # microprice = (10*300 + 12*100) / 400 = (3000 + 1200) / 400 = 10.5
    assert res.iloc[0]["microprice"] == pytest.approx(10.5)


# J. Depth imbalance range within [-1, 1]
def test_depth_imbalance_range(builder):
    df = create_market_df([
        {"bid": 10.0, "ask": 12.0, "bid_size": 100.0, "ask_size": 300.0},  # (100-300)/400 = -0.5
        {"bid": 10.0, "ask": 12.0, "bid_size": 400.0, "ask_size": 100.0},  # (400-100)/500 = +0.6
        {"bid": 10.0, "ask": 12.0, "bid_size": 1000.0, "ask_size": 1000.0},  # 0.0
    ])
    res, report = builder.compute_features(df)
    assert res.iloc[0]["depth_imbalance"] == pytest.approx(-0.5)
    assert res.iloc[1]["depth_imbalance"] == pytest.approx(0.6)
    assert res.iloc[2]["depth_imbalance"] == pytest.approx(0.0)
    assert ((res["depth_imbalance"] >= -1.0) & (res["depth_imbalance"] <= 1.0)).all()


# K. No future-column ingestion
def test_no_future_column_ingestion(builder):
    df = create_market_df([{"bid": 10.0, "ask": 12.0} for _ in range(5)])
    # Add forbidden target-derived columns
    df["future_mid"] = 15.0
    df["target_delta"] = 2.0
    df["label"] = "UP"
    df["physical_horizon_ms"] = 5000

    res, report = builder.compute_features(df)

    # Generated feature list must contain only SAFE_FEATURE_COLUMNS
    for col in SAFE_FEATURE_COLUMNS:
        assert col in res.columns
    for f_col in FORBIDDEN_INPUT_COLUMNS:
        assert f_col not in builder.feature_columns
    assert report.forbidden_column_validation is True


# L. No negative shift
def test_no_negative_shift():
    source = inspect.getsource(CausalFeatureBuilder)
    assert "shift(-" not in source
    assert "shift(-1)" not in source
    assert "shift(-5)" not in source


# M. No centered rolling
def test_no_centered_rolling():
    source = inspect.getsource(CausalFeatureBuilder)
    assert "center=True" not in source


# N. No bfill
def test_no_bfill():
    source = inspect.getsource(CausalFeatureBuilder)
    assert ".bfill" not in source
    assert "bfill()" not in source
    assert "method='bfill'" not in source


# O. Explicit feature whitelist
def test_explicit_feature_whitelist(builder):
    assert len(SAFE_FEATURE_COLUMNS) == 11
    expected = [
        "mid_price", "spread", "spread_bps",
        "mid_return_1s", "mid_return_3s", "mid_return_5s",
        "mid_volatility_5s",
        "bid_change_1s", "ask_change_1s",
        "microprice", "depth_imbalance",
    ]
    assert SAFE_FEATURE_COLUMNS == expected
    assert builder.feature_columns == expected


# P. Chronological output
def test_chronological_output(builder):
    df = create_market_df([
        {"timestamp_ms": 10000, "bid": 10.0, "ask": 11.0},
        {"timestamp_ms": 11000, "bid": 10.0, "ask": 11.0},
        {"timestamp_ms": 12000, "bid": 10.0, "ask": 11.0},
    ])
    res, report = builder.compute_features(df)
    assert report.chronological_validation is True
    assert res["grid_timestamp_ms"].is_monotonic_increasing


# Q. Duplicate timestamp rejection
def test_duplicate_timestamp_flagged(builder):
    df = create_market_df([
        {"timestamp_ms": 10000, "bid": 10.0, "ask": 11.0},
        {"timestamp_ms": 10000, "bid": 10.1, "ask": 11.1},  # duplicate timestamp
    ])
    res, report = builder.compute_features(df)
    assert report.duplicate_timestamp_count == 1


# R. No Inf values
def test_no_inf_values(builder):
    df = create_market_df([{"bid": 10.0, "ask": 11.0} for _ in range(10)])
    res, report = builder.compute_features(df)
    for col in SAFE_FEATURE_COLUMNS:
        assert not np.isinf(res[col]).any()
    assert report.sanity_checks_passed is True


# S. Microprice bounds: bid <= microprice <= ask
def test_microprice_bounds(builder):
    np.random.seed(42)
    rows = []
    for _ in range(50):
        b = np.random.uniform(0.1, 0.8)
        a = b + np.random.uniform(0.01, 0.1)
        bs = np.random.uniform(10, 1000)
        as_ = np.random.uniform(10, 1000)
        rows.append({"bid": b, "ask": a, "bid_size": bs, "ask_size": as_})

    df = create_market_df(rows)
    res, report = builder.compute_features(df)

    assert (res["bid"] <= res["microprice"] + 1e-9).all()
    assert (res["microprice"] <= res["ask"] + 1e-9).all()


# T. SYNTHETIC CAUSALITY TEST (MANDATORY & CRITICAL)
def test_synthetic_causality_test_t(builder):
    """
    Test T: Change a future row and verify that ALL features at earlier
    timestamps remain BIT-FOR-BIT IDENTICAL!
    """
    np.random.seed(999)
    n_rows = 20
    bids = 100.0 + np.cumsum(np.random.normal(0, 0.5, n_rows))
    asks = bids + np.random.uniform(0.5, 2.0, n_rows)
    bid_sizes = np.random.uniform(50, 500, n_rows)
    ask_sizes = np.random.uniform(50, 500, n_rows)

    rows = [
        {"timestamp_ms": 10000 + i * 1000, "bid": bids[i], "ask": asks[i], "bid_size": bid_sizes[i], "ask_size": ask_sizes[i]}
        for i in range(n_rows)
    ]

    original_df = create_market_df(rows)
    res_original, _ = builder.compute_features(original_df)

    # Mutate a future row at index k = 14
    k = 14
    mutated_rows = [dict(r) for r in rows]
    # Drastic shock to row k
    mutated_rows[k]["bid"] = 999.0
    mutated_rows[k]["ask"] = 1050.0
    mutated_rows[k]["bid_size"] = 99999.0
    mutated_rows[k]["ask_size"] = 1.0

    mutated_df = create_market_df(mutated_rows)
    res_mutated, _ = builder.compute_features(mutated_df)

    # Features at indices 0 through k-1 (past rows) MUST BE 100% IDENTICAL!
    past_orig = res_original.iloc[:k][SAFE_FEATURE_COLUMNS]
    past_mut = res_mutated.iloc[:k][SAFE_FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(past_orig, past_mut, check_exact=True)

    # Features at index k MUST reflect the change
    assert res_original.iloc[k]["mid_price"] != res_mutated.iloc[k]["mid_price"]
