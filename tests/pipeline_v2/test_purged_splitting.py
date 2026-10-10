"""
Unit tests for Pipeline V2 Purged Temporal Train/Validation/Test Splitter.

Covers all Phase 15 requirements:
A. Chronological split: train < purge < val < purge < test
B. No timestamp overlap: train ∩ val = empty, train ∩ test = empty, val ∩ test = empty
C. No duplicate rows: zero duplicated row content or hashes across partitions
D. Required purge gap: purge gap >= required minimum (20,000 ms)
E. Label horizon cannot cross boundary: train_ts + label_horizon <= val_min
F. Feature lookback cannot cross boundary: val_ts - feature_lookback >= train_max
G. Session boundary isolation: metadata tracks session and no cross-session bridging
H. No random shuffling: deterministic chronological ordering preserved
I. Deterministic repeated split: identical output across multiple runs
J. Insufficient-data failure: fails fast when rows or time span are too small
K. Duplicate timestamp failure: fails fast when duplicate timestamps exist
L. Verify split metadata: JSON metadata accurately reflects boundaries
M. Verify actual measured purge in milliseconds
N. Verify no sequence construction occurs: outputs remain flat 2D tabular DataFrames
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.splitting.temporal_purged_split import (
    calculate_required_purge_ms,
    PurgedTemporalSplitter,
    SplitMetadata,
    SplitReport,
)


@pytest.fixture
def splitter():
    return PurgedTemporalSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_gap_ms=20000,  # 20 seconds
        max_label_horizon_ms=7000,
        feature_lookback_ms=5000,
        min_rows_per_split=5,
    )


def create_feature_dataset(n_rows=100, start_ts=1000000, step_ms=1000):
    """Generate synthetic chronological feature observations."""
    records = []
    for i in range(n_rows):
        ts = start_ts + i * step_ms
        records.append({
            "grid_timestamp_ms": ts,
            "source_timestamp_ms": ts - 200,
            "source_sequence_id": i + 1,
            "market_id": "mkt_poly_1",
            "asset_id": "ast_yes_1",
            "bid": 0.45 + (i % 5) * 0.01,
            "ask": 0.55 + (i % 5) * 0.01,
            "bid_size": 100.0,
            "ask_size": 100.0,
            "mid_price": 0.50 + (i % 5) * 0.01,
            "spread": 0.10,
            "spread_bps": 2000.0,
            "mid_return_1s": 0.0,
            "mid_return_3s": 0.0,
            "mid_return_5s": 0.0,
            "mid_volatility_5s": 0.0,
            "bid_change_1s": 0.0,
            "ask_change_1s": 0.0,
            "microprice": 0.50,
            "depth_imbalance": 0.0,
            "label": "FLAT",
            "session_id": "sess_1",
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
        "mid_price": "float64",
        "spread": "float64",
        "spread_bps": "float64",
        "mid_return_1s": "float64",
        "mid_return_3s": "float64",
        "mid_return_5s": "float64",
        "mid_volatility_5s": "float64",
        "bid_change_1s": "float64",
        "ask_change_1s": "float64",
        "microprice": "float64",
        "depth_imbalance": "float64",
        "label": "string",
        "session_id": "string",
    })


# A. Chronological split
def test_chronological_split(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, metadata, report = splitter.split(df)

    assert train["grid_timestamp_ms"].max() < val["grid_timestamp_ms"].min()
    assert val["grid_timestamp_ms"].max() < test["grid_timestamp_ms"].min()
    assert report.test_result == "PASS"


# B. No timestamp overlap
def test_no_timestamp_overlap(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, _, report = splitter.split(df)

    train_ts = set(train["grid_timestamp_ms"])
    val_ts = set(val["grid_timestamp_ms"])
    test_ts = set(test["grid_timestamp_ms"])

    assert len(train_ts.intersection(val_ts)) == 0
    assert len(train_ts.intersection(test_ts)) == 0
    assert len(val_ts.intersection(test_ts)) == 0
    assert report.timestamp_overlap_count == 0


# C. No duplicate rows
def test_no_duplicate_rows(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, _, report = splitter.split(df)

    train_hashes = set(train.apply(lambda r: hashlib.md5(r.to_json().encode()).hexdigest(), axis=1))
    val_hashes = set(val.apply(lambda r: hashlib.md5(r.to_json().encode()).hexdigest(), axis=1))
    test_hashes = set(test.apply(lambda r: hashlib.md5(r.to_json().encode()).hexdigest(), axis=1))

    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0
    assert report.duplicate_overlap_count == 0


# D. Required purge gap
def test_required_purge_gap(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, metadata, report = splitter.split(df)

    purge_1 = val["grid_timestamp_ms"].min() - train["grid_timestamp_ms"].max()
    purge_2 = test["grid_timestamp_ms"].min() - val["grid_timestamp_ms"].max()

    assert purge_1 >= 20000
    assert purge_2 >= 20000
    assert report.actual_purge_train_val_ms >= 20000
    assert report.actual_purge_val_test_ms >= 20000


# E. Label horizon cannot cross boundary
def test_label_horizon_cannot_cross_boundary(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, _, report = splitter.split(df)

    # For every row in train: ts + 7000ms <= min(val_ts)
    assert (train["grid_timestamp_ms"] + 7000 <= val["grid_timestamp_ms"].min()).all()
    # For every row in val: ts + 7000ms <= min(test_ts)
    assert (val["grid_timestamp_ms"] + 7000 <= test["grid_timestamp_ms"].min()).all()
    assert report.label_boundary_violations == 0


# F. Feature lookback cannot cross boundary
def test_feature_lookback_cannot_cross_boundary(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, _, report = splitter.split(df)

    # For every row in val: ts - 5000ms >= max(train_ts)
    assert (val["grid_timestamp_ms"] - 5000 >= train["grid_timestamp_ms"].max()).all()
    # For every row in test: ts - 5000ms >= max(val_ts)
    assert (test["grid_timestamp_ms"] - 5000 >= val["grid_timestamp_ms"].max()).all()
    assert report.feature_lookback_violations == 0


# G. Session boundary isolation
def test_session_boundary_isolation(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, metadata, report = splitter.split(df)

    assert "Single continuous recording session" in metadata.session_info
    assert report.session_boundary_violations == 0


# H. No random shuffling
def test_no_random_shuffling(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, _, _ = splitter.split(df)

    assert train["grid_timestamp_ms"].is_monotonic_increasing
    assert val["grid_timestamp_ms"].is_monotonic_increasing
    assert test["grid_timestamp_ms"].is_monotonic_increasing


# I. Deterministic repeated split
def test_deterministic_repeated_split(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    tr1, val1, te1, _, _ = splitter.split(df)
    tr2, val2, te2, _, _ = splitter.split(df)

    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(te1, te2)


# J. Insufficient-data failure
def test_insufficient_data_failure(splitter):
    # Too few rows
    small_df = create_feature_dataset(n_rows=10, step_ms=1000)
    with pytest.raises(ValueError, match="too small"):
        splitter.split(small_df)

    # Too small time span (e.g. 20 rows spaced 100ms = 2 seconds, requires > 40 seconds)
    short_span_df = create_feature_dataset(n_rows=30, step_ms=100)
    with pytest.raises(ValueError, match="time span"):
        splitter.split(short_span_df)


# K. Duplicate timestamp failure
def test_duplicate_timestamp_failure(splitter):
    df = create_feature_dataset(n_rows=100, step_ms=1000)
    # Duplicate row 50
    dup_row = df.iloc[50:51].copy()
    corrupt_df = pd.concat([df.iloc[:51], dup_row, df.iloc[51:]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate timestamps detected"):
        splitter.split(corrupt_df)


# L. Verify split metadata
def test_verify_split_metadata(splitter, tmp_path):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    out_dir = tmp_path / "splits_test"
    in_file = tmp_path / "test_features.parquet"
    df.to_parquet(in_file, index=False)

    train, val, test, metadata, report = splitter.split_file(
        input_path=in_file,
        output_dir=out_dir,
    )

    meta_file = out_dir / "split_metadata.json"
    assert meta_file.exists()
    saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))

    assert saved_meta["input_rows"] == 200
    assert saved_meta["train_rows"] == len(train)
    assert saved_meta["val_rows"] == len(val)
    assert saved_meta["test_rows"] == len(test)
    assert saved_meta["actual_purge_train_val_ms"] >= 20000
    assert saved_meta["actual_purge_val_test_ms"] >= 20000


# M. Verify actual measured purge in milliseconds
def test_verify_actual_measured_purge_in_milliseconds(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, metadata, report = splitter.split(df)

    m1 = int(val["grid_timestamp_ms"].min() - train["grid_timestamp_ms"].max())
    m2 = int(test["grid_timestamp_ms"].min() - val["grid_timestamp_ms"].max())

    assert m1 >= 20000
    assert m2 >= 20000
    assert metadata.actual_purge_train_val_ms == m1
    assert metadata.actual_purge_val_test_ms == m2


# N. Verify no sequence construction occurs
def test_verify_no_sequence_construction_occurs(splitter):
    df = create_feature_dataset(n_rows=200, step_ms=1000)
    train, val, test, _, _ = splitter.split(df)

    # Output must be 2D tabular DataFrames with same number of columns as input
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert train.ndim == 2
    assert val.ndim == 2
    assert test.ndim == 2
    assert set(train.columns) == set(df.columns)
    assert set(val.columns) == set(df.columns)
    assert set(test.columns) == set(df.columns)
