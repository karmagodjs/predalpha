"""Unit and integration tests for Phase 1 Data Pipeline Foundation.

All unit tests use small synthetic fixtures clearly labeled as synthetic test data.
No test data is mixed into real datasets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_data.canonical_pipeline import (
    CanonicalSchemaConfig,
    build_canonical_dataset,
    split_canonical_dataset,
)
from market_data.validation import (
    check_constant_features,
    check_duplicate_records,
    check_duplicate_timestamps,
    check_finite_values,
    check_label_distribution,
    check_market_data_integrity,
    check_missing_values,
    check_target_leakage,
    check_timestamp_integrity,
    validate_canonical_dataset,
    validate_chronological_splits,
)


@pytest.fixture
def synthetic_clean_bbo() -> pd.DataFrame:
    """Create a synthetic clean 1-second BBO series (50 seconds)."""
    n = 50
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=n, freq="1s", tz="UTC")
    # Generate realistic oscillatory mid-price with movements > min_tick
    mids = [0.50 + 0.01 * np.sin(i / 3.0) for i in range(n)]
    spreads = [0.002 + 0.0005 * (i % 2) for i in range(n)]
    bids = [m - s / 2.0 for m, s in zip(mids, spreads)]
    asks = [m + s / 2.0 for m, s in zip(mids, spreads)]

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid": bids,
            "ask": asks,
            "mid_price": mids,
            "spread": spreads,
        },
        index=timestamps,
    )


def test_schema_validation_passes_valid_canonical_data(synthetic_clean_bbo):
    """Verify that a clean synthetic BBO dataset produces a valid canonical dataset."""
    config = CanonicalSchemaConfig(horizon_seconds=2, purge_gap_rows=2)
    canonical, report = build_canonical_dataset(synthetic_clean_bbo, config=config)

    assert report.is_valid
    assert report.failed_count == 0
    assert "timestamp" in canonical.columns
    assert "asset_id" in canonical.columns
    for f in config.feature_cols:
        assert f in canonical.columns
    assert config.target_col in canonical.columns


def test_schema_validation_rejects_missing_required_columns():
    """Verify that omitting required quote columns raises an error."""
    incomplete = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="1s", tz="UTC"),
        "bid": [0.5] * 10,
        # missing ask, mid_price, spread
    })
    with pytest.raises(ValueError, match="Missing required BBO columns"):
        build_canonical_dataset(incomplete)


def test_timestamp_parsing_and_ordering():
    """Verify timestamp validation detects non-monotonic and unparseable timestamps."""
    # 1. Unsorted timestamps
    times = pd.to_datetime(["2026-01-01 00:00:02", "2026-01-01 00:00:01", "2026-01-01 00:00:03"], utc=True)
    df_unsorted = pd.DataFrame({"timestamp": times, "val": [1, 2, 3]})
    res = check_timestamp_integrity(df_unsorted, expected_freq=None)
    assert not res.passed
    assert "not strictly ascending" in res.message

    # 2. Duplicate timestamps
    times_dupe = pd.to_datetime(["2026-01-01 00:00:01", "2026-01-01 00:00:01"], utc=True)
    df_dupe = pd.DataFrame({"timestamp": times_dupe, "val": [1, 2]})
    res_dupe = check_timestamp_integrity(df_dupe, expected_freq=None)
    assert not res_dupe.passed
    assert "duplicate timestamps" in res_dupe.message


def test_timestamp_continuity_detects_unexpected_gaps():
    """Verify that timestamp gaps exceeding expected frequency are detected."""
    times = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:01", "2026-01-01 00:00:05"], utc=True)
    df = pd.DataFrame({"timestamp": times})
    res = check_timestamp_integrity(df, expected_freq="1s", max_gap_seconds=1.0)
    assert not res.passed
    assert "timestamp gaps exceeding" in res.message


def test_duplicate_detection():
    """Verify detection of duplicate records in features or rows."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="1s", tz="UTC"),
        "feature_a": [1.0, 1.0, 1.0],
    })
    # No duplicate rows (timestamps differ)
    assert check_duplicate_records(df).passed

    # Add duplicate row
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    res = check_duplicate_records(df_with_dupe)
    assert not res.passed
    assert res.details["duplicate_count"] == 1


def test_missing_values_detection():
    """Verify that NaNs in features are caught."""
    df = pd.DataFrame({
        "feat_a": [1.0, np.nan, 3.0],
        "feat_b": [4.0, 5.0, 6.0],
    })
    res = check_missing_values(df)
    assert not res.passed
    assert "feat_a" in res.details["by_column"]
    assert res.details["total_missing"] == 1


def test_invalid_market_data_detection():
    """Verify detection of bid > ask, negative prices, and negative spread."""
    # 1. Crossed book (bid > ask)
    df_crossed = pd.DataFrame({
        "bid": [0.55],
        "ask": [0.50],
        "mid_price": [0.525],
        "spread": [-0.05],
    })
    res = check_market_data_integrity(df_crossed)
    assert not res.passed
    assert "crossed book" in res.message or "negative spreads" in res.message

    # 2. Non-positive price
    df_zero = pd.DataFrame({
        "bid": [0.0],
        "ask": [0.05],
        "mid_price": [0.025],
        "spread": [0.05],
    })
    res_zero = check_market_data_integrity(df_zero)
    assert not res_zero.passed
    assert "non-positive" in res_zero.message


def test_finite_values_validation():
    """Verify detection of inf and -inf values."""
    df_inf = pd.DataFrame({"a": [1.0, np.inf, 2.0], "b": [1.0, 2.0, -np.inf]})
    res = check_finite_values(df_inf)
    assert not res.passed
    assert "a" in res.details["non_finite_columns"]
    assert "b" in res.details["non_finite_columns"]


def test_constant_features_detection():
    """Verify detection of constant and near-constant features."""
    df = pd.DataFrame({
        "const_feat": [0.0365] * 100,
        "near_const": [1.0] * 99 + [2.0],
        "varying": list(range(100)),
    })
    res = check_constant_features(df, near_constant_threshold=0.98)
    assert not res.passed
    assert "const_feat" in res.details["constant_columns"]
    assert "near_const" in res.details["near_constant_columns"]


def test_single_class_label_detection():
    """Verify that a single-class dataset is rejected."""
    df_single = pd.DataFrame({"label": ["FLAT"] * 50})
    res = check_label_distribution(df_single, target_col="label", min_classes=2)
    assert not res.passed
    assert "Single-class" in res.message

    # Multi-class passes
    df_multi = pd.DataFrame({"label": ["FLAT"] * 30 + ["UP"] * 10 + ["DOWN"] * 10})
    res_multi = check_label_distribution(df_multi, target_col="label", min_classes=2)
    assert res_multi.passed
    assert res_multi.details["num_classes"] == 3


def test_chronological_splitting_and_purge_gap(synthetic_clean_bbo):
    """Verify chronological split enforces non-overlap and minimum purge gap."""
    config = CanonicalSchemaConfig(horizon_seconds=2, purge_gap_rows=2, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    canonical, _ = build_canonical_dataset(synthetic_clean_bbo, config=config)
    splits, split_res = split_canonical_dataset(canonical, config=config)

    assert split_res.passed
    train = splits["train"]
    val = splits["validation"]
    test = splits["test"]

    # Verify temporal non-overlap
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()

    # Verify purge gap (at least 2s)
    train_val_gap = (val["timestamp"].min() - train["timestamp"].max()).total_seconds()
    val_test_gap = (test["timestamp"].min() - val["timestamp"].max()).total_seconds()
    assert train_val_gap >= 2.0
    assert val_test_gap >= 2.0


def test_chronological_splits_rejects_overlap():
    """Verify that overlapping splits fail validation."""
    t1 = pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:05"], utc=True)
    t2 = pd.to_datetime(["2026-01-01 00:00:04", "2026-01-01 00:00:08"], utc=True)  # Overlaps t1
    t3 = pd.to_datetime(["2026-01-01 00:00:09", "2026-01-01 00:00:12"], utc=True)

    splits = {
        "train": pd.DataFrame({"timestamp": t1, "val": [1, 2]}),
        "validation": pd.DataFrame({"timestamp": t2, "val": [3, 4]}),
        "test": pd.DataFrame({"timestamp": t3, "val": [5, 6]}),
    }
    res = validate_chronological_splits(splits, min_purge_gap_seconds=0.0)
    assert not res.passed
    assert "Temporal overlap" in res.message


def test_leakage_prevention_catches_future_columns():
    """Verify that forbidden future columns in features are detected."""
    feature_cols = ["mid_price", "future_mid", "spread"]
    df = pd.DataFrame({
        "mid_price": [0.5],
        "future_mid": [0.55],
        "spread": [0.01],
        "label": ["UP"],
    })
    res = check_target_leakage(df, feature_cols=feature_cols, target_col="label")
    assert not res.passed
    assert "future_mid" in res.details["forbidden_in_features"]


def test_reproducibility(synthetic_clean_bbo):
    """Verify that running the pipeline multiple times yields identical results."""
    config = CanonicalSchemaConfig(horizon_seconds=2, purge_gap_rows=2)
    canonical_1, _ = build_canonical_dataset(synthetic_clean_bbo, config=config)
    canonical_2, _ = build_canonical_dataset(synthetic_clean_bbo, config=config)

    pd.testing.assert_frame_equal(canonical_1, canonical_2)

    splits_1, _ = split_canonical_dataset(canonical_1, config=config)
    splits_2, _ = split_canonical_dataset(canonical_2, config=config)

    for k in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(splits_1[k], splits_2[k])


def test_preservation_of_existing_files():
    """Verify that baseline existing datasets remain strictly unchanged."""
    baseline_hashes = {
        "data/processed/btc_88000_bbo_1s.parquet": "28a6270cbef34a0d",
        "data/processed/btc_88000_features_5s.parquet": "c89b056464512d8d",
        "data/processed/btc_88000_labeled_5s.parquet": "0306f4c6106923ed",
        "data/processed/market_data.parquet": "2d082c72fe4c6f29",
        "data/processed/labeled_market_data.parquet": "95627294431f1714",
        "data/processed/market_data_extended.parquet": "28ae1c548cbd2aa1",
        "data/raw/btc_observations.jsonl": "ef48ff5842dccc1a",
    }

    for path_str, expected_prefix in baseline_hashes.items():
        p = Path(path_str)
        assert p.exists(), f"Expected baseline file {path_str} does not exist"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        digest = h.hexdigest()
        assert digest.startswith(expected_prefix), (
            f"File {path_str} was modified! Expected hash starting with {expected_prefix}, got {digest[:16]}"
        )
