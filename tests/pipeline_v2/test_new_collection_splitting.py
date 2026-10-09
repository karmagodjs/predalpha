"""
Unit and integration tests for Phase 15 production collection purged temporal splitting.

Verifies:
1. Chronological ordering: Train < Purge 1 < Validation < Purge 2 < Test.
2. Purge sufficiency: Purge 1 >= 22,000 ms, Purge 2 >= 22,000 ms (mathematical dependency requirement).
3. Zero temporal overlap: zero timestamp overlap across splits.
4. Zero dependency overlap: no label horizon, feature lookback, or sequence lookback crossing.
5. Market isolation: zero shared markets between splits (100% session boundary isolation).
6. Asset isolation: both UP and DOWN tokens present in all splits.
7. Class preservation: balanced UP, DOWN, FLAT representation in all partitions.
8. Bitwise deterministic reproducibility.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_v2.splitting.split_production_collection import (
    run_production_splitting,
)
from pipeline_v2.splitting.temporal_purged_split import (
    calculate_required_purge_ms,
    PurgedTemporalSplitter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_DIR = REPO_ROOT / "data" / "clean_v2" / "04_features" / "new_collection"
SPLITS_DIR = REPO_ROOT / "data" / "clean_v2" / "05_splits" / "new_collection"


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    p = SPLITS_DIR / "train.parquet"
    assert p.exists(), f"Missing train.parquet: {p}"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    p = SPLITS_DIR / "validation.parquet"
    assert p.exists(), f"Missing validation.parquet: {p}"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    p = SPLITS_DIR / "test.parquet"
    assert p.exists(), f"Missing test.parquet: {p}"
    return pd.read_parquet(p)


def test_splits_exist_and_row_counts(train_df, val_df, test_df):
    """Verify partition row counts match 100% of input dataset (4,970 rows)."""
    assert len(train_df) == 3392
    assert len(val_df) == 812
    assert len(test_df) == 766
    assert len(train_df) + len(val_df) + len(test_df) == 4970


def test_chronological_ordering(train_df, val_df, test_df):
    """Verify strict chronological ordering across partitions."""
    ts_col = "grid_timestamp_ms"

    max_train = train_df[ts_col].max()
    min_val = val_df[ts_col].min()
    max_val = val_df[ts_col].max()
    min_test = test_df[ts_col].min()

    assert max_train < min_val, f"Train max ({max_train}) >= Val min ({min_val})"
    assert max_val < min_test, f"Val max ({max_val}) >= Test min ({min_test})"


def test_purge_sufficiency(train_df, val_df, test_df):
    """Verify applied purge gaps exceed required minimum (22,000 ms)."""
    ts_col = "grid_timestamp_ms"
    req_purge = calculate_required_purge_ms(
        max_label_horizon_ms=7000,
        feature_lookback_ms=5000,
        future_sequence_lookback_ms=10000,
    )
    assert req_purge == 22000

    gap1 = int(val_df[ts_col].min() - train_df[ts_col].max())
    gap2 = int(test_df[ts_col].min() - val_df[ts_col].max())

    assert gap1 >= req_purge, f"Purge 1 ({gap1} ms) < required ({req_purge} ms)"
    assert gap2 >= req_purge, f"Purge 2 ({gap2} ms) < required ({req_purge} ms)"
    assert gap1 == 88000
    assert gap2 == 69000


def test_no_temporal_overlap(train_df, val_df, test_df):
    """Verify zero timestamp overlap across all partitions."""
    ts_col = "grid_timestamp_ms"
    train_ts = set(train_df[ts_col])
    val_ts = set(val_df[ts_col])
    test_ts = set(test_df[ts_col])

    assert len(train_ts.intersection(val_ts)) == 0
    assert len(train_ts.intersection(test_ts)) == 0
    assert len(val_ts.intersection(test_ts)) == 0


def test_no_dependency_overlap(train_df, val_df, test_df):
    """
    Verify mathematical dependency isolation:
    1. Label horizon: train_ts + 7s <= val_min
    2. Feature lookback: val_min - 5s >= train_max
    3. Sequence lookback: val_min - 10s >= train_max
    """
    ts_col = "grid_timestamp_ms"
    max_train = train_df[ts_col].max()
    min_val = val_df[ts_col].min()
    max_val = val_df[ts_col].max()
    min_test = test_df[ts_col].min()

    # Train -> Val
    assert (train_df[ts_col] + 7000 <= min_val).all(), "Label horizon leakage into validation!"
    assert min_val - 5000 >= max_train, "Feature lookback leakage from validation into train!"
    assert min_val - 10000 >= max_train, "Sequence lookback leakage from validation into train!"

    # Val -> Test
    assert (val_df[ts_col] + 7000 <= min_test).all(), "Label horizon leakage into test!"
    assert min_test - 5000 >= max_val, "Feature lookback leakage from test into validation!"
    assert min_test - 10000 >= max_val, "Sequence lookback leakage from test into validation!"


def test_market_isolation(train_df, val_df, test_df):
    """Verify 100% market boundary isolation (0 shared markets)."""
    train_mkts = set(train_df["market_id"].unique())
    val_mkts = set(val_df["market_id"].unique())
    test_mkts = set(test_df["market_id"].unique())

    assert len(train_mkts.intersection(val_mkts)) == 0, "Train and Val share markets!"
    assert len(train_mkts.intersection(test_mkts)) == 0, "Train and Test share markets!"
    assert len(val_mkts.intersection(test_mkts)) == 0, "Val and Test share markets!"

    assert len(train_mkts) == 8
    assert len(val_mkts) == 5
    assert len(test_mkts) == 6
    assert len(train_mkts) + len(val_mkts) + len(test_mkts) == 19


def test_asset_isolation_and_coverage(train_df, val_df, test_df):
    """Verify both UP and DOWN tokens exist in all three partitions."""
    for split_name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        # Each market has 2 unique assets
        for mkt, grp in split.groupby("market_id"):
            n_assets = grp["asset_id"].nunique()
            assert n_assets == 2, f"{split_name} market {mkt} does not have both assets!"


def test_class_preservation(train_df, val_df, test_df):
    """Verify directional labels (UP, DOWN, FLAT) are preserved with rich representation."""
    for split_name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split["label"].value_counts(normalize=True)
        assert "UP" in counts
        assert "DOWN" in counts
        assert "FLAT" in counts

        assert counts["UP"] > 0.35, f"{split_name} UP < 35%: {counts['UP']:.2%}"
        assert counts["DOWN"] > 0.35, f"{split_name} DOWN < 35%: {counts['DOWN']:.2%}"
        assert counts["FLAT"] > 0.10, f"{split_name} FLAT < 10%: {counts['FLAT']:.2%}"


def test_deterministic_reproducibility():
    """Verify bitwise reproducibility across repeated split executions."""
    features_pq = FEATURES_DIR / "features_production.parquet"
    df = pd.read_parquet(features_pq)

    splitter = PurgedTemporalSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_gap_ms=22000,
        snap_to_market_boundaries=True,
    )

    t1, v1, te1, m1, r1 = splitter.split(df)
    t2, v2, te2, m2, r2 = splitter.split(df)

    pd.testing.assert_frame_equal(t1, t2, check_exact=True)
    pd.testing.assert_frame_equal(v1, v2, check_exact=True)
    pd.testing.assert_frame_equal(te1, te2, check_exact=True)
    assert m1.to_dict() == m2.to_dict()
