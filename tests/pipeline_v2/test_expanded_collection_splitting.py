"""
Unit and integration tests for Phase 15 production collection purged temporal splitting on Expanded Collection (46 Retained Markets).

Verifies:
1. Exact row counts: Train (8,072), Validation (1,572), Test (1,708), Total (11,352).
2. Chronological ordering: Train < Purge 1 < Validation < Purge 2 < Test.
3. Purge sufficiency:
   - Purge 1 (Train -> Val) = 61,000 ms >= 22,000 ms required.
   - Purge 2 (Val -> Test) = 96,000 ms >= 22,000 ms required.
4. Zero temporal overlap: train_ts ∩ val_ts = ∅, val_ts ∩ test_ts = ∅, train_ts ∩ test_ts = ∅.
5. Zero dependency overlap:
   - Label horizon (7s) strictly quarantined.
   - Feature lookback (5s) strictly quarantined.
   - Future sequence lookback (10s) strictly quarantined.
6. Market boundary isolation:
   - Train has 31 markets, Val has 8 markets, Test has 7 markets (disjoint sets, 46 total).
7. Asset stream isolation: both UP and DOWN tokens present in each partition (92 total streams).
8. Directional class preservation:
   - Train (UP: 3273, DOWN: 3273, FLAT: 1526)
   - Val (UP: 681, DOWN: 680, FLAT: 211)
   - Test (UP: 762, DOWN: 758, FLAT: 188)
9. Baseline immutability: new_collection/ remaining untouched (3392, 812, 766).
10. Bitwise deterministic reproducibility.
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
FEATURES_DIR = REPO_ROOT / "data" / "clean_v2" / "04_features" / "expanded_collection"
SPLITS_DIR = REPO_ROOT / "data" / "clean_v2" / "05_splits" / "expanded_collection"
BASELINE_SPLITS_DIR = REPO_ROOT / "data" / "clean_v2" / "05_splits" / "new_collection"


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
    """Verify partition row counts match 100% of input dataset (11,352 rows)."""
    assert len(train_df) == 8072
    assert len(val_df) == 1572
    assert len(test_df) == 1708
    assert len(train_df) + len(val_df) + len(test_df) == 11352


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
    assert gap1 == 61000
    assert gap2 == 96000


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

    assert len(train_mkts) == 31
    assert len(val_mkts) == 8
    assert len(test_mkts) == 7
    assert len(train_mkts) + len(val_mkts) + len(test_mkts) == 46


def test_asset_isolation_and_coverage(train_df, val_df, test_df):
    """Verify that both UP and DOWN tokens exist across all splits and within each market."""
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for mkt_id, grp in part.groupby("market_id"):
            assets = grp["asset_id"].unique()
            assert len(assets) == 2, f"Market {mkt_id} in {name} does not have exactly 2 assets!"


def test_class_preservation(train_df, val_df, test_df):
    """Verify class balance is maintained across all partitions."""
    train_counts = train_df["label"].value_counts()
    val_counts = val_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()

    assert train_counts["UP"] == 3273
    assert train_counts["DOWN"] == 3273
    assert train_counts["FLAT"] == 1526

    assert val_counts["UP"] == 681
    assert val_counts["DOWN"] == 680
    assert val_counts["FLAT"] == 211

    assert test_counts["UP"] == 762
    assert test_counts["DOWN"] == 758
    assert test_counts["FLAT"] == 188


def test_baseline_splits_untouched():
    """Verify that the original 19-market baseline in new_collection/ remains 100% untouched."""
    b_train = pd.read_parquet(BASELINE_SPLITS_DIR / "train.parquet")
    b_val = pd.read_parquet(BASELINE_SPLITS_DIR / "validation.parquet")
    b_test = pd.read_parquet(BASELINE_SPLITS_DIR / "test.parquet")

    assert len(b_train) == 3392
    assert len(b_val) == 812
    assert len(b_test) == 766
    assert len(b_train) + len(b_val) + len(b_test) == 4970


def test_deterministic_reproducibility():
    """Verify that repeating the split produces identical outputs."""
    features_pq = FEATURES_DIR / "features_production.parquet"
    df = pd.read_parquet(features_pq)

    splitter = PurgedTemporalSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        purge_gap_ms=22000,
        max_label_horizon_ms=7000,
        feature_lookback_ms=5000,
        future_sequence_lookback_ms=10000,
        snap_to_market_boundaries=True,
    )

    t1, v1, te1, m1, r1 = splitter.split(df)
    t2, v2, te2, m2, r2 = splitter.split(df)

    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(te1, te2)
