"""
Unit and integration tests for Phase 13 production collection labeling.

Verifies:
1. Strict exclusion of invalid markets (1791204300, 1791208800).
2. Exactly 19 retained production markets present in output directory.
3. Strict physical horizon invariant: 5000 ms <= actual_delta_ms <= 7000 ms on 100% of rows.
4. First valid event selection: target_timestamp >= current_timestamp + 5000 ms.
5. Zero row-shift assumptions (no shift(-5)).
6. Zero target leakage in clean training dataset (no target_*, future_*, delta, threshold).
7. Stale observations excluded (zero rows with is_stale == True).
8. Dual-asset and session isolation (UP tokens never labeled with DOWN events, zero cross-market matching).
9. Rich, non-trivial class balance (UP ~42%, DOWN ~42%, FLAT ~15%).
10. Bitwise deterministic reproducibility.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_v2.labeling.label_production_collection import (
    EXCLUDED_MARKET_STEMS,
    label_single_market,
)
from pipeline_v2.labeling.physical_horizon_labeler import (
    PhysicalHorizonLabeler,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LABELED_DIR = REPO_ROOT / "data" / "clean_v2" / "03_labeled_5s" / "new_collection"
RESAMPLED_DIR = REPO_ROOT / "data" / "clean_v2" / "02_resampled_1s" / "new_collection"
CANONICAL_DIR = REPO_ROOT / "data" / "clean_v2" / "01_canonical_events" / "new_collection"


@pytest.fixture(scope="module")
def combined_clean_df() -> pd.DataFrame:
    p = LABELED_DIR / "labeled_5s_production.parquet"
    assert p.exists(), f"Missing combined labeled parquet: {p}"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def combined_audit_df() -> pd.DataFrame:
    p = LABELED_DIR / "labeling_audit_production.parquet"
    assert p.exists(), f"Missing combined audit parquet: {p}"
    return pd.read_parquet(p)


def test_market_exclusions_strictly_respected(combined_clean_df, combined_audit_df):
    """Verify that both excluded markets are completely absent from outputs."""
    for excluded in EXCLUDED_MARKET_STEMS:
        clean_files = list(LABELED_DIR.glob(f"{excluded}*"))
        assert len(clean_files) == 0, f"Found output file for excluded market: {clean_files}"

        # Check market_id in combined dataframes
        for mkt_id in combined_clean_df["market_id"].unique():
            assert excluded not in mkt_id
        for mkt_id in combined_audit_df["market_id"].unique():
            assert excluded not in mkt_id


def test_19_retained_markets_present(combined_clean_df, combined_audit_df):
    """Verify exactly 19 retained production markets are present in clean and audit outputs."""
    clean_market_files = sorted(list(LABELED_DIR.glob("*_labeled_5s.parquet")))
    audit_market_files = sorted(list(LABELED_DIR.glob("*_labeling_audit.parquet")))

    assert len(clean_market_files) == 19
    assert len(audit_market_files) == 19

    assert combined_clean_df["market_id"].nunique() == 19
    assert combined_audit_df["market_id"].nunique() == 19


def test_physical_horizon_invariants_strictly_enforced(combined_audit_df):
    """Verify that 100% of labeled rows satisfy 5000 ms <= horizon <= 7000 ms."""
    horizons = combined_audit_df["physical_horizon_ms"].to_numpy()
    assert len(horizons) > 0
    assert (horizons >= 5000).all(), f"Found horizons < 5000 ms: {horizons[horizons < 5000]}"
    assert (horizons <= 7000).all(), f"Found horizons > 7000 ms: {horizons[horizons > 7000]}"

    deltas = combined_audit_df["target_timestamp"].to_numpy() - combined_audit_df["current_timestamp"].to_numpy()
    assert (deltas >= 5000).all()
    assert (deltas <= 7000).all()
    assert (deltas == horizons).all()


def test_zero_target_leakage_in_clean_dataset(combined_clean_df):
    """Verify that zero target-derived or future columns exist in clean training dataset."""
    forbidden_prefixes = ("target_", "future_", "delta", "threshold", "physical_horizon", "horizon_")
    for col in combined_clean_df.columns:
        for prefix in forbidden_prefixes:
            assert not col.startswith(prefix), f"Target leakage detected in clean dataset: column '{col}'"

    # Verify explicitly whitelisted/allowed columns
    expected_cols = {
        "grid_timestamp_ms",
        "source_timestamp_ms",
        "source_sequence_id",
        "market_id",
        "asset_id",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "source_file",
        "source_line",
        "source_record_idx",
        "event_age_ms",
        "is_stale",
        "source_events_in_interval",
        "label",
    }
    assert set(combined_clean_df.columns) == expected_cols


def test_stale_observations_excluded(combined_clean_df):
    """Verify that zero stale observations were labeled in the clean training dataset."""
    assert (combined_clean_df["is_stale"] == False).all()


def test_dual_asset_and_session_isolation(combined_clean_df, combined_audit_df):
    """Verify that asset_id and market_id match 100% between clean observations and audit targets."""
    assert len(combined_clean_df) == len(combined_audit_df)
    clean_assets = combined_clean_df["asset_id"].to_numpy()
    audit_assets = combined_audit_df["asset_id"].to_numpy()
    assert (clean_assets == audit_assets).all(), "Asset mismatch between current observation and target!"

    clean_markets = combined_clean_df["market_id"].to_numpy()
    audit_markets = combined_audit_df["market_id"].to_numpy()
    assert (clean_markets == audit_markets).all(), "Market mismatch between current observation and target!"


def test_class_balance_non_trivial(combined_clean_df):
    """Verify that UP, DOWN, and FLAT all have rich, non-trivial representations (> 10%)."""
    counts = combined_clean_df["label"].value_counts()
    total = len(combined_clean_df)

    assert "UP" in counts
    assert "DOWN" in counts
    assert "FLAT" in counts

    up_pct = counts["UP"] / total
    down_pct = counts["DOWN"] / total
    flat_pct = counts["FLAT"] / total

    assert up_pct > 0.35, f"Expected UP > 35%, got {up_pct:.2%}"
    assert down_pct > 0.35, f"Expected DOWN > 35%, got {down_pct:.2%}"
    assert flat_pct > 0.10, f"Expected FLAT > 10%, got {flat_pct:.2%}"


def test_no_shift_minus_5_in_source():
    """Verify that row-based shift(-5) is nowhere in the labeler implementation."""
    src = inspect.getsource(PhysicalHorizonLabeler)
    assert "shift(-5)" not in src
    assert ".shift(" not in src


def test_deterministic_reproducibility():
    """Verify that re-running label_single_market produces identical results."""
    labeler = PhysicalHorizonLabeler(min_horizon_ms=5000, max_horizon_ms=7000)
    test_res_file = RESAMPLED_DIR / "btc-updown-5m-1791204600_resampled_1s.parquet"
    test_can_file = CANONICAL_DIR / "btc-updown-5m-1791204600_canonical.parquet"

    clean_1, audit_1, rep_1 = label_single_market(test_res_file, test_can_file, labeler)
    clean_2, audit_2, rep_2 = label_single_market(test_res_file, test_can_file, labeler)

    pd.testing.assert_frame_equal(clean_1, clean_2)
    pd.testing.assert_frame_equal(audit_1, audit_2)
    assert rep_1.to_dict() == rep_2.to_dict()
