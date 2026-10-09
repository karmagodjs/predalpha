"""
Unit and integration tests for Phase 12 Production Resampling on 19 Retained Markets.

Verifies:
1. Exclusion of 1791204300 and 1791208800.
2. Exact retention of the 19 production markets.
3. Independent dual-asset (UP/DOWN) resampling per market.
4. Strict causal backward as-of semantics (zero future-leakage).
5. Exact integer-second grid alignment and strict monotonicity.
6. Staleness threshold (5000 ms) exact enforcement.
7. Zero cross-market and zero cross-asset contamination.
8. Deterministic reproducibility.
"""

from pathlib import Path
import pandas as pd
import pytest

from pipeline_v2.resampling.point_in_time_grid import PointInTimeResampler
from pipeline_v2.resampling.resample_production_collection import (
    EXCLUDED_MARKET_STEMS,
    resample_single_market,
)

OUTPUT_DIR = Path("data/clean_v2/02_resampled_1s/new_collection")
CANONICAL_DIR = Path("data/clean_v2/01_canonical_events/new_collection")


def test_market_exclusions():
    """Verify that excluded markets never produce Phase 12 resampled datasets."""
    for stem in EXCLUDED_MARKET_STEMS:
        resampled_file = OUTPUT_DIR / f"{stem}_resampled_1s.parquet"
        assert not resampled_file.exists(), f"Excluded market {stem} was unexpectedly resampled!"


def test_19_retained_markets_present():
    """Verify that all 19 retained production markets were resampled."""
    canonical_files = sorted(CANONICAL_DIR.glob("btc-updown-5m-*_canonical.parquet"))
    retained_canonical = [
        f for f in canonical_files if f.name.replace("_canonical.parquet", "") not in EXCLUDED_MARKET_STEMS
    ]
    assert len(retained_canonical) == 19

    for cf in retained_canonical:
        stem = cf.name.replace("_canonical.parquet", "")
        resampled_file = OUTPUT_DIR / f"{stem}_resampled_1s.parquet"
        assert resampled_file.exists(), f"Missing resampled file for retained market {stem}!"


def test_production_dataset_invariants():
    """Verify causal backward semantics, staleness, and monotonicity on production dataset."""
    prod_file = OUTPUT_DIR / "resampled_1s_production.parquet"
    if not prod_file.exists():
        pytest.skip(f"{prod_file} not found; run production resampling first.")

    df = pd.read_parquet(prod_file)
    assert not df.empty
    assert len(df["market_id"].unique()) == 19

    # 1. Zero future event leakage
    assert (df["grid_timestamp_ms"] >= df["source_timestamp_ms"]).all()
    assert (df["event_age_ms"] >= 0).all()
    assert (df["event_age_ms"] == df["grid_timestamp_ms"] - df["source_timestamp_ms"]).all()

    # 2. Integer-second grid alignment
    assert (df["grid_timestamp_ms"] % 1000 == 0).all()

    # 3. Staleness threshold exact check
    expected_stale = df["event_age_ms"] > 5000
    assert (df["is_stale"] == expected_stale).all()

    # 4. Bid < Ask strictly preserved
    assert (df["bid"] < df["ask"]).all()
    assert (df["bid_size"] > 0.0).all()
    assert (df["ask_size"] > 0.0).all()

    # 5. Dual-asset representation per market
    for mkt_id, mkt_group in df.groupby("market_id"):
        assets = mkt_group["asset_id"].unique()
        assert len(assets) == 2, f"Market {mkt_id} has {len(assets)} assets instead of 2!"
        a1_rows = len(mkt_group[mkt_group["asset_id"] == assets[0]])
        a2_rows = len(mkt_group[mkt_group["asset_id"] == assets[1]])
        assert a1_rows == a2_rows, f"Unequal grid rows between tokens in market {mkt_id}: {a1_rows} vs {a2_rows}"

    # 6. Monotonicity and zero duplicate grid timestamps per (market_id, asset_id)
    for _, stream in df.groupby(["market_id", "asset_id"]):
        assert stream["grid_timestamp_ms"].is_monotonic_increasing
        assert not stream["grid_timestamp_ms"].duplicated().any()


def test_deterministic_resampling_reproducibility(tmp_path):
    """Verify that resampling the same canonical input twice produces bit-for-bit identical outputs."""
    resampler = PointInTimeResampler(max_stale_ms=5000, grid_interval_ms=1000)
    sample_canonical = CANONICAL_DIR / "btc-updown-5m-1791204600_canonical.parquet"

    _, df1 = resample_single_market(sample_canonical, tmp_path, resampler)
    _, df2 = resample_single_market(sample_canonical, tmp_path, resampler)

    pd.testing.assert_frame_equal(df1, df2)
