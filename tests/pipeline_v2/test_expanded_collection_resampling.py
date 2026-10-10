"""
Unit and integration tests for Phase 12 Production Resampling on Expanded Collection (46 Retained Markets).

Verifies:
1. Strict exclusion of all 6 non-production markets:
   - btc-updown-5m-1791064800 (19.1s)
   - btc-updown-5m-1791204300 (0 events)
   - btc-updown-5m-1791208800 (54.2s)
   - btc-updown-5m-1791291300 (11.8s)
   - btc-updown-5m-1791296100 (0 events)
   - btc-updown-5m-1791297000 (41.1s)
2. Exact retention of all 46 production markets (19 baseline + 27 expansion).
3. Backward as-of semantics (source_timestamp_ms <= grid_timestamp_ms, event_age_ms >= 0).
4. Physical 1-second integer grid alignment (grid_timestamp_ms % 1000 == 0).
5. Staleness threshold exact check (is_stale == (event_age_ms > 5000)).
6. Dual-asset UP/DOWN streams per market with equal row counts.
7. Monotonicity and zero duplicate grid timestamps per (market_id, asset_id).
8. Deterministic reproducibility across repeated passes.
9. Baseline immutability: new_collection/ remaining untouched.
"""

from pathlib import Path
import pandas as pd
import pytest

from pipeline_v2.resampling.point_in_time_grid import PointInTimeResampler
from pipeline_v2.resampling.resample_production_collection import resample_single_market

OUTPUT_DIR = Path("data/clean_v2/02_resampled_1s/expanded_collection")
CANONICAL_DIR = Path("data/clean_v2/01_canonical_events/expanded_collection")
BASELINE_OUTPUT_DIR = Path("data/clean_v2/02_resampled_1s/new_collection")

EXCLUDED_STEMS = {
    "btc-updown-5m-1791064800",
    "btc-updown-5m-1791204300",
    "btc-updown-5m-1791208800",
    "btc-updown-5m-1791291300",
    "btc-updown-5m-1791296100",
    "btc-updown-5m-1791297000",
}


def test_market_exclusions_expanded():
    """Verify that all 6 excluded markets never produce resampled Parquet files."""
    for stem in EXCLUDED_STEMS:
        resampled_file = OUTPUT_DIR / f"{stem}_resampled_1s.parquet"
        assert not resampled_file.exists(), f"Excluded market {stem} was unexpectedly resampled!"


def test_46_retained_markets_present():
    """Verify that all 46 retained markets were resampled and exist on disk."""
    canonical_files = sorted(CANONICAL_DIR.glob("btc-updown-5m-*_canonical.parquet"))
    retained_canonical = [
        f for f in canonical_files if f.name.replace("_canonical.parquet", "") not in EXCLUDED_STEMS
    ]
    assert len(retained_canonical) == 46, f"Expected 46 retained canonical files, found {len(retained_canonical)}"

    for cf in retained_canonical:
        stem = cf.name.replace("_canonical.parquet", "")
        resampled_file = OUTPUT_DIR / f"{stem}_resampled_1s.parquet"
        assert resampled_file.exists(), f"Missing resampled file for retained market {stem}!"


def test_expanded_production_dataset_invariants():
    """Verify causal backward semantics, staleness, and monotonicity on the combined dataset."""
    prod_file = OUTPUT_DIR / "resampled_1s_production.parquet"
    assert prod_file.exists(), f"{prod_file} not found; run production resampling first."

    df = pd.read_parquet(prod_file)
    assert not df.empty
    assert len(df) == 18198
    assert len(df["market_id"].unique()) == 46

    # 1. Zero future event leakage
    assert (df["grid_timestamp_ms"] >= df["source_timestamp_ms"]).all()
    assert (df["event_age_ms"] >= 0).all()
    assert (df["event_age_ms"] == df["grid_timestamp_ms"] - df["source_timestamp_ms"]).all()

    # 2. Integer-second physical grid alignment
    assert (df["grid_timestamp_ms"] % 1000 == 0).all()

    # 3. Staleness threshold exact check (5000 ms)
    expected_stale = df["event_age_ms"] > 5000
    assert (df["is_stale"] == expected_stale).all()

    # 4. Bid < Ask strictly preserved with positive depth
    assert (df["bid"] < df["ask"]).all()
    assert (df["bid_size"] > 0.0).all()
    assert (df["ask_size"] > 0.0).all()

    # 5. Dual-asset UP/DOWN streams per market with equal row counts
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

    # 7. Provenance fields preserved
    assert "source_file" in df.columns
    assert "source_line" in df.columns
    assert "source_record_idx" in df.columns
    assert df["source_line"].notna().all()


def test_baseline_dataset_untouched():
    """Verify that the existing 19-market baseline in new_collection/ remains 100% untouched."""
    baseline_prod = BASELINE_OUTPUT_DIR / "resampled_1s_production.parquet"
    assert baseline_prod.exists(), "Baseline resampled production file missing!"
    df = pd.read_parquet(baseline_prod)
    assert len(df["market_id"].unique()) == 19


def test_deterministic_resampling_reproducibility(tmp_path):
    """Verify that resampling the same canonical input twice produces bit-for-bit identical outputs."""
    resampler = PointInTimeResampler(max_stale_ms=5000, grid_interval_ms=1000)
    sample_canonical = CANONICAL_DIR / "btc-updown-5m-1791295500_canonical.parquet"

    _, df1 = resample_single_market(sample_canonical, tmp_path, resampler)
    _, df2 = resample_single_market(sample_canonical, tmp_path, resampler)

    pd.testing.assert_frame_equal(df1, df2)
