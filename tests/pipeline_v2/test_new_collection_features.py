"""
Unit and integration tests for Phase 14 production collection causal feature engineering.

Verifies:
1. Future perturbation test: mutating a future observation leaves all earlier features bit-for-bit identical.
2. Zero target-column ingestion: target label and future variables are strictly quarantined.
3. Zero future timestamp access: no shift(-N), center=True, or bfill.
4. Per-market isolation: zero cross-market state leakage.
5. Per-asset isolation: zero cross-asset state leakage between UP and DOWN tokens.
6. Exact label preservation: Phase 13 physical 5-second labels preserved 100%.
7. Microstructure bounds: bid <= mid <= ask, spread > 0, imb in [-1, 1], bid <= microprice <= ask.
8. Zero Infs and zero constant features across all 11 features.
9. Bitwise deterministic reproducibility.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline_v2.features.causal_features import (
    CausalFeatureBuilder,
    FORBIDDEN_INPUT_COLUMNS,
    SAFE_FEATURE_COLUMNS,
)
from pipeline_v2.features.feature_production_collection import (
    FEATURE_LOOKBACK_SPECS,
    run_production_feature_engineering,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LABELED_DIR = REPO_ROOT / "data" / "clean_v2" / "03_labeled_5s" / "new_collection"
FEATURES_DIR = REPO_ROOT / "data" / "clean_v2" / "04_features" / "new_collection"


@pytest.fixture(scope="module")
def production_features_df() -> pd.DataFrame:
    pq = FEATURES_DIR / "features_production.parquet"
    assert pq.exists(), f"Missing features_production.parquet: {pq}"
    return pd.read_parquet(pq)


@pytest.fixture(scope="module")
def production_labeled_df() -> pd.DataFrame:
    pq = LABELED_DIR / "labeled_5s_production.parquet"
    assert pq.exists(), f"Missing labeled_5s_production.parquet: {pq}"
    return pd.read_parquet(pq)


def test_19_retained_markets_features_present(production_features_df):
    """Verify that all 19 retained production markets are present in feature files."""
    per_mkt_files = sorted(list(FEATURES_DIR.glob("*_features.parquet")))
    assert len(per_mkt_files) == 19

    assert production_features_df["market_id"].nunique() == 19
    assert len(production_features_df) == 4970


def test_whitelisted_features_complete(production_features_df):
    """Verify that all 11 whitelisted causal features exist with correct names."""
    for feat in SAFE_FEATURE_COLUMNS:
        assert feat in production_features_df.columns
    assert len(SAFE_FEATURE_COLUMNS) == 11


def test_no_target_column_ingestion(production_features_df):
    """Verify that no forbidden target/future columns are ingested as features."""
    for feat in SAFE_FEATURE_COLUMNS:
        assert feat not in FORBIDDEN_INPUT_COLUMNS
        assert not feat.startswith("future_")
        assert not feat.startswith("target_")

    builder = CausalFeatureBuilder()
    for col in FORBIDDEN_INPUT_COLUMNS:
        assert col not in builder.feature_columns


def test_no_negative_shifts_or_lookahead_syntax():
    """Verify source code contains zero forward-looking shifts or fills."""
    src = inspect.getsource(CausalFeatureBuilder)
    assert "shift(-" not in src
    assert "center=True" not in src
    assert ".bfill" not in src
    assert "bfill()" not in src


def test_future_perturbation_causality():
    """
    Critical Causality Test:
    Mutating an observation at index k must leave all features at indices < k
    100% BIT-FOR-BIT IDENTICAL!
    """
    builder = CausalFeatureBuilder()
    sample_file = LABELED_DIR / "btc-updown-5m-1791204600_labeled_5s.parquet"
    df_orig = pd.read_parquet(sample_file).copy()

    res_orig, _ = builder.compute_features(df_orig)

    # Pick an index in the middle
    k = len(df_orig) // 2
    df_mut = df_orig.copy()

    # Drastically mutate row k
    df_mut.loc[k, "bid"] = df_orig.loc[k, "bid"] * 2.5
    df_mut.loc[k, "ask"] = df_orig.loc[k, "ask"] * 3.0
    df_mut.loc[k, "bid_size"] = 99999.0
    df_mut.loc[k, "ask_size"] = 1.0

    res_mut, _ = builder.compute_features(df_mut)

    # Pre-shock features (indices 0 .. k-1) MUST be identical
    past_orig = res_orig.iloc[:k][SAFE_FEATURE_COLUMNS]
    past_mut = res_mut.iloc[:k][SAFE_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(past_orig, past_mut, check_exact=True)

    # Features at index k must reflect the mutation
    assert res_orig.iloc[k]["mid_price"] != res_mut.iloc[k]["mid_price"]


def test_per_market_isolation():
    """Verify zero cross-market state carry-over."""
    builder = CausalFeatureBuilder()
    f1 = LABELED_DIR / "btc-updown-5m-1791204600_labeled_5s.parquet"
    f2 = LABELED_DIR / "btc-updown-5m-1791204900_labeled_5s.parquet"

    df1 = pd.read_parquet(f1)
    df2 = pd.read_parquet(f2)

    res1, _ = builder.compute_features(df1)
    res2, _ = builder.compute_features(df2)

    combined_input = pd.concat([df1, df2], ignore_index=True)
    res_comb, _ = builder.compute_features(combined_input)

    res1_from_comb = res_comb.iloc[:len(df1)].reset_index(drop=True)
    res2_from_comb = res_comb.iloc[len(df1):].reset_index(drop=True)

    pd.testing.assert_frame_equal(res1[SAFE_FEATURE_COLUMNS], res1_from_comb[SAFE_FEATURE_COLUMNS], check_exact=True)
    pd.testing.assert_frame_equal(res2[SAFE_FEATURE_COLUMNS], res2_from_comb[SAFE_FEATURE_COLUMNS], check_exact=True)


def test_per_asset_isolation():
    """Verify that UP token and DOWN token streams are strictly isolated within a market."""
    builder = CausalFeatureBuilder()
    sample_file = LABELED_DIR / "btc-updown-5m-1791204600_labeled_5s.parquet"
    df = pd.read_parquet(sample_file)

    assets = df["asset_id"].unique()
    assert len(assets) == 2, "Expected 2 assets (UP and DOWN tokens)"
    up_asset, down_asset = assets[0], assets[1]

    res_orig, _ = builder.compute_features(df)

    # Mutate ONLY the UP asset rows
    df_mut = df.copy()
    up_mask = df_mut["asset_id"] == up_asset
    df_mut.loc[up_mask, "bid"] = df_mut.loc[up_mask, "bid"] * 1.5

    res_mut, _ = builder.compute_features(df_mut)

    # The DOWN asset features MUST be completely unaffected
    down_orig = res_orig[res_orig["asset_id"] == down_asset][SAFE_FEATURE_COLUMNS].reset_index(drop=True)
    down_mut = res_mut[res_mut["asset_id"] == down_asset][SAFE_FEATURE_COLUMNS].reset_index(drop=True)

    pd.testing.assert_frame_equal(down_orig, down_mut, check_exact=True)


def test_phase13_labels_preserved_exactly(production_features_df, production_labeled_df):
    """Verify that the target label column is 100% identical to Phase 13 output."""
    assert len(production_features_df) == len(production_labeled_df)
    assert (production_features_df["label"].values == production_labeled_df["label"].values).all()

    counts = production_features_df["label"].value_counts()
    assert counts["UP"] == 2109
    assert counts["DOWN"] == 2108
    assert counts["FLAT"] == 753


def test_microstructure_bounds_and_sanity(production_features_df):
    """Verify that all financial microstructure relationships hold on non-NaN rows."""
    valid_df = production_features_df.dropna(subset=SAFE_FEATURE_COLUMNS)

    # 1. bid <= mid_price <= ask
    assert (valid_df["bid"] <= valid_df["mid_price"] + 1e-9).all()
    assert (valid_df["mid_price"] <= valid_df["ask"] + 1e-9).all()

    # 2. spread > 0
    assert (valid_df["spread"] > 0).all()

    # 3. depth_imbalance in [-1, 1]
    assert (valid_df["depth_imbalance"] >= -1.0 - 1e-9).all()
    assert (valid_df["depth_imbalance"] <= 1.0 + 1e-9).all()

    # 4. microprice between bid and ask
    assert (valid_df["bid"] <= valid_df["microprice"] + 1e-9).all()
    assert (valid_df["microprice"] <= valid_df["ask"] + 1e-9).all()


def test_zero_infs_and_zero_constant_features(production_features_df):
    """Verify zero Infs and zero constant/near-constant features across all 11 features."""
    for col in SAFE_FEATURE_COLUMNS:
        # Zero Infs
        assert not np.isinf(production_features_df[col]).any(), f"Inf detected in feature {col}"

        # Zero constant features
        valid_vals = production_features_df[col].dropna()
        std_val = valid_vals.std()
        nunique_val = valid_vals.nunique()
        assert std_val > 1e-5, f"Feature {col} is near-constant (std={std_val})"
        assert nunique_val > 20, f"Feature {col} has too few unique values ({nunique_val})"


def test_warmup_row_counts_exact(production_features_df):
    """Verify exact warm-up row counts (5 per asset stream x 38 streams = 190)."""
    warmup_mask = production_features_df[SAFE_FEATURE_COLUMNS].isna().any(axis=1)
    assert int(warmup_mask.sum()) == 190

    # Specifically:
    assert int(production_features_df["mid_price"].isna().sum()) == 0
    assert int(production_features_df["spread"].isna().sum()) == 0
    assert int(production_features_df["spread_bps"].isna().sum()) == 0
    assert int(production_features_df["microprice"].isna().sum()) == 0
    assert int(production_features_df["depth_imbalance"].isna().sum()) == 0
    assert int(production_features_df["mid_return_1s"].isna().sum()) == 38
    assert int(production_features_df["bid_change_1s"].isna().sum()) == 38
    assert int(production_features_df["ask_change_1s"].isna().sum()) == 38
    assert int(production_features_df["mid_return_3s"].isna().sum()) == 114
    assert int(production_features_df["mid_return_5s"].isna().sum()) == 190
    assert int(production_features_df["mid_volatility_5s"].isna().sum()) == 190


def test_deterministic_reproducibility():
    """Verify bitwise exact reproducibility of the feature engineering output."""
    builder = CausalFeatureBuilder()
    sample_file = LABELED_DIR / "btc-updown-5m-1791204600_labeled_5s.parquet"
    df = pd.read_parquet(sample_file)

    res1, rep1 = builder.compute_features(df)
    res2, rep2 = builder.compute_features(df)

    pd.testing.assert_frame_equal(res1, res2, check_exact=True)
    assert rep1.to_dict() == rep2.to_dict()
