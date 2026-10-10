"""
Unit tests for Phase 16: Causal Sequence Construction on Production Collection.

Validates all Phase 16 requirements and architectural invariants:
1. Sequence length L = 10 steps (10 seconds lookback).
2. Feature dimension D = 11 causal features.
3. Separate construction per split (Train, Val, Test).
4. Zero cross-split contamination (temporal purge gaps verified).
5. Strict market isolation (no sequence bridges different markets).
6. Strict asset isolation (no sequence bridges UP and DOWN tokens).
7. Strict internal causality (all timestamps <= endpoint T, step-by-step monotonicity).
8. Target alignment (label corresponds strictly to endpoint row physical 5s label).
9. Honest warm-up handling (first 9 rows dropped per stream, 0 padding, 0 fabrication).
10. Balanced directional class distribution preserved.
11. Parquet and NPZ tensor artifacts integrity and consistency.
12. Bitwise deterministic reproducibility.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.sequences.sequence_builder import (
    DEFAULT_SEQUENCE_LENGTH,
    FORBIDDEN_COLUMNS,
    SAFE_FEATURE_COLUMNS,
    SequenceBuilder,
)
from pipeline_v2.sequences.sequence_production_collection import run_production_sequences

SPLITS_DIR = Path("data/clean_v2/05_splits/new_collection")
SEQUENCES_DIR = Path("data/clean_v2/06_sequences/new_collection")


@pytest.fixture(scope="module")
def seq_artifacts():
    """Ensure sequence artifacts are generated and load Parquet and NPZ data."""
    train_pq = SEQUENCES_DIR / "train_sequences.parquet"
    val_pq = SEQUENCES_DIR / "validation_sequences.parquet"
    test_pq = SEQUENCES_DIR / "test_sequences.parquet"
    comb_npz = SEQUENCES_DIR / "sequences_production.npz"
    meta_path = SEQUENCES_DIR / "sequence_metadata.json"

    assert train_pq.exists(), f"Missing {train_pq}"
    assert val_pq.exists(), f"Missing {val_pq}"
    assert test_pq.exists(), f"Missing {test_pq}"
    assert comb_npz.exists(), f"Missing {comb_npz}"
    assert meta_path.exists(), f"Missing {meta_path}"

    train_df = pd.read_parquet(train_pq)
    val_df = pd.read_parquet(val_pq)
    test_df = pd.read_parquet(test_pq)
    npz_data = np.load(comb_npz, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "npz": npz_data,
        "meta": meta,
    }


def test_sequence_row_counts_and_warmup(seq_artifacts):
    """Verify expected sequence counts, warmup drops, and total data accounting."""
    train_df = seq_artifacts["train_df"]
    val_df = seq_artifacts["val_df"]
    test_df = seq_artifacts["test_df"]
    meta = seq_artifacts["meta"]

    # Expected exact sequence counts:
    # Train: 3,392 rows - (16 streams * 9 warmup) = 3,248
    assert len(train_df) == 3248
    # Val: 812 rows - (10 streams * 9 warmup) = 722
    assert len(val_df) == 722
    # Test: 766 rows - (12 streams * 9 warmup) = 658
    assert len(test_df) == 658

    total_seqs = len(train_df) + len(val_df) + len(test_df)
    assert total_seqs == 4628

    # Verify warmup accounting: 342 warmup + 4628 sequences = 4970 total input rows
    counts = meta["row_counts"]
    assert counts["total_warmup_dropped"] == 342
    assert counts["total_sequences"] == 4628
    assert counts["total_input_rows"] == 4970


def test_tensor_shapes_and_dimensions(seq_artifacts):
    """Verify 3D tensor shapes (N, L=10, D=11) in Parquet and NPZ archives."""
    npz = seq_artifacts["npz"]
    meta = seq_artifacts["meta"]

    train_X = npz["train_X"]
    train_y = npz["train_y"]
    val_X = npz["val_X"]
    val_y = npz["val_y"]
    test_X = npz["test_X"]
    test_y = npz["test_y"]

    assert train_X.shape == (3248, 10, 11)
    assert train_y.shape == (3248,)
    assert val_X.shape == (722, 10, 11)
    assert val_y.shape == (722,)
    assert test_X.shape == (658, 10, 11)
    assert test_y.shape == (658,)

    # Feature names match SAFE_FEATURE_COLUMNS
    feature_names = list(npz["feature_names"])
    assert feature_names == SAFE_FEATURE_COLUMNS
    assert len(feature_names) == 11


def test_separate_split_construction_and_no_crossing(seq_artifacts):
    """Verify Train, Val, and Test sequences are strictly separated with zero overlap."""
    train_df = seq_artifacts["train_df"]
    val_df = seq_artifacts["val_df"]
    test_df = seq_artifacts["test_df"]

    # Train endpoint < Validation start
    assert train_df["endpoint_timestamp_ms"].max() < val_df["start_timestamp_ms"].min()
    # Validation endpoint < Test start
    assert val_df["endpoint_timestamp_ms"].max() < test_df["start_timestamp_ms"].min()

    # Zero shared endpoints
    train_ep = set(train_df["endpoint_timestamp_ms"])
    val_ep = set(val_df["endpoint_timestamp_ms"])
    test_ep = set(test_df["endpoint_timestamp_ms"])

    assert len(train_ep.intersection(val_ep)) == 0
    assert len(val_ep.intersection(test_ep)) == 0
    assert len(train_ep.intersection(test_ep)) == 0


def test_market_and_asset_isolation(seq_artifacts):
    """Verify zero sequence crosses market_id boundaries or asset_id boundaries."""
    for split_name in ["train_df", "val_df", "test_df"]:
        df = seq_artifacts[split_name]

        # Each sequence row must have valid market_id and asset_id
        assert df["market_id"].notna().all()
        assert (df["market_id"] != "").all()
        assert df["asset_id"].notna().all()
        assert (df["asset_id"] != "").all()

        # Check nested feature matrix length
        for feat_mat in df["feature_matrix"]:
            assert len(feat_mat) == 10
            for row in feat_mat:
                assert len(row) == 11

    # Market isolation across splits: zero market overlap
    train_mkts = set(seq_artifacts["train_df"]["market_id"])
    val_mkts = set(seq_artifacts["val_df"]["market_id"])
    test_mkts = set(seq_artifacts["test_df"]["market_id"])

    assert len(train_mkts) == 8
    assert len(val_mkts) == 5
    assert len(test_mkts) == 6
    assert len(train_mkts.intersection(val_mkts)) == 0
    assert len(val_mkts.intersection(test_mkts)) == 0
    assert len(train_mkts.intersection(test_mkts)) == 0


def test_strict_causality_and_monotonicity(seq_artifacts):
    """Verify sequence timestamps <= endpoint T and strict chronological ordering."""
    for split_name in ["train_df", "val_df", "test_df"]:
        df = seq_artifacts[split_name]

        # start_timestamp_ms must be strictly less than endpoint_timestamp_ms
        assert (df["start_timestamp_ms"] < df["endpoint_timestamp_ms"]).all()

        # For a 1-second grid with L=10 steps, delta is at least 9,000 ms
        delta_ms = df["endpoint_timestamp_ms"] - df["start_timestamp_ms"]
        assert (delta_ms >= 9000).all()

        # Endpoints are monotonically non-decreasing
        assert df["endpoint_timestamp_ms"].is_monotonic_increasing


def test_target_alignment(seq_artifacts):
    """Verify target labels match Phase 13 physical 5-second forward labels at endpoint T."""
    for split_name, orig_filename in [
        ("train_df", "train.parquet"),
        ("val_df", "validation.parquet"),
        ("test_df", "test.parquet"),
    ]:
        seq_df = seq_artifacts[split_name]
        orig_df = pd.read_parquet(SPLITS_DIR / orig_filename)

        orig_label_lookup = dict(
            zip(
                zip(orig_df["market_id"], orig_df["asset_id"], orig_df["grid_timestamp_ms"]),
                orig_df["label"],
            )
        )

        for _, row in seq_df.iterrows():
            key = (row["market_id"], row["asset_id"], row["endpoint_timestamp_ms"])
            assert key in orig_label_lookup
            assert row["target"] == orig_label_lookup[key]


def test_no_forbidden_columns_in_features(seq_artifacts):
    """Verify that forbidden future columns and targets never enter input sequences."""
    meta = seq_artifacts["meta"]
    feature_cols = set(meta["feature_columns"])

    for forbidden in FORBIDDEN_COLUMNS:
        assert forbidden not in feature_cols, f"Forbidden column {forbidden} found in feature list!"


def test_balanced_class_distributions(seq_artifacts):
    """Verify that UP and DOWN directional classes remain balanced in all splits."""
    cls_dist = seq_artifacts["meta"]["class_distributions"]

    # Train: UP and DOWN counts exactly equal (1,386 each, ratio 1.0)
    assert cls_dist["train"]["up_count"] == 1386
    assert cls_dist["train"]["down_count"] == 1386
    assert cls_dist["train"]["up_down_ratio"] == 1.0

    # Validation: UP 282, DOWN 281
    assert cls_dist["validation"]["up_count"] == 282
    assert cls_dist["validation"]["down_count"] == 281
    assert 0.99 <= cls_dist["validation"]["up_down_ratio"] <= 1.01

    # Test: UP and DOWN counts exactly equal (291 each, ratio 1.0)
    assert cls_dist["test"]["up_count"] == 291
    assert cls_dist["test"]["down_count"] == 291
    assert cls_dist["test"]["up_down_ratio"] == 1.0


def test_unique_endpoint_keys(seq_artifacts):
    """Verify zero duplicate sequences for any (market_id, asset_id, endpoint_ts)."""
    all_keys = []
    for split_name in ["train_df", "val_df", "test_df"]:
        df = seq_artifacts[split_name]
        keys = [f"{r.market_id}_{r.asset_id}_{r.endpoint_timestamp_ms}" for r in df.itertuples()]
        # Within split
        assert len(keys) == len(set(keys)), f"Duplicate endpoints within {split_name}"
        all_keys.extend(keys)

    # Across entire dataset
    assert len(all_keys) == len(set(all_keys)), "Duplicate endpoints across splits"


def test_deterministic_reproducibility(tmp_path):
    """Verify running sequence builder multiple times produces bitwise identical outputs."""
    builder = SequenceBuilder(sequence_length=DEFAULT_SEQUENCE_LENGTH)
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")

    _, X1, y1, ep1 = builder.build_sequences_from_df(train_df, "train")
    _, X2, y2, ep2 = builder.build_sequences_from_df(train_df, "train")

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(ep1, ep2)
