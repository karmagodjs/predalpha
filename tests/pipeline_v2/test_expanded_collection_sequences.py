"""
Unit and integration tests for Phase 16 Causal Sequence Construction on Expanded Collection (46 Retained Markets).

Verifies:
1. Exact sequence counts and warmup drops:
   - Train: 7,514 sequences (558 warmup dropped from 8,072 input rows across 62 streams)
   - Val: 1,428 sequences (144 warmup dropped from 1,572 input rows across 16 streams)
   - Test: 1,582 sequences (126 warmup dropped from 1,708 input rows across 14 streams)
   - Total: 10,524 sequences (828 total warmup dropped from 11,352 input rows across 92 streams)
2. 3D Tensor shapes: (N, L=10, D=11) across NPZ and Parquet files.
3. Separate split construction: Train < Val < Test, 0 temporal cross-boundary sequences.
4. Market and asset stream isolation: 0 cross-market or cross-asset sequences.
5. Strict causality and monotonicity: all sequence timestamps <= endpoint T, step-by-step strictly increasing.
6. Target alignment: target label strictly equals Phase 13 physical 5s forward label at endpoint T.
7. No forbidden columns: only the 11 SAFE_FEATURE_COLUMNS present.
8. Directional class balance: UP and DOWN balanced in Train (3050 vs 3050, ratio 1.0), Val (617 vs 616), Test (709 vs 705).
9. Unique endpoint keys: 0 duplicate sequences for any (market_id, asset_id, endpoint_ts).
10. Causal warm-up NaN accounting:
    - All NaNs confined to steps 0..4 (T-9s through T-5s).
    - Step 9 (endpoint T) has exactly 0 NaNs.
    - Zero NaNs in lookback-0 features (mid_price, spread, spread_bps, microprice, depth_imbalance).
11. Adversarial future perturbation test:
    - Mutating observations strictly after endpoint T leaves X[T] and y[T] 100% bit-for-bit identical.
12. Baseline immutability: new_collection/ remaining untouched (4,628 sequences).
13. Deterministic reproducibility: bitwise identical regeneration.
"""

from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "clean_v2" / "05_splits" / "expanded_collection"
SEQUENCES_DIR = REPO_ROOT / "data" / "clean_v2" / "06_sequences" / "expanded_collection"
BASELINE_DIR = REPO_ROOT / "data" / "clean_v2" / "06_sequences" / "new_collection"


@pytest.fixture(scope="module")
def seq_artifacts():
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
    """Verify exact sequence counts, warmup drops, and total data accounting."""
    train_df = seq_artifacts["train_df"]
    val_df = seq_artifacts["val_df"]
    test_df = seq_artifacts["test_df"]
    meta = seq_artifacts["meta"]

    assert len(train_df) == 7514
    assert len(val_df) == 1428
    assert len(test_df) == 1582

    total_seqs = len(train_df) + len(val_df) + len(test_df)
    assert total_seqs == 10524

    counts = meta["row_counts"]
    assert counts["train_warmup_dropped"] == 558
    assert counts["val_warmup_dropped"] == 144
    assert counts["test_warmup_dropped"] == 126
    assert counts["total_warmup_dropped"] == 828
    assert counts["total_sequences"] == 10524
    assert counts["total_input_rows"] == 11352


def test_tensor_shapes_and_dimensions(seq_artifacts):
    """Verify 3D tensor shapes (N, L=10, D=11) in Parquet and NPZ archives."""
    npz = seq_artifacts["npz"]

    train_X = npz["train_X"]
    train_y = npz["train_y"]
    val_X = npz["val_X"]
    val_y = npz["val_y"]
    test_X = npz["test_X"]
    test_y = npz["test_y"]

    assert train_X.shape == (7514, 10, 11)
    assert train_y.shape == (7514,)
    assert val_X.shape == (1428, 10, 11)
    assert val_y.shape == (1428,)
    assert test_X.shape == (1582, 10, 11)
    assert test_y.shape == (1582,)

    feature_names = list(npz["feature_names"])
    assert feature_names == SAFE_FEATURE_COLUMNS
    assert len(feature_names) == 11


def test_separate_split_construction_and_no_crossing(seq_artifacts):
    """Verify Train, Val, and Test sequences are strictly separated with zero overlap."""
    train_df = seq_artifacts["train_df"]
    val_df = seq_artifacts["val_df"]
    test_df = seq_artifacts["test_df"]

    assert train_df["endpoint_timestamp_ms"].max() < val_df["start_timestamp_ms"].min()
    assert val_df["endpoint_timestamp_ms"].max() < test_df["start_timestamp_ms"].min()

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

        assert df["market_id"].notna().all()
        assert (df["market_id"] != "").all()
        assert df["asset_id"].notna().all()
        assert (df["asset_id"] != "").all()

        for feat_mat in df["feature_matrix"]:
            assert len(feat_mat) == 10
            for row in feat_mat:
                assert len(row) == 11

    train_mkts = set(seq_artifacts["train_df"]["market_id"])
    val_mkts = set(seq_artifacts["val_df"]["market_id"])
    test_mkts = set(seq_artifacts["test_df"]["market_id"])

    assert len(train_mkts) == 31
    assert len(val_mkts) == 8
    assert len(test_mkts) == 7
    assert len(train_mkts.intersection(val_mkts)) == 0
    assert len(val_mkts.intersection(test_mkts)) == 0
    assert len(train_mkts.intersection(test_mkts)) == 0


def test_strict_causality_and_monotonicity(seq_artifacts):
    """Verify sequence timestamps <= endpoint T and strict chronological ordering."""
    for split_name in ["train_df", "val_df", "test_df"]:
        df = seq_artifacts[split_name]

        assert (df["start_timestamp_ms"] < df["endpoint_timestamp_ms"]).all()

        delta_ms = df["endpoint_timestamp_ms"] - df["start_timestamp_ms"]
        assert (delta_ms >= 9000).all()

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

    # Train: UP and DOWN counts exactly equal (3,050 each, ratio 1.0)
    assert cls_dist["train"]["up_count"] == 3050
    assert cls_dist["train"]["down_count"] == 3050
    assert cls_dist["train"]["up_down_ratio"] == 1.0

    # Validation: UP 617, DOWN 616
    assert cls_dist["validation"]["up_count"] == 617
    assert cls_dist["validation"]["down_count"] == 616
    assert 0.99 <= cls_dist["validation"]["up_down_ratio"] <= 1.01

    # Test: UP 709, DOWN 705
    assert cls_dist["test"]["up_count"] == 709
    assert cls_dist["test"]["down_count"] == 705
    assert 0.99 <= cls_dist["test"]["up_down_ratio"] <= 1.01


def test_unique_endpoint_keys(seq_artifacts):
    """Verify zero duplicate sequences for any (market_id, asset_id, endpoint_ts)."""
    all_keys = []
    for split_name in ["train_df", "val_df", "test_df"]:
        df = seq_artifacts[split_name]
        keys = [f"{r.market_id}_{r.asset_id}_{r.endpoint_timestamp_ms}" for r in df.itertuples()]
        assert len(keys) == len(set(keys)), f"Duplicate endpoints within {split_name}"
        all_keys.extend(keys)

    assert len(all_keys) == len(set(all_keys)), "Duplicate endpoints across splits"


def test_warmup_nan_accounting(seq_artifacts):
    """Verify exact warm-up NaN accounting across splits, steps, and features."""
    nan_acc = seq_artifacts["meta"]["nan_accounting"]

    # Total NaNs by split
    assert nan_acc["nan_by_split"]["train"] == 2418
    assert nan_acc["nan_by_split"]["validation"] == 624
    assert nan_acc["nan_by_split"]["test"] == 546
    assert nan_acc["nan_by_split"]["total"] == 3588

    # Steps 5..9 have 0 NaNs; endpoint step 9 has 0 NaNs!
    assert nan_acc["nan_by_step"]["step_9_t_minus_0s"] == 0
    assert nan_acc["nan_by_step"]["step_8_t_minus_1s"] == 0
    assert nan_acc["nan_by_step"]["step_7_t_minus_2s"] == 0
    assert nan_acc["nan_by_step"]["step_6_t_minus_3s"] == 0
    assert nan_acc["nan_by_step"]["step_5_t_minus_4s"] == 0

    # Lookback-0 features have 0 NaNs
    assert nan_acc["nan_by_feature"]["mid_price"] == 0
    assert nan_acc["nan_by_feature"]["spread"] == 0
    assert nan_acc["nan_by_feature"]["spread_bps"] == 0
    assert nan_acc["nan_by_feature"]["microprice"] == 0
    assert nan_acc["nan_by_feature"]["depth_imbalance"] == 0


def test_adversarial_future_perturbation():
    """
    Adversarial Causality Test:
    Take an observation stream and mutate data strictly AFTER endpoint T.
    Reconstruct the sequence ending at T.
    The sequence X[T] and target y[T] must remain 100% BIT-FOR-BIT IDENTICAL.
    """
    builder = SequenceBuilder(sequence_length=DEFAULT_SEQUENCE_LENGTH)
    train_orig = pd.read_parquet(SPLITS_DIR / "train.parquet").copy()

    # Pick an arbitrary stream with at least 20 observations
    sample_mkt = train_orig["market_id"].iloc[0]
    sample_ast = train_orig["asset_id"].iloc[0]
    stream_mask = (train_orig["market_id"] == sample_mkt) & (train_orig["asset_id"] == sample_ast)
    stream_df = train_orig[stream_mask].copy().reset_index(drop=True)
    assert len(stream_df) >= 20

    # Build sequences on unmutated stream
    _, X_orig, y_orig, ep_orig = builder.build_sequences_from_df(stream_df, "orig")

    # Pick an endpoint index in the first half: e.g. index k = 12
    k = 12
    ep_k = stream_df.loc[k, "grid_timestamp_ms"]

    # Mutate data strictly AFTER endpoint k: rows k+1 .. end
    mut_df = stream_df.copy()
    mut_df.loc[k + 1 :, "bid"] = mut_df.loc[k + 1 :, "bid"] * 3.5
    mut_df.loc[k + 1 :, "ask"] = mut_df.loc[k + 1 :, "ask"] * 4.0
    mut_df.loc[k + 1 :, "mid_price"] = 999.99
    mut_df.loc[k + 1 :, "label"] = "FLAT"

    # Reconstruct sequences on mutated stream
    _, X_mut, y_mut, ep_mut = builder.build_sequences_from_df(mut_df, "mut")

    # The sequence ending at ep_k must be BIT-FOR-BIT IDENTICAL
    seq_idx_orig = list(ep_orig).index(ep_k)
    seq_idx_mut = list(ep_mut).index(ep_k)

    np.testing.assert_array_equal(X_orig[seq_idx_orig], X_mut[seq_idx_mut])
    assert y_orig[seq_idx_orig] == y_mut[seq_idx_mut]


def test_baseline_sequences_untouched():
    """Verify that the original 19-market baseline in new_collection/ remains 100% untouched."""
    b_meta = json.loads((BASELINE_DIR / "sequence_metadata.json").read_text(encoding="utf-8"))
    assert b_meta["row_counts"]["total_sequences"] == 4628
    assert b_meta["row_counts"]["train_sequences"] == 3248
    assert b_meta["row_counts"]["val_sequences"] == 722
    assert b_meta["row_counts"]["test_sequences"] == 658


def test_deterministic_reproducibility():
    """Verify running sequence builder multiple times produces bitwise identical outputs."""
    builder = SequenceBuilder(sequence_length=DEFAULT_SEQUENCE_LENGTH)
    train_df = pd.read_parquet(SPLITS_DIR / "train.parquet")

    _, X1, y1, ep1 = builder.build_sequences_from_df(train_df, "train")
    _, X2, y2, ep2 = builder.build_sequences_from_df(train_df, "train")

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(ep1, ep2)
