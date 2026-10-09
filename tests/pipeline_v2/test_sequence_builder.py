"""
Unit tests for Pipeline V2 Causal Sequence Builder.

Covers all Phase 16 requirements:
A. Correct sequence length
B. Chronological ordering
C. Endpoint correctness
D. No future-feature access (causality)
E. No train/validation crossing
F. No validation/test crossing
G. No session crossing
H. No duplicate timestamps
I. Deterministic output
J. Insufficient-data behavior
K. Target comes from endpoint
L. Forbidden future columns rejected/excluded
M. No random shuffling
N. Correct feature dimensionality
O. Metadata correctness
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
    SequenceMetadata,
    SequenceReport,
)


@pytest.fixture
def builder():
    return SequenceBuilder(sequence_length=10)


def create_feature_df(n_rows: int = 50, start_ts: int = 1_000_000, step_ms: int = 1000, session_id: str = "sess_1") -> pd.DataFrame:
    """Generate synthetic chronological tabular feature observations."""
    records = []
    labels = ["FLAT", "UP", "DOWN"]
    for i in range(n_rows):
        ts = start_ts + i * step_ms
        records.append({
            "grid_timestamp_ms": ts,
            "source_timestamp_ms": ts - 200,
            "source_sequence_id": i + 1,
            "market_id": "mkt_test",
            "asset_id": "ast_test",
            "bid": 0.40 + (i % 10) * 0.01,
            "ask": 0.60 + (i % 10) * 0.01,
            "bid_size": 100.0 + i,
            "ask_size": 100.0 + i,
            "mid_price": 0.50 + (i % 10) * 0.01,
            "spread": 0.20,
            "spread_bps": 4000.0,
            "mid_return_1s": 0.001 * (i % 3 - 1),
            "mid_return_3s": 0.002 * (i % 3 - 1),
            "mid_return_5s": 0.003 * (i % 3 - 1),
            "mid_volatility_5s": 0.005,
            "bid_change_1s": 0.01,
            "ask_change_1s": 0.01,
            "microprice": 0.50,
            "depth_imbalance": 0.1,
            "label": labels[i % 3],
            "session_id": session_id,
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


# A. Correct sequence length
def test_correct_sequence_length(builder):
    df = create_feature_df(n_rows=30)
    seq_df, X, y, endpoints = builder.build_sequences_from_df(df, "test_split")

    assert len(seq_df) == 30 - 10 + 1
    assert X.shape == (21, 10, len(SAFE_FEATURE_COLUMNS))
    assert (seq_df["sequence_length"] == 10).all()
    for row_mat in seq_df["feature_matrix"]:
        assert len(row_mat) == 10

    # Test configurable sequence length L=5
    b5 = SequenceBuilder(sequence_length=5)
    seq_df5, X5, _, _ = b5.build_sequences_from_df(df, "test_split")
    assert len(seq_df5) == 30 - 5 + 1
    assert X5.shape == (26, 5, len(SAFE_FEATURE_COLUMNS))


# B. Chronological ordering
def test_chronological_ordering(builder):
    df = create_feature_df(n_rows=25)
    seq_df, X, y, endpoints = builder.build_sequences_from_df(df)

    # Across sequences, endpoints must be strictly increasing
    assert pd.Series(endpoints).is_monotonic_increasing
    assert len(endpoints) == len(set(endpoints))

    # Within each sequence, timestamps must be strictly increasing
    for _, row in seq_df.iterrows():
        assert row["start_timestamp_ms"] < row["endpoint_timestamp_ms"]


# C. Endpoint correctness
def test_endpoint_correctness(builder):
    df = create_feature_df(n_rows=20)
    seq_df, X, y, endpoints = builder.build_sequences_from_df(df)

    # For sequence index 0 (rows 0 to 9), endpoint must be row 9
    assert seq_df.iloc[0]["endpoint_timestamp_ms"] == df.iloc[9]["grid_timestamp_ms"]
    assert seq_df.iloc[0]["start_timestamp_ms"] == df.iloc[0]["grid_timestamp_ms"]

    # For sequence index 5 (rows 5 to 14), endpoint must be row 14
    assert seq_df.iloc[5]["endpoint_timestamp_ms"] == df.iloc[14]["grid_timestamp_ms"]
    assert seq_df.iloc[5]["start_timestamp_ms"] == df.iloc[5]["grid_timestamp_ms"]


# D. No future-feature access
def test_no_future_feature_access(builder):
    df = create_feature_df(n_rows=20)
    seq_df1, X1, _, _ = builder.build_sequences_from_df(df)

    # Mutate row 15 (which is in the future for sequence 0 ending at row 9)
    df_mutated = df.copy()
    df_mutated.loc[15, "mid_price"] = 999.99

    seq_df2, X2, _, _ = builder.build_sequences_from_df(df_mutated)

    # Sequence 0 must be 100% identical between unmutated and mutated
    np.testing.assert_array_equal(X1[0], X2[0])
    assert seq_df1.iloc[0]["endpoint_timestamp_ms"] == seq_df2.iloc[0]["endpoint_timestamp_ms"]


# E. No train/validation crossing
def test_no_train_validation_crossing(tmp_path):
    train_df = create_feature_df(n_rows=20, start_ts=1_000_000, step_ms=1000)
    # 20s purge gap before val
    val_df = create_feature_df(n_rows=20, start_ts=1_050_000, step_ms=1000)
    test_df = create_feature_df(n_rows=20, start_ts=1_100_000, step_ms=1000)

    tr_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    te_p = tmp_path / "test.parquet"
    train_df.to_parquet(tr_p, index=False)
    val_df.to_parquet(val_p, index=False)
    test_df.to_parquet(te_p, index=False)

    out_d = tmp_path / "seq_out"
    b = SequenceBuilder(sequence_length=10)
    meta, rep = b.build_all_splits(tr_p, val_p, te_p, out_d)

    assert meta.cross_split_violations == 0
    assert rep.cross_split_violations == 0

    train_seq = pd.read_parquet(out_d / "train_sequences.parquet")
    val_seq = pd.read_parquet(out_d / "validation_sequences.parquet")

    # Max timestamp in train sequences < Min timestamp in val sequences
    assert train_seq["endpoint_timestamp_ms"].max() < val_seq["start_timestamp_ms"].min()


# F. No validation/test crossing
def test_no_validation_test_crossing(tmp_path):
    train_df = create_feature_df(n_rows=20, start_ts=1_000_000, step_ms=1000)
    val_df = create_feature_df(n_rows=20, start_ts=1_050_000, step_ms=1000)
    test_df = create_feature_df(n_rows=20, start_ts=1_100_000, step_ms=1000)

    tr_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    te_p = tmp_path / "test.parquet"
    train_df.to_parquet(tr_p, index=False)
    val_df.to_parquet(val_p, index=False)
    test_df.to_parquet(te_p, index=False)

    out_d = tmp_path / "seq_out"
    b = SequenceBuilder(sequence_length=10)
    meta, rep = b.build_all_splits(tr_p, val_p, te_p, out_d)

    val_seq = pd.read_parquet(out_d / "validation_sequences.parquet")
    test_seq = pd.read_parquet(out_d / "test_sequences.parquet")

    # Max timestamp in val sequences < Min timestamp in test sequences
    assert val_seq["endpoint_timestamp_ms"].max() < test_seq["start_timestamp_ms"].min()


# G. No session crossing
def test_no_session_crossing(builder):
    df_s1 = create_feature_df(n_rows=15, start_ts=1_000_000, step_ms=1000, session_id="session_A")
    df_s2 = create_feature_df(n_rows=15, start_ts=1_020_000, step_ms=1000, session_id="session_B")
    combined_df = pd.concat([df_s1, df_s2], ignore_index=True)

    seq_df, X, y, endpoints = builder.build_sequences_from_df(combined_df)

    # 15 - 10 + 1 = 6 sequences for session A, 6 sequences for session B -> 12 total
    assert len(seq_df) == 12
    # Verify each sequence only belongs to one session
    assert set(seq_df["session_id"].unique()) == {"session_A", "session_B"}
    assert (seq_df["session_id"] == "session_A").sum() == 6
    assert (seq_df["session_id"] == "session_B").sum() == 6


# H. No duplicate timestamps
def test_no_duplicate_timestamps(builder):
    df = create_feature_df(n_rows=20)
    # Introduce duplicate timestamp
    dup_row = df.iloc[5:6].copy()
    corrupt_df = pd.concat([df.iloc[:6], dup_row, df.iloc[6:]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate timestamps detected"):
        builder.build_sequences_from_df(corrupt_df)


# I. Deterministic output
def test_deterministic_output(builder):
    df = create_feature_df(n_rows=25)
    seq1, X1, y1, ep1 = builder.build_sequences_from_df(df)
    seq2, X2, y2, ep2 = builder.build_sequences_from_df(df)

    pd.testing.assert_frame_equal(seq1, seq2)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(ep1, ep2)


# J. Insufficient-data behavior
def test_insufficient_data_behavior(builder):
    # Only 5 rows (< 10)
    small_df = create_feature_df(n_rows=5)
    seq_df, X, y, endpoints = builder.build_sequences_from_df(small_df)

    assert len(seq_df) == 0
    assert X.shape == (0, 10, len(SAFE_FEATURE_COLUMNS))
    assert len(y) == 0
    assert len(endpoints) == 0


# K. Target comes from endpoint
def test_target_comes_from_endpoint(builder):
    df = create_feature_df(n_rows=15)
    # Set unique target labels
    for i in range(15):
        df.loc[i, "label"] = f"LABEL_{i}"

    seq_df, X, y, endpoints = builder.build_sequences_from_df(df)

    # First sequence ends at row 9 (index 9)
    assert seq_df.iloc[0]["target"] == "LABEL_9"
    assert y[0] == "LABEL_9"

    # Second sequence ends at row 10
    assert seq_df.iloc[1]["target"] == "LABEL_10"
    assert y[1] == "LABEL_10"


# L. Forbidden future columns rejected/excluded
def test_forbidden_future_columns_rejected_excluded():
    # If caller specifies forbidden column in feature_columns, reject immediately
    with pytest.raises(ValueError, match="Forbidden columns detected"):
        SequenceBuilder(feature_columns=["mid_price", "future_mid"])

    with pytest.raises(ValueError, match="Forbidden columns detected"):
        SequenceBuilder(feature_columns=["mid_price", "future_delta"])

    with pytest.raises(ValueError, match="Forbidden columns detected"):
        SequenceBuilder(feature_columns=["mid_price", "label"])


# M. No random shuffling
def test_no_random_shuffling(builder):
    df = create_feature_df(n_rows=30)
    seq_df, _, _, endpoints = builder.build_sequences_from_df(df)

    # Verify monotonic increasing
    assert pd.Series(endpoints).is_monotonic_increasing
    assert seq_df["endpoint_timestamp_ms"].is_monotonic_increasing
    assert seq_df["start_timestamp_ms"].is_monotonic_increasing


# N. Correct feature dimensionality
def test_correct_feature_dimensionality(builder):
    df = create_feature_df(n_rows=25)
    seq_df, X, y, endpoints = builder.build_sequences_from_df(df)

    assert X.ndim == 3
    assert X.shape[1] == 10  # sequence length
    assert X.shape[2] == 11  # 11 safe causal features
    assert X.shape[0] == len(seq_df)

    # Check exact value match for first cell of first sequence
    expected_mid = df.iloc[0]["mid_price"]
    mid_idx = SAFE_FEATURE_COLUMNS.index("mid_price")
    assert np.isclose(X[0, 0, mid_idx], expected_mid)


# O. Metadata correctness
def test_metadata_correctness(tmp_path):
    train_df = create_feature_df(n_rows=20, start_ts=1_000_000, step_ms=1000)
    val_df = create_feature_df(n_rows=20, start_ts=1_050_000, step_ms=1000)
    test_df = create_feature_df(n_rows=20, start_ts=1_100_000, step_ms=1000)

    tr_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    te_p = tmp_path / "test.parquet"
    train_df.to_parquet(tr_p, index=False)
    val_df.to_parquet(val_p, index=False)
    test_df.to_parquet(te_p, index=False)

    out_d = tmp_path / "seq_metadata_test"
    b = SequenceBuilder(sequence_length=10)
    meta, rep = b.build_all_splits(tr_p, val_p, te_p, out_d)

    assert meta.train_sequences == 11
    assert meta.val_sequences == 11
    assert meta.test_sequences == 11
    assert meta.total_sequences == 33
    assert meta.n_features == 11
    assert meta.sequence_length == 10
    assert meta.duplicate_endpoints_count == 0
    assert meta.cross_split_violations == 0
    assert meta.deterministic_verification is True

    # Check JSON file written
    meta_json = json.loads((out_d / "sequence_metadata.json").read_text(encoding="utf-8"))
    assert meta_json["train_sequences"] == 11
    assert meta_json["total_sequences"] == 33
