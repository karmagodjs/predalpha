"""
Unit tests for Pipeline V2 Train-Only Feature Scaler.

Covers all Phase 17 requirements:
1. Scaler fit source is train only
2. Validation cannot influence scaler statistics
3. Test cannot influence scaler statistics
4. Shapes unchanged: (N, 10, 11) preserved across all splits
5. Deterministic output
6. Target remains untouched
7. No leakage
8. Insufficient-data handling
9. Zero-variance feature protection
10. Serialization and deserialization of scaler parameters
11. End-to-end ScalingOrchestrator execution
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.scaling.train_scaler import (
    CausalSequenceScaler,
    ScalingOrchestrator,
    ScalerMetadata,
    ScalingReport,
)


@pytest.fixture
def dummy_sequences():
    """Create synthetic (N, 10, 11) sequences for train, val, test."""
    np.random.seed(42)
    n_features = 11
    seq_len = 10

    # Train: 30 sequences, mean around 5.0, std around 2.0
    # Feature 0 has zero variance (constant 1.0)
    X_train = np.random.normal(loc=5.0, scale=2.0, size=(30, seq_len, n_features))
    X_train[:, :, 0] = 1.0  # Constant zero-variance feature

    # Validation: 10 sequences, mean around 6.0, std around 3.0
    X_val = np.random.normal(loc=6.0, scale=3.0, size=(10, seq_len, n_features))
    X_val[:, :, 0] = 1.0

    # Test: 10 sequences, mean around 4.0, std around 1.5
    X_test = np.random.normal(loc=4.0, scale=1.5, size=(10, seq_len, n_features))
    X_test[:, :, 0] = 1.0

    feat_names = [f"feat_{i}" for i in range(n_features)]
    y_train = np.array(["FLAT"] * 30)
    y_val = np.array(["FLAT"] * 10)
    y_test = np.array(["FLAT"] * 10)

    ep_train = np.arange(1000, 1000 + 30 * 1000, 1000, dtype=np.int64)
    ep_val = np.arange(2000, 2000 + 10 * 1000, 1000, dtype=np.int64)
    ep_test = np.arange(3000, 3000 + 10 * 1000, 1000, dtype=np.int64)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "ep_train": ep_train,
        "X_val": X_val,
        "y_val": y_val,
        "ep_val": ep_val,
        "X_test": X_test,
        "y_test": y_test,
        "ep_test": ep_test,
        "feature_names": feat_names,
    }


# 1. Scaler fit source is train only
def test_scaler_fit_source_is_train_only(dummy_sequences):
    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])
    assert not scaler.is_fitted
    assert scaler.fit_source == "none"

    scaler.fit(dummy_sequences["X_train"], split_name="train")
    assert scaler.is_fitted
    assert scaler.fit_source == "train"

    # Fitting on validation must raise ValueError
    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy_sequences["X_val"], split_name="validation")

    # Fitting on test must raise ValueError
    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy_sequences["X_test"], split_name="test")


# 2. Validation cannot influence scaler statistics
def test_validation_cannot_influence_scaler_statistics(dummy_sequences):
    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])
    scaler.fit(dummy_sequences["X_train"], split_name="train")

    orig_center = scaler.center_.copy()
    orig_scale = scaler.scale_.copy()

    # Extreme validation data with huge outliers
    extreme_val = dummy_sequences["X_val"] * 10_000.0 + 999_999.0
    _ = scaler.transform(extreme_val)

    # Scaler centers and scales must be 100% UNCHANGED
    np.testing.assert_array_equal(scaler.center_, orig_center)
    np.testing.assert_array_equal(scaler.scale_, orig_scale)


# 3. Test cannot influence scaler statistics
def test_test_cannot_influence_scaler_statistics(dummy_sequences):
    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])
    scaler.fit(dummy_sequences["X_train"], split_name="train")

    orig_center = scaler.center_.copy()
    orig_scale = scaler.scale_.copy()

    # Extreme test data
    extreme_test = dummy_sequences["X_test"] * 50_000.0 - 888_888.0
    _ = scaler.transform(extreme_test)

    # Scaler parameters must remain strictly unchanged
    np.testing.assert_array_equal(scaler.center_, orig_center)
    np.testing.assert_array_equal(scaler.scale_, orig_scale)


# 4. Shapes unchanged: (N, 10, 11) preserved across all splits
def test_shapes_unchanged(dummy_sequences):
    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])
    X_tr_scaled = scaler.fit_transform(dummy_sequences["X_train"], split_name="train")
    X_val_scaled = scaler.transform(dummy_sequences["X_val"])
    X_te_scaled = scaler.transform(dummy_sequences["X_test"])

    assert X_tr_scaled.shape == (30, 10, 11)
    assert X_val_scaled.shape == (10, 10, 11)
    assert X_te_scaled.shape == (10, 10, 11)
    assert X_tr_scaled.shape == dummy_sequences["X_train"].shape
    assert X_val_scaled.shape == dummy_sequences["X_val"].shape
    assert X_te_scaled.shape == dummy_sequences["X_test"].shape


# 5. Deterministic output
def test_deterministic_output(dummy_sequences):
    s1 = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])
    s2 = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])

    out1_tr = s1.fit_transform(dummy_sequences["X_train"], split_name="train")
    out1_val = s1.transform(dummy_sequences["X_val"])
    out1_te = s1.transform(dummy_sequences["X_test"])

    out2_tr = s2.fit_transform(dummy_sequences["X_train"], split_name="train")
    out2_val = s2.transform(dummy_sequences["X_val"])
    out2_te = s2.transform(dummy_sequences["X_test"])

    np.testing.assert_array_equal(out1_tr, out2_tr)
    np.testing.assert_array_equal(out1_val, out2_val)
    np.testing.assert_array_equal(out1_te, out2_te)
    np.testing.assert_array_equal(s1.center_, s2.center_)
    np.testing.assert_array_equal(s1.scale_, s2.scale_)


# 6. Target remains untouched
def test_target_remains_untouched(dummy_sequences, tmp_path):
    seq_dir = tmp_path / "sequences"
    out_dir = tmp_path / "scaled"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Save mock sequences
    for split in ["train", "validation", "test"]:
        X = dummy_sequences[f"X_{'val' if split == 'validation' else split}"]
        y = dummy_sequences[f"y_{'val' if split == 'validation' else split}"]
        ep = dummy_sequences[f"ep_{'val' if split == 'validation' else split}"]

        # Write NPZ
        np.savez_compressed(
            seq_dir / f"{split}_sequences.npz",
            X=X,
            y=y,
            endpoints=ep,
            feature_names=np.array(dummy_sequences["feature_names"]),
        )
        # Write Parquet
        df = pd.DataFrame({
            "sequence_id": np.arange(len(X)),
            "endpoint_timestamp_ms": ep,
            "target": y,
            "feature_matrix": [X[i].tolist() for i in range(len(X))],
        })
        df.to_parquet(seq_dir / f"{split}_sequences.parquet", index=False)

    orchestrator = ScalingOrchestrator(scaler_type="standard")
    meta, rep = orchestrator.scale_sequences(seq_dir, out_dir)

    assert meta.train_shape_after == (30, 10, 11)
    assert rep.target_untouched_verification is True

    # Inspect scaled Parquet files
    for split in ["train", "validation", "test"]:
        scaled_df = pd.read_parquet(out_dir / f"{split}_scaled.parquet")
        orig_y = dummy_sequences[f"y_{'val' if split == 'validation' else split}"]
        assert np.array_equal(scaled_df["target"].values, orig_y)


# 7. No leakage
def test_no_leakage(dummy_sequences):
    # Verify scaler params depend strictly on train
    s_train_only = CausalSequenceScaler(scaler_type="standard")
    s_train_only.fit(dummy_sequences["X_train"], split_name="train")

    expected_mean = np.nanmean(dummy_sequences["X_train"], axis=(0, 1))
    expected_std = np.nanstd(dummy_sequences["X_train"], axis=(0, 1))
    expected_scale = np.where(expected_std < 1e-8, 1.0, expected_std)

    np.testing.assert_allclose(s_train_only.center_, expected_mean, rtol=1e-6)
    np.testing.assert_allclose(s_train_only.scale_, expected_scale, rtol=1e-6)


# 8. Insufficient-data handling
def test_insufficient_data_handling():
    scaler = CausalSequenceScaler(scaler_type="standard")
    # Empty array
    empty_X = np.empty((0, 10, 11))
    with pytest.raises(ValueError, match="Cannot fit scaler on empty"):
        scaler.fit(empty_X, split_name="train")


# 9. Zero-variance feature protection
def test_zero_variance_protection():
    scaler = CausalSequenceScaler(scaler_type="standard")
    # Data with constant 3.5 in all cells of feature 0
    X = np.random.normal(size=(20, 10, 5))
    X[:, :, 0] = 3.5

    scaler.fit(X, split_name="train")
    assert scaler.zero_variance_mask_[0] is True or scaler.zero_variance_mask_[0] == 1
    assert scaler.scale_[0] == 1.0

    X_scaled = scaler.transform(X)
    # Feature 0 should be exactly 0.0 without any NaN or Inf
    assert not np.isnan(X_scaled[:, :, 0]).any()
    assert not np.isinf(X_scaled[:, :, 0]).any()
    np.testing.assert_allclose(X_scaled[:, :, 0], 0.0, atol=1e-7)


# 10. Serialization and deserialization
def test_serialization_deserialization(dummy_sequences, tmp_path):
    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=dummy_sequences["feature_names"])
    scaler.fit(dummy_sequences["X_train"], split_name="train")

    save_path = tmp_path / "scaler_test.json"
    scaler.save(save_path)
    assert save_path.exists()

    loaded_scaler = CausalSequenceScaler.load(save_path)
    assert loaded_scaler.is_fitted is True
    assert loaded_scaler.fit_source == "train"
    assert loaded_scaler.scaler_type == "standard"
    np.testing.assert_array_equal(loaded_scaler.center_, scaler.center_)
    np.testing.assert_array_equal(loaded_scaler.scale_, scaler.scale_)

    # Verify identical transform
    t1 = scaler.transform(dummy_sequences["X_val"])
    t2 = loaded_scaler.transform(dummy_sequences["X_val"])
    np.testing.assert_array_equal(t1, t2)
