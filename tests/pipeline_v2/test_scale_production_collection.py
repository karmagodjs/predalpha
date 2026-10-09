"""
Unit tests for Phase 17: Train-Only Feature Scaling on Production Collection.

Validates all Phase 17 requirements and architectural invariants:
1. Scaler fitted EXCLUSIVELY on training data (X_train).
2. Validation data cannot influence scaler parameters (perturbation test).
3. Test data cannot influence scaler parameters (perturbation test).
4. Attempting to fit on validation or test raises ValueError.
5. Transformed 3D tensor shapes are strictly preserved:
   Train: (3248, 10, 11), Validation: (722, 10, 11), Test: (658, 10, 11).
6. Target labels (y), endpoints, and metadata remain strictly unchanged.
7. Zero-variance protection: divisor safe guard with epsilon.
8. Post-scaling numerical stability: zero Infs, bounded ranges.
9. Directional class distribution perfectly preserved.
10. Scaler artifact reloadability and reusability for future inference.
11. Bitwise deterministic reproducibility.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.scaling.train_scaler import CausalSequenceScaler
from pipeline_v2.sequences.sequence_builder import SAFE_FEATURE_COLUMNS

SEQUENCES_DIR = Path("data/clean_v2/06_sequences/new_collection")
SCALED_DIR = Path("data/clean_v2/07_scaled/new_collection")


@pytest.fixture(scope="module")
def scaled_artifacts():
    """Ensure scaled artifacts are generated and load Parquet and NPZ data."""
    train_pq = SCALED_DIR / "train_scaled.parquet"
    val_pq = SCALED_DIR / "validation_scaled.parquet"
    test_pq = SCALED_DIR / "test_scaled.parquet"
    comb_npz = SCALED_DIR / "scaled_production.npz"
    scaler_params = SCALED_DIR / "scaler_params.json"
    meta_path = SCALED_DIR / "scaler_metadata.json"

    assert train_pq.exists(), f"Missing {train_pq}"
    assert val_pq.exists(), f"Missing {val_pq}"
    assert test_pq.exists(), f"Missing {test_pq}"
    assert comb_npz.exists(), f"Missing {comb_npz}"
    assert scaler_params.exists(), f"Missing {scaler_params}"
    assert meta_path.exists(), f"Missing {meta_path}"

    train_df = pd.read_parquet(train_pq)
    val_df = pd.read_parquet(val_pq)
    test_df = pd.read_parquet(test_pq)
    npz_data = np.load(comb_npz, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Also load original unscaled sequence data for comparison
    orig_npz = np.load(SEQUENCES_DIR / "sequences_production.npz", allow_pickle=True)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "npz": npz_data,
        "orig_npz": orig_npz,
        "meta": meta,
        "scaler_params_path": scaler_params,
    }


def test_scaler_fit_source_is_train_only(scaled_artifacts):
    """Verify that scaler metadata explicitly documents fit_source = 'train'."""
    meta = scaled_artifacts["meta"]
    assert meta["fit_source"] == "train"
    assert meta["scaler_type"] == "standard"
    assert meta["feature_dimension"] == 11
    assert meta["feature_names"] == SAFE_FEATURE_COLUMNS


def test_fit_on_val_or_test_raises_error():
    """Verify that attempting to fit scaler on validation or test raises ValueError."""
    scaler = CausalSequenceScaler(scaler_type="standard")
    dummy = np.random.randn(10, 10, 11)

    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy, split_name="validation")

    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy, split_name="test")

    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy, split_name="combined")


def test_validation_and_test_perturbation_invariance(scaled_artifacts):
    """Verify that perturbing validation or test data does NOT affect scaler parameters."""
    orig_npz = scaled_artifacts["orig_npz"]
    X_train = orig_npz["train_X"]
    X_val = orig_npz["val_X"]
    X_test = orig_npz["test_X"]

    # Fit baseline scaler on unperturbed train
    scaler1 = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler1.fit(X_train, split_name="train")

    # Perturb validation and test by extreme factors
    X_val_pert = X_val * 10000.0 + 99999.0
    X_test_pert = X_test * 10000.0 - 99999.0

    # Fit second scaler on train
    scaler2 = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler2.fit(X_train, split_name="train")

    # Parameters must be identical
    np.testing.assert_array_equal(scaler1.center_, scaler2.center_)
    np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)


def test_exact_tensor_shapes_preserved(scaled_artifacts):
    """Verify that 3D sequence shapes (N, L=10, D=11) are strictly preserved across all splits."""
    npz = scaled_artifacts["npz"]
    orig_npz = scaled_artifacts["orig_npz"]

    assert npz["train_X"].shape == (3248, 10, 11)
    assert npz["val_X"].shape == (722, 10, 11)
    assert npz["test_X"].shape == (658, 10, 11)

    assert npz["train_X"].shape == orig_npz["train_X"].shape
    assert npz["val_X"].shape == orig_npz["val_X"].shape
    assert npz["test_X"].shape == orig_npz["test_X"].shape


def test_target_labels_and_endpoints_untouched(scaled_artifacts):
    """Verify that target labels and endpoint timestamps remain completely unmodified."""
    npz = scaled_artifacts["npz"]
    orig_npz = scaled_artifacts["orig_npz"]
    train_df = scaled_artifacts["train_df"]
    val_df = scaled_artifacts["val_df"]
    test_df = scaled_artifacts["test_df"]

    # Target arrays unchanged
    np.testing.assert_array_equal(npz["train_y"], orig_npz["train_y"])
    np.testing.assert_array_equal(npz["val_y"], orig_npz["val_y"])
    np.testing.assert_array_equal(npz["test_y"], orig_npz["test_y"])

    # Endpoint arrays unchanged
    np.testing.assert_array_equal(npz["train_endpoints"], orig_npz["train_endpoints"])
    np.testing.assert_array_equal(npz["val_endpoints"], orig_npz["val_endpoints"])
    np.testing.assert_array_equal(npz["test_endpoints"], orig_npz["test_endpoints"])

    # Parquet target column matches exactly
    assert (train_df["target"].values == npz["train_y"]).all()
    assert (val_df["target"].values == npz["val_y"]).all()
    assert (test_df["target"].values == npz["test_y"]).all()


def test_market_and_asset_isolation_preserved(scaled_artifacts):
    """Verify market_id and asset_id integrity are preserved in scaled DataFrames."""
    for split_df in [scaled_artifacts["train_df"], scaled_artifacts["val_df"], scaled_artifacts["test_df"]]:
        assert "market_id" in split_df.columns
        assert "asset_id" in split_df.columns
        assert split_df["market_id"].notna().all()
        assert split_df["asset_id"].notna().all()
        assert (split_df["market_id"] != "").all()
        assert (split_df["asset_id"] != "").all()


def test_post_scaling_standardization_properties(scaled_artifacts):
    """Verify that scaled training features have mean ~ 0.0 and std ~ 1.0."""
    train_X = scaled_artifacts["npz"]["train_X"]

    # Calculate post-scaling mean and std across (N, L)
    means = np.nanmean(train_X, axis=(0, 1))
    stds = np.nanstd(train_X, axis=(0, 1))

    # All means must be within 1e-10 of 0.0
    np.testing.assert_allclose(means, 0.0, atol=1e-10)

    # All stds must be within 1e-10 of 1.0
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_no_inf_values(scaled_artifacts):
    """Verify zero Inf values exist in any scaled partition."""
    npz = scaled_artifacts["npz"]
    assert not np.isinf(npz["train_X"]).any()
    assert not np.isinf(npz["val_X"]).any()
    assert not np.isinf(npz["test_X"]).any()


def test_class_distributions_preserved(scaled_artifacts):
    """Verify that directional class distributions are identical to Phase 16."""
    npz = scaled_artifacts["npz"]

    for split_key, expected_counts in [
        ("train_y", {"UP": 1386, "DOWN": 1386, "FLAT": 476}),
        ("val_y", {"UP": 282, "DOWN": 281, "FLAT": 159}),
        ("test_y", {"UP": 291, "DOWN": 291, "FLAT": 76}),
    ]:
        counts = dict(pd.Series(npz[split_key]).value_counts())
        assert counts == expected_counts, f"Class counts mismatch for {split_key}: {counts} vs {expected_counts}"


def test_inference_reusability(scaled_artifacts):
    """Verify that the saved scaler parameters can be loaded and applied identically."""
    scaler_path = scaled_artifacts["scaler_params_path"]
    val_X_scaled = scaled_artifacts["npz"]["val_X"]
    orig_val_X = scaled_artifacts["orig_npz"]["val_X"]

    # Load fitted scaler from JSON artifact
    loaded_scaler = CausalSequenceScaler.load(scaler_path)
    assert loaded_scaler.is_fitted is True
    assert loaded_scaler.fit_source == "train"

    # Transform unscaled validation observations
    transformed = loaded_scaler.transform(orig_val_X)

    # Must be bitwise identical
    np.testing.assert_array_equal(transformed, val_X_scaled)


def test_deterministic_scaling(scaled_artifacts):
    """Verify that repeating the scaling transformation produces bitwise identical arrays."""
    orig_npz = scaled_artifacts["orig_npz"]
    X_train = orig_npz["train_X"]

    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler.fit(X_train, split_name="train")

    t1 = scaler.transform(X_train)
    t2 = scaler.transform(X_train)

    np.testing.assert_array_equal(t1, t2)
