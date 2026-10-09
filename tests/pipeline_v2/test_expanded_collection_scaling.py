"""
Unit and integration tests for Phase 17 Train-Only Feature Scaling on Expanded Collection (46 Retained Markets).

Validates all Phase 17 architectural requirements and invariants:
1. Scaler fitted EXCLUSIVELY on training sequences (X_train).
2. Validation and Test splits NEVER influence scaler statistics (adversarial perturbation test).
3. Attempting to fit scaler on validation or test splits raises ValueError.
4. Exact 3D tensor shapes preserved across all splits:
   - Train: (7514, 10, 11)
   - Validation: (1428, 10, 11)
   - Test: (1582, 10, 11)
   - Total: 10,524 sequences
5. Target labels (y), endpoints, and stream identifiers remain strictly unchanged bit-for-bit.
6. Market and asset stream isolation preserved: 31 train, 8 val, 7 test markets with 0 cross-split overlap.
7. Post-scaling standardization properties: train means ~ 0.0, train stds ~ 1.0 on unmasked cells.
8. Validation and Test partitions transformed strictly using frozen train scaler.
9. Causal warm-up NaN accounting & explicit warm-up masks:
   - 3,588 original NaNs (Train: 2418, Val: 624, Test: 546)
   - Confined strictly to lookback warmup steps 0..4
   - Step 9 (endpoint T) has exactly 0 NaNs
   - Explicit boolean masks preserved in npz and parquet artifacts
   - Post-imputation NaNs = 0, Post-imputation Infs = 0
10. Directional class balance strictly preserved:
    - Train: UP: 3050, DOWN: 3050 (ratio 1.0), FLAT: 1414
    - Validation: UP: 617, DOWN: 616, FLAT: 195
    - Test: UP: 709, DOWN: 705, FLAT: 168
11. Machine-readable scaler reloadability from scaler_params.json.
12. Bitwise deterministic reproducibility.
13. Upstream and baseline immutability: new_collection/ remaining untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.scaling.train_scaler import CausalSequenceScaler
from pipeline_v2.sequences.sequence_builder import SAFE_FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SEQUENCES_DIR = REPO_ROOT / "data" / "clean_v2" / "06_sequences" / "expanded_collection"
SCALED_DIR = REPO_ROOT / "data" / "clean_v2" / "07_scaled" / "expanded_collection"
BASELINE_DIR = REPO_ROOT / "data" / "clean_v2" / "07_scaled" / "new_collection"


@pytest.fixture(scope="module")
def scaled_expanded_artifacts():
    """Ensure scaled artifacts exist and load Parquet, NPZ, and metadata."""
    train_pq = SCALED_DIR / "train_scaled.parquet"
    val_pq = SCALED_DIR / "validation_scaled.parquet"
    test_pq = SCALED_DIR / "test_scaled.parquet"
    comb_npz = SCALED_DIR / "scaled_production.npz"
    masks_npz = SCALED_DIR / "warmup_masks.npz"
    scaler_params = SCALED_DIR / "scaler_params.json"
    meta_path = SCALED_DIR / "scaler_metadata.json"

    assert train_pq.exists(), f"Missing {train_pq}"
    assert val_pq.exists(), f"Missing {val_pq}"
    assert test_pq.exists(), f"Missing {test_pq}"
    assert comb_npz.exists(), f"Missing {comb_npz}"
    assert masks_npz.exists(), f"Missing {masks_npz}"
    assert scaler_params.exists(), f"Missing {scaler_params}"
    assert meta_path.exists(), f"Missing {meta_path}"

    train_df = pd.read_parquet(train_pq)
    val_df = pd.read_parquet(val_pq)
    test_df = pd.read_parquet(test_pq)
    npz_data = np.load(comb_npz, allow_pickle=True)
    masks_data = np.load(masks_npz, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Load unscaled Phase 16 sequence data for comparison
    orig_train_npz = np.load(SEQUENCES_DIR / "train_sequences.npz", allow_pickle=True)
    orig_val_npz = np.load(SEQUENCES_DIR / "validation_sequences.npz", allow_pickle=True)
    orig_test_npz = np.load(SEQUENCES_DIR / "test_sequences.npz", allow_pickle=True)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "npz": npz_data,
        "masks": masks_data,
        "meta": meta,
        "scaler_params_path": scaler_params,
        "orig_train": orig_train_npz,
        "orig_val": orig_val_npz,
        "orig_test": orig_test_npz,
    }


def test_scaler_fit_source_is_train_only(scaled_expanded_artifacts):
    """Verify that scaler metadata explicitly documents fit_source = 'train'."""
    meta = scaled_expanded_artifacts["meta"]
    assert meta["fit_source"] == "train"
    assert meta["scaler_type"] == "standard"
    assert meta["feature_dimension"] == 11
    assert meta["feature_names"] == SAFE_FEATURE_COLUMNS
    assert meta["n_samples_seen"] == 7514


def test_fit_on_val_or_test_raises_error():
    """Verify that attempting to fit scaler on validation or test raises ValueError."""
    scaler = CausalSequenceScaler(scaler_type="standard")
    dummy = np.random.randn(20, 10, 11)

    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy, split_name="validation")

    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy, split_name="test")

    with pytest.raises(ValueError, match="Data leakage violation"):
        scaler.fit(dummy, split_name="combined")


def test_adversarial_validation_and_test_perturbation(scaled_expanded_artifacts):
    """Verify that mutating validation and test data leaves train scaler parameters and outputs 100% identical."""
    X_train = scaled_expanded_artifacts["orig_train"]["X"]
    X_val = scaled_expanded_artifacts["orig_val"]["X"]
    X_test = scaled_expanded_artifacts["orig_test"]["X"]

    # Baseline scaler fit on unperturbed train
    scaler1 = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler1.fit(X_train, split_name="train")
    tr_out1 = scaler1.transform(X_train)

    # Perturb validation and test by 10,000x + large offset
    X_val_pert = X_val * 10000.0 + 99999.0
    X_test_pert = X_test * 10000.0 - 99999.0

    # Ensure transforming perturbed validation and test doesn't mutate scaler
    _ = scaler1.transform(X_val_pert)
    _ = scaler1.transform(X_test_pert)

    # Re-fit fresh scaler on train
    scaler2 = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler2.fit(X_train, split_name="train")
    tr_out2 = scaler2.transform(X_train)

    # Parameters and train outputs must be bit-for-bit identical
    np.testing.assert_array_equal(scaler1.center_, scaler2.center_)
    np.testing.assert_array_equal(scaler1.scale_, scaler2.scale_)
    np.testing.assert_array_equal(tr_out1, tr_out2)


def test_exact_tensor_shapes_preserved(scaled_expanded_artifacts):
    """Verify that 3D sequence shapes (N, L=10, D=11) are strictly preserved across all splits."""
    npz = scaled_expanded_artifacts["npz"]
    orig_tr = scaled_expanded_artifacts["orig_train"]
    orig_va = scaled_expanded_artifacts["orig_val"]
    orig_te = scaled_expanded_artifacts["orig_test"]

    assert npz["train_X"].shape == (7514, 10, 11)
    assert npz["val_X"].shape == (1428, 10, 11)
    assert npz["test_X"].shape == (1582, 10, 11)

    assert npz["train_X"].shape == orig_tr["X"].shape
    assert npz["val_X"].shape == orig_va["X"].shape
    assert npz["test_X"].shape == orig_te["X"].shape

    assert len(scaled_expanded_artifacts["train_df"]) == 7514
    assert len(scaled_expanded_artifacts["val_df"]) == 1428
    assert len(scaled_expanded_artifacts["test_df"]) == 1582


def test_target_labels_and_endpoints_untouched(scaled_expanded_artifacts):
    """Verify target labels and endpoint timestamps remain completely unmodified bit-for-bit."""
    npz = scaled_expanded_artifacts["npz"]
    orig_tr = scaled_expanded_artifacts["orig_train"]
    orig_va = scaled_expanded_artifacts["orig_val"]
    orig_te = scaled_expanded_artifacts["orig_test"]

    train_df = scaled_expanded_artifacts["train_df"]
    val_df = scaled_expanded_artifacts["val_df"]
    test_df = scaled_expanded_artifacts["test_df"]

    # Target arrays unchanged
    np.testing.assert_array_equal(npz["train_y"], orig_tr["y"])
    np.testing.assert_array_equal(npz["val_y"], orig_va["y"])
    np.testing.assert_array_equal(npz["test_y"], orig_te["y"])

    # Endpoint arrays unchanged
    np.testing.assert_array_equal(npz["train_endpoints"], orig_tr["endpoints"])
    np.testing.assert_array_equal(npz["val_endpoints"], orig_va["endpoints"])
    np.testing.assert_array_equal(npz["test_endpoints"], orig_te["endpoints"])

    # Parquet target and endpoint columns match exactly
    assert (train_df["target"].values == npz["train_y"]).all()
    assert (val_df["target"].values == npz["val_y"]).all()
    assert (test_df["target"].values == npz["test_y"]).all()

    assert (train_df["endpoint_timestamp_ms"].values == npz["train_endpoints"]).all()
    assert (val_df["endpoint_timestamp_ms"].values == npz["val_endpoints"]).all()
    assert (test_df["endpoint_timestamp_ms"].values == npz["test_endpoints"]).all()


def test_market_and_asset_isolation_preserved(scaled_expanded_artifacts):
    """Verify market_id and asset_id integrity are preserved across splits."""
    train_df = scaled_expanded_artifacts["train_df"]
    val_df = scaled_expanded_artifacts["val_df"]
    test_df = scaled_expanded_artifacts["test_df"]

    for df in [train_df, val_df, test_df]:
        assert "market_id" in df.columns
        assert "asset_id" in df.columns
        assert df["market_id"].notna().all()
        assert df["asset_id"].notna().all()
        assert (df["market_id"] != "").all()
        assert (df["asset_id"] != "").all()

    train_mkts = set(train_df["market_id"])
    val_mkts = set(val_df["market_id"])
    test_mkts = set(test_df["market_id"])

    assert len(train_mkts) == 31
    assert len(val_mkts) == 8
    assert len(test_mkts) == 7
    assert len(train_mkts.intersection(val_mkts)) == 0
    assert len(val_mkts.intersection(test_mkts)) == 0
    assert len(train_mkts.intersection(test_mkts)) == 0


def test_train_standardization_properties(scaled_expanded_artifacts):
    """Verify that scaled training features have mean ~ 0.0 and std ~ 1.0 on unmasked cells."""
    train_X = scaled_expanded_artifacts["npz"]["train_X"]

    means = np.nanmean(train_X, axis=(0, 1))
    stds = np.nanstd(train_X, axis=(0, 1))

    # All means must be within 1e-10 of 0.0
    np.testing.assert_allclose(means, 0.0, atol=1e-10)

    # All stds must be within 1e-10 of 1.0
    np.testing.assert_allclose(stds, 1.0, atol=1e-10)


def test_frozen_scaler_validation_and_test_transform(scaled_expanded_artifacts):
    """Verify validation and test are transformed strictly with frozen train parameters."""
    scaler_params = json.loads(Path(scaled_expanded_artifacts["scaler_params_path"]).read_text(encoding="utf-8"))
    orig_val_X = scaled_expanded_artifacts["orig_val"]["X"]
    orig_test_X = scaled_expanded_artifacts["orig_test"]["X"]
    val_X_scaled = scaled_expanded_artifacts["npz"]["val_X"]
    test_X_scaled = scaled_expanded_artifacts["npz"]["test_X"]

    names = scaler_params["feature_names"]
    centers = np.array([scaler_params["centers"][f] for f in names])
    scales = np.array([scaler_params["scales"][f] for f in names])

    expected_val = (orig_val_X - centers) / scales
    expected_test = (orig_test_X - centers) / scales

    np.testing.assert_allclose(val_X_scaled, expected_val, equal_nan=True)
    np.testing.assert_allclose(test_X_scaled, expected_test, equal_nan=True)


def test_nan_accounting_and_explicit_masks(scaled_expanded_artifacts):
    """Verify exact warm-up NaN accounting, explicit boolean masks, and zero-imputed tensors."""
    npz = scaled_expanded_artifacts["npz"]
    masks = scaled_expanded_artifacts["masks"]
    meta = scaled_expanded_artifacts["meta"]
    nan_acc = meta["nan_accounting"]

    train_X = npz["train_X"]
    val_X = npz["val_X"]
    test_X = npz["test_X"]

    # Original NaN counts
    assert int(np.isnan(train_X).sum()) == 2418
    assert int(np.isnan(val_X).sum()) == 624
    assert int(np.isnan(test_X).sum()) == 546
    total_nans = 2418 + 624 + 546
    assert total_nans == 3588

    # Explicit masks
    assert masks["train_mask"].sum() == 2418
    assert masks["val_mask"].sum() == 624
    assert masks["test_mask"].sum() == 546
    assert (masks["train_mask"] == np.isnan(train_X)).all()
    assert (masks["val_mask"] == np.isnan(val_X)).all()
    assert (masks["test_mask"] == np.isnan(test_X)).all()

    # Step 9 (endpoint T) has exactly 0 NaNs originally and in scaled tensors
    assert int(np.isnan(train_X[:, 9, :]).sum()) == 0
    assert int(np.isnan(val_X[:, 9, :]).sum()) == 0
    assert int(np.isnan(test_X[:, 9, :]).sum()) == 0

    # Post-warmup steps 5..9 have exactly 0 NaNs
    assert int(np.isnan(train_X[:, 5:, :]).sum()) == 0
    assert int(np.isnan(val_X[:, 5:, :]).sum()) == 0
    assert int(np.isnan(test_X[:, 5:, :]).sum()) == 0

    # Zero Infs in scaled arrays
    assert not np.isinf(train_X).any()
    assert not np.isinf(val_X).any()
    assert not np.isinf(test_X).any()

    # Post-imputation tensors: strictly 0 NaNs and 0 Infs
    train_imp = npz["train_X_imputed"]
    val_imp = npz["val_X_imputed"]
    test_imp = npz["test_X_imputed"]

    assert not np.isnan(train_imp).any()
    assert not np.isnan(val_imp).any()
    assert not np.isnan(test_imp).any()
    assert not np.isinf(train_imp).any()
    assert not np.isinf(val_imp).any()
    assert not np.isinf(test_imp).any()

    # Post-imputation values match non-masked cells exactly
    np.testing.assert_array_equal(train_imp[~masks["train_mask"]], train_X[~masks["train_mask"]])
    np.testing.assert_array_equal(val_imp[~masks["val_mask"]], val_X[~masks["val_mask"]])
    np.testing.assert_array_equal(test_imp[~masks["test_mask"]], test_X[~masks["test_mask"]])

    # Masked cells are exactly 0.0
    assert (train_imp[masks["train_mask"]] == 0.0).all()
    assert (val_imp[masks["val_mask"]] == 0.0).all()
    assert (test_imp[masks["test_mask"]] == 0.0).all()


def test_class_distributions_preserved(scaled_expanded_artifacts):
    """Verify that directional class distributions match Phase 16 sequence inputs."""
    npz = scaled_expanded_artifacts["npz"]

    for split_key, expected_counts in [
        ("train_y", {"UP": 3050, "DOWN": 3050, "FLAT": 1414}),
        ("val_y", {"UP": 617, "DOWN": 616, "FLAT": 195}),
        ("test_y", {"UP": 709, "DOWN": 705, "FLAT": 168}),
    ]:
        counts = dict(pd.Series(npz[split_key]).value_counts())
        assert counts == expected_counts, f"Class counts mismatch for {split_key}: {counts} vs {expected_counts}"


def test_inference_reloadability(scaled_expanded_artifacts):
    """Verify that saved scaler_params.json can be reloaded and transforms identically."""
    scaler_path = scaled_expanded_artifacts["scaler_params_path"]
    val_X_scaled = scaled_expanded_artifacts["npz"]["val_X"]
    orig_val_X = scaled_expanded_artifacts["orig_val"]["X"]

    loaded_scaler = CausalSequenceScaler.load(scaler_path)
    assert loaded_scaler.is_fitted is True
    assert loaded_scaler.fit_source == "train"

    transformed = loaded_scaler.transform(orig_val_X)
    np.testing.assert_array_equal(transformed, val_X_scaled)


def test_deterministic_scaling(scaled_expanded_artifacts):
    """Verify that repeated scaling transformations produce bitwise identical outputs."""
    orig_train_X = scaled_expanded_artifacts["orig_train"]["X"]

    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler.fit(orig_train_X, split_name="train")

    t1 = scaler.transform(orig_train_X)
    t2 = scaler.transform(orig_train_X)

    np.testing.assert_array_equal(t1, t2)


def test_baseline_scaled_untouched():
    """Verify that the original 19-market baseline in new_collection/ remains 100% untouched."""
    b_meta = json.loads((BASELINE_DIR / "scaler_metadata.json").read_text(encoding="utf-8"))
    assert b_meta["row_counts"]["total_sequences"] == 4628
    assert b_meta["row_counts"]["train_sequences"] == 3248
    assert b_meta["row_counts"]["val_sequences"] == 722
    assert b_meta["row_counts"]["test_sequences"] == 658
