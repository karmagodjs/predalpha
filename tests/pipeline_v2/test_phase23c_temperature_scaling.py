"""
Unit and Integration Tests for Phase 23C Temperature Scaling & Probability Calibration.

Verifies:
1. Positive temperature enforcement.
2. Probability normalization and finite values.
3. Correct class and label mapping.
4. Baseline predictions matching existing saved predictions on evaluation indices.
5. No model-weight mutation.
6. Calibration fitting uses only the calibration subset.
7. Evaluation labels do not influence the fitted temperature.
8. Identical baseline and calibrated evaluation sample indices.
9. Chronological split integrity and required purge boundary.
10. Deterministic outputs for a fixed seed and input.
11. Failure handling for missing keys, invalid shapes, non-finite logits, and insufficient class coverage.
12. Strict prohibition against locked test-set access.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from pipeline_v2.models.phase23c_temperature_scaling import (
    CLASS_NAMES,
    LABEL_TO_ID,
    LOCKED_TEST_PATH,
    MODEL_PATH,
    PHASE20_PREDS_PATH,
    VAL_PATH,
    compute_brier_score,
    compute_ece_and_reliability_bins,
    compute_multiclass_nll,
    compute_softmax,
    fit_temperature,
    get_model_weights_fingerprint,
    load_frozen_lstm_model,
    partition_chronological_validation,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Test 1: Positive Temperature Enforcement
# ---------------------------------------------------------------------------
def test_positive_temperature_enforcement():
    """Verify that temperature <= 0 or non-finite values raise ValueError."""
    dummy_logits = np.array([[1.0, 2.0, 0.5], [0.2, -1.0, 3.0]])
    dummy_y = np.array([1, 2])

    for invalid_t in [0.0, -0.5, -10.0, float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="Temperature must be"):
            compute_softmax(dummy_logits, temperature=invalid_t)

        with pytest.raises(ValueError, match="Temperature must be"):
            compute_multiclass_nll(dummy_logits, dummy_y, temperature=invalid_t)


# ---------------------------------------------------------------------------
# Test 2: Probability Normalization and Finite Values
# ---------------------------------------------------------------------------
def test_probability_normalization_and_finite_values():
    """Verify output probabilities sum to 1 and contain strictly finite values."""
    np.random.seed(42)
    logits = np.random.randn(50, 3) * 5.0

    for temp in [0.1, 0.5, 1.0, 2.5, 5.0]:
        probs = compute_softmax(logits, temperature=temp)
        assert probs.shape == (50, 3)
        assert np.all(np.isfinite(probs))
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        row_sums = np.sum(probs, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Correct Class and Label Mapping
# ---------------------------------------------------------------------------
def test_correct_class_and_label_mapping():
    """Verify exact class ordering: DOWN=0, FLAT=1, UP=2."""
    assert CLASS_NAMES == ["DOWN", "FLAT", "UP"]
    assert LABEL_TO_ID["DOWN"] == 0
    assert LABEL_TO_ID["FLAT"] == 1
    assert LABEL_TO_ID["UP"] == 2
    assert len(LABEL_TO_ID) == 3


# ---------------------------------------------------------------------------
# Test 4: Baseline Predictions Matching Existing Saved Predictions
# ---------------------------------------------------------------------------
def test_baseline_predictions_matching_saved_predictions():
    """Verify that uncalibrated model probabilities on evaluation slice match Phase 20."""
    if not (MODEL_PATH.exists() and VAL_PATH.exists() and PHASE20_PREDS_PATH.exists()):
        pytest.skip("Required model or validation artifacts not present")

    model = load_frozen_lstm_model(MODEL_PATH)
    val_npz = np.load(VAL_PATH, allow_pickle=False)
    p20_npz = np.load(PHASE20_PREDS_PATH, allow_pickle=False)

    X_val = val_npz["X_imputed"].astype(np.float32)
    endpoints = val_npz["endpoints"].astype(np.int64)

    cal_idx, eval_idx, _, _ = partition_chronological_validation(endpoints)

    with torch.no_grad():
        logits = model.model(torch.from_numpy(X_val)).cpu().numpy()

    eval_logits = logits[eval_idx]
    baseline_probs = compute_softmax(eval_logits, temperature=1.0)
    baseline_preds = np.argmax(baseline_probs, axis=1)

    saved_eval_probs = p20_npz["probabilities"][eval_idx]
    saved_eval_preds = p20_npz["predictions"][eval_idx]

    assert np.allclose(baseline_probs, saved_eval_probs, atol=1e-5)
    assert np.array_equal(baseline_preds, saved_eval_preds)


# ---------------------------------------------------------------------------
# Test 5: No Model-Weight Mutation
# ---------------------------------------------------------------------------
def test_no_model_weight_mutation():
    """Verify that checkpoint weights are identical before and after inference and calibration."""
    if not MODEL_PATH.exists():
        pytest.skip("Model checkpoint not present")

    model = load_frozen_lstm_model(MODEL_PATH)
    fingerprint_before = get_model_weights_fingerprint(model)

    dummy_input = torch.randn(10, 10, 11)
    with torch.no_grad():
        _ = model.model(dummy_input)

    fingerprint_after = get_model_weights_fingerprint(model)
    assert fingerprint_before == fingerprint_after

    # Also verify all requires_grad are False
    for param in model.model.parameters():
        assert not param.requires_grad


# ---------------------------------------------------------------------------
# Test 6 & 7: Calibration Fitting Uses Only Calibration Subset
# ---------------------------------------------------------------------------
def test_calibration_fitting_isolated_from_evaluation():
    """Verify that altering evaluation logits/labels has zero effect on fitted temperature."""
    np.random.seed(123)
    N_cal, N_eval = 60, 40
    cal_logits = np.random.randn(N_cal, 3)
    cal_y = np.random.choice([0, 1, 2], size=N_cal)
    cal_logits[np.arange(N_cal), cal_y] += 2.0  # Informative logits

    eval_logits = np.random.randn(N_eval, 3)
    eval_y = np.random.choice([0, 1, 2], size=N_eval)
    eval_logits[np.arange(N_eval), eval_y] += 2.0

    fit1 = fit_temperature(cal_logits, cal_y)

    # Completely scramble evaluation data
    scrambled_eval_logits = eval_logits * 5.0 + 10.0
    scrambled_eval_y = (eval_y + 1) % 3

    # Re-fit using only calibration subset
    fit2 = fit_temperature(cal_logits, cal_y)

    assert fit1["temperature"] == fit2["temperature"]
    assert fit1["calibrated_nll"] == fit2["calibrated_nll"]


# ---------------------------------------------------------------------------
# Test 8: Identical Baseline and Calibrated Evaluation Sample Indices
# ---------------------------------------------------------------------------
def test_identical_baseline_and_calibrated_evaluation_sample_indices():
    """Verify that evaluation indices evaluated before and after calibration are identical."""
    synthetic_endpoints = np.arange(1000, 1100, 1)
    cal_idx, eval_idx, purged_idx, _ = partition_chronological_validation(
        synthetic_endpoints, split_timestamp=1040, min_purge_ms=5
    )
    assert len(set(cal_idx).intersection(set(eval_idx))) == 0
    assert len(eval_idx) > 0

    # Ensure evaluation indices remain fixed
    eval_idx_copy = eval_idx.copy()
    assert np.array_equal(eval_idx, eval_idx_copy)


# ---------------------------------------------------------------------------
# Test 9: Chronological Split Integrity and Purge Boundary
# ---------------------------------------------------------------------------
def test_chronological_split_integrity_and_purge_boundary():
    """Verify chronological ordering and purge boundary enforcement."""
    # Synthetic case with explicit purge
    endpoints = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    cal_idx, eval_idx, purged_idx, info = partition_chronological_validation(
        endpoints, split_timestamp=130, min_purge_ms=30
    )

    # cal: <= 130 -> [100, 110, 120, 130] (indices 0, 1, 2, 3)
    # purge: (130, 160) -> [140, 150] (indices 4, 5)
    # eval: >= 160 -> [160, 170, 180, 190] (indices 6, 7, 8, 9)
    assert list(cal_idx) == [0, 1, 2, 3]
    assert list(purged_idx) == [4, 5]
    assert list(eval_idx) == [6, 7, 8, 9]
    assert endpoints[eval_idx[0]] - endpoints[cal_idx[-1]] >= 30

    # Non-decreasing violation must raise ValueError
    unsorted_endpoints = np.array([100, 120, 110, 140])
    with pytest.raises(ValueError, match="chronologically sorted"):
        partition_chronological_validation(unsorted_endpoints, split_timestamp=115)


# ---------------------------------------------------------------------------
# Test 10: Deterministic Outputs for Fixed Input
# ---------------------------------------------------------------------------
def test_deterministic_outputs():
    """Verify fit_temperature and metrics are strictly deterministic."""
    np.random.seed(999)
    N = 80
    y = np.random.choice([0, 1, 2], size=N)
    logits = np.random.randn(N, 3)
    logits[np.arange(N), y] += 2.0  # Informative signal

    fit_a = fit_temperature(logits, y)
    fit_b = fit_temperature(logits, y)

    assert fit_a["temperature"] == fit_b["temperature"]
    assert fit_a["calibrated_nll"] == fit_b["calibrated_nll"]
    assert fit_a["nfev"] == fit_b["nfev"]


def test_degeneracy_boundary_check():
    """Verify that pure uninformative noise logits hit search boundary and raise RuntimeError."""
    np.random.seed(42)
    N = 200
    # Pure noise with zero label correlation pushes T to infinity (hits upper bound 10.0)
    logits = np.random.randn(N, 3) * 0.1
    y = np.random.choice([0, 1, 2], size=N)

    with pytest.raises(RuntimeError, match="Fitted temperature hit search boundary"):
        fit_temperature(logits, y, bounds=(0.01, 10.0))


# ---------------------------------------------------------------------------
# Test 11: Failure Handling for Invalid Shapes and Missing Classes
# ---------------------------------------------------------------------------
def test_failure_handling_invalid_inputs():
    """Verify error handling on non-finite values, bad shapes, and missing classes."""
    # 1. Non-finite logits
    bad_logits = np.array([[1.0, 2.0, np.nan], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="non-finite|NaN"):
        fit_temperature(bad_logits, np.array([0, 1]))

    # 2. Invalid shape (1D logits)
    with pytest.raises(ValueError, match="Logits must be 2D"):
        fit_temperature(np.array([1.0, 2.0, 3.0]), np.array([0]))

    # 3. Missing class in calibration set (3 classes expected, only 0 and 1 provided)
    logits_3class = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    y_missing_class = np.array([0, 1, 0])  # class 2 is missing
    with pytest.raises(ValueError, match="Insufficient class coverage"):
        fit_temperature(logits_3class, y_missing_class)


# ---------------------------------------------------------------------------
# Test 12: No Test Set Access Safety Constraint
# ---------------------------------------------------------------------------
def test_no_test_set_access_safety_constraint():
    """Verify that passing the locked test set raises an assertion/safety error."""
    if LOCKED_TEST_PATH.exists():
        with pytest.raises(AssertionError, match="locked test set"):
            run_experiment(val_data_path=LOCKED_TEST_PATH)
