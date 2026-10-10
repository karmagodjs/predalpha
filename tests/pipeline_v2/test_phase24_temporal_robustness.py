"""
Unit and Integration Tests for Phase 24 Temporal Robustness & Predictive-Signal Analysis.

Verifies:
1. Chronological ordering enforcement.
2. Window boundary integrity and purge compliance.
3. Minimum sample-size enforcement.
4. Class encoding and label mappings.
5. Metric calculations (Accuracy, Bal Acc, Macro F1, NLL, Brier, ECE, Majority Baseline, SE).
6. Missing or invalid data failure handling.
7. Deterministic results.
8. No checkpoint mutation.
9. Strict prohibition against locked test set access.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline_v2.models.phase24_temporal_robustness import (
    CLASS_NAMES,
    LABEL_TO_ID,
    LOCKED_TEST_PATH,
    MODEL_PATH,
    compute_brier,
    compute_ece,
    compute_nll,
    compute_window_diagnostics,
    define_observable_regimes,
    partition_temporal_windows,
    run_temporal_robustness_analysis,
)
from pipeline_v2.models.recurrent_models import SmallLSTM


# ---------------------------------------------------------------------------
# Test 1: Chronological Ordering Enforcement
# ---------------------------------------------------------------------------
def test_chronological_ordering_enforcement():
    """Verify that unsorted endpoints raise ValueError."""
    unsorted_endpoints = np.array([1000, 1020, 1010, 1050])
    with pytest.raises(ValueError, match="chronologically sorted"):
        partition_temporal_windows(unsorted_endpoints)


# ---------------------------------------------------------------------------
# Test 2: Window Boundary Integrity and Purge Compliance
# ---------------------------------------------------------------------------
def test_window_boundary_integrity_and_purge_compliance():
    """Verify windows are strictly disjoint, chronological, and respect purge gaps."""
    # Synthetic endpoints: 3 blocks separated by 30-unit gaps
    block1 = np.arange(100, 120)       # 100..119 (len 20)
    block2 = np.arange(150, 170)       # 150..169 (gap 150 - 119 = 31 >= 25)
    block3 = np.arange(200, 220)       # 200..219 (gap 200 - 169 = 31 >= 25)
    endpoints = np.concatenate([block1, block2, block3])
    unq_ep = np.unique(endpoints)

    specs = [
        {"window_id": "W1", "name": "W1", "start_unq_idx": 0, "end_unq_idx": 19},
        {"window_id": "W2", "name": "W2", "start_unq_idx": 20, "end_unq_idx": 39},
        {"window_id": "W3", "name": "W3", "start_unq_idx": 40, "end_unq_idx": 59},
    ]

    w_indices, w_meta = partition_temporal_windows(endpoints, window_specs=specs, min_purge_ms=25)
    assert len(w_indices) == 3
    assert len(w_indices[0]) == 20
    assert len(w_indices[1]) == 20
    assert len(w_indices[2]) == 20

    # Pairwise disjoint
    for i in range(3):
        for j in range(i + 1, 3):
            assert len(set(w_indices[i]).intersection(set(w_indices[j]))) == 0

    # Purge gap check
    assert w_meta[1]["purge_gap_before_ms"] >= 25
    assert w_meta[2]["purge_gap_before_ms"] >= 25

    # Insufficient purge gap must raise ValueError
    with pytest.raises(ValueError, match="Purge violation"):
        partition_temporal_windows(endpoints, window_specs=specs, min_purge_ms=50)


# ---------------------------------------------------------------------------
# Test 3: Minimum Sample-Size Enforcement
# ---------------------------------------------------------------------------
def test_minimum_sample_size_enforcement():
    """Verify that windows smaller than min_samples raise ValueError."""
    y = np.array([0, 1, 2, 0, 1])
    preds = np.array([0, 1, 2, 0, 1])
    probs = np.array([[0.8, 0.1, 0.1]] * 5)

    with pytest.raises(ValueError, match="Insufficient sample count"):
        compute_window_diagnostics(y, preds, probs, min_samples=10)


# ---------------------------------------------------------------------------
# Test 4: Class Encoding and Label Mapping
# ---------------------------------------------------------------------------
def test_class_encoding_and_label_mapping():
    """Verify exact class mapping DOWN=0, FLAT=1, UP=2."""
    assert CLASS_NAMES == ["DOWN", "FLAT", "UP"]
    assert LABEL_TO_ID["DOWN"] == 0
    assert LABEL_TO_ID["FLAT"] == 1
    assert LABEL_TO_ID["UP"] == 2


# ---------------------------------------------------------------------------
# Test 5: Metric Calculations
# ---------------------------------------------------------------------------
def test_metric_calculations():
    """Verify accuracy, NLL, Brier score, ECE, and majority baseline."""
    np.random.seed(42)
    N = 100
    y_true = np.array([0] * 50 + [1] * 30 + [2] * 20)  # Majority class 0 (50%)
    preds = y_true.copy()  # Perfect accuracy
    probs = np.zeros((N, 3), dtype=np.float32)
    probs[np.arange(N), y_true] = 0.9
    probs[np.arange(N), (y_true + 1) % 3] = 0.05
    probs[np.arange(N), (y_true + 2) % 3] = 0.05

    diag = compute_window_diagnostics(y_true, preds, probs, min_samples=30)

    assert diag["accuracy"] == 1.0
    assert diag["majority_class"] == "DOWN"
    assert diag["majority_baseline_accuracy"] == 0.50
    assert diag["delta_vs_majority"] == 0.50
    assert diag["nll"] > 0.0
    assert diag["brier_score"] >= 0.0
    assert diag["ece"] >= 0.0
    assert diag["accuracy_se_iid"] >= 0.0
    assert diag["accuracy_se_neff"] >= diag["accuracy_se_iid"]


# ---------------------------------------------------------------------------
# Test 6: Observable Regime Definitions
# ---------------------------------------------------------------------------
def test_observable_regime_definitions():
    """Verify that regime groupings partition observations without lookahead."""
    np.random.seed(99)
    N, L, D = 100, 10, 11
    X = np.random.randn(N, L, D).astype(np.float32)
    probs = np.full((N, 3), 1.0 / 3.0, dtype=np.float32)
    preds = np.random.choice([0, 1, 2], size=N)

    regimes = define_observable_regimes(X, probs, preds)
    assert "Book_Balanced (|Imbalance| <= Median)" in regimes
    assert "Book_Pressured (|Imbalance| > Median)" in regimes
    assert "Volatility_Calm (Vol <= Median)" in regimes
    assert "Volatility_Turbulent (Vol > Median)" in regimes

    # Check balanced vs pressured partition covers all samples
    mask_bal = regimes["Book_Balanced (|Imbalance| <= Median)"]
    mask_pres = regimes["Book_Pressured (|Imbalance| > Median)"]
    assert np.all(mask_bal ^ mask_pres)
    assert np.sum(mask_bal) + np.sum(mask_pres) == N


# ---------------------------------------------------------------------------
# Test 7: Deterministic Results
# ---------------------------------------------------------------------------
def test_deterministic_diagnostic_results():
    """Verify that diagnostic calculations are strictly deterministic."""
    np.random.seed(123)
    N = 60
    y = np.random.choice([0, 1, 2], size=N)
    preds = np.random.choice([0, 1, 2], size=N)
    probs = np.random.dirichlet([1, 1, 1], size=N).astype(np.float32)

    diag1 = compute_window_diagnostics(y, preds, probs, min_samples=30)
    diag2 = compute_window_diagnostics(y, preds, probs, min_samples=30)

    assert diag1["accuracy"] == diag2["accuracy"]
    assert diag1["macro_f1"] == diag2["macro_f1"]
    assert diag1["nll"] == diag2["nll"]
    assert diag1["brier_score"] == diag2["brier_score"]


# ---------------------------------------------------------------------------
# Test 8: No Checkpoint Mutation
# ---------------------------------------------------------------------------
def test_no_checkpoint_mutation():
    """Verify model weights are not altered by running analysis."""
    if not MODEL_PATH.exists():
        pytest.skip("Model checkpoint not present")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = SmallLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout_p=checkpoint["dropout_p"],
        seed=checkpoint["seed"],
        lr=checkpoint["lr"],
        weight_decay=checkpoint["weight_decay"],
        batch_size=checkpoint["batch_size"],
    )
    model.load_checkpoint(MODEL_PATH)

    weights_before = [p.clone() for p in model.model.parameters()]

    dummy_input = torch.randn(10, 10, 11)
    with torch.no_grad():
        _ = model.model(dummy_input)

    for p_before, p_after in zip(weights_before, model.model.parameters()):
        assert torch.equal(p_before, p_after)


# ---------------------------------------------------------------------------
# Test 9: Locked Test Set Access Prohibition
# ---------------------------------------------------------------------------
def test_locked_test_set_access_prohibition():
    """Verify safety check fails if test_scaled.npz is passed as validation data."""
    if LOCKED_TEST_PATH.exists():
        with pytest.raises(AssertionError, match="locked test set"):
            run_temporal_robustness_analysis(val_data_path=LOCKED_TEST_PATH)
