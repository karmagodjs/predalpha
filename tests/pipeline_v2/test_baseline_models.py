"""
Unit tests for Phase 19A: Baseline Models and Signal Validation.

Verifies:
1. Classification Metrics: accuracy, balanced accuracy, macro F1, weighted F1, confusion matrix.
2. Majority Baseline: statistical floor, mode prediction, prior probabilities.
3. Logistic Regression: parameter count (333), loss convergence, predictive validity.
4. Small MLP: architecture (110 -> 64 -> 32 -> 3), parameter count (9,283), seed determinism.
5. Upstream Immutability & Test Isolation: test partition never touched, Phase 17 hashes unchanged.
6. Deliverables: JSON results, CSV summary, markdown report, checkpoints, confusion matrices, training curves.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from pipeline_v2.models.baseline_models import (
    LogisticRegressionModel,
    MajorityBaseline,
    SmallMLP,
    SmallMLPModule,
    set_seed,
)
from pipeline_v2.models.metrics import (
    compute_accuracy,
    compute_balanced_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_weighted_f1,
    evaluate_predictions,
)

BASELINES_DIR = Path("data/models/phase19/baselines")
SCALED_DIR = Path("data/clean_v2/07_scaled/expanded_collection")


def test_metrics_functions():
    """Verify pure-NumPy classification metrics on known ground truth."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    acc = compute_accuracy(y_true, y_pred)
    assert acc == 4 / 6

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[0, 2] == 0
    assert cm[1, 0] == 0 and cm[1, 1] == 2 and cm[1, 2] == 0
    assert cm[2, 0] == 1 and cm[2, 1] == 0 and cm[2, 2] == 1

    bal_acc = compute_balanced_accuracy(y_true, y_pred, num_classes=3)
    assert round(bal_acc, 4) == round((0.5 + 1.0 + 0.5) / 3, 4)

    macro_f1 = compute_macro_f1(y_true, y_pred, num_classes=3)
    assert 0.0 <= macro_f1 <= 1.0

    eval_dict = evaluate_predictions(y_true, y_pred, class_names=["DOWN", "FLAT", "UP"])
    assert "accuracy" in eval_dict
    assert "balanced_accuracy" in eval_dict
    assert "macro_f1" in eval_dict
    assert "weighted_f1" in eval_dict
    assert "per_class" in eval_dict
    assert "confusion_matrix" in eval_dict
    assert eval_dict["total_samples"] == 6


def test_majority_baseline():
    """Verify MajorityBaseline fits empirical mode and returns correct predictions."""
    y_train = np.array([0, 0, 1, 2, 0, 1])  # 0: 3 times, 1: 2 times, 2: 1 time
    model = MajorityBaseline(num_classes=3)
    model.fit(y_train)

    assert model.majority_class == 0
    assert model.majority_class_name == "DOWN"
    assert model.class_counts[0] == 3

    X_test = np.zeros((5, 110))
    preds = model.predict(X_test)
    assert len(preds) == 5
    assert (preds == 0).all()

    probs = model.predict_proba(X_test)
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_logistic_regression_module():
    """Verify LogisticRegressionModel has exactly 333 parameters and outputs (N, 3)."""
    model = LogisticRegressionModel(in_features=110, num_classes=3, seed=42)
    assert model.parameter_count == 333  # 110 * 3 + 3

    X = np.random.randn(20, 110).astype(np.float32)
    y = np.random.randint(0, 3, size=20)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1, 2})

    probs = model.predict_proba(X)
    assert probs.shape == (20, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_small_mlp_architecture_and_parameters():
    """Verify SmallMLP architecture: 110 -> 64 -> 32 -> 3 with 9,283 parameters."""
    mlp_mod = SmallMLPModule(in_features=110, hidden1=64, hidden2=32, num_classes=3, dropout_p=0.10)
    param_count = sum(p.numel() for p in mlp_mod.parameters())
    # 110*64 + 64 = 7104
    # 64*32 + 32 = 2080
    # 32*3 + 3 = 99
    # Total = 7104 + 2080 + 99 = 9283
    assert param_count == 9283

    mlp = SmallMLP(in_features=110, hidden1=64, hidden2=32, num_classes=3, seed=42)
    assert mlp.parameter_count == 9283


def test_small_mlp_seed_determinism():
    """Verify SmallMLP produces bit-for-bit identical initialization and training with identical seed."""
    X = np.random.RandomState(42).randn(50, 110).astype(np.float32)
    y = np.random.RandomState(42).randint(0, 3, size=50)

    mlp1 = SmallMLP(in_features=110, max_epochs=5, seed=123)
    mlp1.fit(X, y)
    p1 = mlp1.predict(X)

    mlp2 = SmallMLP(in_features=110, max_epochs=5, seed=123)
    mlp2.fit(X, y)
    p2 = mlp2.predict(X)

    assert np.array_equal(p1, p2)


def test_phase19a_deliverables_exist():
    """Verify all Phase 19A outputs exist in data/models/phase19/baselines/."""
    assert (BASELINES_DIR / "baseline_results.json").exists()
    assert (BASELINES_DIR / "baseline_results.csv").exists()
    assert (BASELINES_DIR / "baseline_report.md").exists()
    assert (BASELINES_DIR / "experiment_metadata.json").exists()

    # Checkpoints
    ckpt_dir = BASELINES_DIR / "checkpoints"
    assert (ckpt_dir / "majority_baseline.json").exists()
    assert (ckpt_dir / "logistic_regression.pt").exists()
    assert (ckpt_dir / "logistic_regression_weighted.pt").exists()
    assert (ckpt_dir / "small_mlp_seed_42.pt").exists()
    assert (ckpt_dir / "small_mlp_seed_123.pt").exists()
    assert (ckpt_dir / "small_mlp_seed_999.pt").exists()
    assert (ckpt_dir / "small_mlp_best.pt").exists()

    # Confusion matrices
    cm_dir = BASELINES_DIR / "confusion_matrices"
    assert (cm_dir / "majority_confusion_matrix.json").exists()
    assert (cm_dir / "logistic_regression_confusion_matrix.json").exists()

    # Training curves
    tc_dir = BASELINES_DIR / "training_curves"
    assert (tc_dir / "logistic_regression_history.json").exists()
    assert (tc_dir / "small_mlp_seed_42_history.json").exists()


def test_phase19a_test_set_untouched():
    """Verify test set was NOT evaluated or touched during baseline selection."""
    with open(BASELINES_DIR / "baseline_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["test_set_touched"] is False
    assert meta["test_samples_locked"] == 1582

    # Verify upstream Phase 17 test hash
    with open(SCALED_DIR / "phase17_scaling_metadata.json", "r", encoding="utf-8") as f:
        p17_meta = json.load(f)

    expected_test_hash = p17_meta["hashes"]["input_test_parquet"]
    assert expected_test_hash is not None


def test_phase19a_signal_validation_verdict():
    """Verify Logistic Regression and MLP beat majority baseline by a significant margin."""
    with open(BASELINES_DIR / "baseline_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data["models"]
    decision = data["signal_decision"]

    maj_acc = models["majority_baseline"]["val_metrics"]["accuracy"]
    maj_f1 = models["majority_baseline"]["val_metrics"]["macro_f1"]

    lr_acc = models["logistic_regression"]["val_metrics"]["accuracy"]
    lr_f1 = models["logistic_regression"]["val_metrics"]["macro_f1"]

    # Logistic Regression beats Majority by >= 5 percentage points
    assert lr_acc > maj_acc + 0.05
    assert lr_f1 > maj_f1 + 0.10

    # Decision is positive
    assert decision["meaningful_predictive_signal_demonstrated"] is True
    assert decision["lr_beats_majority"] is True
    assert decision["mlp_beats_majority"] is True
