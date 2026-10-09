"""
Comprehensive unit tests for Pipeline V2 Data Quality and Training Readiness Gate.

Covers all 20 required tests:
1. All checks passing on a synthetic TEST FIXTURE ONLY
2. 100% FLAT dataset fails class-diversity gate
3. Missing UP class fails
4. Missing DOWN class fails
5. Class below 5% fails
6. Unique mid prices below 25 fails
7. Directional transitions below 100 fails
8. Insufficient observations fails
9. Excessive stale data fails
10. NaN fails
11. Inf fails
12. Duplicate sequence endpoints fail
13. Cross-split leakage fails
14. Invalid scaler metadata fails
15. Valid scaler metadata passes
16. Deterministic gate result
17. Configurable thresholds
18. No mutation of input data
19. Constant-feature warning behavior
20. Current real dataset produces honest result (pipeline_valid=True, training_ready=False)
"""

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.validation.data_quality_gate import (
    DataQualityGate,
    QualityGateCheckResult,
    QualityGateConfig,
    QualityGateReport,
)


@pytest.fixture
def perfect_synthetic_fixture():
    """
    Generate synthetic data that passes ALL quality criteria.
    FOR TEST PURPOSES ONLY. Never written to clean_v2.
    """
    np.random.seed(42)
    n_train = 8000
    n_val = 1500
    n_test = 1500
    seq_len = 10
    n_feat = 11

    # Varied features with mean 0, std 1
    X_train = np.random.normal(loc=0.0, scale=1.0, size=(n_train, seq_len, n_feat))
    X_val = np.random.normal(loc=0.0, scale=1.0, size=(n_val, seq_len, n_feat))
    X_test = np.random.normal(loc=0.0, scale=1.0, size=(n_test, seq_len, n_feat))

    # Balanced targets with alternating classes to ensure >100 transitions
    classes = ["UP", "DOWN", "FLAT"]
    y_train = np.array([classes[i % 3] for i in range(n_train)])
    y_val = np.array([classes[i % 3] for i in range(n_val)])
    y_test = np.array([classes[i % 3] for i in range(n_test)])

    # Distinct non-overlapping endpoints
    ep_train = np.arange(10_000_000, 10_000_000 + n_train * 1000, 1000, dtype=np.int64)
    ep_val = np.arange(20_000_000, 20_000_000 + n_val * 1000, 1000, dtype=np.int64)
    ep_test = np.arange(30_000_000, 30_000_000 + n_test * 1000, 1000, dtype=np.int64)

    feat_names = [
        "mid_price", "spread", "spread_bps", "mid_return_1s", "mid_return_3s",
        "mid_return_5s", "mid_volatility_5s", "bid_change_1s", "ask_change_1s",
        "microprice", "depth_imbalance"
    ]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "ep_train": ep_train,
        "ep_val": ep_val,
        "ep_test": ep_test,
        "feature_names": feat_names,
        "canonical_count": 60_000,
        "labeled_count": 15_000,
        "stale_pct": 0.05,
        "unique_mid_prices": 50,
        "scaler_fit_source": "train",
        "cross_split_violations": 0,
        "cross_session_violations": 0,
        "scaler_metadata_valid": True,
    }


# 1. All checks passing on a synthetic TEST FIXTURE ONLY
def test_all_checks_passing_on_synthetic_fixture(perfect_synthetic_fixture):
    gate = DataQualityGate()
    report = gate.evaluate_data(**perfect_synthetic_fixture)

    assert report.pipeline_valid is True
    assert report.data_quality_pass is True
    assert report.training_ready is True
    assert len(report.failed_checks) == 0


# 2. 100% FLAT dataset fails class-diversity gate
def test_flat_dataset_fails_class_diversity(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["y_train"] = np.array(["FLAT"] * len(fixture["y_train"]))
    fixture["y_val"] = np.array(["FLAT"] * len(fixture["y_val"]))
    fixture["y_test"] = np.array(["FLAT"] * len(fixture["y_test"]))

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.pipeline_valid is True
    assert report.data_quality_pass is False
    assert report.training_ready is False

    failed_names = [f["name"] for f in report.failed_checks]
    assert "class_balance_minimum" in failed_names or "required_classes_present" in failed_names


# 3. Missing UP class fails
def test_missing_up_class_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    # Only DOWN and FLAT
    fixture["y_train"] = np.array(["DOWN" if i % 2 == 0 else "FLAT" for i in range(len(fixture["y_train"]))])
    fixture["y_val"] = np.array(["DOWN" if i % 2 == 0 else "FLAT" for i in range(len(fixture["y_val"]))])
    fixture["y_test"] = np.array(["DOWN" if i % 2 == 0 else "FLAT" for i in range(len(fixture["y_test"]))])

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    assert report.training_ready is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "required_classes_present" in failed_names


# 4. Missing DOWN class fails
def test_missing_down_class_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    # Only UP and FLAT
    fixture["y_train"] = np.array(["UP" if i % 2 == 0 else "FLAT" for i in range(len(fixture["y_train"]))])
    fixture["y_val"] = np.array(["UP" if i % 2 == 0 else "FLAT" for i in range(len(fixture["y_val"]))])
    fixture["y_test"] = np.array(["UP" if i % 2 == 0 else "FLAT" for i in range(len(fixture["y_test"]))])

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    assert report.training_ready is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "required_classes_present" in failed_names


# 5. Class below 5% fails
def test_class_below_5_pct_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    n = len(fixture["y_train"])
    # UP is only 1% of the dataset
    y_tr = ["FLAT"] * int(n * 0.5) + ["DOWN"] * int(n * 0.49) + ["UP"] * int(n * 0.01)
    fixture["y_train"] = np.array(y_tr)
    fixture["y_val"] = np.array(["FLAT"] * len(fixture["y_val"]))
    fixture["y_test"] = np.array(["FLAT"] * len(fixture["y_test"]))

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "class_balance_minimum" in failed_names


# 6. Unique mid prices below 25 fails
def test_unique_mid_prices_below_25_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["unique_mid_prices"] = 5  # Only 5 unique mid prices

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "price_diversity" in failed_names


# 7. Directional transitions below 100 fails
def test_directional_transitions_below_100_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    # Long contiguous blocks: very few transitions
    n_tr = len(fixture["y_train"])
    fixture["y_train"] = np.array(["UP"] * (n_tr // 2) + ["DOWN"] * (n_tr - n_tr // 2))
    fixture["y_val"] = np.array(["FLAT"] * len(fixture["y_val"]))
    fixture["y_test"] = np.array(["FLAT"] * len(fixture["y_test"]))

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "directional_transitions" in failed_names


# 8. Insufficient observations fails
def test_insufficient_observations_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["canonical_count"] = 498  # Below 50,000

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "canonical_observations_count" in failed_names


# 9. Excessive stale data fails
def test_excessive_stale_data_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["stale_pct"] = 0.54  # 54% > 20% max

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "staleness_threshold" in failed_names


# 10. NaN fails
def test_nan_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["X_train"][0, 0, 0] = np.nan

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "zero_nan_values" in failed_names


# 11. Inf fails
def test_inf_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["X_test"][0, 0, 0] = np.inf

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.data_quality_pass is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "zero_inf_values" in failed_names


# 12. Duplicate sequence endpoints fail
def test_duplicate_sequence_endpoints_fail(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    # Duplicate endpoint between train and test
    fixture["ep_test"][0] = fixture["ep_train"][0]

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.pipeline_valid is False
    assert report.training_ready is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "duplicate_sequence_endpoints" in failed_names


# 13. Cross-split leakage fails
def test_cross_split_leakage_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["cross_split_violations"] = 3

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.pipeline_valid is False
    assert report.training_ready is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "cross_split_leakage" in failed_names


# 14. Invalid scaler metadata fails
def test_invalid_scaler_metadata_fails(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["scaler_fit_source"] = "validation"  # Leakage!
    fixture["scaler_metadata_valid"] = False

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    assert report.pipeline_valid is False
    assert report.training_ready is False
    failed_names = [f["name"] for f in report.failed_checks]
    assert "scaler_fit_source" in failed_names


# 15. Valid scaler metadata passes
def test_valid_scaler_metadata_passes(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["scaler_fit_source"] = "train"
    fixture["scaler_metadata_valid"] = True

    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    scaler_check = report.checks.get("scaler_fit_source")
    assert scaler_check is not None
    assert scaler_check["status"] == "PASS"


# 16. Deterministic gate result
def test_deterministic_gate_result(perfect_synthetic_fixture):
    gate = DataQualityGate()
    r1 = gate.evaluate_data(**perfect_synthetic_fixture)
    r2 = gate.evaluate_data(**perfect_synthetic_fixture)

    assert r1.to_dict() == r2.to_dict()
    assert r1.to_markdown() == r2.to_markdown()


# 17. Configurable thresholds
def test_configurable_thresholds(perfect_synthetic_fixture):
    # Set threshold to 1 unique mid price
    cfg = QualityGateConfig(min_unique_mid_prices=1, min_canonical_observations=100)
    gate = DataQualityGate(config=cfg)

    fixture = copy.deepcopy(perfect_synthetic_fixture)
    fixture["unique_mid_prices"] = 1
    fixture["canonical_count"] = 150

    report = gate.evaluate_data(**fixture)
    assert report.checks["price_diversity"]["status"] == "PASS"
    assert report.checks["canonical_observations_count"]["status"] == "PASS"


# 18. No mutation of input data
def test_no_mutation_of_input_data(perfect_synthetic_fixture):
    X_tr_orig = perfect_synthetic_fixture["X_train"].copy()
    y_tr_orig = perfect_synthetic_fixture["y_train"].copy()

    gate = DataQualityGate()
    _ = gate.evaluate_data(**perfect_synthetic_fixture)

    np.testing.assert_array_equal(perfect_synthetic_fixture["X_train"], X_tr_orig)
    np.testing.assert_array_equal(perfect_synthetic_fixture["y_train"], y_tr_orig)


# 19. Constant-feature warning behavior
def test_constant_feature_warning_behavior(perfect_synthetic_fixture):
    fixture = copy.deepcopy(perfect_synthetic_fixture)
    # Set 5 features to constant 0
    for col in range(5):
        fixture["X_train"][:, :, col] = 0.0

    # Default policy: produces WARNING, not FAIL
    gate = DataQualityGate()
    report = gate.evaluate_data(**fixture)

    var_check = report.checks.get("feature_variance_audit")
    assert var_check is not None
    assert var_check["status"] == "WARNING"

    # Strict policy: produces FAIL
    strict_gate = DataQualityGate(config=QualityGateConfig(strict_zero_variance=True))
    strict_report = strict_gate.evaluate_data(**fixture)
    assert strict_report.checks["feature_variance_audit"]["status"] == "FAIL"


# 20. Current real dataset produces honest result
def test_current_real_dataset_produces_honest_result():
    real_clean_v2 = Path("data/clean_v2")
    if not (real_clean_v2 / "07_scaled").exists():
        pytest.skip("clean_v2/07_scaled not present")

    gate = DataQualityGate()
    report = gate.evaluate_pipeline(real_clean_v2)

    # Invariants of the real 51-minute dataset:
    # Implementation correctness passes
    assert report.pipeline_valid is True
    # But data quality fails because of 100% FLAT, small sample size, 0 transitions, etc.
    assert report.data_quality_pass is False
    assert report.training_ready is False

    failed_names = [f["name"] for f in report.failed_checks]
    assert "class_balance_minimum" in failed_names
    assert "required_classes_present" in failed_names
    assert "price_diversity" in failed_names
    assert "directional_transitions" in failed_names
    assert "canonical_observations_count" in failed_names
    assert "sequence_counts_sufficiency" in failed_names
