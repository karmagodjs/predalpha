"""Unit and integration tests for Phase 2: Baseline Models & Evaluation.

Covers:
- Correct Phase 1 dataset loading & schema validation
- Feature/target separation and non-feature exclusions
- Non-finite and missing feature detection
- Forbidden target leakage column detection
- Train-only scaler fitting (no data leakage)
- Reproducibility under fixed random seed
- DummyClassifier baseline behavior
- LogisticRegression training and pipeline architecture
- Explicit missing-class metric handling
- Strict validation-only model selection (no test-set tuning)
- Output directory isolation
- Preservation of Phase 1 artifacts (SHA-256 hash checks)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.baseline_models import (
    CONFIG,
    BaselineConfig,
    build_dummy_pipeline,
    build_logistic_regression_pipeline,
    compute_split_metrics,
    extract_features_and_target,
    load_phase1_splits,
    run_phase2_baseline_pipeline,
    select_candidate_model,
)


@pytest.fixture
def synthetic_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate isolated synthetic train, validation, and test dataframes."""
    np.random.seed(42)
    feature_cols = list(CONFIG.feature_cols)

    def make_split(n: int, start_time: str, class_probs: list[float]) -> pd.DataFrame:
        ts = pd.date_range(start_time, periods=n, freq="1s", tz="UTC")
        data: dict[str, Any] = {
            "timestamp": ts,
            "asset_id": "btc_88000",
            "bid": 0.080 + np.random.uniform(0.001, 0.005, size=n),
            "ask": 0.085 + np.random.uniform(0.001, 0.005, size=n),
        }
        for col in feature_cols:
            data[col] = np.random.randn(n)

        # Labels
        classes = np.array(["DOWN", "FLAT", "UP"])
        data["label"] = np.random.choice(classes, size=n, p=class_probs)
        return pd.DataFrame(data, index=ts)

    # Train has all 3 classes
    train_df = make_split(100, "2026-01-01 00:00:00", [0.1, 0.8, 0.1])
    # Val lacks DOWN
    val_df = make_split(30, "2026-01-01 00:02:00", [0.0, 0.9, 0.1])
    # Test lacks UP
    test_df = make_split(30, "2026-01-01 00:03:00", [0.1, 0.9, 0.0])

    return train_df, val_df, test_df


def test_phase1_dataset_loading():
    """Verify loading of Phase 1 parquet splits, row counts, and chronological ordering."""
    phase1_dir = Path("data/processed/phase1")
    splits = load_phase1_splits(data_dir=phase1_dir, config=CONFIG)

    assert set(splits.keys()) == {"train", "validation", "test"}
    assert len(splits["train"]) == 202
    assert len(splits["validation"]) == 38
    assert len(splits["test"]) == 39

    # Verify chronological ordering and non-overlap
    train_max = pd.to_datetime(splits["train"][CONFIG.timestamp_col].max(), utc=True)
    val_min = pd.to_datetime(splits["validation"][CONFIG.timestamp_col].min(), utc=True)
    val_max = pd.to_datetime(splits["validation"][CONFIG.timestamp_col].max(), utc=True)
    test_min = pd.to_datetime(splits["test"][CONFIG.timestamp_col].min(), utc=True)

    assert train_max < val_min, f"Overlap between train ({train_max}) and val ({val_min})"
    assert val_max < test_min, f"Overlap between val ({val_max}) and test ({test_min})"


def test_feature_target_separation_and_exclusions():
    """Verify feature/target extraction strictly excludes non-features and retains all 9 features."""
    train_df = pd.read_parquet("data/processed/phase1/train.parquet")
    X, y = extract_features_and_target(train_df, config=CONFIG)

    # Feature matrix must contain exactly the 9 approved features
    assert list(X.columns) == list(CONFIG.feature_cols)
    assert len(X.columns) == 9
    assert len(X) == len(train_df)
    assert len(y) == len(train_df)

    # Strictly exclude timestamps, identifiers, quotes, and target
    excluded = {"timestamp", "asset_id", "bid", "ask", "label"}
    assert excluded.isdisjoint(set(X.columns)), f"Found excluded columns in X: {set(X.columns) & excluded}"


def test_non_finite_and_missing_features_detection():
    """Verify that NaN and Inf values trigger explicit errors and are never silently passed."""
    df = pd.read_parquet("data/processed/phase1/train.parquet").copy()

    # 1. Test NaN injection
    nan_df = df.copy()
    nan_df.loc[nan_df.index[0], "mid_price"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        extract_features_and_target(nan_df, config=CONFIG)

    # 2. Test positive Inf injection
    inf_df = df.copy()
    inf_df.loc[inf_df.index[0], "spread_bps"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        extract_features_and_target(inf_df, config=CONFIG)

    # 3. Test negative Inf injection
    ninf_df = df.copy()
    ninf_df.loc[ninf_df.index[0], "mid_return_1s"] = -np.inf
    with pytest.raises(ValueError, match="non-finite"):
        extract_features_and_target(ninf_df, config=CONFIG)


def test_forbidden_leakage_columns_detection():
    """Verify that forbidden future-looking columns in input dataframe are immediately flagged."""
    df = pd.read_parquet("data/processed/phase1/train.parquet").copy()

    for forbidden_col in ("future_mid", "future_delta", "future_return", "label_threshold"):
        leak_df = df.copy()
        leak_df[forbidden_col] = 1.0
        with pytest.raises(ValueError, match="Target leakage columns detected"):
            extract_features_and_target(leak_df, config=CONFIG)


def test_train_only_scaler_fitting_no_leakage(synthetic_split_data):
    """Verify that StandardScaler fits strictly on train and validation data does not leak into scaler statistics."""
    train_df, val_df, _ = synthetic_split_data

    # Make train have mean 100.0, val have mean 10,000.0
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["mid_price"] = 100.0 + np.random.randn(len(train_df)) * 0.1
    val_df["mid_price"] = 10000.0 + np.random.randn(len(val_df)) * 0.1

    X_train, y_train = extract_features_and_target(train_df, config=CONFIG)
    X_val, y_val = extract_features_and_target(val_df, config=CONFIG)

    lr_pipe = build_logistic_regression_pipeline(config=CONFIG)
    lr_pipe.fit(X_train, y_train)

    scaler: StandardScaler = lr_pipe.named_steps["scaler"]
    mid_price_idx = list(CONFIG.feature_cols).index("mid_price")
    fitted_mean = scaler.mean_[mid_price_idx]

    # Scaler mean must reflect train (~100.0), NOT validation (~10,000.0)
    assert abs(fitted_mean - 100.0) < 1.0, f"Scaler mean {fitted_mean} leaked validation data!"

    # Transforming or predicting on val must not modify fitted scaler parameters
    old_mean = scaler.mean_.copy()
    lr_pipe.predict(X_val)
    np.testing.assert_array_equal(scaler.mean_, old_mean)


def test_dummy_classifier_baseline(synthetic_split_data):
    """Verify DummyClassifier baseline predicts the most frequent class."""
    train_df, val_df, _ = synthetic_split_data
    X_train, y_train = extract_features_and_target(train_df, config=CONFIG)
    X_val, _ = extract_features_and_target(val_df, config=CONFIG)

    majority_class = y_train.value_counts().idxmax()
    dummy_pipe = build_dummy_pipeline(config=CONFIG)
    dummy_pipe.fit(X_train, y_train)

    preds = dummy_pipe.predict(X_val)
    assert (preds == majority_class).all(), "DummyClassifier predicted a non-majority class!"


def test_logistic_regression_training_and_reproducibility(synthetic_split_data):
    """Verify LogisticRegression pipeline training converges and is bit-for-bit reproducible."""
    train_df, val_df, _ = synthetic_split_data
    X_train, y_train = extract_features_and_target(train_df, config=CONFIG)
    X_val, _ = extract_features_and_target(val_df, config=CONFIG)

    pipe1 = build_logistic_regression_pipeline(config=CONFIG)
    pipe2 = build_logistic_regression_pipeline(config=CONFIG)

    pipe1.fit(X_train, y_train)
    pipe2.fit(X_train, y_train)

    # Predictions must be identical
    p1 = pipe1.predict(X_val)
    p2 = pipe2.predict(X_val)
    np.testing.assert_array_equal(p1, p2)

    # Probabilities must be identical
    proba1 = pipe1.predict_proba(X_val)
    proba2 = pipe2.predict_proba(X_val)
    np.testing.assert_array_almost_equal(proba1, proba2, decimal=10)

    # Probabilities must sum to 1.0
    row_sums = proba1.sum(axis=1)
    np.testing.assert_array_almost_equal(row_sums, np.ones(len(X_val)), decimal=6)


def test_missing_class_metric_handling(synthetic_split_data):
    """Verify metric computation explicitly identifies absent classes and handles log loss safely."""
    train_df, val_df, _ = synthetic_split_data
    X_train, y_train = extract_features_and_target(train_df, config=CONFIG)
    X_val, y_val = extract_features_and_target(val_df, config=CONFIG)

    lr_pipe = build_logistic_regression_pipeline(config=CONFIG)
    lr_pipe.fit(X_train, y_train)

    val_metrics = compute_split_metrics(lr_pipe, X_val, y_val, "validation", config=CONFIG)

    # Check absent ground truth classes detection
    assert "DOWN" in val_metrics["absent_classes_ground_truth"]
    assert val_metrics["per_class"]["DOWN"]["support"] == 0
    assert val_metrics["per_class"]["DOWN"]["present_in_ground_truth"] is False

    # Log loss must be omitted because ground truth lacks DOWN
    assert val_metrics["log_loss"] is None
    assert val_metrics["log_loss_status"] == "OMITTED_MISSING_CLASSES_IN_SPLIT"

    # Accuracy and balanced accuracy must be valid floats between 0 and 1
    assert 0.0 <= val_metrics["accuracy"] <= 1.0
    assert 0.0 <= val_metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= val_metrics["macro_f1"] <= 1.0


def test_no_accidental_test_set_tuning():
    """Verify that model selection logic uses strictly validation split metrics."""
    # Dummy wins on validation
    val_dummy_wins = {
        "accuracy": 0.95,
        "balanced_accuracy": 0.50,
        "macro_f1": 0.40,
    }
    val_lr_loses = {
        "accuracy": 0.30,
        "balanced_accuracy": 0.20,
        "macro_f1": 0.15,
    }

    decision = select_candidate_model(val_dummy_wins, val_lr_loses, config=CONFIG)
    assert decision["selected_model"] == "dummy_classifier"
    assert decision["verdict"] == "FAILED_BASELINE"
    assert decision["beats_baseline"] is False

    # LogisticRegression wins on validation
    val_lr_wins = {
        "accuracy": 0.96,
        "balanced_accuracy": 0.65,
        "macro_f1": 0.60,
    }
    decision2 = select_candidate_model(val_dummy_wins, val_lr_wins, config=CONFIG)
    assert decision2["selected_model"] == "logistic_regression"
    assert decision2["verdict"] == "PASSED_BASELINE"
    assert decision2["beats_baseline"] is True


def test_output_directory_isolation(tmp_path):
    """Verify that Phase 2 outputs are isolated to the requested output directory."""
    phase1_dir = Path("data/processed/phase1")
    custom_out = tmp_path / "phase2_custom"

    res = run_phase2_baseline_pipeline(
        phase1_dir=phase1_dir,
        output_dir=custom_out,
        config=CONFIG,
    )

    expected_files = [
        "baseline_metrics.json",
        "baseline_evaluation_report.md",
        "model_config.json",
        "confusion_matrices.json",
        "dummy_pipeline.joblib",
        "logistic_regression_pipeline.joblib",
        "candidate_pipeline.joblib",
    ]
    for fname in expected_files:
        p = custom_out / fname
        assert p.exists(), f"Expected artifact {fname} not found in {custom_out}"
        assert p.stat().st_size > 0, f"Artifact {fname} is empty"


def test_phase1_artifact_preservation_hashes():
    """Verify that Phase 1 files remain bit-for-bit unchanged (SHA-256 hash preservation)."""
    expected_hashes = {
        "data/processed/phase1/canonical_dataset.parquet": "c9d00febf2941405",
        "data/processed/phase1/train.parquet": "ecfa46a505bb2ec6",
        "data/processed/phase1/validation.parquet": "a183493beea22beb",
        "data/processed/phase1/test.parquet": "7b079c4717807679",
        "data/processed/phase1/dataset_metadata.json": "4a73776acc645c8b",
        "data/processed/phase1/validation_report.json": "d35cb6f8c2c20627",
    }

    for path_str, exp_hash in expected_hashes.items():
        p = Path(path_str)
        assert p.exists(), f"Phase 1 artifact missing: {path_str}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h.startswith(exp_hash), (
            f"Phase 1 artifact {path_str} was modified! Expected prefix {exp_hash}, got {h}"
        )
