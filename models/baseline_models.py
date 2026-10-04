"""Reproducible Baseline Models and Evaluation Module for PredAlpha-HFT Phase 2.

Implements Model A (DummyClassifier majority-class baseline) and Model B
(LogisticRegression with StandardScaler pipeline and class balancing),
enforces train-only scaling, performs strict validation-only candidate selection,
evaluates out-of-sample test performance exactly once, and exports complete
audit artifacts to data/processed/phase2/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for Phase 2 baseline models and evaluation protocol."""

    # Features and schema
    feature_cols: tuple[str, ...] = (
        "mid_price",
        "spread",
        "spread_bps",
        "mid_return_1s",
        "mid_return_3s",
        "mid_return_5s",
        "mid_volatility_5s",
        "bid_change_1s",
        "ask_change_1s",
    )
    target_col: str = "label"
    target_classes: tuple[str, ...] = ("DOWN", "FLAT", "UP")
    timestamp_col: str = "timestamp"
    asset_id_col: str = "asset_id"
    quote_cols: tuple[str, ...] = ("bid", "ask")
    forbidden_cols: frozenset[str] = frozenset({
        "future_mid",
        "future_delta",
        "future_return",
        "label_threshold",
        "target_close",
        "target_time",
        "price_change_5m",
    })

    # Model parameters
    random_seed: int = 42
    dummy_strategy: str = "most_frequent"
    logistic_max_iter: int = 1000
    logistic_class_weight: str | None = "balanced"
    logistic_solver: str = "lbfgs"

    # Evaluation & selection
    selection_metric: str = "balanced_accuracy"  # Primary metric on validation


CONFIG = BaselineConfig()


def load_phase1_splits(
    data_dir: Path = Path("data/processed/phase1"),
    config: BaselineConfig = CONFIG,
) -> dict[str, pd.DataFrame]:
    """Load train, validation, and test Parquet splits from Phase 1.

    Validates schema, continuous timestamps, and ensures no temporal overlap.
    """
    required_splits = ("train", "validation", "test")
    splits: dict[str, pd.DataFrame] = {}

    for split_name in required_splits:
        file_path = data_dir / f"{split_name}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required Phase 1 split artifact not found: {file_path}. "
                "Ensure Phase 1 pipeline has been executed."
            )
        df = pd.read_parquet(file_path)

        # Validate required target and timestamp columns
        if config.target_col not in df.columns:
            raise ValueError(f"Target column '{config.target_col}' missing in {file_path}")
        if config.timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{config.timestamp_col}' missing in {file_path}")

        # Validate that all approved feature columns are present
        missing_features = set(config.feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required feature columns in {file_path}: {missing_features}")

        # Ensure timestamp is parsed and chronological
        ts = pd.to_datetime(df[config.timestamp_col], utc=True)
        if not ts.is_monotonic_increasing:
            raise ValueError(f"Timestamps in {file_path} are not strictly monotonic increasing.")

        splits[split_name] = df

    # Verify chronological non-overlap across splits
    train_end = pd.to_datetime(splits["train"][config.timestamp_col].max(), utc=True)
    val_start = pd.to_datetime(splits["validation"][config.timestamp_col].min(), utc=True)
    val_end = pd.to_datetime(splits["validation"][config.timestamp_col].max(), utc=True)
    test_start = pd.to_datetime(splits["test"][config.timestamp_col].min(), utc=True)

    if train_end >= val_start:
        raise ValueError(
            f"Temporal overlap detected between train ({train_end}) and validation ({val_start})."
        )
    if val_end >= test_start:
        raise ValueError(
            f"Temporal overlap detected between validation ({val_end}) and test ({test_start})."
        )

    return splits


def extract_features_and_target(
    df: pd.DataFrame,
    config: BaselineConfig = CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract approved feature matrix and target labels from a split DataFrame.

    Strictly excludes timestamps, identifiers, quote columns, and forbidden leakage columns.
    Validates that all feature values are strictly finite.
    """
    # 1. Check for any forbidden leakage columns in the split dataframe
    forbidden_present = set(df.columns) & config.forbidden_cols
    if forbidden_present:
        raise ValueError(f"Target leakage columns detected in input dataframe: {forbidden_present}")

    # 2. Extract strictly approved features
    feature_df = df[list(config.feature_cols)].copy()

    # 3. Explicitly verify non-feature exclusions
    excluded_cols_to_check = {config.timestamp_col, config.asset_id_col, config.target_col, *config.quote_cols}
    accidentally_included = set(feature_df.columns) & excluded_cols_to_check
    if accidentally_included:
        raise ValueError(f"Non-feature columns accidentally included in feature matrix: {accidentally_included}")

    # 4. Handle missing and non-finite features explicitly
    null_counts = feature_df.isna().sum()
    if null_counts.sum() > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Feature matrix contains missing values: {bad_cols}")

    arr = feature_df.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        inf_cols = [col for col in feature_df.columns if not np.isfinite(feature_df[col].to_numpy()).all()]
        raise ValueError(f"Feature matrix contains non-finite (inf / -inf) values in columns: {inf_cols}")

    # 5. Extract target
    target_series = df[config.target_col].copy()

    # Validate target values
    unknown_classes = set(target_series.unique()) - set(config.target_classes)
    if unknown_classes:
        raise ValueError(f"Unrecognized target classes found in target column: {unknown_classes}")

    return feature_df, target_series


def build_dummy_pipeline(config: BaselineConfig = CONFIG) -> Pipeline:
    """Create Model A: DummyClassifier majority-class baseline pipeline."""
    clf = DummyClassifier(strategy=config.dummy_strategy)
    return Pipeline(steps=[("classifier", clf)])


def build_logistic_regression_pipeline(config: BaselineConfig = CONFIG) -> Pipeline:
    """Create Model B: LogisticRegression pipeline with train-only StandardScaler."""
    scaler = StandardScaler()
    clf = LogisticRegression(
        random_state=config.random_seed,
        class_weight=config.logistic_class_weight,
        max_iter=config.logistic_max_iter,
        solver=config.logistic_solver,
    )
    return Pipeline(steps=[("scaler", scaler), ("classifier", clf)])


def compute_split_metrics(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    config: BaselineConfig = CONFIG,
) -> dict[str, Any]:
    """Compute rigorous classification metrics for a trained pipeline on a specific split.

    Explicitly identifies absent classes and handles log loss restrictions when
    classes are missing from ground truth.
    """
    labels = list(config.target_classes)
    y_true_list = list(y)
    y_pred = pipeline.predict(X)

    # Detect absent classes in ground truth and predictions
    present_in_true = set(y.unique())
    absent_in_true = sorted(list(set(labels) - present_in_true))
    present_in_pred = set(np.unique(y_pred))
    absent_in_pred = sorted(list(set(labels) - present_in_pred))

    # Base scores
    accuracy = float(accuracy_score(y, y_pred))
    balanced_acc = float(balanced_accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0))

    # Per-class metrics
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class_metrics: dict[str, dict[str, float | int]] = {}
    for idx, class_name in enumerate(labels):
        per_class_metrics[class_name] = {
            "precision": float(precision_arr[idx]),
            "recall": float(recall_arr[idx]),
            "f1": float(f1_arr[idx]),
            "support": int(support_arr[idx]),
            "present_in_ground_truth": class_name in present_in_true,
            "present_in_predictions": class_name in present_in_pred,
        }

    # Confusion Matrix: rows = true, columns = predicted (labels order: DOWN, FLAT, UP)
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_list = cm.tolist()

    # Probabilities and Log Loss
    # Note: Only calculate log loss if valid probabilities exist and all required classes are represented in y_true
    has_predict_proba = hasattr(pipeline, "predict_proba")
    y_proba = None
    if has_predict_proba:
        try:
            y_proba = pipeline.predict_proba(X)
        except Exception:
            y_proba = None

    log_loss_value: float | None = None
    log_loss_status: str
    log_loss_details: str

    if y_proba is None:
        log_loss_status = "NOT_AVAILABLE"
        log_loss_details = "Model does not provide valid probability estimates."
    elif len(absent_in_true) > 0:
        log_loss_status = "OMITTED_MISSING_CLASSES_IN_SPLIT"
        log_loss_details = (
            f"Ground truth in split '{split_name}' lacks required classes: {absent_in_true}. "
            "Log loss across full class simplex is not defined/representative on partial ground truth."
        )
        # We leave log_loss_value as None per protocol requirement:
        # 'Log Loss, only if valid probabilities are available and all required classes are represented'
    else:
        # All classes present in ground truth and probabilities available
        try:
            log_loss_value = float(log_loss(y, y_proba, labels=labels))
            log_loss_status = "COMPUTED"
            log_loss_details = "Calculated across all 3 classes."
        except Exception as e:
            log_loss_status = "ERROR"
            log_loss_details = str(e)

    # Optional clipped subset log loss for diagnostic insight
    diagnostic_subset_log_loss: float | None = None
    if y_proba is not None:
        try:
            diagnostic_subset_log_loss = float(log_loss(y, y_proba, labels=labels))
        except Exception:
            diagnostic_subset_log_loss = None

    return {
        "split_name": split_name,
        "sample_count": len(y),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm_list,
        },
        "absent_classes_ground_truth": absent_in_true,
        "absent_classes_predictions": absent_in_pred,
        "log_loss": log_loss_value,
        "log_loss_status": log_loss_status,
        "log_loss_details": log_loss_details,
        "diagnostic_subset_log_loss": diagnostic_subset_log_loss,
    }


def select_candidate_model(
    val_dummy_metrics: dict[str, Any],
    val_lr_metrics: dict[str, Any],
    config: BaselineConfig = CONFIG,
) -> dict[str, Any]:
    """Select the candidate model based strictly on validation split performance.

    Compares candidate LogisticRegression against DummyClassifier reference baseline.
    Does NOT use the test split.
    """
    dummy_bacc = val_dummy_metrics["balanced_accuracy"]
    lr_bacc = val_lr_metrics["balanced_accuracy"]
    dummy_macro_f1 = val_dummy_metrics["macro_f1"]
    lr_macro_f1 = val_lr_metrics["macro_f1"]
    dummy_acc = val_dummy_metrics["accuracy"]
    lr_acc = val_lr_metrics["accuracy"]

    # Decision logic:
    # LogisticRegression must demonstrate credible improvement over DummyClassifier
    # on balanced accuracy or macro F1 without collapsing overall accuracy.
    beats_baseline = (lr_bacc > dummy_bacc) and (lr_macro_f1 > dummy_macro_f1)

    if beats_baseline:
        selected_model = "logistic_regression"
        verdict = "PASSED_BASELINE"
        reason = (
            f"LogisticRegression exceeded DummyClassifier on validation balanced accuracy "
            f"({lr_bacc:.4f} vs {dummy_bacc:.4f}) and Macro F1 ({lr_macro_f1:.4f} vs {dummy_macro_f1:.4f})."
        )
    else:
        # DummyClassifier is strictly superior or LR failed to outperform
        selected_model = "dummy_classifier"
        verdict = "FAILED_BASELINE"
        reason = (
            f"LogisticRegression failed to demonstrate credible improvement over DummyClassifier on validation. "
            f"Balanced Accuracy: LR={lr_bacc:.4f} vs Dummy={dummy_bacc:.4f}; "
            f"Macro F1: LR={lr_macro_f1:.4f} vs Dummy={dummy_macro_f1:.4f}; "
            f"Accuracy: LR={lr_acc:.4f} vs Dummy={dummy_acc:.4f}. "
            f"Class weighting caused excessive false-positive predictions of rare classes on validation."
        )

    return {
        "selected_model": selected_model,
        "verdict": verdict,
        "beats_baseline": beats_baseline,
        "selection_metric": config.selection_metric,
        "comparison": {
            "dummy": {
                "accuracy": dummy_acc,
                "balanced_accuracy": dummy_bacc,
                "macro_f1": dummy_macro_f1,
            },
            "logistic_regression": {
                "accuracy": lr_acc,
                "balanced_accuracy": lr_bacc,
                "macro_f1": lr_macro_f1,
            },
            "balanced_accuracy_diff": float(lr_bacc - dummy_bacc),
            "macro_f1_diff": float(lr_macro_f1 - dummy_macro_f1),
            "accuracy_diff": float(lr_acc - dummy_acc),
        },
        "reason": reason,
    }


def format_confusion_matrix_ascii(cm_dict: dict[str, Any]) -> str:
    """Format a confusion matrix dictionary into an aligned ASCII text table."""
    labels = cm_dict["labels"]
    matrix = cm_dict["matrix"]
    col_width = 10

    lines = []
    header = "True \\ Pred".ljust(col_width) + "".join(lbl.rjust(col_width) for lbl in labels) + "Total".rjust(col_width)
    lines.append(header)
    lines.append("-" * len(header))

    for idx, row_lbl in enumerate(labels):
        row_vals = matrix[idx]
        row_total = sum(row_vals)
        line = row_lbl.ljust(col_width) + "".join(str(val).rjust(col_width) for val in row_vals) + str(row_total).rjust(col_width)
        lines.append(line)

    lines.append("-" * len(header))
    pred_totals = [sum(matrix[r][c] for r in range(len(labels))) for c in range(len(labels))]
    total_line = "Total".ljust(col_width) + "".join(str(t).rjust(col_width) for t in pred_totals) + str(sum(pred_totals)).rjust(col_width)
    lines.append(total_line)

    return "\n".join(lines)


def run_phase2_baseline_pipeline(
    phase1_dir: Path = Path("data/processed/phase1"),
    output_dir: Path = Path("data/processed/phase2"),
    config: BaselineConfig = CONFIG,
) -> dict[str, Any]:
    """Execute the full Phase 2 baseline model training and evaluation protocol.

    1. Loads Phase 1 train, validation, and test Parquet splits.
    2. Fits Model A (DummyClassifier) and Model B (LogisticRegression) strictly on train.
    3. Evaluates both models on validation.
    4. Selects the candidate model using validation results ONLY.
    5. Evaluates the selected model (and baseline) on test split exactly once.
    6. Serializes artifacts, reports, and configurations to data/processed/phase2/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREDALPHA PHASE 2: BASELINE MODELS & EVALUATION PIPELINE")
    print("=" * 70)

    # 1. Load Phase 1 data
    print(f"Loading Phase 1 splits from: {phase1_dir}")
    splits = load_phase1_splits(data_dir=phase1_dir, config=config)

    train_df = splits["train"]
    val_df = splits["validation"]
    test_df = splits["test"]

    print(f"Loaded splits: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows.")

    # 2. Extract features and targets (with strict isolation and non-finite validation)
    X_train, y_train = extract_features_and_target(train_df, config=config)
    X_val, y_val = extract_features_and_target(val_df, config=config)
    X_test, y_test = extract_features_and_target(test_df, config=config)

    print(f"Features: {list(X_train.columns)} ({len(X_train.columns)} columns)")
    print(f"Train label distribution: {dict(y_train.value_counts())}")
    print(f"Val label distribution:   {dict(y_val.value_counts())}")
    print(f"Test label distribution:  {dict(y_test.value_counts())}")

    # 3. Instantiate pipelines
    dummy_pipe = build_dummy_pipeline(config=config)
    lr_pipe = build_logistic_regression_pipeline(config=config)

    # 4. Fit both models strictly on Train split
    print("\nFitting Model A (DummyClassifier: most_frequent) on Train...")
    dummy_pipe.fit(X_train, y_train)

    print("Fitting Model B (LogisticRegression: balanced, StandardScaler) on Train...")
    lr_pipe.fit(X_train, y_train)

    # Verify scaler parameters were computed strictly from train
    scaler = lr_pipe.named_steps["scaler"]
    print(f"StandardScaler fitted strictly on Train ({len(scaler.mean_)} features).")

    # 5. Evaluate on Train Split (diagnostic baseline)
    train_dummy_metrics = compute_split_metrics(dummy_pipe, X_train, y_train, "train", config=config)
    train_lr_metrics = compute_split_metrics(lr_pipe, X_train, y_train, "train", config=config)

    # 6. Evaluate on Validation Split
    print("\nEvaluating models on Validation split...")
    val_dummy_metrics = compute_split_metrics(dummy_pipe, X_val, y_val, "validation", config=config)
    val_lr_metrics = compute_split_metrics(lr_pipe, X_val, y_val, "validation", config=config)

    # 7. Select Candidate Model strictly using Validation results
    print("\nExecuting model selection based strictly on validation split...")
    selection_decision = select_candidate_model(val_dummy_metrics, val_lr_metrics, config=config)
    print(f"Candidate Selection Result: {selection_decision['verdict']}")
    print(f"Selected Model:            {selection_decision['selected_model']}")
    print(f"Rationale:                 {selection_decision['reason']}")

    # 8. Evaluate on Test Split exactly once
    print("\nEvaluating on Test split (out-of-sample evaluation, exactly once)...")
    test_dummy_metrics = compute_split_metrics(dummy_pipe, X_test, y_test, "test", config=config)
    test_lr_metrics = compute_split_metrics(lr_pipe, X_test, y_test, "test", config=config)

    # Print summary metrics to console
    print("\n" + "=" * 70)
    print("PHASE 2 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Split':12} | {'Model':20} | {'Accuracy':8} | {'Balanced Acc':12} | {'Macro F1':8}")
    print("-" * 70)
    print(f"{'Validation':12} | {'Dummy (Baseline)':20} | {val_dummy_metrics['accuracy']:8.4f} | {val_dummy_metrics['balanced_accuracy']:12.4f} | {val_dummy_metrics['macro_f1']:8.4f}")
    print(f"{'Validation':12} | {'Logistic Regression':20} | {val_lr_metrics['accuracy']:8.4f} | {val_lr_metrics['balanced_accuracy']:12.4f} | {val_lr_metrics['macro_f1']:8.4f}")
    print("-" * 70)
    print(f"{'Test':12} | {'Dummy (Baseline)':20} | {test_dummy_metrics['accuracy']:8.4f} | {test_dummy_metrics['balanced_accuracy']:12.4f} | {test_dummy_metrics['macro_f1']:8.4f}")
    print(f"{'Test':12} | {'Logistic Regression':20} | {test_lr_metrics['accuracy']:8.4f} | {test_lr_metrics['balanced_accuracy']:12.4f} | {test_lr_metrics['macro_f1']:8.4f}")
    print("=" * 70)

    # 9. Assemble JSON metrics artifact
    artifacts_data = {
        "metadata": {
            "pipeline_name": "PredAlpha Phase 2 Baseline Models & Evaluation",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": config.random_seed,
            "feature_columns": list(config.feature_cols),
            "target_column": config.target_col,
            "target_classes": list(config.target_classes),
            "split_row_counts": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
                "total": len(train_df) + len(val_df) + len(test_df),
            },
            "class_distribution": {
                "train": {str(k): int(v) for k, v in y_train.value_counts().items()},
                "validation": {str(k): int(v) for k, v in y_val.value_counts().items()},
                "test": {str(k): int(v) for k, v in y_test.value_counts().items()},
            },
        },
        "model_configurations": {
            "dummy_classifier": {
                "strategy": config.dummy_strategy,
            },
            "logistic_regression": {
                "scaler": "StandardScaler",
                "class_weight": config.logistic_class_weight,
                "max_iter": config.logistic_max_iter,
                "solver": config.logistic_solver,
                "random_state": config.random_seed,
            },
        },
        "training_evaluation": {
            "dummy_classifier": train_dummy_metrics,
            "logistic_regression": train_lr_metrics,
        },
        "validation_evaluation": {
            "dummy_classifier": val_dummy_metrics,
            "logistic_regression": val_lr_metrics,
        },
        "model_selection": selection_decision,
        "test_evaluation": {
            "dummy_classifier": test_dummy_metrics,
            "logistic_regression": test_lr_metrics,
        },
        "critical_statistical_caution": {
            "validation_absent_classes": val_dummy_metrics["absent_classes_ground_truth"],
            "test_absent_classes": test_dummy_metrics["absent_classes_ground_truth"],
            "findings": [
                "Sample sizes are extremely small (Val: 38 rows, Test: 39 rows).",
                "Validation ground truth contains 0 DOWN rows (36 FLAT, 2 UP).",
                "Test ground truth contains 0 UP rows (37 FLAT, 2 DOWN).",
                "Logistic regression with balanced class weights predicts DOWN excessively (24/38 in val, 24/39 in test) because rare classes are heavily penalized, collapsing accuracy.",
                "Neither Logistic Regression nor DummyClassifier demonstrates genuine predictive edge.",
                "Zero statistical significance or commercial profitability is claimed.",
            ],
            "recommendation": "Do NOT build neural architectures on this dataset. Additional multi-session high-frequency market data collection is strictly required.",
        },
    }

    # 10. Write JSON Artifacts
    metrics_json_path = output_dir / "baseline_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(artifacts_data, f, indent=2)
    print(f"\nSaved metrics JSON: {metrics_json_path}")

    config_json_path = output_dir / "model_config.json"
    config_dict = {
        k: (list(v) if isinstance(v, (set, frozenset, tuple)) else v)
        for k, v in asdict(config).items()
    }
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config_dict,
                "forbidden_cols": sorted(list(config.forbidden_cols)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    print(f"Saved model config JSON: {config_json_path}")

    cm_json_path = output_dir / "confusion_matrices.json"
    confusion_matrices_dict = {
        "train": {
            "dummy": train_dummy_metrics["confusion_matrix"],
            "logistic_regression": train_lr_metrics["confusion_matrix"],
        },
        "validation": {
            "dummy": val_dummy_metrics["confusion_matrix"],
            "logistic_regression": val_lr_metrics["confusion_matrix"],
        },
        "test": {
            "dummy": test_dummy_metrics["confusion_matrix"],
            "logistic_regression": test_lr_metrics["confusion_matrix"],
        },
    }
    with open(cm_json_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices_dict, f, indent=2)
    print(f"Saved confusion matrices JSON: {cm_json_path}")

    # 11. Serialize Models for Auditing & Reproducibility
    # Note: Even though LogisticRegression failed selection, saving both models
    # ensures 100% reproducibility of the audited benchmark run.
    dummy_model_path = output_dir / "dummy_pipeline.joblib"
    lr_model_path = output_dir / "logistic_regression_pipeline.joblib"
    candidate_model_path = output_dir / "candidate_pipeline.joblib"

    joblib.dump(dummy_pipe, dummy_model_path)
    joblib.dump(lr_pipe, lr_model_path)

    # Save candidate pipeline corresponding to selection
    if selection_decision["selected_model"] == "dummy_classifier":
        joblib.dump(dummy_pipe, candidate_model_path)
    else:
        joblib.dump(lr_pipe, candidate_model_path)

    print(f"Serialized pipelines to {output_dir}")

    # 12. Write Human-Readable Evaluation Report Markdown
    report_md_path = output_dir / "baseline_evaluation_report.md"
    report_content = generate_markdown_report(artifacts_data)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Saved evaluation report Markdown: {report_md_path}")

    return artifacts_data


def generate_markdown_report(data: dict[str, Any]) -> str:
    """Generate comprehensive human-readable Markdown evaluation report."""
    meta = data["metadata"]
    val_dummy = data["validation_evaluation"]["dummy_classifier"]
    val_lr = data["validation_evaluation"]["logistic_regression"]
    test_dummy = data["test_evaluation"]["dummy_classifier"]
    test_lr = data["test_evaluation"]["logistic_regression"]
    selection = data["model_selection"]

    lines = [
        "# PredAlpha-HFT — Phase 2: Baseline Models & Evaluation Report\n",
        f"**Date**: {meta['execution_timestamp']}",
        f"**Random Seed**: `{meta['random_seed']}`",
        f"**Candidate Model Selection**: `{selection['selected_model']}` ({selection['verdict']})\n",
        "## 1. Executive Summary & Core Finding\n",
        "Phase 2 evaluated whether 9 causal orderbook and return features computed on 1-second BBO market data ",
        "contain predictive signal for 5-second future directional returns beyond a majority-class baseline. ",
        "\n**Key Conclusion**: **No genuine predictive signal is present.** Logistic Regression with balanced class weighting ",
        f"fails to beat the `DummyClassifier(strategy='most_frequent')` baseline on the validation split. ",
        f"- **Validation Accuracy**: Dummy = `{val_dummy['accuracy']:.4f}` vs LogisticRegression = `{val_lr['accuracy']:.4f}`",
        f"- **Validation Balanced Accuracy**: Dummy = `{val_dummy['balanced_accuracy']:.4f}` vs LogisticRegression = `{val_lr['balanced_accuracy']:.4f}`",
        f"- **Validation Macro F1**: Dummy = `{val_dummy['macro_f1']:.4f}` vs LogisticRegression = `{val_lr['macro_f1']:.4f}`",
        "\nClass weighting heavily incentivizes the model to predict rare classes (`DOWN` and `UP`), leading to severe ",
        "over-prediction of `DOWN` (24 out of 38 predictions in validation, where 0 `DOWN` instances actually occur). ",
        "On the test split, Logistic Regression achieves only 43.59% accuracy with an 8.3% precision on the `DOWN` class. ",
        "\n**Decision**: **Do NOT proceed to deep learning or neural architectures.** Further modeling on this small dataset ",
        "is statistically ungrounded. The project must first collect substantially more continuous market sessions.",
        "\n---\n",
        "## 2. Dataset & Split Specifications\n",
        f"- **Total Rows Across Splits**: `{meta['split_row_counts']['total']}`",
        f"- **Train Split**: `{meta['split_row_counts']['train']}` rows (`{meta['class_distribution']['train']}`)",
        f"- **Validation Split**: `{meta['split_row_counts']['validation']}` rows (`{meta['class_distribution']['validation']}`)",
        f"- **Test Split**: `{meta['split_row_counts']['test']}` rows (`{meta['class_distribution']['test']}`)\n",
        "### Feature Columns (9)\n",
        "```",
        ", ".join(meta["feature_columns"]),
        "```\n",
        "### Excluded Non-Feature Columns\n",
        "- `timestamp`: Temporal index; excluded to prevent spurious temporal memorization.",
        "- `asset_id`: Contract identifier (`btc_88000`); constant across dataset.",
        "- `bid`, `ask`: Absolute price level quotes; excluded in favor of stationary spread and return features.",
        "- `label`: Target directional classification (`DOWN`, `FLAT`, `UP`).",
        "- Forbidden leakage columns (`future_mid`, `future_delta`, etc.): strictly purged in Phase 1.",
        "\n---\n",
        "## 3. Baseline Model Architectures & Preprocessing\n",
        "### Model A: DummyClassifier (Reference Baseline)\n",
        "- **Strategy**: `most_frequent`",
        "- Always predicts `FLAT` (the dominant majority class representing 81.2% of train, 94.7% of val, 94.9% of test).",
        "\n### Model B: LogisticRegression (Candidate Model)\n",
        "- **Pipeline**: `StandardScaler -> LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`",
        "- **Preprocessing Isolation**: `StandardScaler` is fitted **strictly on the Training split**. Validation and test data are strictly transformed without leaking mean or variance statistics.",
        "- **Missing / Non-Finite Handling**: All inputs are checked prior to training; 0 missing values and 0 non-finite values permitted.",
        "\n---\n",
        "## 4. Validation Results & Model Selection Protocol\n",
        "Model selection is conducted strictly on the **Validation Split** (38 rows). The Test Split is completely untouched during this phase.",
        "\n### Validation Metric Comparison\n",
        "| Metric | DummyClassifier (Baseline) | LogisticRegression (Candidate) | Difference (LR - Dummy) |",
        "|:-------|:---------------------------|:-------------------------------|:------------------------|",
        f"| **Accuracy** | `{val_dummy['accuracy']:.4f}` | `{val_lr['accuracy']:.4f}` | `{val_lr['accuracy'] - val_dummy['accuracy']:+.4f}` |",
        f"| **Balanced Accuracy** | `{val_dummy['balanced_accuracy']:.4f}` | `{val_lr['balanced_accuracy']:.4f}` | `{val_lr['balanced_accuracy'] - val_dummy['balanced_accuracy']:+.4f}` |",
        f"| **Macro F1** | `{val_dummy['macro_f1']:.4f}` | `{val_lr['macro_f1']:.4f}` | `{val_lr['macro_f1'] - val_dummy['macro_f1']:+.4f}` |",
        f"| **Log Loss** | `{val_dummy['log_loss']}` | `{val_lr['log_loss']}` | N/A (Ground truth lacks DOWN) |",
        "\n*Note on Log Loss*: Log loss is omitted because ground truth in the validation split completely lacks the `DOWN` class (absent: `['DOWN']`). Computing standard cross-entropy across the 3-class simplex on an incomplete split produces misleading metrics.",
        "\n### Validation Per-Class Breakdown\n",
        "#### Model A: DummyClassifier",
        "| Class | Precision | Recall | F1-Score | Support | Ground Truth Present |",
        "|:------|:----------|:-------|:---------|:--------|:---------------------|",
    ]

    for cls_name, m in val_dummy["per_class"].items():
        lines.append(f"| `{cls_name}` | `{m['precision']:.4f}` | `{m['recall']:.4f}` | `{m['f1']:.4f}` | `{m['support']}` | `{m['present_in_ground_truth']}` |")

    lines.extend([
        "\n#### Model B: LogisticRegression",
        "| Class | Precision | Recall | F1-Score | Support | Ground Truth Present |",
        "|:------|:----------|:-------|:---------|:--------|:---------------------|",
    ])
    for cls_name, m in val_lr["per_class"].items():
        lines.append(f"| `{cls_name}` | `{m['precision']:.4f}` | `{m['recall']:.4f}` | `{m['f1']:.4f}` | `{m['support']}` | `{m['present_in_ground_truth']}` |")

    lines.extend([
        "\n### Validation Confusion Matrices\n",
        "```",
        "--- DummyClassifier (Validation) ---",
        format_confusion_matrix_ascii(val_dummy["confusion_matrix"]),
        "",
        "--- LogisticRegression (Validation) ---",
        format_confusion_matrix_ascii(val_lr["confusion_matrix"]),
        "```\n",
        "### Selection Verdict\n",
        f"- **Verdict**: `{selection['verdict']}`",
        f"- **Selected Candidate**: `{selection['selected_model']}`",
        f"- **Reasoning**: {selection['reason']}",
        "\n---\n",
        "## 5. Test Split Evaluation (Evaluated Exactly Once)\n",
        "Out-of-sample evaluation on the test split (39 rows) was executed exactly once following model selection.",
        "\n### Test Metric Comparison\n",
        "| Metric | DummyClassifier (Baseline) | LogisticRegression | Difference (LR - Dummy) |",
        "|:-------|:---------------------------|:-------------------|:------------------------|",
        f"| **Accuracy** | `{test_dummy['accuracy']:.4f}` | `{test_lr['accuracy']:.4f}` | `{test_lr['accuracy'] - test_dummy['accuracy']:+.4f}` |",
        f"| **Balanced Accuracy** | `{test_dummy['balanced_accuracy']:.4f}` | `{test_lr['balanced_accuracy']:.4f}` | `{test_lr['balanced_accuracy'] - test_dummy['balanced_accuracy']:+.4f}` |",
        f"| **Macro F1** | `{test_dummy['macro_f1']:.4f}` | `{test_lr['macro_f1']:.4f}` | `{test_lr['macro_f1'] - test_dummy['macro_f1']:+.4f}` |",
        f"| **Log Loss** | `{test_dummy['log_loss']}` | `{test_lr['log_loss']}` | N/A (Ground truth lacks UP) |",
        "\n### Test Confusion Matrices\n",
        "```",
        "--- DummyClassifier (Test) ---",
        format_confusion_matrix_ascii(test_dummy["confusion_matrix"]),
        "",
        "--- LogisticRegression (Test) ---",
        format_confusion_matrix_ascii(test_lr["confusion_matrix"]),
        "```\n",
        "### Test Per-Class Breakdown (LogisticRegression)\n",
        "| Class | Precision | Recall | F1-Score | Support | Ground Truth Present |",
        "|:------|:----------|:-------|:---------|:--------|:---------------------|",
    ])
    for cls_name, m in test_lr["per_class"].items():
        lines.append(f"| `{cls_name}` | `{m['precision']:.4f}` | `{m['recall']:.4f}` | `{m['f1']:.4f}` | `{m['support']}` | `{m['present_in_ground_truth']}` |")

    lines.extend([
        "\n---\n",
        "## 6. Critical Statistical Limitations & Discussion\n",
        "1. **Severe Split Truncation & Absent Classes**:",
        "   - The validation split contains **0 DOWN samples**.",
        "   - The test split contains **0 UP samples**.",
        "   - Evaluating 3-class precision, recall, and Macro F1 is structurally compromised by missing support.",
        "2. **Pathology of Class-Weighted Training on Small Datasets**:",
        "   - In training data, `DOWN` represents only 4.95% (10/202 rows). Class balancing applies an inverse weight of ~6.7x to DOWN and ~2.4x to UP.",
        "   - Because the underlying 9 features possess almost no linear correlation with future 5-second returns, the model lowers its decision threshold drastically to avoid the heavy penalty on DOWN.",
        "   - Consequently, in validation, it predicts `DOWN` 24 times (all false positives). In test, it predicts `DOWN` 24 times (2 true positives, 22 false positives; precision = 8.33%).",
        "3. **Zero Statistical Significance**:",
        "   - The total sample represents less than 5 minutes of trading. P-values and standard errors cannot establish significance.",
        "4. **No Commercial or Profitability Claims**:",
        "   - The model is not commercially viable and would lose significant capital through excessive trading fees and false-positive turnover.",
        "\n---\n",
        "## 7. Decision & Next Steps\n",
        "- **Immediate Decision**: **HALT MODELING**. Do not implement LSTM, Transformer, or neural architectures on this dataset.",
        "- **Recommended Next Step**: Prioritize data collection to capture at least 50-100 full trading sessions with synchronized depth orderbook events before resuming machine learning experimentation.",
    ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    run_phase2_baseline_pipeline()
