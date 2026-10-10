"""
Classification Metrics for Phase 19 Baseline Modeling.

Implements pure NumPy metrics for multi-class classification:
- Accuracy
- Balanced Accuracy (macro-averaged recall)
- Macro F1
- Weighted F1
- Per-class Precision, Recall, F1, and Support
- Confusion Matrix

All operations are deterministic, standalone, and safely handle zero-division edge cases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Compute confusion matrix where row i represents true class and column j represents predicted class.
    Shape: (num_classes, num_classes).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute precision, recall, F1, and support for each class.
    """
    if class_names is None:
        class_names = ["DOWN", "FLAT", "UP"]
    num_classes = len(class_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return metrics


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> float:
    """
    Compute balanced accuracy (unweighted macro-average of recall across all classes).
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    recalls = []
    for i in range(num_classes):
        denom = np.sum(cm[i, :])
        rec = float(cm[i, i] / denom) if denom > 0 else 0.0
        recalls.append(rec)
    return float(np.mean(recalls))


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> float:
    """Compute unweighted macro-averaged F1 score."""
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=[str(i) for i in range(num_classes)])
    f1s = [m["f1"] for m in per_class.values()]
    return float(np.mean(f1s))


def compute_weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3,
) -> float:
    """Compute support-weighted averaged F1 score."""
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=[str(i) for i in range(num_classes)])
    total_support = sum(m["support"] for m in per_class.values())
    if total_support == 0:
        return 0.0
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
    return float(weighted_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation dictionary for model predictions.
    """
    if class_names is None:
        class_names = ["DOWN", "FLAT", "UP"]
    num_classes = len(class_names)

    acc = compute_accuracy(y_true, y_pred)
    bal_acc = compute_balanced_accuracy(y_true, y_pred, num_classes=num_classes)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class = compute_per_class_metrics(y_true, y_pred, class_names=class_names)
    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))

    total_supp = sum(m["support"] for m in per_class.values())
    weighted_f1 = (
        float(sum(m["f1"] * m["support"] for m in per_class.values()) / total_supp)
        if total_supp > 0
        else 0.0
    )

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
    }
