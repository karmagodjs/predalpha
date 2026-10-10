"""
Pipeline V2 Models Package.
Contains baseline models, deep neural sequence architectures, and evaluation utilities.
"""

from pipeline_v2.models.baseline_models import (
    LogisticRegressionModel,
    MajorityBaseline,
    SmallMLP,
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

__all__ = [
    "MajorityBaseline",
    "LogisticRegressionModel",
    "SmallMLP",
    "set_seed",
    "compute_accuracy",
    "compute_balanced_accuracy",
    "compute_confusion_matrix",
    "compute_macro_f1",
    "compute_per_class_metrics",
    "compute_weighted_f1",
    "evaluate_predictions",
]
