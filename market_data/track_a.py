"""Leakage-safe utilities for the small Oct. 3 BTC direction research set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TrackAConfig:
    timestamp_column: str = "timestamp"
    target_column: str = "label_5m"
    feature_columns: tuple[str, ...] = (
        "mid_price_up", "spread_up", "spread_down", "mid_price_change",
    )
    forbidden_columns: frozenset[str] = frozenset({
        "target_close", "target_time", "price_change_5m", "future_mid",
        "target_cutoff", "current_close", "current_close_time",
    })


CONFIG = TrackAConfig()


def validate_feature_frame(frame: pd.DataFrame, config: TrackAConfig = CONFIG) -> None:
    """Raise if selected features are missing, unsafe, or not prediction-time clean."""
    selected = set(config.feature_columns)
    missing = selected - set(frame.columns)
    unsafe = selected & config.forbidden_columns
    if missing:
        raise ValueError(f"Missing configured features: {sorted(missing)}")
    if unsafe:
        raise ValueError(f"Forbidden future-derived features selected: {sorted(unsafe)}")
    if frame.loc[:, config.feature_columns].isna().any().any():
        raise ValueError("Configured feature frame contains missing values")
    if config.target_column not in frame.columns:
        raise ValueError(f"Missing target column: {config.target_column}")


def validate_chronological_splits(splits: dict[str, pd.DataFrame], config: TrackAConfig = CONFIG) -> None:
    """Require sorted timestamps and strict non-overlap in train/validation/test order."""
    previous_end = None
    for name in ("train", "validation", "test"):
        frame = splits[name]
        times = pd.to_datetime(frame[config.timestamp_column], utc=True)
        if times.duplicated().any() or not times.is_monotonic_increasing:
            raise ValueError(f"{name} timestamps are not unique and ascending")
        if previous_end is not None and times.min() <= previous_end:
            raise ValueError(f"{name} overlaps or precedes the prior split")
        previous_end = times.max()


def evaluate_classifier(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, object]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_distribution": pd.Series(y_true).value_counts().sort_index().to_dict(),
    }


def run_baselines(splits: dict[str, pd.DataFrame], config: TrackAConfig = CONFIG) -> dict[str, dict[str, dict[str, object]]]:
    """Fit once on chronological train data and score validation and untouched test."""
    validate_chronological_splits(splits, config)
    for frame in splits.values():
        validate_feature_frame(frame, config)
    train = splits["train"]
    x_train, y_train = train.loc[:, config.feature_columns], train[config.target_column]
    models = {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=7, class_weight="balanced")),
        ]),
    }
    results: dict[str, dict[str, dict[str, object]]] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        results[name] = {}
        for split_name in ("validation", "test"):
            frame = splits[split_name]
            results[name][split_name] = evaluate_classifier(
                frame[config.target_column], model.predict(frame.loc[:, config.feature_columns])
            )
    return results
