"""Reusable data quality, integrity, and leakage validation functions for PredAlpha."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "ERROR"  # "ERROR" or "WARNING"


@dataclass
class ValidationReport:
    timestamp: str
    dataset_name: str
    total_checks: int
    passed_count: int
    failed_count: int
    warning_count: int
    is_valid: bool
    results: list[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "total_checks": self.total_checks,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "warning_count": self.warning_count,
            "is_valid": self.is_valid,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


def check_missing_values(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    allow_missing: bool = False,
) -> ValidationResult:
    """Check for missing (NaN/None/pd.NA) values in specified columns."""
    cols = list(columns) if columns is not None else list(df.columns)
    missing_counts = df[cols].isna().sum().to_dict()
    total_missing = sum(missing_counts.values())

    cols_with_missing = {k: int(v) for k, v in missing_counts.items() if v > 0}
    passed = total_missing == 0 or allow_missing
    msg = (
        f"No missing values found across {len(cols)} columns."
        if total_missing == 0
        else f"Found {total_missing} missing values in columns: {cols_with_missing}."
    )
    return ValidationResult(
        check_name="missing_values",
        passed=passed,
        message=msg,
        details={"total_missing": total_missing, "by_column": cols_with_missing},
        severity="ERROR" if not allow_missing else "WARNING",
    )


def check_duplicate_records(
    df: pd.DataFrame,
    subset: Iterable[str] | None = None,
) -> ValidationResult:
    """Check for duplicate rows across specified columns or all columns."""
    cols = list(subset) if subset is not None else list(df.columns)
    # Handle unhashable types if any
    try:
        dupes = int(df.duplicated(subset=cols).sum())
    except TypeError:
        dupes = int(df[cols].astype(str).duplicated().sum())

    passed = dupes == 0
    msg = "No duplicate records found." if passed else f"Found {dupes} duplicate records."
    return ValidationResult(
        check_name="duplicate_records",
        passed=passed,
        message=msg,
        details={"duplicate_count": dupes, "subset_columns": cols},
        severity="ERROR",
    )


def check_duplicate_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> ValidationResult:
    """Check for duplicate timestamps."""
    if timestamp_col in df.columns:
        ts_series = df[timestamp_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_series = pd.Series(df.index, index=df.index)
    else:
        return ValidationResult(
            check_name="duplicate_timestamps",
            passed=False,
            message=f"Timestamp column '{timestamp_col}' not found and index is not DatetimeIndex.",
            details={},
            severity="ERROR",
        )

    dupes = int(ts_series.duplicated().sum())
    passed = dupes == 0
    msg = "No duplicate timestamps found." if passed else f"Found {dupes} duplicate timestamps."
    return ValidationResult(
        check_name="duplicate_timestamps",
        passed=passed,
        message=msg,
        details={"duplicate_count": dupes},
        severity="ERROR",
    )


def check_timestamp_integrity(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    expected_freq: str | None = "1s",
    max_gap_seconds: float | None = None,
) -> ValidationResult:
    """Verify timestamp parsing, ordering, uniqueness, and interval continuity."""
    if timestamp_col in df.columns:
        ts_series = df[timestamp_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_series = pd.Series(df.index, index=df.index)
    else:
        return ValidationResult(
            check_name="timestamp_integrity",
            passed=False,
            message=f"Timestamp column '{timestamp_col}' not found and index is not DatetimeIndex.",
            details={},
            severity="ERROR",
        )

    # Convert to UTC datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(ts_series):
        try:
            ts_series = pd.to_datetime(ts_series, utc=True)
        except Exception as e:
            return ValidationResult(
                check_name="timestamp_integrity",
                passed=False,
                message=f"Failed to parse timestamp column to datetime: {e}",
                details={"error": str(e)},
                severity="ERROR",
            )

    dupes = int(ts_series.duplicated().sum())
    monotonic = bool(ts_series.is_monotonic_increasing)

    details: dict[str, Any] = {
        "min_time": str(ts_series.min()),
        "max_time": str(ts_series.max()),
        "is_monotonic_increasing": monotonic,
        "duplicate_timestamps": dupes,
    }

    issues = []
    if not monotonic:
        issues.append("Timestamps are not strictly ascending.")
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate timestamps.")

    # Check gaps if expected_freq or max_gap_seconds is provided
    gaps_found = 0
    if len(ts_series) > 1 and (expected_freq or max_gap_seconds):
        diffs = ts_series.diff().dropna()
        diff_seconds = diffs.dt.total_seconds() if hasattr(diffs, "dt") else diffs / np.timedelta64(1, "s")
        max_observed_gap = float(diff_seconds.max())
        details["max_observed_gap_seconds"] = max_observed_gap

        threshold = max_gap_seconds or (pd.Timedelta(expected_freq).total_seconds() if expected_freq else 1.0)
        gaps = diff_seconds[diff_seconds > threshold]
        gaps_found = len(gaps)
        details["gaps_over_threshold"] = gaps_found

        if gaps_found > 0:
            issues.append(f"Found {gaps_found} timestamp gaps exceeding {threshold}s (max gap: {max_observed_gap}s).")

    passed = len(issues) == 0
    msg = "Timestamps are ordered, unique, and continuous." if passed else " | ".join(issues)
    return ValidationResult(
        check_name="timestamp_integrity",
        passed=passed,
        message=msg,
        details=details,
        severity="ERROR",
    )


def check_market_data_integrity(
    df: pd.DataFrame,
    bid_col: str = "bid",
    ask_col: str = "ask",
    mid_col: str = "mid_price",
    spread_col: str = "spread",
) -> ValidationResult:
    """Validate positive prices, spread non-negativity, and bid <= ask (no crossed books)."""
    issues = []
    details: dict[str, Any] = {}

    has_bid = bid_col in df.columns
    has_ask = ask_col in df.columns
    has_mid = mid_col in df.columns
    has_spread = spread_col in df.columns

    if not (has_bid and has_ask):
        # Check alternative column names: best_bid, best_ask
        if "best_bid" in df.columns and "best_ask" in df.columns:
            bid_col, ask_col = "best_bid", "best_ask"
            has_bid, has_ask = True, True

    if has_bid and has_ask:
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        ask = pd.to_numeric(df[ask_col], errors="coerce")

        non_positive_bids = int((bid <= 0).sum())
        non_positive_asks = int((ask <= 0).sum())
        crossed_books = int((bid > ask).sum())

        details["non_positive_bids"] = non_positive_bids
        details["non_positive_asks"] = non_positive_asks
        details["crossed_books"] = crossed_books

        if non_positive_bids > 0:
            issues.append(f"Found {non_positive_bids} non-positive {bid_col} values.")
        if non_positive_asks > 0:
            issues.append(f"Found {non_positive_asks} non-positive {ask_col} values.")
        if crossed_books > 0:
            issues.append(f"Found {crossed_books} crossed book occurrences ({bid_col} > {ask_col}).")

        # Spread check
        if has_spread:
            spread = pd.to_numeric(df[spread_col], errors="coerce")
            negative_spreads = int((spread < 0).sum())
            details["negative_spreads"] = negative_spreads
            if negative_spreads > 0:
                issues.append(f"Found {negative_spreads} negative spreads.")

        # Mid-price consistency check
        if has_mid:
            mid = pd.to_numeric(df[mid_col], errors="coerce")
            non_positive_mids = int((mid <= 0).sum())
            details["non_positive_mids"] = non_positive_mids
            if non_positive_mids > 0:
                issues.append(f"Found {non_positive_mids} non-positive {mid_col} values.")

            calc_mid = (bid + ask) / 2.0
            mid_diff = (mid - calc_mid).abs()
            mid_inconsistencies = int((mid_diff > 1e-6).sum())
            details["mid_price_inconsistencies"] = mid_inconsistencies
            if mid_inconsistencies > 0:
                issues.append(f"Found {mid_inconsistencies} mid_price values deviating from (bid+ask)/2.")
    else:
        issues.append(f"Missing required price columns: {bid_col}, {ask_col}.")

    passed = len(issues) == 0
    msg = "Market prices and orderbook integrity valid." if passed else " | ".join(issues)
    return ValidationResult(
        check_name="market_data_integrity",
        passed=passed,
        message=msg,
        details=details,
        severity="ERROR",
    )


def check_finite_values(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] | None = None,
) -> ValidationResult:
    """Verify that numeric columns do not contain inf or -inf."""
    if numeric_cols is not None:
        cols = [c for c in numeric_cols if c in df.columns]
    else:
        cols = list(df.select_dtypes(include=[np.number]).columns)

    inf_counts = {}
    for col in cols:
        s = df[col]
        infs = int(np.isinf(s).sum())
        if infs > 0:
            inf_counts[col] = infs

    passed = len(inf_counts) == 0
    msg = (
        f"All {len(cols)} numeric columns contain only finite values."
        if passed
        else f"Found non-finite values in columns: {inf_counts}."
    )
    return ValidationResult(
        check_name="finite_values",
        passed=passed,
        message=msg,
        details={"non_finite_columns": inf_counts},
        severity="ERROR",
    )


def check_constant_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    near_constant_threshold: float = 0.98,
) -> ValidationResult:
    """Identify constant features (nunique <= 1) and near-constant features."""
    cols = list(feature_cols) if feature_cols is not None else list(df.select_dtypes(include=[np.number]).columns)
    constant_cols = []
    near_constant_cols = {}

    n = len(df)
    for col in cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 1:
            constant_cols.append(col)
        elif n > 0:
            top_freq = float(df[col].value_counts(normalize=True, dropna=False).iloc[0])
            if top_freq >= near_constant_threshold:
                near_constant_cols[col] = top_freq

    details = {
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "threshold": near_constant_threshold,
    }

    if constant_cols:
        msg = f"Found constant columns: {constant_cols}."
        passed = False
        sev = "ERROR"
    elif near_constant_cols:
        msg = f"Found near-constant columns (>{near_constant_threshold*100:.0f}% same value): {list(near_constant_cols.keys())}."
        passed = True
        sev = "WARNING"
    else:
        msg = f"All {len(cols)} features exhibit sufficient variability."
        passed = True
        sev = "INFO"

    return ValidationResult(
        check_name="constant_features",
        passed=passed,
        message=msg,
        details=details,
        severity=sev,
    )


def check_label_distribution(
    df: pd.DataFrame,
    target_col: str = "label",
    min_classes: int = 2,
    min_samples_per_class: int = 1,
) -> ValidationResult:
    """Validate label column presence, class count, single-class detection, and distribution."""
    if target_col not in df.columns:
        return ValidationResult(
            check_name="label_distribution",
            passed=False,
            message=f"Target column '{target_col}' not found.",
            details={},
            severity="ERROR",
        )

    counts = df[target_col].value_counts(dropna=False).to_dict()
    pcts = (df[target_col].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
    class_count = len(counts)

    details = {
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_percentages": {str(k): float(v) for k, v in pcts.items()},
        "num_classes": class_count,
    }

    if class_count < min_classes:
        return ValidationResult(
            check_name="label_distribution",
            passed=False,
            message=f"Single-class or empty label set detected: only {class_count} class ({counts}). Must have at least {min_classes} classes.",
            details=details,
            severity="ERROR",
        )

    # Check for empty classes
    underrepresented = {str(k): int(v) for k, v in counts.items() if v < min_samples_per_class}
    if underrepresented:
        return ValidationResult(
            check_name="label_distribution",
            passed=False,
            message=f"Classes under min threshold ({min_samples_per_class}): {underrepresented}.",
            details=details,
            severity="ERROR",
        )

    return ValidationResult(
        check_name="label_distribution",
        passed=True,
        message=f"Label distribution valid with {class_count} classes: {details['class_counts']}.",
        details=details,
        severity="INFO",
    )


def check_target_leakage(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "label",
    forbidden_cols: Iterable[str] | None = None,
    forbidden_patterns: Iterable[str] | None = None,
) -> ValidationResult:
    """Verify that no future target or forbidden leakage columns exist in the feature set or feature frame."""
    default_forbidden = {
        "future_mid",
        "future_delta",
        "future_return",
        "future_price",
        "label_threshold",
        "target_close",
        "target_time",
        "target_cutoff",
        "price_change_5m",
        "price_difference",
    }
    if forbidden_cols:
        default_forbidden.update(forbidden_cols)

    default_patterns = [r"^future_.*", r"^target_.*", r"^next_.*"]
    if forbidden_patterns:
        default_patterns.extend(forbidden_patterns)

    feature_set = set(feature_cols)
    all_cols = set(df.columns)

    # 1. Target column must not be in feature set
    target_in_features = target_col in feature_set

    # 2. Forbidden column exact match in feature_cols
    forbidden_in_features = sorted(feature_set.intersection(default_forbidden))

    # 3. Forbidden pattern match in feature_cols
    pattern_matches_in_features = []
    for col in feature_cols:
        for pat in default_patterns:
            if re.match(pat, col, re.IGNORECASE):
                pattern_matches_in_features.append(col)
                break

    # 4. Check if forbidden columns exist in the dataframe itself (outside of explicit target)
    forbidden_in_df = sorted((all_cols - {target_col}).intersection(default_forbidden))

    details = {
        "target_in_features": target_in_features,
        "forbidden_in_features": forbidden_in_features,
        "pattern_matches_in_features": pattern_matches_in_features,
        "forbidden_in_df": forbidden_in_df,
    }

    issues = []
    if target_in_features:
        issues.append(f"Target column '{target_col}' is listed as a feature.")
    if forbidden_in_features:
        issues.append(f"Forbidden future columns in features: {forbidden_in_features}.")
    if pattern_matches_in_features:
        issues.append(f"Future pattern matched in features: {pattern_matches_in_features}.")
    if forbidden_in_df:
        issues.append(f"Forbidden leakage columns present in dataframe: {forbidden_in_df}.")

    passed = len(issues) == 0
    msg = "No target leakage or future information detected in features." if passed else " | ".join(issues)
    return ValidationResult(
        check_name="target_leakage",
        passed=passed,
        message=msg,
        details=details,
        severity="ERROR",
    )


def validate_chronological_splits(
    splits: dict[str, pd.DataFrame],
    timestamp_col: str = "timestamp",
    min_purge_gap_seconds: float = 0.0,
) -> ValidationResult:
    """Verify chronological split order, strict non-overlap, and purge gap enforcement."""
    split_order = ("train", "validation", "test")
    missing_splits = [s for s in split_order if s not in splits]
    if missing_splits:
        return ValidationResult(
            check_name="chronological_splits",
            passed=False,
            message=f"Missing required splits: {missing_splits}.",
            details={"missing_splits": missing_splits},
            severity="ERROR",
        )

    details: dict[str, Any] = {"splits": {}}
    previous_end = None
    previous_name = None
    issues = []

    for name in split_order:
        frame = splits[name]
        if frame.empty:
            issues.append(f"Split '{name}' is empty.")
            continue

        if timestamp_col in frame.columns:
            times = pd.to_datetime(frame[timestamp_col], utc=True)
        elif isinstance(frame.index, pd.DatetimeIndex):
            times = frame.index.tz_convert("UTC") if frame.index.tz else frame.index.tz_localize("UTC")
        else:
            issues.append(f"Split '{name}' has no '{timestamp_col}' column or DatetimeIndex.")
            continue

        if times.duplicated().any():
            issues.append(f"Split '{name}' contains duplicate timestamps.")
        if not times.is_monotonic_increasing:
            issues.append(f"Split '{name}' timestamps are not strictly monotonically increasing.")

        start_time = times.min()
        end_time = times.max()

        details["splits"][name] = {
            "rows": len(frame),
            "start": str(start_time),
            "end": str(end_time),
        }

        if previous_end is not None:
            if start_time <= previous_end:
                issues.append(
                    f"Temporal overlap: '{name}' start ({start_time}) <= '{previous_name}' end ({previous_end})."
                )
            else:
                gap_sec = (start_time - previous_end).total_seconds()
                details[f"{previous_name}_to_{name}_gap_seconds"] = gap_sec
                if gap_sec < min_purge_gap_seconds:
                    issues.append(
                        f"Purge gap between '{previous_name}' and '{name}' is {gap_sec}s, which is less than required {min_purge_gap_seconds}s."
                    )

        previous_end = end_time
        previous_name = name

    passed = len(issues) == 0
    msg = "Chronological splits are strictly ordered, non-overlapping, and respect purge gaps." if passed else " | ".join(issues)
    return ValidationResult(
        check_name="chronological_splits",
        passed=passed,
        message=msg,
        details=details,
        severity="ERROR",
    )


def validate_canonical_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "label",
    timestamp_col: str = "timestamp",
    expected_freq: str | None = "1s",
    bid_col: str = "bid",
    ask_col: str = "ask",
    mid_col: str = "mid_price",
    spread_col: str = "spread",
    dataset_name: str = "canonical_dataset",
) -> ValidationReport:
    """Run full validation suite on a prepared canonical dataset."""
    results: list[ValidationResult] = []

    # 1. Missing values
    required_cols = [timestamp_col, *feature_cols, target_col]
    present_cols = [c for c in required_cols if c in df.columns]
    results.append(check_missing_values(df, columns=present_cols))

    # 2. Duplicate records
    results.append(check_duplicate_records(df, subset=[timestamp_col] if timestamp_col in df.columns else None))

    # 3. Timestamp integrity
    results.append(check_timestamp_integrity(df, timestamp_col=timestamp_col, expected_freq=expected_freq))

    # 4. Market data integrity (if price columns present)
    if all(c in df.columns for c in [bid_col, ask_col, mid_col, spread_col]):
        results.append(check_market_data_integrity(
            df, bid_col=bid_col, ask_col=ask_col, mid_col=mid_col, spread_col=spread_col
        ))

    # 5. Finite values in features
    results.append(check_finite_values(df, numeric_cols=feature_cols))

    # 6. Constant / near-constant features
    results.append(check_constant_features(df, feature_cols=feature_cols))

    # 7. Label distribution
    results.append(check_label_distribution(df, target_col=target_col, min_classes=2, min_samples_per_class=1))

    # 8. Target leakage
    results.append(check_target_leakage(df, feature_cols=feature_cols, target_col=target_col))

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed and r.severity == "ERROR")
    warning_count = sum(1 for r in results if not r.passed and r.severity == "WARNING")
    is_valid = failed_count == 0

    return ValidationReport(
        timestamp=datetime.now().isoformat(),
        dataset_name=dataset_name,
        total_checks=total,
        passed_count=passed_count,
        failed_count=failed_count,
        warning_count=warning_count,
        is_valid=is_valid,
        results=results,
    )
