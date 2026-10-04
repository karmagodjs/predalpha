"""Reproducible Canonical Dataset Preparation Pipeline for PredAlpha Phase 1.

Prepares a clean, verified, leakage-free canonical dataset from verified raw/BBO sources,
generates deterministic chronological splits with purge gaps, and outputs complete audit reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_data.validation import (
    ValidationReport,
    validate_canonical_dataset,
    validate_chronological_splits,
)


@dataclass(frozen=True)
class CanonicalSchemaConfig:
    timestamp_col: str = "timestamp"
    asset_id_col: str = "asset_id"
    asset_id_value: str = "btc_88000"
    market_symbol: str = "BTC-88000-BBO"
    target_col: str = "label"
    classes: tuple[str, ...] = ("DOWN", "FLAT", "UP")
    horizon_seconds: int = 5
    min_tick: float = 0.001
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
    quote_cols: tuple[str, ...] = ("bid", "ask", "mid_price", "spread")
    forbidden_cols: frozenset[str] = frozenset({
        "future_mid",
        "future_delta",
        "future_return",
        "label_threshold",
        "target_close",
        "target_time",
        "price_change_5m",
    })
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_gap_rows: int = 5  # 5 rows = 5s for 1Hz sampling, matches horizon


CONFIG = CanonicalSchemaConfig()


def build_canonical_dataset(
    bbo_df: pd.DataFrame,
    config: CanonicalSchemaConfig = CONFIG,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Build canonical features and labels from raw 1-second BBO market data.

    All features are strictly backward-looking (causal).
    Target labels are computed at horizon_seconds and then all intermediate
    future-looking columns are purged.
    """
    df = bbo_df.copy()

    # Ensure UTC Datetime timestamp
    if config.timestamp_col in df.columns:
        df[config.timestamp_col] = pd.to_datetime(df[config.timestamp_col], utc=True)
        df = df.sort_values(config.timestamp_col).reset_index(drop=True)
        df.index = pd.DatetimeIndex(df[config.timestamp_col], name=config.timestamp_col)
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df[config.timestamp_col] = df.index
    else:
        raise ValueError("Input dataframe must have a 'timestamp' column or DatetimeIndex.")

    # Validate required BBO quote columns
    required_quotes = {"bid", "ask", "mid_price", "spread"}
    missing_quotes = required_quotes - set(df.columns)
    if missing_quotes:
        raise ValueError(f"Missing required BBO columns: {missing_quotes}")

    # Check 1-second continuity
    expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1s", tz="UTC")
    missing_seconds = len(expected_index.difference(df.index))
    if missing_seconds > 0:
        raise ValueError(f"Timestamps are not continuous 1-second intervals: {missing_seconds} missing seconds.")

    # 1. Target generation (future horizon)
    future_mid = df["mid_price"].shift(-config.horizon_seconds)
    future_delta = future_mid - df["mid_price"]
    label_threshold = (df["spread"] / 2.0).clip(lower=config.min_tick)

    # Classify directional movements
    label = pd.Series("FLAT", index=df.index)
    label.loc[future_delta > label_threshold] = "UP"
    label.loc[future_delta < -label_threshold] = "DOWN"

    # 2. Causal Feature Engineering (past information only)
    mid = df["mid_price"]
    spread = df["spread"]
    bid = df["bid"]
    ask = df["ask"]

    mid_return_1s = mid.pct_change(1)
    mid_return_3s = mid.pct_change(3)
    mid_return_5s = mid.pct_change(5)
    spread_bps = (spread / mid) * 10_000.0
    mid_volatility_5s = mid_return_1s.rolling(5).std()
    bid_change_1s = bid.diff(1)
    ask_change_1s = ask.diff(1)

    # 3. Assemble canonical frame
    canonical = pd.DataFrame(
        {
            config.timestamp_col: df[config.timestamp_col],
            config.asset_id_col: config.asset_id_value,
            # Quote features
            "bid": bid,
            "ask": ask,
            "mid_price": mid,
            "spread": spread,
            "spread_bps": spread_bps,
            # Causal return / momentum / volatility features
            "mid_return_1s": mid_return_1s,
            "mid_return_3s": mid_return_3s,
            "mid_return_5s": mid_return_5s,
            "mid_volatility_5s": mid_volatility_5s,
            "bid_change_1s": bid_change_1s,
            "ask_change_1s": ask_change_1s,
            # Target
            config.target_col: label,
        },
        index=df.index,
    )

    # 4. Drop boundary rows where causal features (first 5 rows) or target labels (last 5 rows) are undefined
    valid_mask = (
        future_mid.notna()
        & mid_return_5s.notna()
        & mid_volatility_5s.notna()
        & bid_change_1s.notna()
        & ask_change_1s.notna()
    )
    canonical = canonical.loc[valid_mask].copy()

    # Verify no leakage columns exist
    leakage_present = set(canonical.columns) & config.forbidden_cols
    if leakage_present:
        raise RuntimeError(f"Target leakage detected in canonical dataset: {leakage_present}")

    # Run comprehensive data validation
    report = validate_canonical_dataset(
        canonical,
        feature_cols=config.feature_cols,
        target_col=config.target_col,
        timestamp_col=config.timestamp_col,
        expected_freq="1s",
        bid_col="bid",
        ask_col="ask",
        mid_col="mid_price",
        spread_col="spread",
        dataset_name="Canonical PredAlpha Dataset (Phase 1)",
    )

    return canonical, report


def split_canonical_dataset(
    df: pd.DataFrame,
    config: CanonicalSchemaConfig = CONFIG,
) -> tuple[dict[str, pd.DataFrame], Any]:
    """Perform deterministic chronological train/validation/test split with purge gaps."""
    n = len(df)
    train_end = int(n * config.train_ratio)
    val_end = int(n * (config.train_ratio + config.val_ratio))
    purge = config.purge_gap_rows

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end + purge : val_end].copy()
    test = df.iloc[val_end + purge :].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError(
            f"Split produced empty dataset (train={len(train)}, val={len(val)}, test={len(test)}). Check dataset size or purge gap."
        )

    splits = {"train": train, "validation": val, "test": test}

    split_validation = validate_chronological_splits(
        splits,
        timestamp_col=config.timestamp_col,
        min_purge_gap_seconds=float(purge),
    )

    if not split_validation.passed:
        raise ValueError(f"Split validation failed: {split_validation.message}")

    return splits, split_validation


def run_phase1_pipeline(
    input_bbo_path: Path = Path("data/processed/btc_88000_bbo_1s.parquet"),
    output_dir: Path = Path("data/processed/phase1"),
    config: CanonicalSchemaConfig = CONFIG,
) -> dict[str, Any]:
    """Execute the full Phase 1 dataset preparation, validation, and serialization pipeline."""
    if not input_bbo_path.exists():
        raise FileNotFoundError(f"Source BBO dataset not found: {input_bbo_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PREDALPHA PHASE 1: CANONICAL DATASET PREPARATION PIPELINE")
    print("=" * 70)
    print(f"Loading source BBO: {input_bbo_path}")

    bbo_df = pd.read_parquet(input_bbo_path)
    print(f"Loaded source BBO rows: {len(bbo_df)}")

    # 1. Build Canonical Dataset
    canonical_df, val_report = build_canonical_dataset(bbo_df, config)
    print(f"Canonical dataset generated: {len(canonical_df)} rows, {len(canonical_df.columns)} columns.")
    print(f"Validation checks: {val_report.passed_count}/{val_report.total_checks} passed. Valid: {val_report.is_valid}")

    # 2. Split into Train / Validation / Test
    splits, split_res = split_canonical_dataset(canonical_df, config)
    print(f"Chronological splits created (purge gap = {config.purge_gap_rows}s):")
    for name, s_df in splits.items():
        print(f"  - {name:12}: {len(s_df):4} rows | classes: {dict(s_df[config.target_col].value_counts())}")

    # 3. Save Canonical Artifacts
    canonical_path = output_dir / "canonical_dataset.parquet"
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "validation.parquet"
    test_path = output_dir / "test.parquet"

    canonical_df.to_parquet(canonical_path, index=True)
    splits["train"].to_parquet(train_path, index=True)
    splits["validation"].to_parquet(val_path, index=True)
    splits["test"].to_parquet(test_path, index=True)

    # 4. Generate Metadata
    metadata = {
        "dataset_name": "PredAlpha Phase 1 Canonical BBO Dataset",
        "market_symbol": config.market_symbol,
        "asset_id": config.asset_id_value,
        "source_file": str(input_bbo_path),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "total_rows": len(canonical_df),
        "total_columns": len(canonical_df.columns),
        "start_time": str(canonical_df[config.timestamp_col].min()),
        "end_time": str(canonical_df[config.timestamp_col].max()),
        "frequency": "1s",
        "prediction_horizon_seconds": config.horizon_seconds,
        "target_column": config.target_col,
        "target_classes": list(config.classes),
        "class_distribution": {
            str(k): int(v) for k, v in canonical_df[config.target_col].value_counts().items()
        },
        "feature_columns": list(config.feature_cols),
        "quote_columns": list(config.quote_cols),
        "forbidden_leakage_columns_checked": list(config.forbidden_cols),
        "train_rows": len(splits["train"]),
        "train_start": str(splits["train"][config.timestamp_col].min()),
        "train_end": str(splits["train"][config.timestamp_col].max()),
        "train_classes": {str(k): int(v) for k, v in splits["train"][config.target_col].value_counts().items()},
        "val_rows": len(splits["validation"]),
        "val_start": str(splits["validation"][config.timestamp_col].min()),
        "val_end": str(splits["validation"][config.timestamp_col].max()),
        "val_classes": {str(k): int(v) for k, v in splits["validation"][config.target_col].value_counts().items()},
        "test_rows": len(splits["test"]),
        "test_start": str(splits["test"][config.timestamp_col].min()),
        "test_end": str(splits["test"][config.timestamp_col].max()),
        "test_classes": {str(k): int(v) for k, v in splits["test"][config.target_col].value_counts().items()},
        "purge_gap_seconds": config.purge_gap_rows,
        "status": "VALID_FOR_EXPERIMENTATION_ONLY",
        "limitations": [
            "Sample size of 289 rows (~4.8 minutes) is insufficient for production model training.",
            "Validation split lacks DOWN class due to short continuous session length.",
            "Test split lacks UP class due to short continuous session length.",
            "Autocorrelation in 1-second sampling requires multi-session cross-validation before deployment.",
        ],
    }
    meta_path = output_dir / "dataset_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 5. Generate Validation Reports
    val_report_json = output_dir / "validation_report.json"
    with open(val_report_json, "w", encoding="utf-8") as f:
        json.dump(val_report.to_dict(), f, indent=2)

    val_report_md = output_dir / "validation_report.md"
    md_content = [
        "# PredAlpha Phase 1 — Dataset Validation Report\n",
        f"- **Dataset**: {val_report.dataset_name}",
        f"- **Validation Time**: {val_report.timestamp}",
        f"- **Overall Status**: {'✅ PASSED' if val_report.is_valid else '❌ FAILED'}",
        f"- **Checks Passed**: {val_report.passed_count}/{val_report.total_checks}",
        f"- **Errors**: {val_report.failed_count} | **Warnings**: {val_report.warning_count}\n",
        "## Summary of Validation Checks\n",
        "| Check Name | Status | Details |",
        "|:-----------|:-------|:--------|",
    ]
    for r in val_report.results:
        status_icon = "✅ PASS" if r.passed else ("⚠️ WARN" if r.severity == "WARNING" else "❌ FAIL")
        md_content.append(f"| `{r.check_name}` | {status_icon} | {r.message} |")

    md_content.append("\n## Split Boundary Verification\n")
    md_content.append(f"- **Train Bound**: {metadata['train_start']} to {metadata['train_end']} ({metadata['train_rows']} rows)")
    md_content.append(f"- **Train -> Val Gap**: {split_res.details.get('train_to_validation_gap_seconds', 0.0)}s (minimum purge required: {config.purge_gap_rows}s)")
    md_content.append(f"- **Val Bound**: {metadata['val_start']} to {metadata['val_end']} ({metadata['val_rows']} rows)")
    md_content.append(f"- **Val -> Test Gap**: {split_res.details.get('validation_to_test_gap_seconds', 0.0)}s (minimum purge required: {config.purge_gap_rows}s)")
    md_content.append(f"- **Test Bound**: {metadata['test_start']} to {metadata['test_end']} ({metadata['test_rows']} rows)")

    with open(val_report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")

    # 6. Generate Dataset Selection Decision Report
    selection_report_md = output_dir / "dataset_selection_decision.md"
    selection_content = f"""# PredAlpha Phase 1 — Canonical Dataset Selection Decision

## Decision Summary

- **Selected Canonical Dataset**: `data/processed/phase1/canonical_dataset.parquet` (prepared from `data/processed/btc_88000_bbo_1s.parquet`)
- **Suitability Classification**: **SUITABLE FOR EXPERIMENTATION & TESTING ONLY**
- **Production Status**: **NOT SUITABLE FOR PRODUCTION MODEL TRAINING**
- **Reason for Selection**: It is the only continuous, high-frequency (1s) market dataset in the repository that exhibits genuine mid-price variability (29 distinct price levels), valid positive spreads, no crossed books, all three directional target classes (FLAT: 84.4%, UP: 11.4%, DOWN: 4.2%), and clean causal feature generation once leakage columns are purged.

## Alternatives Evaluated & Reasons for Rejection

1. **Older Experiment (`market_data.parquet`, `labeled_market_data.parquet`, root `train.parquet`, `val.parquet`, `test.parquet`)**:
   - **Fatal Defect**: 100% CONSTANT mid-price ($0.03650), best bid ($0.036), best ask ($0.037), and spread ($0.001) across all rows.
   - **Invalid Synthetic Labels**: Labels were generated from tiny orderbook depth fluctuations moving microprice by $0.00001, violating the project's explicit rule in `training/data_quality.py` ("Mid-price is constant. Do not generate directional labels.").
   - **Target Leakage**: `future_return`, `future_return_50`, and `future_return_100` were saved directly alongside feature matrices.
   - **Type Corruption**: `asset_id` was cast to float64, losing integer token precision.
   - **Rejection**: **Fatal defects; rejected permanently.**

2. **`market_data_extended.parquet` & `market_data_test.parquet`**:
   - **Fatal Defect**: `market_data_extended.parquet` has 100% constant mid-price ($0.02550); `market_data_test.parquet` has 100% constant mid-price ($0.03650). Both lack target labels.
   - **Rejection**: **Unlabeled and zero price movement; rejected.**

3. **`btc_oct3_up_down_session_*` (Sessions 01-04)**:
   - **Defect**: Interleaved UP/DOWN token book events without on-disk settlement/outcome labels.
   - **Alignment Artifacts**: Aligning with Binance 1m candles produced only 94 valid rows with variable event-to-target horizons (1 to 5 minutes) and only 7 label transitions.
   - **Rejection**: **Unsuitable for order-book HFT training; rejected.**

4. **`btc_88000_session_20261002_01_features_5s.parquet` (Session 1)**:
   - **Defect**: Extreme class imbalance (96.3% FLAT, 2.1% DOWN, 1.7% UP).
   - **Single-Class Validation Set**: Chronological splitting produces a validation set with 100% FLAT labels (0 UP, 0 DOWN), making validation uninformative.
   - **Target Leakage**: Future columns (`future_mid`, `future_delta`, `label_threshold`) were saved into feature files.
   - **Rejection**: **Single-class validation set and extreme imbalance; rejected.**

5. **Track A Pilot (`data/raw/btc_observations.jsonl`)**:
   - **Defect**: Contains only 12 rows of 1-minute BTC candles (1 hour of sparse data).
   - **Rejection**: **Insufficient sample size for any statistical or ML pipeline.**

6. **Track B Settlement Evidence (`data/raw/track_b/`)**:
   - **Defect**: Contains 16 market settlement outcome records but zero event or order-book streaming files.
   - **Rejection**: **Settlement metadata only; no tick/BBO data available for training.**

## Known Limitations of Selected Canonical Dataset

1. **Ultra-Short Duration**: 289 rows of 1-second BBO observations spans only 4 minutes and 48 seconds of continuous trading.
2. **Split Class Coverage**: Chronological splitting with a 5-second purge gap creates a validation split lacking DOWN samples and a test split lacking UP samples due to regime clustering in this short session.
3. **Autocorrelation**: Consecutive 1-second snapshots exhibit high serial dependency.

## Canonical Schema Definition

- `timestamp`: UTC Datetime (`datetime64[ns, UTC]`)
- `asset_id`: String (`btc_88000`)
- `bid`: Float64
- `ask`: Float64
- `mid_price`: Float64
- `spread`: Float64
- `spread_bps`: Float64
- `mid_return_1s`: Float64
- `mid_return_3s`: Float64
- `mid_return_5s`: Float64
- `mid_volatility_5s`: Float64
- `bid_change_1s`: Float64
- `ask_change_1s`: Float64
- `label`: Categorical string (`FLAT`, `UP`, `DOWN`)
"""
    with open(selection_report_md, "w", encoding="utf-8") as f:
        f.write(selection_content)

    print(f"\nAll Phase 1 artifacts saved successfully to: {output_dir}")
    print(f"  - Canonical: {canonical_path}")
    print(f"  - Train:     {train_path}")
    print(f"  - Val:       {val_path}")
    print(f"  - Test:      {test_path}")
    print(f"  - Metadata:  {meta_path}")
    print(f"  - Report:    {val_report_md}")
    print(f"  - Decision:  {selection_report_md}")

    return {
        "canonical_path": canonical_path,
        "metadata": metadata,
        "validation_report": val_report.to_dict(),
        "splits": {k: len(v) for k, v in splits.items()},
    }


if __name__ == "__main__":
    run_phase1_pipeline()
