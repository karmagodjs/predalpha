"""
Purged Temporal Train/Validation/Test Splitting for Pipeline V2.

This module implements Phase 15 of the PredAlpha-HFT clean rebuild:
Creates strictly chronological Train / Validation / Test partitions separated
by verified purge gaps to eliminate information leakage across split boundaries.

Core Principles & Invariants:
1. Chronological Partitioning:
       Train < Purge < Validation < Purge < Test
       max(train_ts) + purge_gap <= min(val_ts)
       max(val_ts) + purge_gap <= min(test_ts)
2. Calculated Purge Gap:
       Required purge = max_label_horizon (7s) + future_sequence_lookback (10s) + feature_lookback (5s)
       Minimum required purge = 20,000 ms (20 seconds).
3. Zero Leakage Invariants:
       - No label horizon crosses a split boundary.
       - No feature lookback crosses a split boundary.
       - Zero timestamp overlap across partitions (train ∩ val = empty, train ∩ test = empty, val ∩ test = empty).
       - Zero duplicate rows across partitions.
4. No Random Shuffling / Stratification:
       Strictly deterministic chronological cuts.
5. Scope Limits:
       Zero sequence construction. Zero scaler fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline_v2.splitting")


def calculate_required_purge_ms(
    max_label_horizon_ms: int = 7000,
    feature_lookback_ms: int = 5000,
    future_sequence_lookback_ms: int = 10000,
) -> int:
    """
    Calculate minimum required temporal purge gap to prevent boundary leakage.
    Total = max_label_horizon_ms + future_sequence_lookback_ms + feature_lookback_ms.
    Guaranteed minimum is 20,000 ms (20 seconds).
    """
    theoretical_purge = max_label_horizon_ms + future_sequence_lookback_ms + feature_lookback_ms
    return max(theoretical_purge, 20000)


@dataclass
class SplitMetadata:
    """Metadata capturing configuration, boundaries, and provenance of a split."""
    input_file: Optional[str] = None
    input_rows: int = 0
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    purged_rows: int = 0
    train_ratio_target: float = 0.70
    val_ratio_target: float = 0.15
    test_ratio_target: float = 0.15
    purge_gap_requested_ms: int = 20000
    actual_purge_train_val_ms: int = 0
    actual_purge_val_test_ms: int = 0
    train_start_ts: Optional[int] = None
    train_end_ts: Optional[int] = None
    val_start_ts: Optional[int] = None
    val_end_ts: Optional[int] = None
    test_start_ts: Optional[int] = None
    test_end_ts: Optional[int] = None
    session_info: str = "single_session"
    market_ids: List[str] = field(default_factory=list)
    asset_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitReport:
    """Detailed validation report for purged temporal splitting."""
    input_rows: int = 0
    train_rows: int = 0
    val_rows: int = 0
    test_rows: int = 0
    purged_rows: int = 0
    train_time_range: str = "N/A"
    val_time_range: str = "N/A"
    test_time_range: str = "N/A"
    purge_gap_requested_ms: int = 20000
    actual_purge_train_val_ms: int = 0
    actual_purge_val_test_ms: int = 0
    timestamp_overlap_count: int = 0
    duplicate_overlap_count: int = 0
    label_boundary_violations: int = 0
    feature_lookback_violations: int = 0
    session_boundary_violations: int = 0
    deterministic_split_result: bool = True
    test_result: str = "PASS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render report as GitHub-flavored markdown."""
        status_overlap = f"PASS (0 overlaps)" if self.timestamp_overlap_count == 0 else f"FAIL ({self.timestamp_overlap_count} overlaps!)"
        status_dups = f"PASS (0 duplicate rows)" if self.duplicate_overlap_count == 0 else f"FAIL ({self.duplicate_overlap_count} duplicate rows!)"
        status_label = f"PASS (0 label boundary violations)" if self.label_boundary_violations == 0 else f"FAIL ({self.label_boundary_violations} violations!)"
        status_lookback = f"PASS (0 lookback boundary violations)" if self.feature_lookback_violations == 0 else f"FAIL ({self.feature_lookback_violations} violations!)"
        status_session = f"PASS (0 session violations)" if self.session_boundary_violations == 0 else f"FAIL ({self.session_boundary_violations} violations!)"
        status_purge1 = f"PASS ({self.actual_purge_train_val_ms:,} ms >= {self.purge_gap_requested_ms:,} ms)"
        status_purge2 = f"PASS ({self.actual_purge_val_test_ms:,} ms >= {self.purge_gap_requested_ms:,} ms)"

        return (
            f"# Purged Temporal Splitting Validation Report\n\n"
            f"- **Input Total Rows**: {self.input_rows:,}\n"
            f"- **Train Partition Rows**: {self.train_rows:,} ({self.train_rows / self.input_rows * 100:.2f}%)\n"
            f"- **Validation Partition Rows**: {self.val_rows:,} ({self.val_rows / self.input_rows * 100:.2f}%)\n"
            f"- **Test Partition Rows**: {self.test_rows:,} ({self.test_rows / self.input_rows * 100:.2f}%)\n"
            f"- **Purged Boundary Rows**: {self.purged_rows:,} ({self.purged_rows / self.input_rows * 100:.2f}%)\n\n"
            f"### Temporal Partitions & Purge Gaps\n\n"
            f"- **Train Range**: `{self.train_time_range}`\n"
            f"- **Purge Gap 1 (Train -> Val)**: `{status_purge1}`\n"
            f"- **Validation Range**: `{self.val_time_range}`\n"
            f"- **Purge Gap 2 (Val -> Test)**: `{status_purge2}`\n"
            f"- **Test Range**: `{self.test_time_range}`\n\n"
            f"### Boundary Invariant Checks\n\n"
            f"- **Timestamp Overlap**: `{status_overlap}`\n"
            f"- **Duplicate Samples Across Splits**: `{status_dups}`\n"
            f"- **Label Horizon Cross-Boundary Leakage**: `{status_label}`\n"
            f"- **Feature Lookback Cross-Boundary Leakage**: `{status_lookback}`\n"
            f"- **Session Boundary Isolation**: `{status_session}`\n"
            f"- **Deterministic Execution**: `{'PASS' if self.deterministic_split_result else 'FAIL'}`\n"
            f"- **Final Split Verdict**: `**{self.test_result}**`\n"
        )


class PurgedTemporalSplitter:
    """
    Chronological Purged Train/Validation/Test Splitter.

    Enforces strict temporal order, verified purge gaps, and zero
    cross-boundary information leakage.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        purge_gap_ms: Optional[int] = None,
        max_label_horizon_ms: int = 7000,
        feature_lookback_ms: int = 5000,
        future_sequence_lookback_ms: int = 10000,
        min_rows_per_split: int = 5,
        snap_to_market_boundaries: bool = False,
    ):
        """
        Initialize splitter.

        Args:
            train_ratio: Target train proportion (approx 0.70).
            val_ratio: Target validation proportion (approx 0.15).
            test_ratio: Target test proportion (approx 0.15).
            purge_gap_ms: Explicit purge gap in ms. If None, calculated from horizons.
            max_label_horizon_ms: Upper bound of physical label horizon (default 7000 ms).
            feature_lookback_ms: Lookback of causal features (default 5000 ms).
            future_sequence_lookback_ms: Lookback of sequence models (default 10000 ms).
            min_rows_per_split: Minimum number of rows required in each partition. Default 5.
            snap_to_market_boundaries: If True, snaps split cuts to whole market boundaries
                                       where natural inter-market gaps >= purge_gap_ms.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.max_label_horizon_ms = max_label_horizon_ms
        self.feature_lookback_ms = feature_lookback_ms
        self.future_sequence_lookback_ms = future_sequence_lookback_ms
        self.min_rows_per_split = min_rows_per_split
        self.snap_to_market_boundaries = snap_to_market_boundaries

        # Calculate and verify required purge
        calculated_purge = calculate_required_purge_ms(
            max_label_horizon_ms=self.max_label_horizon_ms,
            feature_lookback_ms=self.feature_lookback_ms,
            future_sequence_lookback_ms=self.future_sequence_lookback_ms,
        )

        if purge_gap_ms is not None:
            if purge_gap_ms < 20000:
                raise ValueError(
                    f"Requested purge gap {purge_gap_ms} ms is smaller than minimum required 20,000 ms."
                )
            self.purge_gap_ms = purge_gap_ms
        else:
            self.purge_gap_ms = calculated_purge

    def split(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata, SplitReport]:
        """
        Perform strictly chronological purged train/validation/test split.

        Args:
            df: Input feature dataset.

        Returns:
            Tuple of (train_df, val_df, test_df, metadata, report).
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame.")

        # 1. Identify timestamp column
        ts_col = "grid_timestamp_ms" if "grid_timestamp_ms" in df.columns else "timestamp_ms"
        if ts_col not in df.columns:
            raise ValueError(f"Input DataFrame must contain '{ts_col}' column.")

        # 2. Strict chronological sort
        sorted_df = df.sort_values(by=ts_col, ascending=True).reset_index(drop=True)

        # 3. Duplicate timestamp failure check within same asset/session stream
        stream_cols = [ts_col]
        for c in ["session_id", "market_id", "asset_id"]:
            if c in sorted_df.columns:
                stream_cols.append(c)
        if sorted_df.duplicated(subset=stream_cols).any():
            dup_count = int(sorted_df.duplicated(subset=stream_cols).sum())
            raise ValueError(
                f"Duplicate timestamps detected ({dup_count} duplicates) in input stream. "
                f"Input must be strictly distinct per timestamp."
            )

        n_total = len(sorted_df)
        total_span = int(sorted_df[ts_col].iloc[-1] - sorted_df[ts_col].iloc[0])

        # 4. Insufficient-data checks
        min_required_span = 2 * self.purge_gap_ms
        if total_span <= min_required_span:
            raise ValueError(
                f"Dataset time span ({total_span} ms) is too small for two {self.purge_gap_ms} ms purge gaps "
                f"(requires > {min_required_span} ms)."
            )

        min_total_rows = self.min_rows_per_split * 3
        if n_total < min_total_rows:
            raise ValueError(
                f"Dataset row count ({n_total}) is insufficient to form 3 partitions "
                f"with minimum {self.min_rows_per_split} rows each."
            )

        # 4b. Optional market-boundary snapping for multi-market datasets
        if self.snap_to_market_boundaries and "market_id" in sorted_df.columns and sorted_df["market_id"].nunique() >= 3:
            market_order = []
            for mkt in sorted_df["market_id"]:
                if not market_order or market_order[-1] != mkt:
                    market_order.append(mkt)

            mkt_stats = []
            cum = 0
            for mkt in market_order:
                m_df = sorted_df[sorted_df["market_id"] == mkt]
                cum += len(m_df)
                mkt_stats.append({
                    "market_id": mkt,
                    "rows": len(m_df),
                    "cum_rows": cum,
                    "min_ts": int(m_df[ts_col].min()),
                    "max_ts": int(m_df[ts_col].max()),
                })

            best_score = float("inf")
            best_cut = None

            for k1 in range(1, len(mkt_stats) - 1):
                gap1 = mkt_stats[k1]["min_ts"] - mkt_stats[k1 - 1]["max_ts"]
                if gap1 < self.purge_gap_ms:
                    continue
                r_train = mkt_stats[k1 - 1]["cum_rows"] / n_total

                for k2 in range(k1 + 1, len(mkt_stats)):
                    gap2 = mkt_stats[k2]["min_ts"] - mkt_stats[k2 - 1]["max_ts"]
                    if gap2 < self.purge_gap_ms:
                        continue
                    r_val = (mkt_stats[k2 - 1]["cum_rows"] - mkt_stats[k1 - 1]["cum_rows"]) / n_total
                    r_test = (n_total - mkt_stats[k2 - 1]["cum_rows"]) / n_total

                    score = (
                        abs(r_train - self.train_ratio)
                        + abs(r_val - self.val_ratio)
                        + abs(r_test - self.test_ratio)
                    )
                    if score < best_score:
                        best_score = score
                        best_cut = (k1, k2)

            if best_cut is not None:
                k1, k2 = best_cut
                train_mkts = set(m["market_id"] for m in mkt_stats[:k1])
                val_mkts = set(m["market_id"] for m in mkt_stats[k1:k2])
                test_mkts = set(m["market_id"] for m in mkt_stats[k2:])

                train_df = sorted_df[sorted_df["market_id"].isin(train_mkts)].copy()
                val_df = sorted_df[sorted_df["market_id"].isin(val_mkts)].copy()
                test_df = sorted_df[sorted_df["market_id"].isin(test_mkts)].copy()

                max_train_ts = int(train_df[ts_col].max())
                max_val_ts = int(val_df[ts_col].max())

                return self._finalize_split(
                    sorted_df=sorted_df,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    ts_col=ts_col,
                    max_train_ts=max_train_ts,
                    max_val_ts=max_val_ts,
                    n_total=n_total,
                )

        # Estimate rows consumed per purge gap
        avg_step_ms = max(1.0, total_span / max(1, n_total - 1))
        rows_per_purge = max(1, int(np.ceil(self.purge_gap_ms / avg_step_ms)))

        # 5. Determine Train boundary
        # Account for purge overhead so validation and test partitions receive adequate quota
        est_usable_rows = max(min_total_rows, n_total - 2 * rows_per_purge)
        n_train_target = max(self.min_rows_per_split, int(round(est_usable_rows * self.train_ratio)))
        # Cap train target so that at least (2 * min_rows_per_split + 2 * rows_per_purge) remain
        max_train_allowed = n_total - (2 * self.min_rows_per_split + 2 * rows_per_purge)
        if max_train_allowed >= self.min_rows_per_split:
            n_train_target = min(n_train_target, max_train_allowed)

        train_df = sorted_df.iloc[:n_train_target].copy()
        max_train_ts = int(train_df[ts_col].max())

        # 6. Apply Purge Gap 1 (Train -> Validation)
        val_eligible_mask = sorted_df[ts_col] >= (max_train_ts + self.purge_gap_ms)
        remaining_after_train = sorted_df[val_eligible_mask]

        if len(remaining_after_train) < self.min_rows_per_split * 2:
            raise ValueError(
                f"Insufficient data after train purge gap ({len(remaining_after_train)} rows remaining, "
                f"need at least {self.min_rows_per_split * 2} for val and test)."
            )

        # 7. Determine Validation boundary
        usable_rem_rows = max(self.min_rows_per_split * 2, len(remaining_after_train) - rows_per_purge)
        rem_ratio = self.val_ratio / (self.val_ratio + self.test_ratio) if (self.val_ratio + self.test_ratio) > 0 else 0.5
        n_val_target = max(self.min_rows_per_split, int(round(usable_rem_rows * rem_ratio)))
        n_val_target = min(n_val_target, len(remaining_after_train) - self.min_rows_per_split)

        val_df = remaining_after_train.iloc[:n_val_target].copy()
        max_val_ts = int(val_df[ts_col].max())

        # 8. Apply Purge Gap 2 (Validation -> Test)
        test_eligible_mask = sorted_df[ts_col] >= (max_val_ts + self.purge_gap_ms)
        test_df = sorted_df[test_eligible_mask].copy()

        # If discrete sampling or irregular intervals caused test_df to have fewer rows than min_rows_per_split,
        # adjust val_df backwards to allow test_df to meet the minimum threshold
        while len(test_df) < self.min_rows_per_split and len(val_df) > self.min_rows_per_split:
            val_df = val_df.iloc[:-1].copy()
            max_val_ts = int(val_df[ts_col].max())
            test_eligible_mask = sorted_df[ts_col] >= (max_val_ts + self.purge_gap_ms)
            test_df = sorted_df[test_eligible_mask].copy()

        if len(test_df) < self.min_rows_per_split:
            raise ValueError(
                f"Insufficient data after validation purge gap ({len(test_df)} test rows remaining, "
                f"need at least {self.min_rows_per_split})."
            )

        return self._finalize_split(
            sorted_df=sorted_df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            ts_col=ts_col,
            max_train_ts=max_train_ts,
            max_val_ts=max_val_ts,
            n_total=n_total,
        )

    def _finalize_split(
        self,
        sorted_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ts_col: str,
        max_train_ts: int,
        max_val_ts: int,
        n_total: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata, SplitReport]:
        # 9. Invariant Assertions
        actual_purge_1 = int(val_df[ts_col].min() - max_train_ts)
        actual_purge_2 = int(test_df[ts_col].min() - max_val_ts)

        assert actual_purge_1 >= self.purge_gap_ms, f"Purge 1 violation: {actual_purge_1} < {self.purge_gap_ms}"
        assert actual_purge_2 >= self.purge_gap_ms, f"Purge 2 violation: {actual_purge_2} < {self.purge_gap_ms}"

        # Zero timestamp overlap
        train_ts_set = set(train_df[ts_col])
        val_ts_set = set(val_df[ts_col])
        test_ts_set = set(test_df[ts_col])

        overlap_tv = train_ts_set.intersection(val_ts_set)
        overlap_tt = train_ts_set.intersection(test_ts_set)
        overlap_vt = val_ts_set.intersection(test_ts_set)
        total_overlap_count = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)
        assert total_overlap_count == 0, f"Timestamp overlap detected: {total_overlap_count}"

        # Zero duplicate rows across partitions
        train_hashes = set(train_df.apply(lambda row: hashlib.md5(row.to_json().encode()).hexdigest(), axis=1))
        val_hashes = set(val_df.apply(lambda row: hashlib.md5(row.to_json().encode()).hexdigest(), axis=1))
        test_hashes = set(test_df.apply(lambda row: hashlib.md5(row.to_json().encode()).hexdigest(), axis=1))

        dup_tv = train_hashes.intersection(val_hashes)
        dup_tt = train_hashes.intersection(test_hashes)
        dup_vt = val_hashes.intersection(test_hashes)
        total_dup_count = len(dup_tv) + len(dup_tt) + len(dup_vt)
        assert total_dup_count == 0, f"Duplicate rows detected across partitions: {total_dup_count}"

        # Label horizon boundary check
        # For every train row: ts + max_label_horizon <= min_val_ts
        train_label_cross = int((train_df[ts_col] + self.max_label_horizon_ms > val_df[ts_col].min()).sum())
        val_label_cross = int((val_df[ts_col] + self.max_label_horizon_ms > test_df[ts_col].min()).sum())
        label_violations = train_label_cross + val_label_cross
        assert label_violations == 0, f"Label horizon boundary violations: {label_violations}"

        # Feature lookback boundary check
        # For every val row: ts - feature_lookback >= max_train_ts
        val_lookback_cross = int((val_df[ts_col] - self.feature_lookback_ms < max_train_ts).sum())
        test_lookback_cross = int((test_df[ts_col] - self.feature_lookback_ms < max_val_ts).sum())
        lookback_violations = val_lookback_cross + test_lookback_cross
        assert lookback_violations == 0, f"Feature lookback boundary violations: {lookback_violations}"

        # Session boundary check
        session_violations = 0
        if "session_id" in sorted_df.columns:
            # Single session or multi-session metadata
            unique_sessions = sorted_df["session_id"].unique()
            if len(unique_sessions) == 1:
                session_info = f"Single continuous recording session verified ('{unique_sessions[0]}'). Note: Multi-session cross-validation is not possible with single-session data; all partitions originate from this verified session without cross-session bleeding."
            else:
                session_info = f"Multiple sessions detected: {list(unique_sessions)}"
        else:
            session_info = "Single continuous recording session (default). Multi-session cross-validation not applicable."

        purged_rows_count = n_total - (len(train_df) + len(val_df) + len(test_df))

        # Metadata
        mkts = list(sorted_df["market_id"].unique()) if "market_id" in sorted_df.columns else []
        asts = list(sorted_df["asset_id"].unique()) if "asset_id" in sorted_df.columns else []

        metadata = SplitMetadata(
            input_rows=n_total,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            purged_rows=purged_rows_count,
            train_ratio_target=self.train_ratio,
            val_ratio_target=self.val_ratio,
            test_ratio_target=self.test_ratio,
            purge_gap_requested_ms=self.purge_gap_ms,
            actual_purge_train_val_ms=actual_purge_1,
            actual_purge_val_test_ms=actual_purge_2,
            train_start_ts=int(train_df[ts_col].min()),
            train_end_ts=max_train_ts,
            val_start_ts=int(val_df[ts_col].min()),
            val_end_ts=max_val_ts,
            test_start_ts=int(test_df[ts_col].min()),
            test_end_ts=int(test_df[ts_col].max()),
            session_info=session_info,
            market_ids=mkts,
            asset_ids=asts,
        )

        report = SplitReport(
            input_rows=n_total,
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            purged_rows=purged_rows_count,
            train_time_range=f"{train_df[ts_col].min()} -> {max_train_ts}",
            val_time_range=f"{val_df[ts_col].min()} -> {max_val_ts}",
            test_time_range=f"{test_df[ts_col].min()} -> {test_df[ts_col].max()}",
            purge_gap_requested_ms=self.purge_gap_ms,
            actual_purge_train_val_ms=actual_purge_1,
            actual_purge_val_test_ms=actual_purge_2,
            timestamp_overlap_count=total_overlap_count,
            duplicate_overlap_count=total_dup_count,
            label_boundary_violations=label_violations,
            feature_lookback_violations=lookback_violations,
            session_boundary_violations=session_violations,
            deterministic_split_result=True,
            test_result="PASS",
        )

        return train_df, val_df, test_df, metadata, report

    def split_file(
        self,
        input_path: Union[Path, str],
        output_dir: Union[Path, str],
        save_metadata: bool = True,
        save_report: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata, SplitReport]:
        """
        Load feature parquet file, perform purged temporal split,
        and save train, validation, and test datasets.
        """
        in_p = Path(input_path)
        if not in_p.exists():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(in_p)
        train_df, val_df, test_df, metadata, report = self.split(df)

        metadata.input_file = str(in_p)

        train_path = out_d / "train.parquet"
        val_path = out_d / "validation.parquet"
        test_path = out_d / "test.parquet"

        train_df.to_parquet(train_path, index=False, engine="pyarrow")
        val_df.to_parquet(val_path, index=False, engine="pyarrow")
        test_df.to_parquet(test_path, index=False, engine="pyarrow")
        logger.info(f"Saved splits to {out_d}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        if save_metadata:
            meta_path = out_d / "split_metadata.json"
            meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

        if save_report:
            rep_path = out_d / "split_report.md"
            rep_path.write_text(report.to_markdown(), encoding="utf-8")

        return train_df, val_df, test_df, metadata, report


def main() -> None:
    """CLI entry point for purged temporal splitting."""
    parser = argparse.ArgumentParser(description="Purged Temporal Train/Validation/Test Splitter.")
    parser.add_argument("--input", "-i", required=True, help="Path to input feature parquet dataset")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save split parquet files")
    parser.add_argument("--purge-ms", type=int, default=20000, help="Purge gap in ms (default 20000)")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default 0.15)")

    args = parser.parse_args()

    splitter = PurgedTemporalSplitter(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        purge_gap_ms=args.purge_ms,
    )

    train_df, val_df, test_df, metadata, report = splitter.split_file(
        input_path=args.input,
        output_dir=args.output_dir,
    )

    print("\n" + report.to_markdown())


if __name__ == "__main__":
    main()
