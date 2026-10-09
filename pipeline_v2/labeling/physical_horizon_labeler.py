"""
Physical Elapsed-Time 5-Second Labeler for Pipeline V2.

This module implements Phase 13 of the PredAlpha-HFT clean rebuild:
Assigns directional price labels based on physical elapsed time horizons
(5000 ms to 7000 ms) rather than row shifts (shift(-5) is strictly forbidden).

Core Invariants & Rules:
1. Target Horizon:
       5000 ms <= target_timestamp - current_timestamp <= 7000 ms
   Selects the FIRST valid physical event where target_timestamp >= current_timestamp + 5000 ms.
2. If no valid physical target exists within [T+5000ms, T+7000ms]:
       label = NaN (dropped from training output).
3. Stale Data Filtering:
   Observations with `is_stale == True` are strictly excluded from label generation.
4. Session Isolation:
   Targets are never matched across session boundaries.
5. Classification Threshold:
       current_mid = (bid + ask) / 2
       future_mid = (target_bid + target_ask) / 2
       delta = future_mid - current_mid
       threshold = max(current_spread / 2, 0.001)
       delta > threshold  => UP
       delta < -threshold => DOWN
       otherwise          => FLAT
6. Target Leakage Prevention:
   The exported training dataset contains ONLY causal market state and the `label` column.
   All future_* and target_* columns are quarantined to a detached audit dataset.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline_v2.labeling")


@dataclass
class LabelingValidationReport:
    """Detailed audit report for physical 5-second horizon labeling."""
    input_rows: int = 0
    stale_rows_excluded: int = 0
    labeled_rows: int = 0
    dropped_rows: int = 0
    min_horizon_ms: Optional[int] = None
    max_horizon_ms: Optional[int] = None
    mean_horizon_ms: Optional[float] = None
    horizon_distribution: Dict[str, int] = field(default_factory=dict)
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0
    up_percentage: float = 0.0
    down_percentage: float = 0.0
    flat_percentage: float = 0.0
    session_boundary_violations: int = 0
    physical_horizon_violations: int = 0
    future_column_violations: int = 0
    output_file: Optional[str] = None
    audit_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render report as GitHub-flavored markdown."""
        status_horizon = (
            "PASS (0 violations, all in [5000, 7000] ms)"
            if self.physical_horizon_violations == 0
            else f"FAIL ({self.physical_horizon_violations} violations!)"
        )
        status_leakage = (
            "PASS (0 future columns in training dataset)"
            if self.future_column_violations == 0
            else f"FAIL ({self.future_column_violations} target/future columns detected!)"
        )
        status_session = (
            "PASS (0 cross-session target matches)"
            if self.session_boundary_violations == 0
            else f"FAIL ({self.session_boundary_violations} cross-session violations!)"
        )

        dist_rows = ""
        for bucket, count in sorted(self.horizon_distribution.items()):
            dist_rows += f"| `{bucket}` | {count:,} |\n"
        if not dist_rows:
            dist_rows = "| None | 0 |\n"

        mean_str = f"{self.mean_horizon_ms:.2f} ms" if self.mean_horizon_ms is not None else "N/A"

        return (
            f"# Physical 5-Second Labeling Validation Report\n\n"
            f"- **Input Observations**: {self.input_rows:,}\n"
            f"- **Stale Observations Excluded**: {self.stale_rows_excluded:,}\n"
            f"- **Successfully Labeled Rows**: {self.labeled_rows:,}\n"
            f"- **Dropped Rows (Stale + No Target in [5s, 7s])**: {self.dropped_rows:,}\n\n"
            f"### Horizon Statistics\n\n"
            f"- **Min Physical Horizon**: {self.min_horizon_ms} ms\n"
            f"- **Max Physical Horizon**: {self.max_horizon_ms} ms\n"
            f"- **Mean Physical Horizon**: {mean_str}\n"
            f"- **Physical Horizon Invariant [5000, 7000] ms**: `{status_horizon}`\n"
            f"- **Session Boundary Isolation**: `{status_session}`\n"
            f"- **Target Leakage / Future Column Invariant**: `{status_leakage}`\n\n"
            f"### Class Distribution\n\n"
            f"| Class | Count | Percentage |\n"
            f"| :--- | :--- | :--- |\n"
            f"| `UP` | {self.up_count:,} | {self.up_percentage:.2f}% |\n"
            f"| `DOWN` | {self.down_count:,} | {self.down_percentage:.2f}% |\n"
            f"| `FLAT` | {self.flat_count:,} | {self.flat_percentage:.2f}% |\n\n"
            f"### Horizon Interval Distribution\n\n"
            f"| Horizon Bucket (ms) | Count |\n"
            f"| :--- | :--- |\n"
            f"{dist_rows}\n"
            f"- **Training Dataset Output**: `{self.output_file or 'In-Memory'}`\n"
            f"- **Detached Audit Output**: `{self.audit_file or 'In-Memory'}`\n"
        )


class PhysicalHorizonLabeler:
    """
    Physical-Time 5-Second Horizon Labeler.

    Assigns directional labels based strictly on physical elapsed milliseconds.
    Zero row-based indexing or row count assumptions.
    Quarantines future target columns to a detached audit dataset.
    """

    FORBIDDEN_TRAIN_COLS = {
        "target_timestamp",
        "target_timestamp_ms",
        "future_timestamp",
        "future_timestamp_ms",
        "target_mid",
        "future_mid",
        "target_bid",
        "target_ask",
        "future_bid",
        "future_ask",
        "target_delta",
        "future_delta",
        "delta",
        "threshold",
        "label_threshold",
        "physical_horizon_ms",
        "horizon_ms",
    }

    def __init__(
        self,
        min_horizon_ms: int = 5000,
        max_horizon_ms: int = 7000,
        min_spread_fraction: float = 0.5,
        min_threshold_abs: float = 0.001,
    ):
        """
        Initialize labeler.

        Args:
            min_horizon_ms: Minimum elapsed physical horizon in ms. Default 5000 ms.
            max_horizon_ms: Maximum allowed physical horizon in ms. Default 7000 ms.
            min_spread_fraction: Fraction of current spread used for threshold. Default 0.5.
            min_threshold_abs: Minimum absolute delta threshold. Default 0.001.
        """
        self.min_horizon_ms = min_horizon_ms
        self.max_horizon_ms = max_horizon_ms
        self.min_spread_fraction = min_spread_fraction
        self.min_threshold_abs = min_threshold_abs

    def compute_threshold(self, bid: float, ask: float) -> float:
        """Compute minimum movement threshold: max(spread / 2, 0.001)."""
        spread = ask - bid
        return max(spread * self.min_spread_fraction, self.min_threshold_abs)

    def classify_delta(self, delta: float, threshold: float) -> str:
        """
        Classify price return delta:
            delta > threshold       => UP
            delta < -threshold      => DOWN
            otherwise               => FLAT
        """
        if delta > threshold:
            return "UP"
        elif delta < -threshold:
            return "DOWN"
        else:
            return "FLAT"

    def label_dataset(
        self,
        resampled_df: pd.DataFrame,
        events_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, LabelingValidationReport]:
        """
        Label observations in resampled_df using physical future quote events.

        Args:
            resampled_df: 1-second grid observations (must contain grid_timestamp_ms, bid, ask, is_stale).
            events_df: Canonical market events pool. If None, uses non-stale rows of resampled_df.

        Returns:
            Tuple of:
            1. clean_labeled_df: Training dataset containing causal market features and `label`.
                                 Guaranteed zero target_* / future_* columns.
            2. audit_df: Detached audit artifact containing target prices, horizons, deltas, and thresholds.
            3. report: LabelingValidationReport summarizing horizon and class statistics.
        """
        if resampled_df.empty:
            empty_clean = pd.DataFrame()
            empty_audit = pd.DataFrame()
            return empty_clean, empty_audit, LabelingValidationReport()

        total_input_rows = len(resampled_df)

        # 1. Identify timestamp column in resampled input
        ts_col = "grid_timestamp_ms" if "grid_timestamp_ms" in resampled_df.columns else "timestamp_ms"
        if ts_col not in resampled_df.columns:
            raise ValueError(f"Input resampled DataFrame must contain 'grid_timestamp_ms' or 'timestamp_ms'.")

        # 2. Filter out stale current observations
        if "is_stale" in resampled_df.columns:
            stale_mask = resampled_df["is_stale"].astype(bool)
            fresh_df = resampled_df[~stale_mask].copy()
            stale_rows_excluded = int(stale_mask.sum())
        else:
            fresh_df = resampled_df.copy()
            stale_rows_excluded = 0

        # 3. Determine target events pool
        has_session = "session_id" in resampled_df.columns
        partition_cols = ["session_id", "market_id", "asset_id"] if has_session else ["market_id", "asset_id"]

        if events_df is not None and not events_df.empty:
            target_source_df = events_df.copy()
            target_ts_col = "timestamp_ms" if "timestamp_ms" in target_source_df.columns else "grid_timestamp_ms"
        else:
            # Fallback to fresh resampled observations
            target_source_df = fresh_df.copy()
            target_ts_col = ts_col

        labeled_records: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []
        session_violations = 0
        physical_horizon_violations = 0

        # Group by partition to guarantee strict session and asset isolation
        for part_key, part_obs in fresh_df.groupby(partition_cols, sort=True):
            # Filter target pool strictly to the matching partition
            if has_session:
                sess_val, mkt_val, ast_val = part_key
                mask_target = (
                    (target_source_df["session_id"] == sess_val)
                    & (target_source_df["market_id"] == mkt_val)
                    & (target_source_df["asset_id"] == ast_val)
                )
            else:
                mkt_val, ast_val = part_key
                mask_target = (
                    (target_source_df["market_id"] == mkt_val)
                    & (target_source_df["asset_id"] == ast_val)
                )

            part_targets = target_source_df[mask_target]
            if part_targets.empty:
                continue

            # Ensure targets are deterministically sorted by physical timestamp ASC, sequence_id ASC
            sort_cols = [target_ts_col]
            if "sequence_id" in part_targets.columns:
                sort_cols.append("sequence_id")
            elif "source_sequence_id" in part_targets.columns:
                sort_cols.append("source_sequence_id")

            part_targets = part_targets.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

            target_ts_arr = part_targets[target_ts_col].to_numpy(dtype=np.int64)
            target_bids = part_targets["bid"].to_numpy(dtype=np.float64)
            target_asks = part_targets["ask"].to_numpy(dtype=np.float64)
            target_seqs = (
                part_targets["sequence_id"].to_numpy(dtype=np.int64)
                if "sequence_id" in part_targets.columns
                else (
                    part_targets["source_sequence_id"].to_numpy(dtype=np.int64)
                    if "source_sequence_id" in part_targets.columns
                    else np.zeros(len(part_targets), dtype=np.int64)
                )
            )

            # Iterate over fresh observations in this partition
            for _, obs_row in part_obs.iterrows():
                curr_ts = int(obs_row[ts_col])
                target_min = curr_ts + self.min_horizon_ms
                target_max = curr_ts + self.max_horizon_ms

                # Select the FIRST valid physical event where target_timestamp >= T + 5000 ms
                idx = np.searchsorted(target_ts_arr, target_min, side="left")

                if idx < len(target_ts_arr):
                    target_ts = int(target_ts_arr[idx])

                    # Target must fall strictly within [T + 5000 ms, T + 7000 ms]
                    if target_ts <= target_max:
                        horizon_ms = target_ts - curr_ts

                        if not (self.min_horizon_ms <= horizon_ms <= self.max_horizon_ms):
                            physical_horizon_violations += 1
                            continue

                        curr_bid = float(obs_row["bid"])
                        curr_ask = float(obs_row["ask"])
                        curr_mid = (curr_bid + curr_ask) / 2.0

                        tgt_bid = float(target_bids[idx])
                        tgt_ask = float(target_asks[idx])
                        tgt_mid = (tgt_bid + tgt_ask) / 2.0

                        delta = tgt_mid - curr_mid
                        threshold = self.compute_threshold(curr_bid, curr_ask)
                        label = self.classify_delta(delta, threshold)

                        # Clean training observation (causal features + label ONLY)
                        clean_row = obs_row.to_dict()
                        # Exclude is_stale flag if present (all remaining rows are fresh)
                        clean_row["label"] = label
                        labeled_records.append(clean_row)

                        # Detached audit observation
                        audit_row = {
                            "current_timestamp": curr_ts,
                            "target_timestamp": target_ts,
                            "physical_horizon_ms": horizon_ms,
                            "current_mid": curr_mid,
                            "target_mid": tgt_mid,
                            "delta": delta,
                            "threshold": threshold,
                            "label": label,
                            "market_id": str(mkt_val),
                            "asset_id": str(ast_val),
                        }
                        if has_session:
                            audit_row["session_id"] = str(sess_val)
                        if "sequence_id" in obs_row:
                            audit_row["current_sequence_id"] = int(obs_row["sequence_id"])
                        elif "source_sequence_id" in obs_row:
                            audit_row["current_sequence_id"] = int(obs_row["source_sequence_id"])
                        audit_row["target_sequence_id"] = int(target_seqs[idx])

                        audit_records.append(audit_row)

        labeled_count = len(labeled_records)
        dropped_count = total_input_rows - labeled_count

        if labeled_records:
            clean_labeled_df = pd.DataFrame(labeled_records)
            audit_df = pd.DataFrame(audit_records)

            # Strip any forbidden future-derived columns from training dataset
            forbidden_found = [c for c in clean_labeled_df.columns if c in self.FORBIDDEN_TRAIN_COLS or c.startswith("future_") or c.startswith("target_")]
            future_column_violations = len(forbidden_found)
            if forbidden_found:
                clean_labeled_df = clean_labeled_df.drop(columns=forbidden_found)

            # Types
            clean_labeled_df["label"] = clean_labeled_df["label"].astype("string")
            audit_df["label"] = audit_df["label"].astype("string")
            audit_df["physical_horizon_ms"] = audit_df["physical_horizon_ms"].astype("int64")

            horizons = audit_df["physical_horizon_ms"].to_numpy()
            min_h = int(horizons.min())
            max_h = int(horizons.max())
            mean_h = float(horizons.mean())

            # Distribution buckets in 500ms intervals
            horizon_buckets: Dict[str, int] = {}
            for h in horizons:
                bucket_start = (int(h) // 500) * 500
                bucket_str = f"[{bucket_start}, {bucket_start + 500}) ms"
                horizon_buckets[bucket_str] = horizon_buckets.get(bucket_str, 0) + 1

            up_c = int((clean_labeled_df["label"] == "UP").sum())
            down_c = int((clean_labeled_df["label"] == "DOWN").sum())
            flat_c = int((clean_labeled_df["label"] == "FLAT").sum())

            up_pct = (up_c / labeled_count) * 100.0
            down_pct = (down_c / labeled_count) * 100.0
            flat_pct = (flat_c / labeled_count) * 100.0
        else:
            clean_labeled_df = pd.DataFrame()
            audit_df = pd.DataFrame()
            future_column_violations = 0
            min_h = None
            max_h = None
            mean_h = None
            horizon_buckets = {}
            up_c = down_c = flat_c = 0
            up_pct = down_pct = flat_pct = 0.0

        report = LabelingValidationReport(
            input_rows=total_input_rows,
            stale_rows_excluded=stale_rows_excluded,
            labeled_rows=labeled_count,
            dropped_rows=dropped_count,
            min_horizon_ms=min_h,
            max_horizon_ms=max_h,
            mean_horizon_ms=mean_h,
            horizon_distribution=horizon_buckets,
            up_count=up_c,
            down_count=down_c,
            flat_count=flat_c,
            up_percentage=up_pct,
            down_percentage=down_pct,
            flat_percentage=flat_pct,
            session_boundary_violations=session_violations,
            physical_horizon_violations=physical_horizon_violations,
            future_column_violations=future_column_violations,
        )

        return clean_labeled_df, audit_df, report

    def label_file(
        self,
        resampled_path: Union[Path, str],
        events_path: Optional[Union[Path, str]] = None,
        output_labeled_path: Optional[Union[Path, str]] = None,
        output_audit_path: Optional[Union[Path, str]] = None,
        report_path: Optional[Union[Path, str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, LabelingValidationReport]:
        """
        Label a resampled parquet dataset and write clean training dataset,
        detached audit artifact, and markdown validation report.
        """
        r_path = Path(resampled_path)
        if not r_path.exists():
            raise FileNotFoundError(f"Resampled input file not found: {r_path}")

        resampled_df = pd.read_parquet(r_path)

        events_df = None
        if events_path is not None:
            e_path = Path(events_path)
            if not e_path.exists():
                raise FileNotFoundError(f"Canonical events input file not found: {e_path}")
            events_df = pd.read_parquet(e_path)

        clean_df, audit_df, report = self.label_dataset(resampled_df, events_df=events_df)

        if output_labeled_path is not None:
            out_lab = Path(output_labeled_path)
            out_lab.parent.mkdir(parents=True, exist_ok=True)
            clean_df.to_parquet(out_lab, index=False, engine="pyarrow")
            report.output_file = str(out_lab)
            logger.info(f"Saved {len(clean_df)} clean labeled rows to {out_lab}")

        if output_audit_path is not None:
            out_aud = Path(output_audit_path)
            out_aud.parent.mkdir(parents=True, exist_ok=True)
            audit_df.to_parquet(out_aud, index=False, engine="pyarrow")
            report.audit_file = str(out_aud)
            logger.info(f"Saved {len(audit_df)} detached audit rows to {out_aud}")

        if report_path is not None:
            rep_path = Path(report_path)
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(report.to_markdown(), encoding="utf-8")
            logger.info(f"Saved labeling report to {rep_path}")

        return clean_df, audit_df, report


def main() -> None:
    """CLI entry point for physical elapsed-time 5-second labeling."""
    parser = argparse.ArgumentParser(description="Physical Elapsed-Time 5-Second Labeler.")
    parser.add_argument("--resampled", "-r", required=True, help="Path to input 1s resampled parquet")
    parser.add_argument("--events", "-e", default=None, help="Path to input canonical events parquet (optional)")
    parser.add_argument("--output-labeled", "-o", default=None, help="Path to output clean labeled parquet")
    parser.add_argument("--output-audit", "-a", default=None, help="Path to output detached audit parquet")
    parser.add_argument("--report", default=None, help="Path to save markdown report")
    parser.add_argument("--min-horizon-ms", type=int, default=5000, help="Min physical horizon in ms (default 5000)")
    parser.add_argument("--max-horizon-ms", type=int, default=7000, help="Max physical horizon in ms (default 7000)")

    args = parser.parse_args()

    labeler = PhysicalHorizonLabeler(
        min_horizon_ms=args.min_horizon_ms,
        max_horizon_ms=args.max_horizon_ms,
    )
    clean_df, audit_df, report = labeler.label_file(
        resampled_path=args.resampled,
        events_path=args.events,
        output_labeled_path=args.output_labeled,
        output_audit_path=args.output_audit,
        report_path=args.report,
    )

    print("\n" + report.to_markdown())


if __name__ == "__main__":
    main()
