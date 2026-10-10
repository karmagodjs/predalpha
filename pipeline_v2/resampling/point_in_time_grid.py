"""
Causal 1-Second Point-In-Time Grid Resampler for Pipeline V2.

This module implements Phase 12 of the PredAlpha-HFT clean rebuild:
Resamples canonical L2 quote events to an exact integer-second grid using
strict backward point-in-time as-of semantics.

Core Rules & Invariants:
1. For every integer-second grid timestamp T:
       selected_event_timestamp <= T
   The selected event is strictly the latest canonical event available at or before T.
2. Zero future-event leakage:
       event_age_ms = grid_timestamp_ms - source_timestamp_ms >= 0
   Strictly asserted for 100% of rows.
3. Staleness tracking:
   Quotes older than 5,000 ms are explicitly marked `is_stale = True`.
4. Provenance preservation:
   Retains source_timestamp_ms, source_sequence_id, market_id, asset_id,
   and source file/line provenance.
5. Session isolation:
   Never matches events across session boundaries.
6. Zero extrapolation:
   Grid starts at first valid integer second >= session start and ends at
   last valid integer second <= session end.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline_v2.resampling")


@dataclass
class ResamplingValidationReport:
    """Detailed audit and validation report for causal 1-second grid resampling."""
    input_events: int = 0
    output_grid_rows: int = 0
    min_grid_timestamp_ms: Optional[int] = None
    max_grid_timestamp_ms: Optional[int] = None
    min_source_timestamp_ms: Optional[int] = None
    max_source_timestamp_ms: Optional[int] = None
    max_event_age_ms: int = 0
    mean_event_age_ms: float = 0.0
    stale_row_count: int = 0
    stale_percentage: float = 0.0
    future_event_violations: int = 0
    duplicate_grid_timestamps: int = 0
    monotonicity: bool = True
    session_boundary_violations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Render validation report as GitHub-flavored markdown."""
        status_future = "PASS (0 violations)" if self.future_event_violations == 0 else f"FAIL ({self.future_event_violations} violations!)"
        status_dup = "PASS (0 duplicates)" if self.duplicate_grid_timestamps == 0 else f"FAIL ({self.duplicate_grid_timestamps} duplicates!)"
        status_mono = "PASS (strictly monotonic)" if self.monotonicity else "FAIL (non-monotonic!)"
        status_session = "PASS (0 cross-session matches)" if self.session_boundary_violations == 0 else f"FAIL ({self.session_boundary_violations} cross-session matches!)"

        return (
            f"# Causal 1-Second Resampling Validation Report\n\n"
            f"- **Input Canonical Events**: {self.input_events:,}\n"
            f"- **Output 1-Second Grid Rows**: {self.output_grid_rows:,}\n"
            f"- **Grid Timestamp Range**: {self.min_grid_timestamp_ms} -> {self.max_grid_timestamp_ms}\n"
            f"- **Matched Source Timestamp Range**: {self.min_source_timestamp_ms} -> {self.max_source_timestamp_ms}\n"
            f"- **Mean Event Age**: {self.mean_event_age_ms:.2f} ms\n"
            f"- **Maximum Event Age**: {self.max_event_age_ms:,} ms\n"
            f"- **Stale Rows (`age > 5000 ms`)**: {self.stale_row_count:,} ({self.stale_percentage:.2f}%)\n"
            f"- **Future-Event Violations (`age < 0`)**: `{status_future}`\n"
            f"- **Duplicate Grid Timestamps**: `{status_dup}`\n"
            f"- **Timestamp Monotonicity**: `{status_mono}`\n"
            f"- **Session Boundary Isolation**: `{status_session}`\n"
        )


class PointInTimeResampler:
    """
    Causal Point-In-Time 1-Second Grid Resampler.

    Guarantees backward-only event alignment with zero lookahead,
    strict staleness tracking, and provenance preservation.
    """

    MANDATORY_CANONICAL_COLS = [
        "timestamp_ms",
        "sequence_id",
        "market_id",
        "asset_id",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
    ]

    def __init__(
        self,
        max_stale_ms: int = 5000,
        grid_interval_ms: int = 1000,
    ):
        """
        Initialize resampler.

        Args:
            max_stale_ms: Threshold in ms beyond which an observation is flagged stale. Default 5000 ms.
            grid_interval_ms: Integer grid spacing in ms. Default 1000 ms (1 second).
        """
        self.max_stale_ms = max_stale_ms
        self.grid_interval_ms = grid_interval_ms

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Perform strict schema and invariant validation on input canonical events.
        Raises ValueError or TypeError if any check fails.
        """
        if df.empty:
            raise ValueError("Input canonical events DataFrame is empty.")

        # 1. Canonical schema verification
        missing_cols = [col for col in self.MANDATORY_CANONICAL_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required canonical columns: {missing_cols}")

        # 2. Timestamp type check
        if not np.issubdtype(df["timestamp_ms"].dtype, np.integer):
            raise TypeError(
                f"Column 'timestamp_ms' must be integer dtype, got {df['timestamp_ms'].dtype}. "
                f"Float timestamps are forbidden to prevent precision loss."
            )

        # 3. Monotonicity within each asset stream
        group_cols = ["session_id", "market_id", "asset_id"] if "session_id" in df.columns else ["market_id", "asset_id"]
        for _, group in df.groupby(group_cols):
            if not group["timestamp_ms"].is_monotonic_increasing:
                raise ValueError("Input events are not monotonically ordered by timestamp_ms within an asset stream.")

        # 4. Valid quotes: bid < ask
        crossed = df[df["bid"] >= df["ask"]]
        if not crossed.empty:
            raise ValueError(f"Input contains {len(crossed)} crossed book records where bid >= ask.")

        # 5. Positive depth
        non_positive_depth = df[(df["bid_size"] <= 0.0) | (df["ask_size"] <= 0.0)]
        if not non_positive_depth.empty:
            raise ValueError(f"Input contains {len(non_positive_depth)} records with non-positive depth (bid_size/ask_size <= 0).")

    def generate_grid_timestamps(
        self,
        session_start_ms: int,
        session_end_ms: int,
    ) -> np.ndarray:
        """
        Generate integer-second grid boundaries strictly inside [session_start_ms, session_end_ms].

        - Starts at the first valid integer second >= session_start_ms.
        - Ends at the last valid integer second <= session_end_ms.
        - Zero extrapolation before session start or after session end.
        """
        start_grid = math.ceil(session_start_ms / self.grid_interval_ms) * self.grid_interval_ms
        end_grid = (session_end_ms // self.grid_interval_ms) * self.grid_interval_ms

        if start_grid > end_grid:
            return np.empty(0, dtype=np.int64)

        return np.arange(start_grid, end_grid + self.grid_interval_ms, self.grid_interval_ms, dtype=np.int64)

    def resample_partition(
        self,
        events: pd.DataFrame,
        session_start_ms: Optional[int] = None,
        session_end_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Resample a single partition (isolated by session_id, market_id, asset_id)
        to the causal integer-second grid.
        """
        if events.empty:
            return pd.DataFrame()

        # Deterministic sort within partition
        sorted_events = events.sort_values(
            by=["timestamp_ms", "sequence_id"],
            ascending=[True, True],
        ).reset_index(drop=True)

        first_event_ts = int(sorted_events["timestamp_ms"].iloc[0])
        last_event_ts = int(sorted_events["timestamp_ms"].iloc[-1])

        # Enforce boundary bounds: grid cannot start before first observed event
        eff_start = max(session_start_ms, first_event_ts) if session_start_ms is not None else first_event_ts
        eff_end = min(session_end_ms, last_event_ts) if session_end_ms is not None else last_event_ts

        grid_ts = self.generate_grid_timestamps(eff_start, eff_end)
        if len(grid_ts) == 0:
            return pd.DataFrame()

        grid_df = pd.DataFrame({"grid_timestamp_ms": grid_ts})

        # Backward as-of point-in-time matching: latest event where timestamp_ms <= grid_timestamp_ms
        merged = pd.merge_asof(
            grid_df,
            sorted_events,
            left_on="grid_timestamp_ms",
            right_on="timestamp_ms",
            direction="backward",
        )

        # Invariant check: backward match must not leave nulls since grid_start >= first_event_ts
        if merged["timestamp_ms"].isna().any():
            raise RuntimeError("Backward as-of merge produced unaligned grid rows with no prior event.")

        # Compute physical event age in milliseconds
        merged["event_age_ms"] = merged["grid_timestamp_ms"] - merged["timestamp_ms"]

        # HARD ASSERTION: Zero future event leakage
        future_violations = merged[merged["event_age_ms"] < 0]
        if not future_violations.empty:
            raise RuntimeError(
                f"CRITICAL FUTURE EVENT LEAKAGE DETECTED! "
                f"{len(future_violations)} rows have source_timestamp_ms > grid_timestamp_ms! "
                f"Sample:\n{future_violations[['grid_timestamp_ms', 'timestamp_ms', 'event_age_ms']].head()}"
            )

        # Staleness flag
        merged["is_stale"] = merged["event_age_ms"] > self.max_stale_ms

        # Rename columns to maintain explicit provenance
        merged = merged.rename(columns={
            "timestamp_ms": "source_timestamp_ms",
            "sequence_id": "source_sequence_id",
        })

        return merged

    def resample_events(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, ResamplingValidationReport]:
        """
        Resample the complete dataset across all partitions.
        Returns resampled DataFrame and ResamplingValidationReport.
        """
        self.validate_input(df)

        has_session = "session_id" in df.columns
        partition_cols = ["session_id", "market_id", "asset_id"] if has_session else ["market_id", "asset_id"]

        resampled_parts = []
        session_violations = 0

        for key, group in df.groupby(partition_cols, sort=True):
            part_resampled = self.resample_partition(group)
            if not part_resampled.empty:
                # Verify session isolation
                if has_session:
                    expected_session = key[0]
                    if (part_resampled["session_id"] != expected_session).any():
                        session_violations += 1
                resampled_parts.append(part_resampled)

        if not resampled_parts:
            empty_df = pd.DataFrame(columns=[
                "grid_timestamp_ms",
                "source_timestamp_ms",
                "source_sequence_id",
                "market_id",
                "asset_id",
                "bid",
                "ask",
                "bid_size",
                "ask_size",
                "event_age_ms",
                "is_stale",
            ])
            report = ResamplingValidationReport(
                input_events=len(df),
                output_grid_rows=0,
            )
            return empty_df, report

        result_df = pd.concat(resampled_parts, ignore_index=True)

        # Deterministic sorting
        sort_fields = ["grid_timestamp_ms"] + partition_cols
        result_df = result_df.sort_values(by=sort_fields, ascending=True).reset_index(drop=True)

        # Explicit typing
        type_dict: Dict[str, str] = {
            "grid_timestamp_ms": "int64",
            "source_timestamp_ms": "int64",
            "source_sequence_id": "int64",
            "market_id": "string",
            "asset_id": "string",
            "bid": "float64",
            "ask": "float64",
            "bid_size": "float64",
            "ask_size": "float64",
            "event_age_ms": "int64",
            "is_stale": "bool",
        }
        if "source_file" in result_df.columns:
            type_dict["source_file"] = "string"
        if "source_line" in result_df.columns:
            type_dict["source_line"] = "int64"
        if "source_record_idx" in result_df.columns:
            type_dict["source_record_idx"] = "int64"
        if has_session:
            type_dict["session_id"] = "string"

        result_df = result_df.astype(type_dict)

        # Metrics for report
        total_rows = len(result_df)
        min_grid = int(result_df["grid_timestamp_ms"].min())
        max_grid = int(result_df["grid_timestamp_ms"].max())
        min_source = int(result_df["source_timestamp_ms"].min())
        max_source = int(result_df["source_timestamp_ms"].max())
        max_age = int(result_df["event_age_ms"].max())
        mean_age = float(result_df["event_age_ms"].mean())
        stale_count = int(result_df["is_stale"].sum())
        stale_pct = (stale_count / total_rows) * 100.0 if total_rows > 0 else 0.0

        # Invariant checks
        future_violations = int((result_df["event_age_ms"] < 0).sum())
        dup_keys = ["grid_timestamp_ms"] + partition_cols
        duplicates = int(result_df.duplicated(subset=dup_keys).sum())

        monotonicity = True
        for _, stream in result_df.groupby(partition_cols):
            if not stream["grid_timestamp_ms"].is_monotonic_increasing:
                monotonicity = False
                break

        report = ResamplingValidationReport(
            input_events=len(df),
            output_grid_rows=total_rows,
            min_grid_timestamp_ms=min_grid,
            max_grid_timestamp_ms=max_grid,
            min_source_timestamp_ms=min_source,
            max_source_timestamp_ms=max_source,
            max_event_age_ms=max_age,
            mean_event_age_ms=mean_age,
            stale_row_count=stale_count,
            stale_percentage=stale_pct,
            future_event_violations=future_violations,
            duplicate_grid_timestamps=duplicates,
            monotonicity=monotonicity,
            session_boundary_violations=session_violations,
        )

        return result_df, report

    def resample_file(
        self,
        input_parquet: Union[Path, str],
        output_parquet: Optional[Union[Path, str]] = None,
    ) -> Tuple[pd.DataFrame, ResamplingValidationReport]:
        """
        Load canonical events from Parquet, resample to 1-second grid,
        and optionally save to output Parquet.
        """
        in_path = Path(input_parquet)
        if not in_path.exists():
            raise FileNotFoundError(f"Input canonical parquet file not found: {in_path}")

        df = pd.read_parquet(in_path)
        resampled_df, report = self.resample_events(df)

        if output_parquet is not None:
            out_path = Path(output_parquet)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            resampled_df.to_parquet(out_path, index=False, engine="pyarrow")
            logger.info(f"Saved {len(resampled_df)} resampled grid rows to {out_path}")

        return resampled_df, report


def main() -> None:
    """CLI entry point for causal 1-second grid resampling."""
    parser = argparse.ArgumentParser(description="Causal Point-In-Time 1-Second Grid Resampler.")
    parser.add_argument("--input", "-i", required=True, help="Path to input canonical events parquet file")
    parser.add_argument("--output", "-o", default=None, help="Path to output resampled 1s parquet file")
    parser.add_argument("--max-stale-ms", type=int, default=5000, help="Max stale threshold in ms (default 5000)")
    parser.add_argument("--report", "-r", default=None, help="Path to save markdown report (optional)")

    args = parser.parse_args()

    resampler = PointInTimeResampler(max_stale_ms=args.max_stale_ms)
    df, report = resampler.resample_file(args.input, output_parquet=args.output)

    print("\n" + report.to_markdown())

    if args.report:
        rep_path = Path(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\nReport written to {rep_path}")


if __name__ == "__main__":
    main()
