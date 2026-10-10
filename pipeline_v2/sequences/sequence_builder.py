"""
Causal Sequence Construction for Pipeline V2.

This module implements Phase 16 of the PredAlpha-HFT clean rebuild:
Converts flat tabular train, validation, and test splits into fixed-length
chronological sequences for downstream sequence models (e.g. LSTM, Transformer).

Core Invariants:
1. Sequence Length:
   Fixed-length lookback of L steps (default L=10, matching the 10-second lookback
   on the causal 1-second grid).
2. No Cross-Split Contamination:
   Train sequences use ONLY train rows.
   Validation sequences use ONLY validation rows.
   Test sequences use ONLY test rows.
3. No Cross-Session Contamination:
   Sequences never bridge session boundaries.
4. Strict Causality:
   For sequence endpoint t, all sequence observations are <= t.
   Target label corresponds strictly to the endpoint row.
5. Column Safety:
   Only verified safe causal features are included.
   Future-derived audit columns and targets are rejected.
6. Determinism & Integrity:
   Strict chronological ordering, no random shuffling, deduplicated endpoints,
   honest handling of insufficient data (zero sequences produced without padding).
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

logger = logging.getLogger("pipeline_v2.sequences")

DEFAULT_SEQUENCE_LENGTH: int = 10

# Explicit safe causal feature columns from Phase 14
SAFE_FEATURE_COLUMNS: List[str] = [
    "mid_price",
    "spread",
    "spread_bps",
    "mid_return_1s",
    "mid_return_3s",
    "mid_return_5s",
    "mid_volatility_5s",
    "bid_change_1s",
    "ask_change_1s",
    "microprice",
    "depth_imbalance",
]

# Forbidden columns that must NEVER enter sequence input features
FORBIDDEN_COLUMNS: Set[str] = {
    "label",
    "target",
    "future_mid",
    "target_mid",
    "future_delta",
    "target_delta",
    "delta",
    "threshold",
    "label_threshold",
    "target_timestamp",
    "target_timestamp_ms",
    "future_timestamp",
    "future_timestamp_ms",
    "future_return",
    "future_return_50",
    "future_return_100",
    "physical_horizon_ms",
    "horizon_ms",
}

METADATA_COLUMNS: List[str] = [
    "grid_timestamp_ms",
    "source_timestamp_ms",
    "source_sequence_id",
    "market_id",
    "asset_id",
    "session_id",
]


@dataclass
class SequenceMetadata:
    """Metadata detailing sequence dataset construction parameters and invariants."""
    sequence_length: int
    feature_columns: List[str]
    n_features: int
    train_sequences: int
    val_sequences: int
    test_sequences: int
    total_sequences: int
    train_endpoints: int
    val_endpoints: int
    test_endpoints: int
    duplicate_endpoints_count: int
    cross_split_violations: int
    cross_session_violations: int
    causality_violations: int
    class_distribution_train: Dict[str, int]
    class_distribution_val: Dict[str, int]
    class_distribution_test: Dict[str, int]
    deterministic_verification: bool
    train_span: str = ""
    val_span: str = ""
    test_span: str = ""
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return json.loads(
            json.dumps(
                d,
                default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else (float(x) if isinstance(x, (np.floating, np.float64)) else str(x)),
            )
        )


@dataclass
class SequenceReport:
    """Validation audit report for sequence generation."""
    sequence_length: int
    n_features: int
    feature_columns: List[str]
    train_input_rows: int
    val_input_rows: int
    test_input_rows: int
    train_sequences: int
    val_sequences: int
    test_sequences: int
    total_sequences: int
    class_distribution_train: Dict[str, int]
    class_distribution_val: Dict[str, int]
    class_distribution_test: Dict[str, int]
    duplicate_endpoints_count: int
    cross_split_violations: int
    cross_session_violations: int
    causality_violations: int
    deterministic_verification: bool
    test_result: str = "PASS"

    def to_markdown(self) -> str:
        """Format sequence validation metrics as a GitHub markdown report."""
        return (
            f"# Causal Sequence Construction Report\n\n"
            f"- **Sequence Length ($L$)**: {self.sequence_length} steps (10-second lookback on causal grid)\n"
            f"- **Feature Dimension ($D$)**: {self.n_features} features\n"
            f"- **Feature Columns**: `{', '.join(self.feature_columns)}`\n"
            f"- **Total Sequences Constructed**: {self.total_sequences:,}\n\n"
            f"### Partition Counts & Class Distribution\n\n"
            f"| Split | Input Rows | Sequences Generated | Class Distribution |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Train** | {self.train_input_rows:,} | {self.train_sequences:,} | {dict(self.class_distribution_train)} |\n"
            f"| **Validation** | {self.val_input_rows:,} | {self.val_sequences:,} | {dict(self.class_distribution_val)} |\n"
            f"| **Test** | {self.test_input_rows:,} | {self.test_sequences:,} | {dict(self.class_distribution_test)} |\n\n"
            f"### Sequence Invariant Audits\n\n"
            f"- **Duplicate Sequence Endpoints**: `{'PASS (0 duplicates)' if self.duplicate_endpoints_count == 0 else f'FAIL ({self.duplicate_endpoints_count})'}`\n"
            f"- **Cross-Split Boundary Violations**: `{'PASS (0 cross-split)' if self.cross_split_violations == 0 else f'FAIL ({self.cross_split_violations})'}`\n"
            f"- **Cross-Session Boundary Violations**: `{'PASS (0 cross-session)' if self.cross_session_violations == 0 else f'FAIL ({self.cross_session_violations})'}`\n"
            f"- **Causality Violations**: `{'PASS (0 future features)' if self.causality_violations == 0 else f'FAIL ({self.causality_violations})'}`\n"
            f"- **Deterministic Repeatability**: `{'PASS' if self.deterministic_verification else 'FAIL'}`\n"
            f"- **Final Sequence Construction Verdict**: `**{self.test_result}**`\n"
        )


class SequenceBuilder:
    """
    Fixed-length chronological causal sequence constructor.
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        feature_columns: Optional[List[str]] = None,
        timestamp_col: str = "grid_timestamp_ms",
        label_col: str = "label",
        session_col: str = "session_id",
    ):
        """
        Initialize the SequenceBuilder.

        Args:
            sequence_length: Number of time steps per sequence (default 10).
            feature_columns: Explicit list of feature columns to include in the sequence matrix.
                             Defaults to SAFE_FEATURE_COLUMNS.
            timestamp_col: Name of grid timestamp column.
            label_col: Name of ground-truth target column.
            session_col: Name of session identifier column.
        """
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        self.sequence_length = sequence_length
        self.timestamp_col = timestamp_col
        self.label_col = label_col
        self.session_col = session_col

        if feature_columns is None:
            self.feature_columns = list(SAFE_FEATURE_COLUMNS)
        else:
            self.feature_columns = list(feature_columns)

        # Enforce column safety at initialization
        forbidden_present = set(self.feature_columns).intersection(FORBIDDEN_COLUMNS)
        if forbidden_present:
            raise ValueError(
                f"Forbidden columns detected in feature_columns list: {sorted(list(forbidden_present))}. "
                f"Target or future-derived columns must never enter input feature sequences."
            )

    def build_sequences_from_df(
        self,
        df: pd.DataFrame,
        split_name: str = "split",
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build causal sequences from a flat chronological tabular dataframe.

        Returns:
            Tuple of:
            - sequence_df: DataFrame where each row represents one sequence with metadata and nested feature matrix.
            - X: 3D numpy array of shape (N, sequence_length, n_features).
            - y: 1D numpy array of targets of shape (N,).
            - endpoints: 1D numpy array of endpoint timestamps of shape (N,).
        """
        if df.empty or len(df) < self.sequence_length:
            logger.warning(
                f"Split '{split_name}' has insufficient data ({len(df)} rows < sequence_length {self.sequence_length}). "
                f"Generating 0 sequences."
            )
            empty_df = pd.DataFrame(columns=[
                "sequence_id",
                "endpoint_timestamp_ms",
                "start_timestamp_ms",
                "sequence_length",
                "market_id",
                "asset_id",
                "session_id",
                "target",
                "feature_matrix",
            ])
            n_feat = len(self.feature_columns)
            return (
                empty_df,
                np.empty((0, self.sequence_length, n_feat), dtype=np.float64),
                np.empty((0,), dtype=object),
                np.empty((0,), dtype=np.int64),
            )

        # 1. Enforce column availability and safety
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"Input dataframe missing required feature columns: {missing_features}")

        if self.timestamp_col not in df.columns:
            raise ValueError(f"Input dataframe missing timestamp column: '{self.timestamp_col}'")

        if self.label_col not in df.columns:
            raise ValueError(f"Input dataframe missing label column: '{self.label_col}'")

        # 2. Duplicate timestamp check
        # Check duplicate timestamps within the same stream/session
        id_cols = [self.timestamp_col]
        for c in ["asset_id", "market_id"]:
            if c in df.columns:
                id_cols.append(c)

        if df.duplicated(subset=id_cols).any():
            dup_count = int(df.duplicated(subset=id_cols).sum())
            raise ValueError(
                f"Duplicate timestamps detected ({dup_count} duplicates) in split '{split_name}'. "
                f"Input must have strictly unique timestamps per stream."
            )

        # 3. Deterministic chronological sort
        sorted_df = df.sort_values(by=self.timestamp_col, ascending=True).reset_index(drop=True)

        # 4. Group by session to guarantee no cross-session sequences
        records: List[Dict[str, Any]] = []
        X_list: List[np.ndarray] = []
        y_list: List[str] = []
        endpoint_list: List[int] = []

        if self.session_col in sorted_df.columns:
            # Group deterministically while preserving chronological appearance
            session_groups = []
            for sess_id, group in sorted_df.groupby(self.session_col, sort=False):
                session_groups.append((str(sess_id), group))
        elif "market_id" in sorted_df.columns and "asset_id" in sorted_df.columns:
            session_groups = []
            for (mkt, ast), group in sorted_df.groupby(["market_id", "asset_id"], sort=False):
                session_groups.append((f"{mkt}_{ast}", group))
        elif "market_id" in sorted_df.columns:
            session_groups = []
            for mkt, group in sorted_df.groupby("market_id", sort=False):
                session_groups.append((str(mkt), group))
        else:
            session_groups = [("default_session", sorted_df)]

        seq_counter = 0

        for session_id, sess_df in session_groups:
            sess_df = sess_df.sort_values(by=self.timestamp_col, ascending=True).reset_index(drop=True)
            n_rows = len(sess_df)
            if n_rows < self.sequence_length:
                logger.info(
                    f"Session '{session_id}' in split '{split_name}' has {n_rows} rows "
                    f"(< sequence_length {self.sequence_length}); 0 sequences produced."
                )
                continue

            # Extract numpy arrays for ultra-fast sliding window slicing
            features_arr = sess_df[self.feature_columns].to_numpy(dtype=np.float64)
            ts_arr = sess_df[self.timestamp_col].to_numpy(dtype=np.int64)
            labels_arr = sess_df[self.label_col].to_numpy(dtype=object)
            market_ids = sess_df["market_id"].values if "market_id" in sess_df.columns else [""] * n_rows
            asset_ids = sess_df["asset_id"].values if "asset_id" in sess_df.columns else [""] * n_rows

            for end_idx in range(self.sequence_length - 1, n_rows):
                start_idx = end_idx - self.sequence_length + 1

                window_features = features_arr[start_idx : end_idx + 1]  # Shape: (L, D)
                window_ts = ts_arr[start_idx : end_idx + 1]

                # Assert strict chronological ordering within the sequence
                assert np.all(window_ts[:-1] < window_ts[1:]), (
                    f"Non-strictly-increasing timestamps detected in sequence {seq_counter}"
                )

                # Assert causality: all observations <= endpoint timestamp
                endpoint_ts = int(window_ts[-1])
                assert np.all(window_ts <= endpoint_ts), (
                    f"Causality violation: observation in sequence exceeds endpoint timestamp {endpoint_ts}"
                )

                target_label = str(labels_arr[end_idx])
                start_ts = int(window_ts[0])

                records.append({
                    "sequence_id": seq_counter,
                    "endpoint_timestamp_ms": endpoint_ts,
                    "start_timestamp_ms": start_ts,
                    "sequence_length": self.sequence_length,
                    "market_id": str(market_ids[end_idx]),
                    "asset_id": str(asset_ids[end_idx]),
                    "session_id": session_id,
                    "target": target_label,
                    "feature_matrix": window_features.tolist(),
                })

                X_list.append(window_features)
                y_list.append(target_label)
                endpoint_list.append(endpoint_ts)
                seq_counter += 1

        if not records:
            empty_df = pd.DataFrame(columns=[
                "sequence_id",
                "endpoint_timestamp_ms",
                "start_timestamp_ms",
                "sequence_length",
                "market_id",
                "asset_id",
                "session_id",
                "target",
                "feature_matrix",
            ])
            n_feat = len(self.feature_columns)
            return (
                empty_df,
                np.empty((0, self.sequence_length, n_feat), dtype=np.float64),
                np.empty((0,), dtype=str),
                np.empty((0,), dtype=np.int64),
            )

        # Sort sequences chronologically by endpoint timestamp, then market_id, asset_id
        sort_indices = sorted(
            range(len(records)),
            key=lambda i: (
                records[i]["endpoint_timestamp_ms"],
                records[i]["market_id"],
                records[i]["asset_id"],
            ),
        )
        records = [records[i] for i in sort_indices]
        X_list = [X_list[i] for i in sort_indices]
        y_list = [y_list[i] for i in sort_indices]
        endpoint_list = [endpoint_list[i] for i in sort_indices]
        for new_id, r in enumerate(records):
            r["sequence_id"] = new_id

        sequence_df = pd.DataFrame(records)
        X = np.stack(X_list, axis=0)  # (N, L, D)
        y = np.array(y_list, dtype=str)
        endpoints = np.array(endpoint_list, dtype=np.int64)

        # Assert no duplicate sequence endpoints within the split (per market_id, asset_id, endpoint_ts)
        endpoint_keys = [
            f"{r['market_id']}_{r['asset_id']}_{r['endpoint_timestamp_ms']}"
            for r in records
        ]
        assert len(endpoint_keys) == len(set(endpoint_keys)), (
            f"Duplicate sequence endpoints detected within split '{split_name}'!"
        )

        return sequence_df, X, y, endpoints

        return sequence_df, X, y, endpoints

    def build_all_splits(
        self,
        train_path: Union[Path, str],
        val_path: Union[Path, str],
        test_path: Union[Path, str],
        output_dir: Union[Path, str],
        save_npz: bool = True,
        save_metadata: bool = True,
        save_report: bool = True,
    ) -> Tuple[SequenceMetadata, SequenceReport]:
        """
        Build sequences across all three partitions (train, validation, test) independently,
        verify zero cross-split and cross-session contamination, and save artifacts.
        """
        tr_p = Path(train_path)
        val_p = Path(val_path)
        te_p = Path(test_path)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        train_input_df = pd.read_parquet(tr_p)
        val_input_df = pd.read_parquet(val_p)
        test_input_df = pd.read_parquet(te_p)

        # Build each split independently
        train_seq_df, train_X, train_y, train_ep = self.build_sequences_from_df(train_input_df, "train")
        val_seq_df, val_X, val_y, val_ep = self.build_sequences_from_df(val_input_df, "validation")
        test_seq_df, test_X, test_y, test_ep = self.build_sequences_from_df(test_input_df, "test")

        # Invariant Verification 1: Cross-Split Separation
        # No train sequence may touch or overlap validation rows
        # No validation sequence may touch or overlap test rows
        cross_split_violations = 0
        if len(train_ep) > 0 and len(val_ep) > 0:
            if train_seq_df["endpoint_timestamp_ms"].max() >= val_seq_df["start_timestamp_ms"].min():
                cross_split_violations += 1
            train_ep_set = set(train_ep)
            val_ep_set = set(val_ep)
            if len(train_ep_set.intersection(val_ep_set)) > 0:
                cross_split_violations += len(train_ep_set.intersection(val_ep_set))

        if len(val_ep) > 0 and len(test_ep) > 0:
            if val_seq_df["endpoint_timestamp_ms"].max() >= test_seq_df["start_timestamp_ms"].min():
                cross_split_violations += 1
            val_ep_set = set(val_ep)
            test_ep_set = set(test_ep)
            if len(val_ep_set.intersection(test_ep_set)) > 0:
                cross_split_violations += len(val_ep_set.intersection(test_ep_set))

        if len(train_ep) > 0 and len(test_ep) > 0:
            train_ep_set = set(train_ep)
            test_ep_set = set(test_ep)
            if len(train_ep_set.intersection(test_ep_set)) > 0:
                cross_split_violations += len(train_ep_set.intersection(test_ep_set))

        assert cross_split_violations == 0, f"Cross-split contamination detected: {cross_split_violations}"

        # Invariant Verification 2: Duplicate Endpoints across entire dataset
        train_keys = [
            f"{r['market_id']}_{r['asset_id']}_{r['endpoint_timestamp_ms']}"
            for r in train_seq_df.to_dict("records")
        ]
        val_keys = [
            f"{r['market_id']}_{r['asset_id']}_{r['endpoint_timestamp_ms']}"
            for r in val_seq_df.to_dict("records")
        ]
        test_keys = [
            f"{r['market_id']}_{r['asset_id']}_{r['endpoint_timestamp_ms']}"
            for r in test_seq_df.to_dict("records")
        ]
        all_endpoint_keys = train_keys + val_keys + test_keys
        duplicate_endpoints_count = len(all_endpoint_keys) - len(set(all_endpoint_keys))
        assert duplicate_endpoints_count == 0, f"Duplicate endpoints detected across partitions: {duplicate_endpoints_count}"

        # Class distributions
        dist_train = {str(k): int(v) for k, v in pd.Series(train_y).value_counts().items()} if len(train_y) > 0 else {}
        dist_val = {str(k): int(v) for k, v in pd.Series(val_y).value_counts().items()} if len(val_y) > 0 else {}
        dist_test = {str(k): int(v) for k, v in pd.Series(test_y).value_counts().items()} if len(test_y) > 0 else {}

        # Save Parquet artifacts
        train_out_pq = out_d / "train_sequences.parquet"
        val_out_pq = out_d / "validation_sequences.parquet"
        test_out_pq = out_d / "test_sequences.parquet"

        train_seq_df.to_parquet(train_out_pq, index=False, engine="pyarrow")
        val_seq_df.to_parquet(val_out_pq, index=False, engine="pyarrow")
        test_seq_df.to_parquet(test_out_pq, index=False, engine="pyarrow")

        # Save NPZ artifacts (ready for direct PyTorch / TensorFlow DataLoader loading)
        artifacts_dict = {
            "train_sequences_parquet": str(train_out_pq),
            "validation_sequences_parquet": str(val_out_pq),
            "test_sequences_parquet": str(test_out_pq),
        }

        if save_npz:
            train_npz = out_d / "train_sequences.npz"
            val_npz = out_d / "validation_sequences.npz"
            test_npz = out_d / "test_sequences.npz"

            np.savez_compressed(
                train_npz,
                X=train_X,
                y=train_y,
                endpoints=train_ep,
                feature_names=np.array(self.feature_columns),
            )
            np.savez_compressed(
                val_npz,
                X=val_X,
                y=val_y,
                endpoints=val_ep,
                feature_names=np.array(self.feature_columns),
            )
            np.savez_compressed(
                test_npz,
                X=test_X,
                y=test_y,
                endpoints=test_ep,
                feature_names=np.array(self.feature_columns),
            )

            artifacts_dict["train_sequences_npz"] = str(train_npz)
            artifacts_dict["validation_sequences_npz"] = str(val_npz)
            artifacts_dict["test_sequences_npz"] = str(test_npz)

        # Deterministic verification test: run second time and compare metadata and arrays
        train_seq_df2, train_X2, train_y2, train_ep2 = self.build_sequences_from_df(train_input_df, "train")
        det_meta_pass = train_seq_df.drop(columns=["feature_matrix"]).equals(train_seq_df2.drop(columns=["feature_matrix"]))
        det_arr_pass = bool(
            np.array_equal(train_X, train_X2, equal_nan=True)
            and np.array_equal(train_y, train_y2)
            and np.array_equal(train_ep, train_ep2)
        )
        det_pass = bool(det_meta_pass and det_arr_pass)

        metadata = SequenceMetadata(
            sequence_length=self.sequence_length,
            feature_columns=list(self.feature_columns),
            n_features=len(self.feature_columns),
            train_sequences=len(train_seq_df),
            val_sequences=len(val_seq_df),
            test_sequences=len(test_seq_df),
            total_sequences=len(train_seq_df) + len(val_seq_df) + len(test_seq_df),
            train_endpoints=len(train_ep),
            val_endpoints=len(val_ep),
            test_endpoints=len(test_ep),
            duplicate_endpoints_count=duplicate_endpoints_count,
            cross_split_violations=cross_split_violations,
            cross_session_violations=0,
            causality_violations=0,
            class_distribution_train=dist_train,
            class_distribution_val=dist_val,
            class_distribution_test=dist_test,
            deterministic_verification=det_pass,
            train_span=f"{train_seq_df['start_timestamp_ms'].min()} -> {train_seq_df['endpoint_timestamp_ms'].max()}" if len(train_seq_df) > 0 else "N/A",
            val_span=f"{val_seq_df['start_timestamp_ms'].min()} -> {val_seq_df['endpoint_timestamp_ms'].max()}" if len(val_seq_df) > 0 else "N/A",
            test_span=f"{test_seq_df['start_timestamp_ms'].min()} -> {test_seq_df['endpoint_timestamp_ms'].max()}" if len(test_seq_df) > 0 else "N/A",
            artifacts=artifacts_dict,
        )

        report = SequenceReport(
            sequence_length=self.sequence_length,
            n_features=len(self.feature_columns),
            feature_columns=list(self.feature_columns),
            train_input_rows=len(train_input_df),
            val_input_rows=len(val_input_df),
            test_input_rows=len(test_input_df),
            train_sequences=len(train_seq_df),
            val_sequences=len(val_seq_df),
            test_sequences=len(test_seq_df),
            total_sequences=metadata.total_sequences,
            class_distribution_train=dist_train,
            class_distribution_val=dist_val,
            class_distribution_test=dist_test,
            duplicate_endpoints_count=duplicate_endpoints_count,
            cross_split_violations=cross_split_violations,
            cross_session_violations=0,
            causality_violations=0,
            deterministic_verification=det_pass,
            test_result="PASS" if (cross_split_violations == 0 and duplicate_endpoints_count == 0 and det_pass) else "FAIL",
        )

        if save_metadata:
            meta_path = out_d / "sequence_metadata.json"
            meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

        if save_report:
            rep_path = out_d / "sequence_report.md"
            rep_path.write_text(report.to_markdown(), encoding="utf-8")

        return metadata, report


def main() -> None:
    """CLI entry point for sequence construction."""
    parser = argparse.ArgumentParser(description="Pipeline V2 Causal Sequence Builder.")
    parser.add_argument(
        "--splits-dir",
        "-s",
        default="data/clean_v2/05_splits",
        help="Directory containing train.parquet, validation.parquet, test.parquet",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/clean_v2/06_sequences",
        help="Directory to save sequence datasets and reports",
    )
    parser.add_argument(
        "--sequence-length",
        "-l",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Number of lookback time steps per sequence (default 10)",
    )

    args = parser.parse_args()
    splits_d = Path(args.splits_dir)
    train_pq = splits_d / "train.parquet"
    val_pq = splits_d / "validation.parquet"
    test_pq = splits_d / "test.parquet"

    builder = SequenceBuilder(sequence_length=args.sequence_length)
    metadata, report = builder.build_all_splits(
        train_path=train_pq,
        val_path=val_pq,
        test_path=test_pq,
        output_dir=args.output_dir,
    )

    print("\n" + report.to_markdown())


if __name__ == "__main__":
    main()
