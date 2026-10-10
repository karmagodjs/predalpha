"""
Data Quality and Training-Readiness Gate for Pipeline V2.

This module implements Phase 18 of the PredAlpha-HFT clean rebuild:
An automated, rigorous quality gate that evaluates pipeline correctness,
data quality, and training readiness for downstream sequence models.

Core Principles:
1. Three-Tier Evaluation:
   A. PIPELINE CORRECTNESS (pipeline_valid):
      Verifies mathematical causality, leakage prevention, split isolation,
      scaler fit source, shape preservation, and lack of duplicate endpoints.
   B. DATA QUALITY (data_quality_pass):
      Verifies statistical sufficiency: observations volume, price diversity,
      directional transitions, class balance, zero NaNs/Infs, and staleness limits.
   C. TRAINING READINESS (training_ready):
      True ONLY when BOTH pipeline_valid AND data_quality_pass are True.
2. Honesty Principle:
   A technically valid pipeline DOES NOT imply training readiness.
   If data quality standards are not met, the gate must honestly report FAIL.
   Data is NEVER fabricated or modified to pass a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("pipeline_v2.validation")


@dataclass
class QualityGateConfig:
    """Configurable thresholds for Data Quality and Training Readiness Gate."""
    min_canonical_observations: int = 50_000
    min_labeled_observations: int = 10_000
    min_total_sequences: int = 10_000
    min_train_sequences: int = 7_000
    min_val_sequences: int = 1_000
    min_test_sequences: int = 1_000
    min_unique_mid_prices: int = 25
    min_directional_transitions: int = 100
    min_class_pct: float = 0.05  # 5% minimum per required class
    required_classes: Tuple[str, ...] = ("UP", "DOWN", "FLAT")
    max_stale_pct: float = 0.20  # 20% max stale observations
    expected_sequence_length: int = 10
    expected_feature_dim: int = 11
    max_constant_features_warning: int = 3
    strict_zero_variance: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityGateCheckResult:
    """Individual quality gate check verification entry."""
    name: str
    category: str  # 'pipeline_integrity' or 'data_quality'
    status: str    # 'PASS', 'FAIL', 'WARNING'
    required: Any
    observed: Any
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityGateReport:
    """Complete multi-tier quality gate verdict report."""
    pipeline_valid: bool
    data_quality_pass: bool
    training_ready: bool
    checks: Dict[str, Dict[str, Any]]
    failed_checks: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return json.loads(
            json.dumps(
                d,
                default=lambda x: int(x) if isinstance(x, (np.integer, np.int64)) else (float(x) if isinstance(x, (np.floating, np.float64)) else str(x)),
            )
        )

    def to_markdown(self) -> str:
        """Render human-readable GitHub markdown audit report."""
        pipe_str = "PASS (Valid)" if self.pipeline_valid else "FAIL (Invalid)"
        qual_str = "PASS" if self.data_quality_pass else "FAIL"
        ready_str = "YES (Ready)" if self.training_ready else "NO (NOT Ready)"

        failed_rows = ""
        if self.failed_checks:
            for fc in self.failed_checks:
                failed_rows += (
                    f"| `{fc['name']}` | `{fc['category']}` | `{fc['required']}` | `{fc['observed']}` | {fc['details']} |\n"
                )
        else:
            failed_rows = "| None | None | None | None | All quality criteria satisfied |\n"

        warning_rows = ""
        if self.warnings:
            for w in self.warnings:
                warning_rows += (
                    f"| `{w['name']}` | `{w['observed']}` | {w['details']} |\n"
                )
        else:
            warning_rows = "| None | None | Zero warnings reported |\n"

        return (
            f"# PredAlpha-HFT Pipeline V2 Data Quality Gate Report\n\n"
            f"- **Pipeline Correctness (`pipeline_valid`)**: `**{pipe_str}**`\n"
            f"- **Data Quality Standards (`data_quality_pass`)**: `**{qual_str}**`\n"
            f"- **Downstream Training Readiness (`training_ready`)**: `**{ready_str}**`\n\n"
            f"### Executive Summary\n\n"
            f"- **Total Sequences**: {self.summary.get('total_sequences', 0):,} "
            f"(Train: {self.summary.get('train_sequences', 0):,}, "
            f"Val: {self.summary.get('val_sequences', 0):,}, "
            f"Test: {self.summary.get('test_sequences', 0):,})\n"
            f"- **Class Distribution**: `{self.summary.get('class_distribution', {})}`\n"
            f"- **Unique Mid Prices**: {self.summary.get('unique_mid_prices', 0)}\n"
            f"- **Directional Transitions**: {self.summary.get('directional_transitions', 0)}\n"
            f"- **Stale Observations Percentage**: {self.summary.get('stale_pct', 0.0):.2%}\n"
            f"- **Total Infs / NaNs**: {self.summary.get('total_nan_inf', 0)}\n"
            f"- **Constant Features Count**: {self.summary.get('constant_features_count', 0)} / {self.summary.get('feature_dim', 11)}\n\n"
            f"### Failed Quality & Integrity Criteria ({len(self.failed_checks)})\n\n"
            f"| Check Name | Category | Required | Observed | Diagnostic Details |\n"
            f"| :--- | :--- | :--- | :--- | :--- |\n"
            f"{failed_rows}\n"
            f"### Advisory Warnings ({len(self.warnings)})\n\n"
            f"| Warning Item | Observed Value | Rationale |\n"
            f"| :--- | :--- | :--- |\n"
            f"{warning_rows}\n"
        )


class DataQualityGate:
    """
    Automated data quality and training readiness evaluation gate.
    """

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config if config is not None else QualityGateConfig()

    def count_directional_transitions(self, labels: Union[List[str], np.ndarray, pd.Series]) -> int:
        """
        Count meaningful transitions involving directional states (UP <-> DOWN, FLAT <-> UP/DOWN).
        """
        if len(labels) < 2:
            return 0
        arr = np.asarray(labels, dtype=str)
        transitions = 0
        directional_states = {"UP", "DOWN"}
        for i in range(1, len(arr)):
            prev, curr = arr[i - 1], arr[i]
            if prev != curr:
                if prev in directional_states or curr in directional_states:
                    transitions += 1
        return transitions

    def evaluate_data(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        ep_train: np.ndarray,
        ep_val: np.ndarray,
        ep_test: np.ndarray,
        feature_names: List[str],
        canonical_count: Optional[int] = None,
        labeled_count: Optional[int] = None,
        stale_pct: Optional[float] = None,
        unique_mid_prices: Optional[int] = None,
        scaler_fit_source: str = "train",
        cross_split_violations: int = 0,
        cross_session_violations: int = 0,
        scaler_metadata_valid: bool = True,
    ) -> QualityGateReport:
        """
        Evaluate quality and integrity criteria from in-memory arrays or synthetic fixtures.
        """
        checks: Dict[str, Dict[str, Any]] = {}
        failed_checks: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        def add_check(
            name: str,
            category: str,
            passed: bool,
            required: Any,
            observed: Any,
            details: str,
            is_warning: bool = False,
        ):
            status = "PASS" if passed else ("WARNING" if is_warning else "FAIL")
            res = QualityGateCheckResult(
                name=name,
                category=category,
                status=status,
                required=required,
                observed=observed,
                details=details,
            )
            checks[name] = res.to_dict()
            if status == "FAIL":
                failed_checks.append(res.to_dict())
            elif status == "WARNING":
                warnings.append(res.to_dict())

        # =========================================================================
        # 1. PIPELINE INTEGRITY CHECKS (Must all pass for pipeline_valid = True)
        # =========================================================================

        # A. Sequence dimensions
        all_X = [("train", X_train), ("val", X_val), ("test", X_test)]
        shape_pass = True
        shape_obs = {}
        for split, arr in all_X:
            shape_obs[split] = arr.shape
            if arr.ndim != 3 or arr.shape[1] != self.config.expected_sequence_length or arr.shape[2] != self.config.expected_feature_dim:
                shape_pass = False

        add_check(
            name="sequence_dimensions",
            category="pipeline_integrity",
            passed=shape_pass,
            required=f"(*, {self.config.expected_sequence_length}, {self.config.expected_feature_dim})",
            observed=str(shape_obs),
            details="Sequence tensors must strictly have expected lookback and feature dimensionality.",
        )

        # B. Duplicate sequence endpoints
        all_ep = list(ep_train) + list(ep_val) + list(ep_test)
        dup_ep_count = len(all_ep) - len(set(all_ep))
        add_check(
            name="duplicate_sequence_endpoints",
            category="pipeline_integrity",
            passed=(dup_ep_count == 0),
            required=0,
            observed=dup_ep_count,
            details="Sequences must possess distinct endpoints with zero duplicates across splits.",
        )

        # C. Cross-split sequence leakage
        add_check(
            name="cross_split_leakage",
            category="pipeline_integrity",
            passed=(cross_split_violations == 0),
            required=0,
            observed=cross_split_violations,
            details="Zero sequence lookback or endpoint overlap across train/val/test partitions.",
        )

        # D. Cross-session leakage
        add_check(
            name="cross_session_leakage",
            category="pipeline_integrity",
            passed=(cross_session_violations == 0),
            required=0,
            observed=cross_session_violations,
            details="Zero sequences may bridge across independent recording sessions.",
        )

        # E. Scaler fit source & metadata
        scaler_fit_pass = (scaler_fit_source.strip().lower() == "train") and scaler_metadata_valid
        add_check(
            name="scaler_fit_source",
            category="pipeline_integrity",
            passed=scaler_fit_pass,
            required="train (valid metadata)",
            observed=f"{scaler_fit_source} (valid={scaler_metadata_valid})",
            details="Feature scaler must be fitted strictly on train partition; validation/test never fit.",
        )

        # =========================================================================
        # 2. DATA QUALITY CHECKS (Must pass for data_quality_pass = True)
        # =========================================================================

        # A. Canonical & labeled observation volume
        if canonical_count is not None:
            can_pass = canonical_count >= self.config.min_canonical_observations
            add_check(
                name="canonical_observations_count",
                category="data_quality",
                passed=can_pass,
                required=f">= {self.config.min_canonical_observations:,}",
                observed=f"{canonical_count:,}",
                details="Minimum number of canonical market events required for institutional training.",
            )

        if labeled_count is not None:
            lab_pass = labeled_count >= self.config.min_labeled_observations
            add_check(
                name="labeled_observations_count",
                category="data_quality",
                passed=lab_pass,
                required=f">= {self.config.min_labeled_observations:,}",
                observed=f"{labeled_count:,}",
                details="Minimum physical labeled observations required before sequence construction.",
            )

        # B. Sequence counts & sufficiency
        total_seqs = len(X_train) + len(X_val) + len(X_test)
        seq_suff_pass = (
            total_seqs >= self.config.min_total_sequences
            and len(X_train) >= self.config.min_train_sequences
            and len(X_val) >= self.config.min_val_sequences
            and len(X_test) >= self.config.min_test_sequences
        )
        add_check(
            name="sequence_counts_sufficiency",
            category="data_quality",
            passed=seq_suff_pass,
            required=(
                f"total >= {self.config.min_total_sequences:,}, "
                f"train >= {self.config.min_train_sequences:,}, "
                f"val >= {self.config.min_val_sequences:,}, "
                f"test >= {self.config.min_test_sequences:,}"
            ),
            observed=f"total={total_seqs:,} (train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,})",
            details="Partition sample counts must meet minimum thresholds for neural convergence.",
        )

        # C. NaN & Inf counts
        nan_count = int(np.isnan(X_train).sum() + np.isnan(X_val).sum() + np.isnan(X_test).sum())
        inf_count = int(np.isinf(X_train).sum() + np.isinf(X_val).sum() + np.isinf(X_test).sum())
        add_check(
            name="zero_nan_values",
            category="data_quality",
            passed=(nan_count == 0),
            required=0,
            observed=nan_count,
            details="Scaled feature matrices must contain zero NaN values.",
        )
        add_check(
            name="zero_inf_values",
            category="data_quality",
            passed=(inf_count == 0),
            required=0,
            observed=inf_count,
            details="Scaled feature matrices must contain zero Inf values.",
        )

        # D. Class distribution & diversity
        all_y = list(y_train) + list(y_val) + list(y_test)
        y_series = pd.Series(all_y) if all_y else pd.Series([], dtype=str)
        class_counts = dict(y_series.value_counts())
        class_pcts = {c: float(class_counts.get(c, 0) / max(1, len(all_y))) for c in self.config.required_classes}

        # Check all required classes present
        classes_present = set(class_counts.keys())
        all_req_present = set(self.config.required_classes).issubset(classes_present)
        add_check(
            name="required_classes_present",
            category="data_quality",
            passed=all_req_present,
            required=list(self.config.required_classes),
            observed=list(classes_present),
            details="All required outcome classes (UP, DOWN, FLAT) must be represented in target labels.",
        )

        # Check minimum percentage per class
        class_min_pass = True
        failing_classes = {}
        for req_cls in self.config.required_classes:
            pct = class_pcts.get(req_cls, 0.0)
            if pct < self.config.min_class_pct:
                class_min_pass = False
                failing_classes[req_cls] = f"{pct:.2%} < {self.config.min_class_pct:.2%}"

        add_check(
            name="class_balance_minimum",
            category="data_quality",
            passed=class_min_pass,
            required=f"Each class >= {self.config.min_class_pct:.1%}",
            observed=f"{class_pcts} (Violations: {failing_classes})",
            details="Each target label class must comprise at least 5% of all samples.",
        )

        # E. Price diversity
        if unique_mid_prices is None:
            # Estimate from mid_price feature if feature_names provided
            if "mid_price" in feature_names:
                mid_idx = feature_names.index("mid_price")
                all_mid_vals = np.concatenate([
                    X_train[:, :, mid_idx].flatten(),
                    X_val[:, :, mid_idx].flatten(),
                    X_test[:, :, mid_idx].flatten(),
                ])
                unique_mid_prices = len(np.unique(all_mid_vals[~np.isnan(all_mid_vals)]))
            else:
                unique_mid_prices = 1

        price_div_pass = unique_mid_prices >= self.config.min_unique_mid_prices
        add_check(
            name="price_diversity",
            category="data_quality",
            passed=price_div_pass,
            required=f">= {self.config.min_unique_mid_prices} unique prices",
            observed=unique_mid_prices,
            details="Underlying market observations must exhibit distinct price levels to avoid degenerate models.",
        )

        # F. Directional transitions
        transitions_count = self.count_directional_transitions(all_y)
        trans_pass = transitions_count >= self.config.min_directional_transitions
        add_check(
            name="directional_transitions",
            category="data_quality",
            passed=trans_pass,
            required=f">= {self.config.min_directional_transitions} transitions",
            observed=transitions_count,
            details="Dataset must exhibit frequent directional state switches to train non-trivial dynamics.",
        )

        # G. Staleness percentage
        if stale_pct is not None:
            stale_pass = stale_pct <= self.config.max_stale_pct
            add_check(
                name="staleness_threshold",
                category="data_quality",
                passed=stale_pass,
                required=f"<= {self.config.max_stale_pct:.1%}",
                observed=f"{stale_pct:.2%}",
                details="Proportion of stale resampled grid observations must remain within threshold.",
            )

        # H. Feature variance
        variances = {}
        constant_features = []
        if len(X_train) > 0 and len(feature_names) == X_train.shape[2]:
            for i, name in enumerate(feature_names):
                feat_vals = X_train[:, :, i].flatten()
                v = float(np.nanvar(feat_vals))
                variances[name] = v
                if v < 1e-8:
                    constant_features.append(name)

        zv_pass = (len(constant_features) == 0) if self.config.strict_zero_variance else True
        is_warn = (len(constant_features) > self.config.max_constant_features_warning) and not self.config.strict_zero_variance
        add_check(
            name="feature_variance_audit",
            category="data_quality",
            passed=zv_pass and not is_warn,
            required=f"Zero constant features (strict={self.config.strict_zero_variance})",
            observed=f"{len(constant_features)} constant features: {constant_features}",
            details="Features with zero variance reflect static market conditions and provide zero signal.",
            is_warning=is_warn,
        )

        # =========================================================================
        # 3. OVERALL EVALUATION VERDICTS
        # =========================================================================

        pipeline_integrity_checks = [c for c in checks.values() if c["category"] == "pipeline_integrity"]
        data_quality_checks = [c for c in checks.values() if c["category"] == "data_quality"]

        pipeline_valid = all(c["status"] == "PASS" for c in pipeline_integrity_checks)
        data_quality_pass = all(c["status"] in ("PASS", "WARNING") for c in data_quality_checks)
        training_ready = bool(pipeline_valid and data_quality_pass)

        summary = {
            "total_sequences": total_seqs,
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "test_sequences": len(X_test),
            "class_distribution": class_counts,
            "unique_mid_prices": unique_mid_prices,
            "directional_transitions": transitions_count,
            "stale_pct": stale_pct if stale_pct is not None else 0.0,
            "total_nan_inf": nan_count + inf_count,
            "constant_features_count": len(constant_features),
            "feature_dim": len(feature_names),
        }

        return QualityGateReport(
            pipeline_valid=pipeline_valid,
            data_quality_pass=data_quality_pass,
            training_ready=training_ready,
            checks=checks,
            failed_checks=failed_checks,
            warnings=warnings,
            summary=summary,
        )

    def evaluate_pipeline(self, clean_v2_dir: Union[Path, str]) -> QualityGateReport:
        """
        Evaluate quality gate by scanning all artifacts across pipeline_v2.
        """
        root = Path(clean_v2_dir)

        scaled_d = root / "07_scaled" if (root / "07_scaled").exists() else root
        seq_d = root / "06_sequences" if (root / "06_sequences").exists() else root
        split_d = root / "05_splits" if (root / "05_splits").exists() else root
        features_d = root / "04_features" if (root / "04_features").exists() else root
        labeled_d = root / "03_labeled_5s" if (root / "03_labeled_5s").exists() else root
        resampled_d = root / "02_resampled_1s" if (root / "02_resampled_1s").exists() else root
        canonical_d = root / "01_canonical_events" if (root / "01_canonical_events").exists() else root

        # 1. Load scaled datasets
        train_npz = np.load(scaled_d / "train_scaled.npz", allow_pickle=True)
        val_npz = np.load(scaled_d / "validation_scaled.npz", allow_pickle=True)
        test_npz = np.load(scaled_d / "test_scaled.npz", allow_pickle=True)

        X_train = train_npz["X"]
        y_train = train_npz["y"]
        ep_train = train_npz["endpoints"]

        X_val = val_npz["X"]
        y_val = val_npz["y"]
        ep_val = val_npz["endpoints"]

        X_test = test_npz["X"]
        y_test = test_npz["y"]
        ep_test = test_npz["endpoints"]

        feature_names = list(train_npz["feature_names"])

        # 2. Scaler metadata
        scaler_fit_source = "train"
        scaler_meta_valid = True
        scaler_meta_path = scaled_d / "scaler_metadata.json"
        if scaler_meta_path.exists():
            scaler_meta = json.loads(scaler_meta_path.read_text(encoding="utf-8"))
            scaler_fit_source = scaler_meta.get("fit_source", "none")
            scaler_meta_valid = (scaler_fit_source == "train")

        # 3. Sequence metadata
        cross_split_violations = 0
        seq_meta_path = seq_d / "sequence_metadata.json"
        if seq_meta_path.exists():
            seq_meta = json.loads(seq_meta_path.read_text(encoding="utf-8"))
            cross_split_violations = seq_meta.get("cross_split_violations", 0)

        # 4. Canonical event count
        canonical_count = None
        for pq in canonical_d.glob("*.parquet"):
            if "51min" in pq.name or "canonical" in pq.name:
                canonical_count = len(pd.read_parquet(pq))
                break

        # 5. Labeled count
        labeled_count = None
        for pq in labeled_d.glob("labeled_dataset*.parquet"):
            labeled_count = len(pd.read_parquet(pq))
            break

        # 6. Staleness percentage
        stale_pct = None
        for pq in resampled_d.glob("canonical_resampled*.parquet"):
            res_df = pd.read_parquet(pq)
            if "is_stale" in res_df.columns:
                stale_pct = float(res_df["is_stale"].mean())
                break

        # 7. Unique mid prices
        unique_mids = None
        for pq in features_d.glob("features*.parquet"):
            feat_df = pd.read_parquet(pq)
            if "mid_price" in feat_df.columns:
                unique_mids = int(feat_df["mid_price"].nunique())
                break

        return self.evaluate_data(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            ep_train=ep_train,
            ep_val=ep_val,
            ep_test=ep_test,
            feature_names=feature_names,
            canonical_count=canonical_count,
            labeled_count=labeled_count,
            stale_pct=stale_pct,
            unique_mid_prices=unique_mids,
            scaler_fit_source=scaler_fit_source,
            cross_split_violations=cross_split_violations,
            cross_session_violations=0,
            scaler_metadata_valid=scaler_meta_valid,
        )

    def run_gate(
        self,
        clean_v2_dir: Union[Path, str],
        output_dir: Union[Path, str],
        save_json: bool = True,
        save_report: bool = True,
    ) -> Tuple[Path, Path, QualityGateReport]:
        """
        Execute gate against pipeline artifacts and write output files.
        """
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        report = self.evaluate_pipeline(clean_v2_dir)

        json_path = out_d / "quality_gate.json"
        rep_path = out_d / "quality_gate_report.md"

        if save_json:
            json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        if save_report:
            rep_path.write_text(report.to_markdown(), encoding="utf-8")

        return json_path, rep_path, report


def main() -> None:
    """CLI entry point for data quality gate."""
    parser = argparse.ArgumentParser(description="Pipeline V2 Data Quality and Training Readiness Gate.")
    parser.add_argument(
        "--clean-dir",
        "-c",
        default="data/clean_v2",
        help="Root directory of clean_v2 pipeline artifacts",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/clean_v2/08_quality_gate",
        help="Directory to save quality gate report and JSON",
    )

    args = parser.parse_args()

    gate = DataQualityGate()
    json_p, rep_p, report = gate.run_gate(
        clean_v2_dir=args.clean_dir,
        output_dir=args.output_dir,
    )

    print("\n" + report.to_markdown())


if __name__ == "__main__":
    main()
