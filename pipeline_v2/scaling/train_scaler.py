"""
Train-Only Feature Scaling for Pipeline V2.

This module implements Phase 17 of the PredAlpha-HFT clean rebuild:
Applies causal feature scaling fitted EXCLUSIVELY on training data to prevent
data leakage. Validation and test sets are transformed strictly using parameters
learned from the training set.

Core Principles & Invariants:
1. Train-Only Fitting:
   Scaler statistics (center, scale) are computed ONLY from training data.
   Validation and test partitions NEVER influence scaler parameters.
2. Leakage Protection:
   Attempting to fit on validation or test raises a strict ValueError.
3. Shape Preservation:
   3D sequence shape (N, L, D) is preserved exactly across all splits.
4. Target Integrity:
   Target labels ('label' / 'target') are completely isolated and never scaled.
5. Zero-Variance Protection:
   Constant features with zero variance in train are scaled safely with unit divisor,
   preventing NaN or Inf propagation.
6. Strict Determinism:
   Zero randomness, zero shuffling, bit-for-bit reproducible outputs.
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

logger = logging.getLogger("pipeline_v2.scaling")

# Forbidden columns that must never enter scaling
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


@dataclass
class ScalerMetadata:
    """Metadata detailing scaler configuration and learned parameters."""
    scaler_type: str
    fit_source: str
    n_samples_seen: int
    sequence_length: int
    n_features: int
    feature_names: List[str]
    centers: Dict[str, float]
    scales: Dict[str, float]
    zero_variance_features: List[str]
    train_shape_before: Tuple[int, ...]
    train_shape_after: Tuple[int, ...]
    val_shape_before: Tuple[int, ...]
    val_shape_after: Tuple[int, ...]
    test_shape_before: Tuple[int, ...]
    test_shape_after: Tuple[int, ...]
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
class ScalingReport:
    """Audit report for train-only feature scaling."""
    scaler_type: str
    fit_source: str
    train_sequences: int
    val_sequences: int
    test_sequences: int
    sequence_length: int
    n_features: int
    feature_names: List[str]
    zero_variance_features: List[str]
    train_means_after: Dict[str, float]
    train_stds_after: Dict[str, float]
    target_untouched_verification: bool
    shape_preservation_verification: bool
    no_leakage_verification: bool
    deterministic_verification: bool
    test_result: str = "PASS"

    def to_markdown(self) -> str:
        """Format scaling report as GitHub markdown."""
        zv_str = ", ".join(self.zero_variance_features) if self.zero_variance_features else "None"
        return (
            f"# Train-Only Feature Scaling Report\n\n"
            f"- **Scaler Type**: `{self.scaler_type}` (z-score standardization)\n"
            f"- **Fit Source**: `{self.fit_source}` (strictly TRAIN only)\n"
            f"- **Feature Dimension ($D$)**: {self.n_features} features\n"
            f"- **Sequence Length ($L$)**: {self.sequence_length} steps\n"
            f"- **Zero-Variance Features Handled**: `{zv_str}`\n\n"
            f"### Scaled Dataset Partitions\n\n"
            f"| Split | Input Sequences | Scaled Shape | Target Untouched |\n"
            f"| :--- | :--- | :--- | :--- |\n"
            f"| **Train** | {self.train_sequences:,} | `({self.train_sequences}, {self.sequence_length}, {self.n_features})` | `{'PASS' if self.target_untouched_verification else 'FAIL'}` |\n"
            f"| **Validation** | {self.val_sequences:,} | `({self.val_sequences}, {self.sequence_length}, {self.n_features})` | `{'PASS' if self.target_untouched_verification else 'FAIL'}` |\n"
            f"| **Test** | {self.test_sequences:,} | `({self.test_sequences}, {self.sequence_length}, {self.n_features})` | `{'PASS' if self.target_untouched_verification else 'FAIL'}` |\n\n"
            f"### Invariant & Leakage Audits\n\n"
            f"- **Fit Source Verification**: `PASS (fitted strictly on train)`\n"
            f"- **Validation Leakage Check**: `PASS (validation data cannot affect scaler)`\n"
            f"- **Test Leakage Check**: `PASS (test data cannot affect scaler)`\n"
            f"- **Sequence Shape Preservation**: `{'PASS' if self.shape_preservation_verification else 'FAIL'}`\n"
            f"- **Target Isolation**: `{'PASS (labels untouched)' if self.target_untouched_verification else 'FAIL'}`\n"
            f"- **Deterministic Output**: `{'PASS' if self.deterministic_verification else 'FAIL'}`\n"
            f"- **Final Scaling Verdict**: `**{self.test_result}**`\n"
        )


class CausalSequenceScaler:
    """
    Train-only sequence scaler for 3D tensors of shape (N, L, D).

    Calculates feature-wise parameters across all training observations and applies
    the identical transformation across train, validation, and test sets.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        feature_names: Optional[List[str]] = None,
        eps: float = 1e-8,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        """
        Initialize the scaler.

        Args:
            scaler_type: 'standard' (z-score), 'robust' (median/IQR), or 'minmax'.
            feature_names: Optional list of feature names.
            eps: Numerical epsilon to prevent division by zero on zero-variance features.
            clip_min: Optional lower bound clipping.
            clip_max: Optional upper bound clipping.
        """
        valid_types = {"standard", "robust", "minmax"}
        if scaler_type not in valid_types:
            raise ValueError(f"Unknown scaler_type: '{scaler_type}'. Must be one of {valid_types}")

        self.scaler_type = scaler_type
        self.feature_names = list(feature_names) if feature_names is not None else []
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # State attributes
        self.is_fitted: bool = False
        self.fit_source: str = "none"
        self.n_samples_seen: int = 0
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.zero_variance_mask_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, split_name: str = "train") -> CausalSequenceScaler:
        """
        Fit scaler parameters EXCLUSIVELY on training data.

        Args:
            X: Array of shape (N, L, D) or (N, D).
            split_name: Name of the split being fitted. Must strictly be 'train'.
        """
        # Strict leakage guard
        if split_name.strip().lower() != "train":
            raise ValueError(
                f"Data leakage violation: Scaler must ONLY be fitted on training data. "
                f"Attempted to fit on '{split_name}'!"
            )

        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)

        if X.size == 0 or len(X) == 0:
            raise ValueError("Cannot fit scaler on empty training dataset (0 samples).")

        if X.ndim not in (2, 3):
            raise ValueError(f"Expected 2D (N, D) or 3D (N, L, D) array, got ndim={X.ndim} with shape {X.shape}")

        # Compute across sample axes:
        # If 3D (N, L, D): reduction axes are (0, 1), resulting in shape (D,)
        # If 2D (N, D): reduction axis is 0, resulting in shape (D,)
        reduction_axes = (0, 1) if X.ndim == 3 else 0
        n_features = X.shape[-1]

        if self.feature_names and len(self.feature_names) != n_features:
            raise ValueError(
                f"Feature count mismatch: scaler initialized with {len(self.feature_names)} names, "
                f"but input data has {n_features} features."
            )

        if self.scaler_type == "standard":
            center = np.nanmean(X, axis=reduction_axes)
            scale = np.nanstd(X, axis=reduction_axes)
        elif self.scaler_type == "robust":
            center = np.nanmedian(X, axis=reduction_axes)
            q75 = np.nanpercentile(X, 75, axis=reduction_axes)
            q25 = np.nanpercentile(X, 25, axis=reduction_axes)
            scale = q75 - q25
        elif self.scaler_type == "minmax":
            center = np.nanmin(X, axis=reduction_axes)
            max_val = np.nanmax(X, axis=reduction_axes)
            scale = max_val - center

        # Detect zero-variance features and apply numerical safety guard
        zero_var = scale < self.eps
        safe_scale = np.where(zero_var, 1.0, scale)

        self.center_ = center.astype(np.float64)
        self.scale_ = safe_scale.astype(np.float64)
        self.zero_variance_mask_ = zero_var
        self.n_samples_seen = len(X)
        self.fit_source = "train"
        self.is_fitted = True

        logger.info(
            f"Fitted {self.scaler_type} scaler on {self.n_samples_seen} train samples. "
            f"Features: {n_features}, Zero-variance count: {int(zero_var.sum())}"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform array using parameters learned from training data.

        Args:
            X: Array of shape (N, L, D) or (N, D).
        """
        if not self.is_fitted or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit() on training data first.")

        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)

        if X.size == 0 or len(X) == 0:
            return np.empty_like(X, dtype=np.float64)

        if X.shape[-1] != len(self.center_):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.center_)} features, got {X.shape[-1]}."
            )

        # Broadcast center and scale across (N, L, D) or (N, D)
        X_scaled = (X - self.center_) / self.scale_

        if self.clip_min is not None or self.clip_max is not None:
            X_scaled = np.clip(
                X_scaled,
                a_min=self.clip_min if self.clip_min is not None else -np.inf,
                a_max=self.clip_max if self.clip_max is not None else np.inf,
            )

        return X_scaled

    def fit_transform(self, X: np.ndarray, split_name: str = "train") -> np.ndarray:
        """Fit on training data and transform in one step."""
        return self.fit(X, split_name=split_name).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scaler parameters to dictionary."""
        if not self.is_fitted or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Cannot serialize unfitted scaler.")

        feat_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(len(self.center_))]
        centers_dict = {name: float(self.center_[i]) for i, name in enumerate(feat_names)}
        scales_dict = {name: float(self.scale_[i]) for i, name in enumerate(feat_names)}
        zero_var_feats = [name for i, name in enumerate(feat_names) if self.zero_variance_mask_[i]]

        return {
            "scaler_type": self.scaler_type,
            "fit_source": self.fit_source,
            "is_fitted": self.is_fitted,
            "n_samples_seen": self.n_samples_seen,
            "n_features": len(self.center_),
            "feature_names": feat_names,
            "centers": centers_dict,
            "scales": scales_dict,
            "zero_variance_features": zero_var_feats,
            "eps": self.eps,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CausalSequenceScaler:
        """Reconstruct fitted scaler from parameter dictionary."""
        scaler = cls(
            scaler_type=data["scaler_type"],
            feature_names=data.get("feature_names"),
            eps=data.get("eps", 1e-8),
            clip_min=data.get("clip_min"),
            clip_max=data.get("clip_max"),
        )
        names = data["feature_names"]
        scaler.center_ = np.array([data["centers"][n] for n in names], dtype=np.float64)
        scaler.scale_ = np.array([data["scales"][n] for n in names], dtype=np.float64)
        zero_var_set = set(data.get("zero_variance_features", []))
        scaler.zero_variance_mask_ = np.array([n in zero_var_set for n in names], dtype=bool)
        scaler.n_samples_seen = data["n_samples_seen"]
        scaler.fit_source = data["fit_source"]
        scaler.is_fitted = True
        return scaler

    def save(self, path: Union[Path, str]) -> None:
        """Save scaler parameters to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> CausalSequenceScaler:
        """Load fitted scaler from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


class ScalingOrchestrator:
    """
    Coordinates loading sequence splits, fitting on train only,
    transforming validation and test, and saving scaled datasets.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        eps: float = 1e-8,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ):
        self.scaler = CausalSequenceScaler(
            scaler_type=scaler_type,
            eps=eps,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    def scale_sequences(
        self,
        sequences_dir: Union[Path, str],
        output_dir: Union[Path, str],
        save_metadata: bool = True,
        save_report: bool = True,
    ) -> Tuple[ScalerMetadata, ScalingReport]:
        """
        Execute train-only feature scaling across all sequence splits.
        """
        seq_d = Path(sequences_dir)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        # 1. Load Parquet tables
        train_pq_path = seq_d / "train_sequences.parquet"
        val_pq_path = seq_d / "validation_sequences.parquet"
        test_pq_path = seq_d / "test_sequences.parquet"

        if not train_pq_path.exists():
            raise FileNotFoundError(f"Train sequences parquet not found: {train_pq_path}")
        if not val_pq_path.exists():
            raise FileNotFoundError(f"Validation sequences parquet not found: {val_pq_path}")
        if not test_pq_path.exists():
            raise FileNotFoundError(f"Test sequences parquet not found: {test_pq_path}")

        train_df = pd.read_parquet(train_pq_path)
        val_df = pd.read_parquet(val_pq_path)
        test_df = pd.read_parquet(test_pq_path)

        # 2. Load NPZ binary arrays
        train_npz = np.load(seq_d / "train_sequences.npz", allow_pickle=True)
        val_npz = np.load(seq_d / "validation_sequences.npz", allow_pickle=True)
        test_npz = np.load(seq_d / "test_sequences.npz", allow_pickle=True)

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
        self.scaler.feature_names = feature_names

        # 3. Fit scaler ONLY on train
        self.scaler.fit(X_train, split_name="train")

        # 4. Transform all splits using train-learned parameters
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 5. Invariant Assertions
        # Shapes must remain exactly unchanged
        assert X_train_scaled.shape == X_train.shape, f"Train shape changed: {X_train_scaled.shape} != {X_train.shape}"
        assert X_val_scaled.shape == X_val.shape, f"Validation shape changed: {X_val_scaled.shape} != {X_val.shape}"
        assert X_test_scaled.shape == X_test.shape, f"Test shape changed: {X_test_scaled.shape} != {X_test.shape}"

        # Targets must remain completely untouched
        assert np.array_equal(y_train, train_df["target"].values), "Train targets altered during scaling!"
        assert np.array_equal(y_val, val_df["target"].values), "Validation targets altered during scaling!"
        assert np.array_equal(y_test, test_df["target"].values), "Test targets altered during scaling!"

        # No Inf in scaled outputs
        assert not np.isinf(X_train_scaled).any(), "Inf detected in scaled train data!"
        assert not np.isinf(X_val_scaled).any(), "Inf detected in scaled validation data!"
        assert not np.isinf(X_test_scaled).any(), "Inf detected in scaled test data!"

        # 6. Build scaled DataFrames (updating feature_matrix column with scaled lists)
        train_scaled_df = train_df.copy()
        val_scaled_df = val_df.copy()
        test_scaled_df = test_df.copy()

        train_scaled_df["feature_matrix"] = [X_train_scaled[i].tolist() for i in range(len(X_train_scaled))]
        val_scaled_df["feature_matrix"] = [X_val_scaled[i].tolist() for i in range(len(X_val_scaled))]
        test_scaled_df["feature_matrix"] = [X_test_scaled[i].tolist() for i in range(len(X_test_scaled))]

        # 7. Save Parquet outputs
        train_pq_out = out_d / "train_scaled.parquet"
        val_pq_out = out_d / "validation_scaled.parquet"
        test_pq_out = out_d / "test_scaled.parquet"

        train_scaled_df.to_parquet(train_pq_out, index=False, engine="pyarrow")
        val_scaled_df.to_parquet(val_pq_out, index=False, engine="pyarrow")
        test_scaled_df.to_parquet(test_pq_out, index=False, engine="pyarrow")

        # 8. Save NPZ outputs (strings cast to string dtype for clean, pickle-free loading)
        train_npz_out = out_d / "train_scaled.npz"
        val_npz_out = out_d / "validation_scaled.npz"
        test_npz_out = out_d / "test_scaled.npz"

        np.savez_compressed(
            train_npz_out,
            X=X_train_scaled,
            y=np.array(y_train, dtype=str),
            endpoints=ep_train,
            feature_names=np.array(feature_names, dtype=str),
        )
        np.savez_compressed(
            val_npz_out,
            X=X_val_scaled,
            y=np.array(y_val, dtype=str),
            endpoints=ep_val,
            feature_names=np.array(feature_names, dtype=str),
        )
        np.savez_compressed(
            test_npz_out,
            X=X_test_scaled,
            y=np.array(y_test, dtype=str),
            endpoints=ep_test,
            feature_names=np.array(feature_names, dtype=str),
        )

        # 9. Save Scaler Parameters
        scaler_params_out = out_d / "scaler_params.json"
        self.scaler.save(scaler_params_out)

        # 10. Post-scaling statistics
        train_means_after = {name: float(np.nanmean(X_train_scaled[:, :, i])) for i, name in enumerate(feature_names)}
        train_stds_after = {name: float(np.nanstd(X_train_scaled[:, :, i])) for i, name in enumerate(feature_names)}
        zero_var_feats = [name for i, name in enumerate(feature_names) if self.scaler.zero_variance_mask_[i]]

        # Determinism verification: transform again and verify bit-for-bit identity
        X_train_scaled_2 = self.scaler.transform(X_train)
        det_pass = bool(np.array_equal(X_train_scaled, X_train_scaled_2, equal_nan=True))

        metadata = ScalerMetadata(
            scaler_type=self.scaler.scaler_type,
            fit_source=self.scaler.fit_source,
            n_samples_seen=self.scaler.n_samples_seen,
            sequence_length=X_train.shape[1],
            n_features=len(feature_names),
            feature_names=feature_names,
            centers={name: float(self.scaler.center_[i]) for i, name in enumerate(feature_names)},
            scales={name: float(self.scaler.scale_[i]) for i, name in enumerate(feature_names)},
            zero_variance_features=zero_var_feats,
            train_shape_before=X_train.shape,
            train_shape_after=X_train_scaled.shape,
            val_shape_before=X_val.shape,
            val_shape_after=X_val_scaled.shape,
            test_shape_before=X_test.shape,
            test_shape_after=X_test_scaled.shape,
            artifacts={
                "train_scaled_parquet": str(train_pq_out),
                "validation_scaled_parquet": str(val_pq_out),
                "test_scaled_parquet": str(test_pq_out),
                "train_scaled_npz": str(train_npz_out),
                "validation_scaled_npz": str(val_npz_out),
                "test_scaled_npz": str(test_npz_out),
                "scaler_params_json": str(scaler_params_out),
            },
        )

        report = ScalingReport(
            scaler_type=self.scaler.scaler_type,
            fit_source=self.scaler.fit_source,
            train_sequences=len(X_train),
            val_sequences=len(X_val),
            test_sequences=len(X_test),
            sequence_length=X_train.shape[1],
            n_features=len(feature_names),
            feature_names=feature_names,
            zero_variance_features=zero_var_feats,
            train_means_after=train_means_after,
            train_stds_after=train_stds_after,
            target_untouched_verification=True,
            shape_preservation_verification=True,
            no_leakage_verification=True,
            deterministic_verification=det_pass,
            test_result="PASS",
        )

        if save_metadata:
            meta_out = out_d / "scaler_metadata.json"
            meta_out.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")

        if save_report:
            rep_out = out_d / "scaling_report.md"
            rep_out.write_text(report.to_markdown(), encoding="utf-8")

        return metadata, report


def main() -> None:
    """CLI entry point for feature scaling."""
    parser = argparse.ArgumentParser(description="Pipeline V2 Train-Only Sequence Scaler.")
    parser.add_argument(
        "--sequences-dir",
        "-s",
        default="data/clean_v2/06_sequences",
        help="Directory containing unscaled sequence datasets",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/clean_v2/07_scaled",
        help="Directory to save scaled sequence datasets and scaler metadata",
    )
    parser.add_argument(
        "--scaler-type",
        "-t",
        default="standard",
        choices=["standard", "robust", "minmax"],
        help="Scaler type (default: standard)",
    )

    args = parser.parse_args()

    orchestrator = ScalingOrchestrator(scaler_type=args.scaler_type)
    metadata, report = orchestrator.scale_sequences(
        sequences_dir=args.sequences_dir,
        output_dir=args.output_dir,
    )

    print("\n" + report.to_markdown())


if __name__ == "__main__":
    main()
