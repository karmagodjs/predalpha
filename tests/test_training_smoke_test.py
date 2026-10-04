"""Unit and workflow tests for Phase 3 Minimal Training Smoke Test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from models.baseline_models import CONFIG as BASELINE_CONFIG
from models.smoke_test import (
    DEFAULT_SMOKE_CONFIG,
    SmokeTestConfig,
    build_smoke_test_pipeline,
    get_file_hashes,
    run_training_smoke_test,
)
from scripts.run_smoke_test import build_parser, main


def test_cli_parser_defaults():
    """Verify CLI argument parser configurations and defaults."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.phase1_dir == Path("data/processed/phase1")
    assert args.output_dir == Path("data/processed/phase3_smoke_test")
    assert args.class_weight == "balanced"
    assert args.random_seed == 42
    assert args.max_iter == 1000


def test_smoke_test_execution_and_artifacts(tmp_path: Path):
    """Verify end-to-end execution of training smoke test and all exported artifacts."""
    output_dir = tmp_path / "smoke_test_out"
    config = SmokeTestConfig(
        phase1_dir=Path("data/processed/phase1"),
        output_dir=output_dir,
        class_weight="balanced",
        random_seed=42,
    )

    results = run_training_smoke_test(config=config, output_dir=output_dir)

    # 1. Verify expected files exist
    assert (output_dir / "smoke_test_model.joblib").exists()
    assert (output_dir / "smoke_test_metrics.json").exists()
    assert (output_dir / "smoke_test_config.json").exists()
    assert (output_dir / "confusion_matrices.json").exists()
    assert (output_dir / "smoke_test_report.md").exists()

    # 2. Verify model loading and inference
    loaded_pipeline = joblib.load(output_dir / "smoke_test_model.joblib")
    scaler = loaded_pipeline.named_steps["scaler"]
    clf = loaded_pipeline.named_steps["classifier"]
    assert scaler.n_samples_seen_ == 202
    assert clf.n_features_in_ == 9

    # 3. Verify metrics content
    with open(output_dir / "smoke_test_metrics.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    meta = metrics_data["smoke_test_metadata"]
    assert meta["is_smoke_test"] is True
    assert "PIPELINE SMOKE TEST ONLY" in meta["disclaimer"]
    assert meta["split_row_counts"] == {"train": 202, "validation": 38, "test": 39, "total": 279}
    assert meta["target_classes"] == ["DOWN", "FLAT", "UP"]

    eval_model = metrics_data["evaluation_metrics"]["model"]
    assert "train" in eval_model and "validation" in eval_model and "test" in eval_model
    assert eval_model["train"]["accuracy"] > 0.0
    assert eval_model["validation"]["accuracy"] > 0.0
    assert eval_model["test"]["accuracy"] > 0.0

    # 4. Verify markdown report contains required disclaimer
    with open(output_dir / "smoke_test_report.md", "r", encoding="utf-8") as f:
        report_text = f.read()
    assert "PIPELINE SMOKE TEST ONLY" in report_text
    assert "Dataset & Label Confirmation" in report_text
    assert "Confusion Matrices" in report_text


def test_split_boundary_preservation():
    """Verify that train, validation, and test splits preserve original Phase 1 boundaries."""
    train_df = pd.read_parquet("data/processed/phase1/train.parquet")
    val_df = pd.read_parquet("data/processed/phase1/validation.parquet")
    test_df = pd.read_parquet("data/processed/phase1/test.parquet")

    t_max = pd.to_datetime(train_df["timestamp"].max(), utc=True)
    v_min = pd.to_datetime(val_df["timestamp"].min(), utc=True)
    v_max = pd.to_datetime(val_df["timestamp"].max(), utc=True)
    te_min = pd.to_datetime(test_df["timestamp"].min(), utc=True)

    # 5-second purge gaps between splits
    assert (v_min - t_max).total_seconds() == 6.0  # timestamps 12:47:33 to 12:47:39 = 6s delta (5s gap)
    assert (te_min - v_max).total_seconds() == 6.0  # timestamps 12:48:16 to 12:48:22 = 6s delta (5s gap)


def test_train_only_scaling_isolation():
    """Verify that StandardScaler computes mean and std strictly on training set."""
    train_df = pd.read_parquet("data/processed/phase1/train.parquet")
    val_df = pd.read_parquet("data/processed/phase1/validation.parquet")

    feature_cols = list(BASELINE_CONFIG.feature_cols)
    X_train = train_df[feature_cols].to_numpy(dtype=float)

    pipeline = build_smoke_test_pipeline()
    pipeline.fit(train_df[feature_cols], train_df["label"])

    scaler = pipeline.named_steps["scaler"]
    expected_means = np.mean(X_train, axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)


def test_phase1_and_phase2_artifacts_not_overwritten(tmp_path: Path):
    """Verify that running the smoke test does not modify Phase 1 or Phase 2 artifacts."""
    p1_dir = Path("data/processed/phase1")
    p2_dir = Path("data/processed/phase2")

    p1_before = get_file_hashes(p1_dir)
    p2_before = get_file_hashes(p2_dir)

    # Run smoke test targeting a temporary output directory
    run_training_smoke_test(output_dir=tmp_path / "smoke_out")

    p1_after = get_file_hashes(p1_dir)
    p2_after = get_file_hashes(p2_dir)

    assert p1_before == p1_after, "Phase 1 files were modified!"
    assert p2_before == p2_after, "Phase 2 files were modified!"


def test_cli_main_invocation(tmp_path: Path):
    """Verify CLI main entrypoint executes cleanly."""
    out_dir = tmp_path / "cli_smoke_out"
    ret = main(["--output-dir", str(out_dir), "--class-weight", "none"])
    assert ret == 0
    assert (out_dir / "smoke_test_model.joblib").exists()
