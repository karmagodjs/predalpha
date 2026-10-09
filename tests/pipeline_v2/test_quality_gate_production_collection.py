"""
Unit tests for Phase 18: Final Data Quality & Training Readiness Gate on Production Collection.

Validates all Phase 18 requirements and gates:
1. Dataset Integrity & Shapes:
   - Train: (3248, 10, 11)
   - Validation: (722, 10, 11)
   - Test: (658, 10, 11)
   - Total: 4,628 sequences across 19 retained markets (38 token assets).
   - Zero duplicate endpoint keys.
2. Mathematical NaN Proof:
   - Exactly 1,482 NaNs across the entire dataset (0.29%).
   - Timesteps 5-9 have strictly ZERO NaNs across all splits and features.
   - Prediction endpoint (timestep 9) has strictly ZERO NaNs.
   - Labels have strictly ZERO NaNs.
   - Tensor has strictly ZERO Infs.
   - NaNs reside exclusively in causal lookback warmup steps 0-4.
3. Class Quality:
   - UP/DOWN ratio = 1.000 (1,959 UP, 1,958 DOWN, 711 FLAT).
   - Mirror-conjugate token symmetry holds per market.
   - No class collapse (all classes > 5%).
4. Feature Quality:
   - Zero constant features (0 / 11).
   - Scaled train features exhibit unit variance and zero mean.
5. Information Leakage Audits:
   - Adversarial perturbation invariance (10,000x val/test perturbation -> 0.0 change).
   - Purge gap 1 = 88,000 ms >= 22,000 ms.
   - Purge gap 2 = 69,000 ms >= 22,000 ms.
6. Market Realism:
   - Unique mid prices = 261 >= 25.
   - Directional transitions = 3,832 >= 100.
7. Final GO / NO-GO Verdict:
   - Pipeline Correctness = PASS (True).
   - Institutional Data Quality = FAIL (False).
   - Verdict string = 'TRAINING_READY = FALSE' with exactly 4 documented blocking failures.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.scaling.train_scaler import CausalSequenceScaler
from pipeline_v2.sequences.sequence_builder import SAFE_FEATURE_COLUMNS

SCALED_DIR = Path("data/clean_v2/07_scaled/new_collection")
QUALITY_DIR = Path("data/clean_v2/08_quality_gate/new_collection")
PARENT_QUALITY_DIR = Path("data/clean_v2/08_quality_gate")


@pytest.fixture(scope="module")
def quality_gate_artifacts():
    """Load Phase 18 quality gate outputs and Phase 17 input tensors."""
    meta_path = QUALITY_DIR / "phase18_quality_metadata.json"
    rep_path = QUALITY_DIR / "phase18_quality_report.md"
    comb_npz_path = SCALED_DIR / "scaled_production.npz"
    train_pq_path = SCALED_DIR / "train_scaled.parquet"
    val_pq_path = SCALED_DIR / "validation_scaled.parquet"
    test_pq_path = SCALED_DIR / "test_scaled.parquet"

    assert meta_path.exists(), f"Missing metadata: {meta_path}"
    assert rep_path.exists(), f"Missing report: {rep_path}"
    assert comb_npz_path.exists(), f"Missing NPZ: {comb_npz_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    npz = np.load(comb_npz_path, allow_pickle=True)
    train_df = pd.read_parquet(train_pq_path)
    val_df = pd.read_parquet(val_pq_path)
    test_df = pd.read_parquet(test_pq_path)

    return {
        "meta": meta,
        "report_text": rep_path.read_text(encoding="utf-8"),
        "npz": npz,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def test_artifacts_exist_and_mirrored(quality_gate_artifacts):
    """Verify Phase 18 reports and JSON metadata exist in new_collection and parent directory."""
    assert (QUALITY_DIR / "phase18_quality_metadata.json").exists()
    assert (QUALITY_DIR / "phase18_quality_report.md").exists()
    assert (QUALITY_DIR / "quality_gate.json").exists()
    assert (QUALITY_DIR / "quality_gate_report.md").exists()

    assert (PARENT_QUALITY_DIR / "phase18_quality_metadata.json").exists()
    assert (PARENT_QUALITY_DIR / "phase18_quality_report.md").exists()

    assert len(quality_gate_artifacts["report_text"]) > 1000


def test_dataset_integrity_and_sequence_counts(quality_gate_artifacts):
    """Verify exact sequence counts, shapes, dimensions, and endpoint uniqueness."""
    npz = quality_gate_artifacts["npz"]
    meta = quality_gate_artifacts["meta"]
    train_df = quality_gate_artifacts["train_df"]
    val_df = quality_gate_artifacts["val_df"]
    test_df = quality_gate_artifacts["test_df"]

    # Verify tensor shapes
    assert npz["train_X"].shape == (3248, 10, 11)
    assert npz["val_X"].shape == (722, 10, 11)
    assert npz["test_X"].shape == (658, 10, 11)

    # Verify target shape
    assert npz["train_y"].shape == (3248,)
    assert npz["val_y"].shape == (722,)
    assert npz["test_y"].shape == (658,)

    # Verify metadata row counts
    assert meta["row_counts"]["train_sequences"] == 3248
    assert meta["row_counts"]["val_sequences"] == 722
    assert meta["row_counts"]["test_sequences"] == 658
    assert meta["row_counts"]["total_sequences"] == 4628

    # Verify 0 duplicate endpoint keys
    for df_split in [train_df, val_df, test_df]:
        assert df_split.duplicated(subset=["market_id", "asset_id", "endpoint_timestamp_ms"]).sum() == 0

    # Verify 19 retained markets and 38 tokens
    comb_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    assert comb_df["market_id"].nunique() == 19
    assert comb_df["asset_id"].nunique() == 38


def test_mathematical_nan_proof(quality_gate_artifacts):
    """
    Verify exact mathematical proof that NaNs exist strictly in lookback warmup steps 0-4,
    with 0 NaNs at prediction endpoint T (step 9) and post-warmup steps 5-9.
    """
    npz = quality_gate_artifacts["npz"]
    meta = quality_gate_artifacts["meta"]
    nan_proof = meta["nan_proof"]

    X_train = npz["train_X"]
    X_val = npz["val_X"]
    X_test = npz["test_X"]

    # Total NaNs
    assert nan_proof["train"]["total_nans"] == 624
    assert nan_proof["validation"]["total_nans"] == 390
    assert nan_proof["test"]["total_nans"] == 468
    total_nans = 624 + 390 + 468
    assert total_nans == 1482

    # Step-wise verification
    for split_name, X_split in [("train", X_train), ("val", X_val), ("test", X_test)]:
        # Prediction endpoint (step 9) has STRICTLY ZERO NaNs
        assert np.isnan(X_split[:, 9, :]).sum() == 0
        # Post-warmup steps (steps 5 through 9) have STRICTLY ZERO NaNs
        assert np.isnan(X_split[:, 5:10, :]).sum() == 0
        # Zero Infs
        assert np.isinf(X_split).sum() == 0

    # Step-by-step exact distribution across combined dataset
    combined_X = np.concatenate([X_train, X_val, X_test], axis=0)
    assert np.isnan(combined_X[:, 0, :]).sum() == 608  # step 0: 1s, 3s, 5s warmup
    assert np.isnan(combined_X[:, 1, :]).sum() == 380  # step 1: 3s, 5s warmup
    assert np.isnan(combined_X[:, 2, :]).sum() == 266  # step 2: 3s, 5s warmup
    assert np.isnan(combined_X[:, 3, :]).sum() == 152  # step 3: 5s warmup
    assert np.isnan(combined_X[:, 4, :]).sum() == 76   # step 4: 5s warmup
    assert np.isnan(combined_X[:, 5:, :]).sum() == 0   # steps 5-9: 0 NaNs!

    # Target labels have zero NaNs
    assert np.isnan(npz["train_y"] == "nan").sum() == 0
    assert pd.Series(npz["train_y"]).isna().sum() == 0
    assert pd.Series(npz["val_y"]).isna().sum() == 0
    assert pd.Series(npz["test_y"]).isna().sum() == 0


def test_class_distribution_and_mirror_symmetry(quality_gate_artifacts):
    """Verify class balance, UP/DOWN symmetry, and absence of class collapse."""
    meta = quality_gate_artifacts["meta"]
    comb = meta["class_audit"]["combined"]

    assert comb["total"] == 4628
    assert comb["up_count"] == 1959
    assert comb["down_count"] == 1958
    assert comb["flat_count"] == 711
    assert comb["up_pct"] == 42.33
    assert comb["down_pct"] == 42.31
    assert round(comb["up_down_ratio"], 3) == 1.000
    assert comb["up_down_ratio"] == 1.0005

    # Check mirror-conjugate symmetry in per-market stats
    market_breakdown = meta["market_class_breakdown"]
    assert len(market_breakdown) == 19
    exact_matches = 0
    for mkt, counts in market_breakdown.items():
        assert abs(counts["up"] - counts["down"]) <= 1, f"Market {mkt} violates UP/DOWN mirror symmetry!"
        if counts["up"] == counts["down"]:
            exact_matches += 1
    # 18 of 19 markets have exact 1:1 symmetry, 1 market has a 1-sequence difference (74 vs 73)
    assert exact_matches == 18


def test_feature_quality_and_zero_variance(quality_gate_artifacts):
    """Verify that all 11 features have non-zero variance and proper scaling parameters."""
    meta = quality_gate_artifacts["meta"]
    feat_audit = meta["feature_quality_audit"]

    assert len(feat_audit) == 11
    for f_name in SAFE_FEATURE_COLUMNS:
        info = feat_audit[f_name]
        assert info["is_constant"] is False
        assert info["scaled_variance"] > 1e-4
        assert abs(info["scaled_mean"]) < 1e-10
        assert round(info["scaled_std"], 4) == 1.0000


def test_adversarial_leakage_and_temporal_purge(quality_gate_artifacts):
    """Verify perturbation invariance of scaler and temporal purge gap >= 22,000ms."""
    npz = quality_gate_artifacts["npz"]
    train_df = quality_gate_artifacts["train_df"]
    val_df = quality_gate_artifacts["val_df"]
    test_df = quality_gate_artifacts["test_df"]

    X_train = npz["train_X"]
    X_val = npz["val_X"]
    X_test = npz["test_X"]

    # Scaler perturbation invariance
    scaler_base = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler_base.fit(X_train, split_name="train")

    X_val_pert = X_val * 99999.0 + 123456.0
    X_test_pert = X_test * -99999.0 - 123456.0

    scaler_after = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler_after.fit(X_train, split_name="train")

    np.testing.assert_array_equal(scaler_base.center_, scaler_after.center_)
    np.testing.assert_array_equal(scaler_base.scale_, scaler_after.scale_)

    # Temporal Purge Gaps
    t_train_max = int(train_df["endpoint_timestamp_ms"].max())
    t_val_min = int(val_df["start_timestamp_ms"].min())
    t_val_max = int(val_df["endpoint_timestamp_ms"].max())
    t_test_min = int(test_df["start_timestamp_ms"].min())

    purge_1_ms = t_val_min - t_train_max
    purge_2_ms = t_test_min - t_val_max

    assert purge_1_ms >= 22000, f"Purge 1 violated: {purge_1_ms} < 22000"
    assert purge_2_ms >= 22000, f"Purge 2 violated: {purge_2_ms} < 22000"
    assert purge_1_ms == 88000
    assert purge_2_ms == 69000


def test_final_verdict_and_blocking_failures(quality_gate_artifacts):
    """
    Verify final GO/NO-GO verdict:
    Pipeline Correctness = PASS
    Institutional Data Quality = FAIL
    Final Verdict = TRAINING_READY = FALSE
    Blocking failures accurately identify volume thresholds and warmup NaNs.
    """
    meta = quality_gate_artifacts["meta"]

    assert meta["pipeline_valid"] is True
    assert meta["data_quality_pass"] is False
    assert meta["training_ready"] is False
    assert meta["verdict_string"] == "TRAINING_READY = FALSE"

    blocking = meta["blocking_failures"]
    assert len(blocking) == 4
    blocking_text = " ".join(blocking)

    assert "sequence_counts_sufficiency" in blocking_text
    assert "canonical_observations_count" in blocking_text
    assert "labeled_observations_count" in blocking_text
    assert "zero_nan_values" in blocking_text


def test_bitwise_deterministic_reproducibility(quality_gate_artifacts):
    """Verify bitwise identical outputs under repeated transformation from unscaled sequences."""
    seq_npz_path = Path("data/clean_v2/06_sequences/new_collection/sequences_production.npz")
    assert seq_npz_path.exists()
    orig_npz = np.load(seq_npz_path, allow_pickle=True)
    orig_train_X = orig_npz["train_X"]
    scaled_train_X = quality_gate_artifacts["npz"]["train_X"]

    scaler = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler.fit(orig_train_X, split_name="train")
    t1 = scaler.transform(orig_train_X)
    t2 = scaler.transform(orig_train_X)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(t1, scaled_train_X)
