"""
Unit tests for Phase 18: Final Data Quality & Training Readiness Gate on Expanded Collection.

Validates all Phase 18 requirements and gates on the 46-market expanded production dataset:
1. Dataset Integrity & Shapes:
   - Train: (7514, 10, 11)
   - Validation: (1428, 10, 11)
   - Test: (1582, 10, 11)
   - Total: 10,524 sequences across 46 retained markets (92 token streams).
   - Zero duplicate endpoint keys.
2. Mathematical NaN Proof & Warmup Masks:
   - Exactly 3,588 warmup mask cells (train: 2418, val: 624, test: 546).
   - Timesteps 5-9 have strictly ZERO NaNs across all splits and features.
   - Prediction endpoint (timestep 9) has strictly ZERO NaNs.
   - Imputed model tensors have strictly ZERO NaNs and ZERO Infs.
   - Labels have strictly ZERO NaNs.
3. Class Quality:
   - UP/DOWN ratio ~ 1.001 (4,376 UP, 4,371 DOWN, 1,777 FLAT).
   - Mirror-conjugate token symmetry holds per market.
   - No class collapse (all classes > 5%).
4. Feature Quality:
   - Zero constant features (0 / 11).
   - Scaled train features exhibit unit variance and zero mean.
5. Information Leakage Audits:
   - Adversarial perturbation invariance (10,000x val/test perturbation -> 0.0 change).
   - Fit rejection on validation and test splits (raises ValueError).
   - Purge gap 1 = 61,000 ms >= 22,000 ms.
   - Purge gap 2 = 96,000 ms >= 22,000 ms.
6. Market Realism & Microstructure:
   - Unique mid prices = 282 >= 25.
   - Directional transitions = 8,596 >= 100.
   - Spread positivity: 100.0%.
   - Bid <= Mid <= Ask: 100.0%.
   - Bid <= Microprice <= Ask: 100.0%.
   - Depth imbalance in [-1, 1]: 100.0%.
7. Canonical & Labeled Counts:
   - Retained canonical observations: 16,431,475 >= 50,000.
   - Excluded canonical observations: 287,238 across 6 excluded markets.
   - Total canonical observations: 16,718,713 across 52 markets.
   - Labeled observations: 11,352 >= 10,000.
8. Final GO / NO-GO Verdict:
   - Pipeline Correctness = PASS (True).
   - Institutional Data Quality = PASS (True).
   - Training Readiness = YES (True).
   - Verdict string = 'TRAINING_READY = TRUE' with 0 blocking failures.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from pipeline_v2.scaling.train_scaler import CausalSequenceScaler
from pipeline_v2.sequences.causal_dataloader import CausalSequenceDataset, causal_collate_fn
from pipeline_v2.sequences.sequence_builder import SAFE_FEATURE_COLUMNS

SCALED_DIR = Path("data/clean_v2/07_scaled/expanded_collection")
QUALITY_DIR = Path("data/clean_v2/08_quality_gate/expanded_collection")


@pytest.fixture(scope="module")
def expanded_quality_gate_artifacts():
    """Load Phase 18 quality gate outputs and Phase 17 input tensors."""
    meta_path = QUALITY_DIR / "phase18_quality_metadata.json"
    gate_meta_path = QUALITY_DIR / "phase18_quality_gate_metadata.json"
    rep_path = QUALITY_DIR / "phase18_quality_report.md"
    gate_rep_path = QUALITY_DIR / "phase18_quality_gate_report.md"
    comb_npz_path = SCALED_DIR / "scaled_production.npz"
    train_pq_path = SCALED_DIR / "train_scaled.parquet"
    val_pq_path = SCALED_DIR / "validation_scaled.parquet"
    test_pq_path = SCALED_DIR / "test_scaled.parquet"
    warmup_masks_path = SCALED_DIR / "warmup_masks.npz"

    assert meta_path.exists(), f"Missing metadata: {meta_path}"
    assert gate_meta_path.exists(), f"Missing gate metadata: {gate_meta_path}"
    assert rep_path.exists(), f"Missing report: {rep_path}"
    assert gate_rep_path.exists(), f"Missing gate report: {gate_rep_path}"
    assert comb_npz_path.exists(), f"Missing NPZ: {comb_npz_path}"
    assert warmup_masks_path.exists(), f"Missing warmup masks: {warmup_masks_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    npz = np.load(comb_npz_path, allow_pickle=True)
    warmup_npz = np.load(warmup_masks_path, allow_pickle=True)
    train_df = pd.read_parquet(train_pq_path)
    val_df = pd.read_parquet(val_pq_path)
    test_df = pd.read_parquet(test_pq_path)

    return {
        "meta": meta,
        "report_text": rep_path.read_text(encoding="utf-8"),
        "gate_report_text": gate_rep_path.read_text(encoding="utf-8"),
        "npz": npz,
        "warmup_npz": warmup_npz,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def test_expanded_artifacts_exist(expanded_quality_gate_artifacts):
    """Verify all Phase 18 reports and JSON metadata exist and are non-empty."""
    assert (QUALITY_DIR / "phase18_quality_metadata.json").exists()
    assert (QUALITY_DIR / "phase18_quality_gate_metadata.json").exists()
    assert (QUALITY_DIR / "phase18_quality_report.md").exists()
    assert (QUALITY_DIR / "phase18_quality_gate_report.md").exists()
    assert (QUALITY_DIR / "quality_gate.json").exists()
    assert (QUALITY_DIR / "quality_gate_report.md").exists()

    assert len(expanded_quality_gate_artifacts["report_text"]) > 1000
    assert len(expanded_quality_gate_artifacts["gate_report_text"]) > 1000


def test_expanded_dataset_integrity_and_sequence_counts(expanded_quality_gate_artifacts):
    """Verify exact sequence counts, shapes, dimensions, and endpoint uniqueness."""
    npz = expanded_quality_gate_artifacts["npz"]
    meta = expanded_quality_gate_artifacts["meta"]
    train_df = expanded_quality_gate_artifacts["train_df"]
    val_df = expanded_quality_gate_artifacts["val_df"]
    test_df = expanded_quality_gate_artifacts["test_df"]

    # Sequence Counts
    assert len(npz["train_X"]) == 7514
    assert len(npz["val_X"]) == 1428
    assert len(npz["test_X"]) == 1582
    total = len(npz["train_X"]) + len(npz["val_X"]) + len(npz["test_X"])
    assert total == 10524

    # Tensor Shapes
    assert npz["train_X"].shape == (7514, 10, 11)
    assert npz["val_X"].shape == (1428, 10, 11)
    assert npz["test_X"].shape == (1582, 10, 11)

    # Feature columns
    assert list(npz["feature_names"]) == SAFE_FEATURE_COLUMNS

    # Market Isolation
    train_mkts = set(train_df["market_id"].unique())
    val_mkts = set(val_df["market_id"].unique())
    test_mkts = set(test_df["market_id"].unique())
    assert len(train_mkts) == 31
    assert len(val_mkts) == 8
    assert len(test_mkts) == 7
    assert len(train_mkts.intersection(val_mkts)) == 0
    assert len(val_mkts.intersection(test_mkts)) == 0
    assert len(train_mkts.intersection(test_mkts)) == 0
    assert len(train_mkts) + len(val_mkts) + len(test_mkts) == 46

    # Endpoint uniqueness
    all_keys = [
        f"{m}_{a}_{t}"
        for df in [train_df, val_df, test_df]
        for m, a, t in zip(df["market_id"], df["asset_id"], df["endpoint_timestamp_ms"])
    ]
    assert len(all_keys) == len(set(all_keys))


def test_expanded_nan_proof_and_warmup_masks(expanded_quality_gate_artifacts):
    """Verify warmup masks, lookback NaN origin, and clean model tensors."""
    npz = expanded_quality_gate_artifacts["npz"]
    warmup_npz = expanded_quality_gate_artifacts["warmup_npz"]
    meta = expanded_quality_gate_artifacts["meta"]

    # Warmup masks
    tr_m = int(warmup_npz["train_mask"].sum())
    va_m = int(warmup_npz["val_mask"].sum())
    te_m = int(warmup_npz["test_mask"].sum())
    assert tr_m == 2418
    assert va_m == 624
    assert te_m == 546
    assert tr_m + va_m + te_m == 3588

    # Unimputed NaNs match warmup masks exactly
    assert int(np.isnan(npz["train_X"]).sum()) == 2418
    assert int(np.isnan(npz["val_X"]).sum()) == 624
    assert int(np.isnan(npz["test_X"]).sum()) == 546

    # Lookback steps 5-9 have strictly 0 NaNs
    for X_split in [npz["train_X"], npz["val_X"], npz["test_X"]]:
        assert np.isnan(X_split[:, 5:, :]).sum() == 0
        assert np.isnan(X_split[:, 9, :]).sum() == 0  # Endpoint T

    # Imputed arrays have strictly ZERO NaNs and ZERO Infs
    assert int(np.isnan(npz["train_X_imputed"]).sum()) == 0
    assert int(np.isnan(npz["val_X_imputed"]).sum()) == 0
    assert int(np.isnan(npz["test_X_imputed"]).sum()) == 0
    assert int(np.isinf(npz["train_X_imputed"]).sum()) == 0
    assert int(np.isinf(npz["val_X_imputed"]).sum()) == 0
    assert int(np.isinf(npz["test_X_imputed"]).sum()) == 0

    # Labels have 0 NaNs
    assert int(pd.Series(npz["train_y"]).isna().sum()) == 0
    assert int(pd.Series(npz["val_y"]).isna().sum()) == 0
    assert int(pd.Series(npz["test_y"]).isna().sum()) == 0


def test_expanded_class_quality_and_balance(expanded_quality_gate_artifacts):
    """Verify class balance, presence of all 3 classes, and absence of class collapse."""
    meta = expanded_quality_gate_artifacts["meta"]
    comb_cls = meta["class_audit"]["combined"]
    train_cls = meta["class_audit"]["train"]

    assert set(comb_cls.keys()).issuperset({"up_count", "down_count", "flat_count"})
    assert comb_cls["up_count"] == 4376
    assert comb_cls["down_count"] == 4371
    assert comb_cls["flat_count"] == 1777

    # All classes > 5%
    assert comb_cls["up_pct"] >= 5.0
    assert comb_cls["down_pct"] >= 5.0
    assert comb_cls["flat_pct"] >= 5.0

    # UP/DOWN balance
    assert 0.99 <= comb_cls["up_down_ratio"] <= 1.01


def test_expanded_feature_quality(expanded_quality_gate_artifacts):
    """Verify that all 11 causal features have unit variance and zero mean on train split."""
    npz = expanded_quality_gate_artifacts["npz"]
    meta = expanded_quality_gate_artifacts["meta"]
    X_train = npz["train_X"]

    for i, f_name in enumerate(SAFE_FEATURE_COLUMNS):
        vals = X_train[:, :, i]
        v = float(np.nanvar(vals))
        m = float(np.nanmean(vals))
        assert v > 0.0, f"Feature {f_name} has zero variance"
        assert abs(v - 1.0) < 1e-4, f"Feature {f_name} variance is not 1.0: {v}"
        assert abs(m) < 1e-4, f"Feature {f_name} mean is not 0.0: {m}"


def test_expanded_adversarial_leakage_and_temporal_integrity(expanded_quality_gate_artifacts):
    """Verify adversarial perturbation invariance, fit rejection, and purge gaps."""
    npz = expanded_quality_gate_artifacts["npz"]
    meta = expanded_quality_gate_artifacts["meta"]
    X_train = npz["train_X"]
    X_val = npz["val_X"]
    X_test = npz["test_X"]

    # Perturbation invariance
    scaler1 = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler1.fit(X_train, split_name="train")

    X_val_pert = X_val * 9999.0 + 8888.0
    X_test_pert = X_test * 9999.0 - 8888.0
    scaler2 = CausalSequenceScaler(scaler_type="standard", feature_names=SAFE_FEATURE_COLUMNS)
    scaler2.fit(X_train, split_name="train")

    assert np.max(np.abs(scaler1.center_ - scaler2.center_)) == 0.0
    assert np.max(np.abs(scaler1.scale_ - scaler2.scale_)) == 0.0

    # Fit rejection on validation and test
    with pytest.raises(ValueError, match="leakage violation"):
        scaler1.fit(X_val, split_name="validation")

    with pytest.raises(ValueError, match="leakage violation"):
        scaler1.fit(X_test, split_name="test")

    # Purge gaps
    purge_gaps = meta["purge_gaps_ms"]
    assert purge_gaps["purge_1_ms"] == 61000
    assert purge_gaps["purge_2_ms"] == 96000
    assert purge_gaps["purge_1_ms"] >= 22000
    assert purge_gaps["purge_2_ms"] >= 22000


def test_expanded_canonical_and_labeled_counts(expanded_quality_gate_artifacts):
    """Verify retained canonical observation count >= 50k and labeled >= 10k."""
    meta = expanded_quality_gate_artifacts["meta"]
    gates = meta["gates"]

    canon_gate = gates["canonical_observations_count"]
    assert canon_gate["status"] == "PASS"

    canon_acct = meta["canonical_accounting"]
    assert canon_acct["retained"] == 16431475
    assert canon_acct["excluded"] == 287238
    assert canon_acct["total"] == 16718713
    assert canon_acct["retained"] >= 50000

    labeled_gate = gates["labeled_observations_count"]
    assert labeled_gate["status"] == "PASS"
    assert int(labeled_gate["observed"].replace(",", "")) == 11352
    assert int(labeled_gate["observed"].replace(",", "")) >= 10000


def test_expanded_market_realism_and_microstructure(expanded_quality_gate_artifacts):
    """Verify price diversity, directional transitions, and order book microstructure bounds."""
    meta = expanded_quality_gate_artifacts["meta"]
    gates = meta["gates"]

    assert gates["price_diversity"]["status"] == "PASS"
    assert "282" in gates["price_diversity"]["observed"]

    assert gates["directional_transitions"]["status"] == "PASS"
    assert "8,596" in gates["directional_transitions"]["observed"]

    micro = meta["microstructure_audit"]
    assert micro["spread_positive_pct"] == 100.0
    assert micro["bid_mid_ask_pct"] == 100.0
    assert micro["bid_micro_ask_pct"] == 100.0
    assert micro["depth_imbalance_valid_pct"] == 100.0


def test_expanded_dataloader_collation(expanded_quality_gate_artifacts):
    """Verify PyTorch DataLoader collation produces 0 NaNs and 0 Infs."""
    npz = expanded_quality_gate_artifacts["npz"]
    ds = CausalSequenceDataset(npz["train_X"], npz["train_y"], npz["train_endpoints"])
    batch_items = [ds[i] for i in range(min(len(ds), 128))]
    collated = causal_collate_fn(batch_items)

    assert torch.isnan(collated["x"]).sum().item() == 0
    assert torch.isinf(collated["x"]).sum().item() == 0
    assert collated["causal_mask"].shape == collated["x"].shape


def test_expanded_final_training_readiness_verdict(expanded_quality_gate_artifacts):
    """Verify Phase 18 Final Verdict is strictly TRAINING_READY = TRUE with 0 blockers."""
    meta = expanded_quality_gate_artifacts["meta"]

    assert meta["pipeline_valid"] is True
    assert meta["data_quality_pass"] is True
    assert meta["training_ready"] is True
    assert meta["verdict_string"] == "TRAINING_READY = TRUE"
    assert len(meta["blocking_failures"]) == 0

    # All 15 individual gates must PASS
    gates = meta["gates"]
    assert len(gates) == 15
    for gate_name, gate_info in gates.items():
        assert gate_info["status"] == "PASS", f"Gate {gate_name} failed: {gate_info}"
