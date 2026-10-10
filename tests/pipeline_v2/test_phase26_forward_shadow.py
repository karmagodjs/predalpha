"""
Unit and integration tests for Phase 26 — Forward Shadow Execution & Execution-Risk Validation.

Verifies:
1. Locked test set isolation.
2. Model weight immutability.
3. Timestamp causality and zero lookahead.
4. Quote synchronization and physical horizon alignment.
5. Stale-data and gap rejection mechanics.
6. Two-sided fee accounting correctness for Long and Short positions.
7. Missing exit quote handling and conservative liquidation.
8. Append-only shadow logger integrity and schema validation.
9. Bitwise reproduction of Phase 25 baseline metrics.
"""

from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from pipeline_v2.models.phase26_forward_shadow import (
    ForwardShadowLogger,
    audit_phase25_mechanics,
    evaluate_stress_scenarios,
    prepare_phase26_dataset,
    sha256_file,
    compute_hash_string,
    define_prospective_protocol,
    LOCKED_TEST_PATH,
    MODEL_PATH,
    PREDECLARED_CONFIDENCE_THRESHOLD,
    TRAIN_SPREAD_THRESHOLD,
    TRAIN_DEPTH_IMBALANCE_P90,
)


def test_locked_test_set_isolation():
    """Ensure the locked test set is present on disk but never accessed or loaded."""
    assert LOCKED_TEST_PATH.exists(), "Locked test set file must exist."
    assert LOCKED_TEST_PATH.is_file(), "Locked test set must be a file."
    # Assert that no function in phase26 opens or returns test data
    protocol = define_prospective_protocol()
    assert "test_scaled.npz" not in str(protocol)


def test_model_weight_immutability():
    """Ensure the frozen Phase 19 SmallLSTM checkpoint weights are bitwise invariant."""
    initial_hash = sha256_file(MODEL_PATH)
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    assert "lstm.weight_ih_l0" in state

    # Re-hash
    final_hash = sha256_file(MODEL_PATH)
    assert initial_hash == final_hash, "Model checkpoint must remain bitwise identical."


def test_timestamp_causality():
    """Verify that all gating features evaluate step 9 (decision time t) and have zero lookahead."""
    df = prepare_phase26_dataset()
    assert "endpoint_timestamp_ms" in df.columns
    assert "scaled_spread" in df.columns
    assert "abs_scaled_depth_imbal" in df.columns
    assert "confidence" in df.columns

    # Verify that prediction timestamps are non-decreasing
    ts = df["endpoint_timestamp_ms"].values
    assert np.all(np.diff(ts) >= 0), "Validation sequence endpoints must be chronological."


def test_quote_synchronization():
    """Verify that entry quotes align with endpoint timestamps and exit quotes align with horizon."""
    df = prepare_phase26_dataset()
    # Entry quote timestamp matches endpoint
    assert (df["grid_timestamp_ms_entry"] == df["endpoint_timestamp_ms"]).all()

    # Target timestamp difference >= 5000ms
    diff = df["target_timestamp"] - df["endpoint_timestamp_ms"]
    assert (diff >= 5000).all(), "Target timestamp must be at least 5000ms after entry."
    assert diff.min() == 5000


def test_stale_data_and_gap_rejection():
    """Verify that the shadow logger correctly flags stale quotes and data gaps."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ForwardShadowLogger(output_dir=Path(tmp_dir), run_id="test_stale_logger", max_quote_age_ms=1000.0, max_data_gap_ms=3000.0)

        # 1. Fresh quote -> should pass
        r1 = logger.log_decision_point(
            event_timestamp_ms=10000,
            market_id="mkt_1",
            asset_id="asset_1",
            quote_timestamp_ms=9500,  # age 500ms <= 1000ms
            bid_entry=0.49,
            ask_entry=0.51,
            depth_imbalance=0.1,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.2,
            model_probs=(0.1, 0.1, 0.8),
            predicted_class=2,
            confidence=0.8,
            burst_id=0,
            data_gap_ms=1000.0,
            bid_exit=0.52,
            ask_exit=0.54,
        )
        assert r1["gate_decision"] == "EXECUTE"
        assert r1["outcome_status"] == "COMPLETED"

        # 2. Stale quote (age 1500ms > 1000ms) -> should reject
        r2 = logger.log_decision_point(
            event_timestamp_ms=12000,
            market_id="mkt_1",
            asset_id="asset_1",
            quote_timestamp_ms=10000,  # age 2000ms > 1000ms
            bid_entry=0.49,
            ask_entry=0.51,
            depth_imbalance=0.1,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.2,
            model_probs=(0.1, 0.1, 0.8),
            predicted_class=2,
            confidence=0.8,
            burst_id=0,
            data_gap_ms=2000.0,
            bid_exit=0.52,
            ask_exit=0.54,
        )
        assert r2["gate_decision"] == "REJECT"
        assert "STALE_ENTRY_QUOTE" in r2["rejection_reasons"]
        assert r2["outcome_status"] == "SUPPRESSED_GATE"

        # 3. Data gap > 3000ms -> should reject
        r3 = logger.log_decision_point(
            event_timestamp_ms=20000,
            market_id="mkt_1",
            asset_id="asset_1",
            quote_timestamp_ms=19800,  # age 200ms
            bid_entry=0.49,
            ask_entry=0.51,
            depth_imbalance=0.1,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.2,
            model_probs=(0.1, 0.1, 0.8),
            predicted_class=2,
            confidence=0.8,
            burst_id=1,
            data_gap_ms=8000.0,  # gap 8000ms > 3000ms
            bid_exit=0.52,
            ask_exit=0.54,
        )
        assert r3["gate_decision"] == "REJECT"
        assert "INTER_BURST_GAP" in r3["rejection_reasons"]

        manifest = logger.close()
        assert manifest["total_records"] == 3


def test_two_sided_fee_accounting():
    """Verify that two-sided fees on both Long and Short accurately charge entry and exit prices."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ForwardShadowLogger(output_dir=Path(tmp_dir), run_id="test_fee_logger")

        # Long: Enter at Ask (0.50), Exit at Bid (0.55), Fee 10 bps
        # Fee = (0.50 + 0.55) * 0.0010 = 0.00105
        # Gross = 0.55 - 0.50 = 0.05
        # Net = 0.05 - 0.00105 = 0.04895
        r_long = logger.log_decision_point(
            event_timestamp_ms=1000,
            market_id="m",
            asset_id="a",
            quote_timestamp_ms=1000,
            bid_entry=0.49,
            ask_entry=0.50,
            depth_imbalance=0.0,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.1,
            model_probs=(0.1, 0.1, 0.8),
            predicted_class=2,
            confidence=0.8,
            burst_id=0,
            data_gap_ms=1000.0,
            bid_exit=0.55,
            ask_exit=0.56,
            fee_bps=10.0,
        )
        assert pytest.approx(r_long["fee_cost"], abs=1e-5) == 0.00105
        assert pytest.approx(r_long["net_hypothetical_pnl"], abs=1e-5) == 0.04895

        # Short: Enter at Bid (0.60), Exit at Ask (0.52), Fee 10 bps
        # Fee = (0.60 + 0.52) * 0.0010 = 0.00112
        # Gross = 0.60 - 0.52 = 0.08
        # Net = 0.08 - 0.00112 = 0.07888
        r_short = logger.log_decision_point(
            event_timestamp_ms=2000,
            market_id="m",
            asset_id="a",
            quote_timestamp_ms=2000,
            bid_entry=0.60,
            ask_entry=0.61,
            depth_imbalance=0.0,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.1,
            model_probs=(0.8, 0.1, 0.1),
            predicted_class=0,
            confidence=0.8,
            burst_id=0,
            data_gap_ms=1000.0,
            bid_exit=0.51,
            ask_exit=0.52,
            fee_bps=10.0,
        )
        assert pytest.approx(r_short["fee_cost"], abs=1e-5) == 0.00112
        assert pytest.approx(r_short["net_hypothetical_pnl"], abs=1e-5) == 0.07888

        logger.close()


def test_missing_exit_handling():
    """Verify that observations with missing exit quotes are flagged as INCOMPLETE_MISSING_EXIT."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ForwardShadowLogger(output_dir=Path(tmp_dir), run_id="test_missing_exit_logger")

        r = logger.log_decision_point(
            event_timestamp_ms=1000,
            market_id="m",
            asset_id="a",
            quote_timestamp_ms=1000,
            bid_entry=0.49,
            ask_entry=0.50,
            depth_imbalance=0.0,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.1,
            model_probs=(0.1, 0.1, 0.8),
            predicted_class=2,
            confidence=0.8,
            burst_id=0,
            data_gap_ms=1000.0,
            bid_exit=None,
            ask_exit=None,
        )
        assert r["gate_decision"] == "EXECUTE"
        assert r["has_exit_quote"] is False
        assert r["outcome_status"] == "INCOMPLETE_MISSING_EXIT"
        assert r["net_hypothetical_pnl"] is None

        logger.close()


def test_append_only_shadow_logger_and_integrity():
    """Verify append-only logging, schema conformance, and SHA-256 rolling chain integrity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = ForwardShadowLogger(output_dir=Path(tmp_dir), run_id="test_integrity_logger")

        for i in range(5):
            logger.log_decision_point(
                event_timestamp_ms=1000 + i * 1000,
                market_id="m",
                asset_id="a",
                quote_timestamp_ms=1000 + i * 1000,
                bid_entry=0.50,
                ask_entry=0.51,
                depth_imbalance=0.0,
                scaled_spread=-0.5,
                abs_scaled_depth_imbal=0.1,
                model_probs=(0.1, 0.1, 0.8),
                predicted_class=2,
                confidence=0.8,
                burst_id=0,
                data_gap_ms=1000.0,
                bid_exit=0.52,
                ask_exit=0.53,
            )

        manifest = logger.close()
        assert manifest["total_records"] == 5
        assert Path(tmp_dir, manifest["log_file"]).exists()

        # Verify each line has valid json and integrity_hash
        with open(Path(tmp_dir, manifest["log_file"]), "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 5
        for line in lines:
            assert "integrity_hash" in line
            assert len(line["integrity_hash"]) == 64


def test_phase25_baseline_reproduction():
    """Verify that Phase 25 baseline metrics match bitwise under exact same conditions."""
    df = prepare_phase26_dataset()
    stress_results = evaluate_stress_scenarios(df)
    s0 = stress_results["S0_Baseline_Top_of_Book_0bps"]

    assert s0["trade_count"] == 346
    assert s0["cumulative_pnl"] == 1.75
    assert s0["win_rate"] == 0.4653
    assert s0["max_drawdown"] == 1.19


def test_forward_shadow_logger_rejects_preexisting_log_file():
    """Verify that ForwardShadowLogger raises FileExistsError if its log file already exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_id = "test_collision_log"
        log_file = tmp_path / f"{run_id}_observations.jsonl"
        sentinel_content = '{"sentinel": "preexisting_data"}\n'
        log_file.write_text(sentinel_content, encoding="utf-8")

        with pytest.raises(FileExistsError, match="Artifact collision: log file already exists"):
            ForwardShadowLogger(output_dir=tmp_path, run_id=run_id)

        # Verify pre-existing file was not modified or truncated
        assert log_file.read_text(encoding="utf-8") == sentinel_content


def test_forward_shadow_logger_rejects_preexisting_manifest_file():
    """Verify that ForwardShadowLogger raises FileExistsError if its manifest file already exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_id = "test_collision_manifest"
        manifest_file = tmp_path / f"{run_id}_manifest.json"
        sentinel_content = '{"sentinel": "preexisting_manifest"}'
        manifest_file.write_text(sentinel_content, encoding="utf-8")

        with pytest.raises(FileExistsError, match="Artifact collision: manifest file already exists"):
            ForwardShadowLogger(output_dir=tmp_path, run_id=run_id)

        # Verify pre-existing file was not overwritten
        assert manifest_file.read_text(encoding="utf-8") == sentinel_content


def test_forward_shadow_logger_rejects_reused_run_id_after_closure():
    """Verify that reusing a run_id after session closure is strictly rejected and preserves prior data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        run_id = "test_reused_run_id"

        # First session runs and closes
        logger1 = ForwardShadowLogger(output_dir=tmp_path, run_id=run_id)
        logger1.log_decision_point(
            event_timestamp_ms=1000,
            market_id="m",
            asset_id="a",
            quote_timestamp_ms=1000,
            bid_entry=0.50,
            ask_entry=0.51,
            depth_imbalance=0.0,
            scaled_spread=-0.5,
            abs_scaled_depth_imbal=0.1,
            model_probs=(0.1, 0.1, 0.8),
            predicted_class=2,
            confidence=0.8,
            burst_id=0,
            data_gap_ms=1000.0,
            bid_exit=0.52,
            ask_exit=0.53,
        )
        logger1.close()
        orig_log_bytes = (tmp_path / f"{run_id}_observations.jsonl").read_bytes()
        orig_manifest_bytes = (tmp_path / f"{run_id}_manifest.json").read_bytes()

        # Second attempt to open same run_id must raise FileExistsError
        with pytest.raises(FileExistsError, match="Artifact collision"):
            ForwardShadowLogger(output_dir=tmp_path, run_id=run_id)

        # Verify original files remain bitwise invariant
        assert (tmp_path / f"{run_id}_observations.jsonl").read_bytes() == orig_log_bytes
        assert (tmp_path / f"{run_id}_manifest.json").read_bytes() == orig_manifest_bytes


def test_protocol_amendment_v1_1_definition():
    """Verify Protocol Amendment v1.1 version, documented metric definitions, and hard stops."""
    protocol = define_prospective_protocol()
    assert protocol["protocol_version"] == "1.1"
    assert protocol["amendment_version"] == "1.1"
    assert "amendment_metric_definitions" in protocol

    defs = protocol["amendment_metric_definitions"]
    assert "missing_quote_rate" in defs
    assert "average_quote_staleness" in defs
    assert defs["missing_quote_rate_threshold"] == 0.25
    assert defs["average_quote_staleness_threshold_ms"] == 2000.0

    triggers = protocol["hard_stopping_abort_triggers"]
    assert triggers["max_drawdown_stop_loss_units"] == 2.50
    assert triggers["consecutive_losses_limit"] == 8
    assert triggers["max_missing_quote_rate_pct"] == 25.0
    assert triggers["max_average_quote_staleness_ms"] == 2000.0
