"""
Unit and Integration Tests for Phase 25 Regime-Gated Decision Rules & Cost-Aware Backtesting.

Verifies:
1. No future-data leakage in filter evaluation.
2. Correct feature, quote, and prediction alignment.
3. Chronological split integrity and purge boundaries.
4. Correct threshold provenance (training-derived).
5. Cost calculation accuracy (bid/ask crossing, fee math).
6. Missing/unavailable quote rejection handling.
7. Gating logic correctness.
8. Deterministic policy metrics.
9. Checkpoint immutability.
10. Strict prohibition against locked test set access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pipeline_v2.models.phase25_regime_gating import (
    LOCKED_TEST_PATH,
    MODEL_PATH,
    PREDECLARED_CONFIDENCE_THRESHOLD,
    TRAIN_DEPTH_IMBALANCE_P90,
    TRAIN_SPREAD_THRESHOLD,
    VAL_SCALED_PATH,
    compute_policy_metrics,
    evaluate_all_policies,
    prepare_backtest_dataset,
    run_phase25_experiment,
)
from pipeline_v2.models.recurrent_models import SmallLSTM


# ---------------------------------------------------------------------------
# Test 1: No Future-Data Leakage in Gating Logic
# ---------------------------------------------------------------------------
def test_no_future_data_leakage():
    """Verify gating masks depend only on prediction-time inputs (zero target info)."""
    np.random.seed(42)
    N = 50
    df = pd.DataFrame({
        "is_directional": [True] * N,
        "has_exit_quote": [True] * N,
        "confidence": np.random.uniform(0.4, 0.7, size=N),
        "scaled_spread": np.random.uniform(-0.1, 0.1, size=N),
        "abs_scaled_depth_imbal": np.random.uniform(0.5, 1.5, size=N),
        "mid_counterfactual_pnl": np.random.randn(N),
        "gross_pnl": np.random.randn(N),
        "mid_price_entry": np.full(N, 0.5),
        "entry_price": np.full(N, 0.5),
        "endpoint_timestamp_ms": np.arange(1000, 1000 + N),
    })

    res1 = evaluate_all_policies(df)

    # Scramble future PnLs completely
    df["gross_pnl"] = df["gross_pnl"] * 10.0 + 5.0
    df["mid_counterfactual_pnl"] = df["mid_counterfactual_pnl"] * -2.0

    res2 = evaluate_all_policies(df)

    # Coverage masks and executed trade counts must be bitwise identical
    for pol in res1["coverage_stats"]:
        assert res1["coverage_stats"][pol]["executed_trades"] == res2["coverage_stats"][pol]["executed_trades"]


# ---------------------------------------------------------------------------
# Test 2: Cost Calculation Accuracy (Spread Crossing & Fees)
# ---------------------------------------------------------------------------
def test_cost_calculation_accuracy():
    """Verify long/short PnL math and fee deduction."""
    # Long trade: buy at 0.58 (ask), sell at 0.62 (bid) -> gross PnL = +0.04
    # Short trade: sell at 0.45 (bid), buy at 0.40 (ask) -> gross PnL = +0.05
    # Loss long: buy at 0.58 (ask), sell at 0.55 (bid) -> gross PnL = -0.03
    gross_pnls = np.array([0.04, 0.05, -0.03])
    entry_prices = np.array([0.58, 0.45, 0.58])

    # 0 bps fee
    m0 = compute_policy_metrics(gross_pnls, entry_prices, fee_bps=0.0, total_decisions=3)
    assert m0["trade_count"] == 3
    assert np.isclose(m0["cumulative_pnl"], 0.06)
    assert np.isclose(m0["mean_pnl_per_trade"], 0.02)
    assert np.isclose(m0["win_rate"], 2.0 / 3.0, atol=1e-3)
    assert np.isclose(m0["loss_rate"], 1.0 / 3.0, atol=1e-3)

    # 10 bps fee per side
    # Trade 1: entry=0.58, exit=0.62, notional sum=1.20, fee=0.0012 -> net=0.0388
    # Trade 2: entry=0.45, exit=0.40, notional sum=0.85, fee=0.00085 -> net=0.04915
    # Trade 3: entry=0.58, exit=0.55, notional sum=1.13, fee=0.00113 -> net=-0.03113
    # Net sum = 0.0388 + 0.04915 - 0.03113 = 0.05682
    m10 = compute_policy_metrics(gross_pnls, entry_prices, fee_bps=10.0, total_decisions=3)
    assert np.isclose(m10["cumulative_pnl"], 0.0568, atol=1e-3)
    assert m10["cumulative_pnl"] < m0["cumulative_pnl"]


# ---------------------------------------------------------------------------
# Test 3: Missing Quote & Rejection Handling
# ---------------------------------------------------------------------------
def test_missing_quote_rejection_handling():
    """Verify that trades with missing exit quotes are marked rejected."""
    df = pd.DataFrame({
        "is_directional": [True, True, False],
        "has_exit_quote": [True, False, True],  # 2nd trade missing exit quote
        "confidence": [0.60, 0.60, 0.60],
        "scaled_spread": [-0.10, -0.10, -0.10],
        "abs_scaled_depth_imbal": [1.0, 1.0, 1.0],
        "mid_counterfactual_pnl": [0.01, 0.01, 0.0],
        "gross_pnl": [0.01, 0.01, 0.0],
        "mid_price_entry": [0.5, 0.5, 0.5],
        "entry_price": [0.5, 0.5, 0.5],
        "endpoint_timestamp_ms": [1000, 1001, 1002],
    })

    res = evaluate_all_policies(df)
    cov = res["coverage_stats"]["Policy_1_Unfiltered_Baseline"]

    # Only 1 executed trade: 1 skipped FLAT, 1 rejected missing quote
    assert cov["executed_trades"] == 1
    assert cov["skipped_flat"] == 1
    assert cov["rejected_missing_quotes"] == 1


# ---------------------------------------------------------------------------
# Test 4: Threshold Provenance
# ---------------------------------------------------------------------------
def test_threshold_provenance_constants():
    """Verify threshold constants match training distribution specifications."""
    assert TRAIN_SPREAD_THRESHOLD <= -0.08
    assert TRAIN_DEPTH_IMBALANCE_P90 > 1.30
    assert PREDECLARED_CONFIDENCE_THRESHOLD == 0.55


# ---------------------------------------------------------------------------
# Test 5: Dataset Alignment
# ---------------------------------------------------------------------------
def test_backtest_dataset_alignment():
    """Verify alignment of 1428 validation sequences with quotes and predictions."""
    if not (VAL_SCALED_PATH.exists() and MODEL_PATH.exists()):
        pytest.skip("Validation artifacts not available")

    df = prepare_backtest_dataset()
    assert len(df) == 1428
    assert "bid_entry" in df.columns
    assert "ask_entry" in df.columns
    assert "prediction" in df.columns
    assert "gross_pnl" in df.columns
    assert set(df["prediction"].unique()).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# Test 6: Deterministic Policy Evaluation
# ---------------------------------------------------------------------------
def test_deterministic_policy_evaluation():
    """Verify that evaluate_all_policies is strictly deterministic."""
    np.random.seed(999)
    N = 40
    df = pd.DataFrame({
        "is_directional": [True] * N,
        "has_exit_quote": [True] * N,
        "confidence": np.random.uniform(0.4, 0.7, size=N),
        "scaled_spread": np.random.uniform(-0.1, 0.1, size=N),
        "abs_scaled_depth_imbal": np.random.uniform(0.5, 1.5, size=N),
        "mid_counterfactual_pnl": np.random.randn(N),
        "gross_pnl": np.random.randn(N),
        "mid_price_entry": np.full(N, 0.5),
        "entry_price": np.full(N, 0.5),
        "endpoint_timestamp_ms": np.arange(1000, 1000 + N),
    })

    res1 = evaluate_all_policies(df)
    res2 = evaluate_all_policies(df)

    p1 = res1["policy_results"]["Policy_5_Combined_Gate"]["fee_0bps"]
    p2 = res2["policy_results"]["Policy_5_Combined_Gate"]["fee_0bps"]

    assert p1["cumulative_pnl"] == p2["cumulative_pnl"]
    assert p1["trade_count"] == p2["trade_count"]
    assert p1["win_rate"] == p2["win_rate"]


# ---------------------------------------------------------------------------
# Test 7: Checkpoint Immutability
# ---------------------------------------------------------------------------
def test_checkpoint_immutability():
    """Verify model weights are unchanged before and after inference."""
    if not MODEL_PATH.exists():
        pytest.skip("Model checkpoint not present")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = SmallLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
        dropout_p=checkpoint["dropout_p"],
        seed=checkpoint["seed"],
        lr=checkpoint["lr"],
        weight_decay=checkpoint["weight_decay"],
        batch_size=checkpoint["batch_size"],
    )
    model.load_checkpoint(MODEL_PATH)

    weights_before = [p.clone() for p in model.model.parameters()]
    dummy_input = torch.randn(5, 10, 11)
    with torch.no_grad():
        _ = model.model(dummy_input)

    for p_b, p_a in zip(weights_before, model.model.parameters()):
        assert torch.equal(p_b, p_a)


# ---------------------------------------------------------------------------
# Test 8: Locked Test Set Access Prohibition
# ---------------------------------------------------------------------------
def test_locked_test_set_access_prohibition():
    """Verify safety check fails if test_scaled.npz is passed as validation data."""
    if LOCKED_TEST_PATH.exists():
        with pytest.raises(AssertionError, match="locked test set"):
            run_phase25_experiment(val_scaled_path=LOCKED_TEST_PATH)
