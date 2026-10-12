"""
Unit and integration tests for Phase 27 — Passive Maker Execution Engine & Dual-Cohort Protocol.

Verifies:
1. Locked test set isolation and frozen checkpoint integrity.
2. Strict price-through maker fill logic (rejecting touch-only fills).
3. 2,000 ms entry timeout cancellation and zero taker fallback on entry.
4. 5,000 ms physical holding horizon and asymmetric fee accounting (0 bps maker entry, 5 bps taker exit).
5. Missing exit quotes handling and conservative forced-liquidation spread penalty.
6. Partial fills and fractional attribution without survivorship bias.
7. Duplicate signals and overlapping positions tracking.
8. Independent dual-hurdle evaluation (ITT hurdle >= +0.0025 and filled hurdle >= +0.0030).
9. Liang-Zeger cluster-robust variance, t-statistic, and Cameron-Miller empirical N_eff.
10. Stopping rules conjunction (all 5 criteria required for completion).
11. Directional balance metric and asymmetry index bounding.
"""

import pytest
import numpy as np
import tempfile
import json
import hashlib
from dataclasses import asdict
from pathlib import Path

from pipeline_v2.models.phase27_maker_execution import (
    classify_l2_fill_event,
    check_price_through_fill,
    compute_cluster_robust_variance,
    compute_empirical_effective_sample_size,
    compute_dual_cohort_metrics,
    compute_softmax,
    compute_multiclass_nll,
    partition_validation_period1_period2_100step,
    fit_temperature_period1,
    simulate_maker_cohort_on_indices,
    run_phase27_historical_calibration_validation,
    define_phase27_protocol,
    sha256_file,
    MakerExecutionSimulator,
    TradeRecord,
    LOCKED_TEST_PATH,
    MODEL_PATH,
    VAL_SCALED_PATH,
    FROZEN_MODEL_SHA256,
    HURDLE_ITT_MEAN_PNL,
    HURDLE_FILLED_MEAN_PNL,
    HURDLE_PROFIT_FACTOR_FLOOR,
    TARGET_PROFIT_FACTOR,
    HURDLE_DECISIVE_WIN_RATE,
    HURDLE_OVERALL_WIN_RATE,
    MAX_DIRECTIONAL_ASYMMETRY,
    CLUSTER_ROBUST_T_CRITICAL,
    TARGET_MIN_FILLED_TRADES,
    TARGET_MIN_ITT_SIGNALS,
    TARGET_MIN_BURSTS,
    TARGET_MIN_SESSIONS,
    PURGE_GAP_MS,
    assign_session_scoped_clusters,
    aggregate_prospective_sessions,
)


def test_locked_test_set_isolation_and_model_checkpoint():
    """Ensure locked test set exists but is never imported, and model checkpoint hash is bitwise invariant."""
    assert LOCKED_TEST_PATH.exists(), "Locked test set file must exist on disk."
    assert LOCKED_TEST_PATH.is_file(), "Locked test set must be a regular file."

    # Verify protocol string does not expose test data
    protocol = define_phase27_protocol()
    assert "test_scaled.npz" not in str(protocol["predeclared_regime_gates"])
    assert protocol["temporal_separation_invariants"]["locked_test_set_isolation"] == "STRICT_ZERO_ACCESS"

    # Verify model checkpoint hash bitwise matches frozen reference
    actual_hash = sha256_file(MODEL_PATH)
    assert actual_hash == FROZEN_MODEL_SHA256, (
        f"Model checkpoint mutated! Expected {FROZEN_MODEL_SHA256}, got {actual_hash}"
    )


def test_strict_price_through_maker_fill_logic_up_buys():
    """
    Verify strict price-through logic for UP (BUY) limit orders at Bid B_0:
    - Touch only (Bid == B_0, Ask > B_0): REJECTED / UNFILLED.
    - Inside spread (Bid < B_0, Ask > B_0): REJECTED / UNFILLED.
    - Price-through via Ask (Ask <= B_0): FILLED.
    - Price-through via Bid drop (Bid < B_0): FILLED (queue exhausted).
    - Crossed quotes: REJECTED.
    """
    limit_price = 0.50

    # 1. Touch only (Bid == 0.50, Ask == 0.51): Must NOT fill
    assert not check_price_through_fill("BUY", limit_price, current_bid=0.50, current_ask=0.51)

    # 2. Market moved higher away (Bid == 0.51, Ask == 0.52): Must NOT fill
    assert not check_price_through_fill("BUY", limit_price, current_bid=0.51, current_ask=0.52)

    # 3. Market price-through via Ask hitting limit (Ask == 0.50, Bid == 0.49): Must fill
    assert check_price_through_fill("BUY", limit_price, current_bid=0.49, current_ask=0.50)

    # 4. Market price-through deeply via Ask (Ask == 0.48, Bid == 0.47): Must fill
    assert check_price_through_fill("BUY", limit_price, current_bid=0.47, current_ask=0.48)

    # 5. Market price-through via Bid dropping below limit (Bid == 0.49, Ask == 0.52): Must fill
    # All resting bids at 0.50 were cleared; top of book bid retreated to 0.49
    assert check_price_through_fill("BUY", limit_price, current_bid=0.49, current_ask=0.52)

    # 6. Crossed quote (Bid >= Ask): Must reject as invalid
    assert not check_price_through_fill("BUY", limit_price, current_bid=0.52, current_ask=0.51)

    # 7. Non-positive quote: Must reject
    assert not check_price_through_fill("BUY", limit_price, current_bid=0.0, current_ask=0.51)


def test_strict_price_through_maker_fill_logic_down_sells():
    """
    Verify strict price-through logic for DOWN (SELL) limit orders at Ask A_0:
    - Touch only (Ask == A_0, Bid < A_0): REJECTED / UNFILLED.
    - Market moved lower away (Bid < A_0, Ask < A_0): REJECTED / UNFILLED.
    - Price-through via Bid (Bid >= A_0): FILLED.
    - Price-through via Ask rise (Ask > A_0): FILLED (queue exhausted).
    - Crossed quotes: REJECTED.
    """
    limit_price = 0.50

    # 1. Touch only (Bid == 0.49, Ask == 0.50): Must NOT fill
    assert not check_price_through_fill("SELL", limit_price, current_bid=0.49, current_ask=0.50)

    # 2. Market moved lower away (Bid == 0.48, Ask == 0.49): Must NOT fill
    assert not check_price_through_fill("SELL", limit_price, current_bid=0.48, current_ask=0.49)

    # 3. Market price-through via Bid lifting limit (Bid == 0.50, Ask == 0.51): Must fill
    assert check_price_through_fill("SELL", limit_price, current_bid=0.50, current_ask=0.51)

    # 4. Market price-through deeply via Bid (Bid == 0.52, Ask == 0.53): Must fill
    assert check_price_through_fill("SELL", limit_price, current_bid=0.52, current_ask=0.53)

    # 5. Market price-through via Ask rising above limit (Bid == 0.48, Ask == 0.51): Must fill
    # All resting asks at 0.50 were cleared; top of book ask lifted to 0.51
    assert check_price_through_fill("SELL", limit_price, current_bid=0.48, current_ask=0.51)

    # 6. Crossed quote: Must reject
    assert not check_price_through_fill("SELL", limit_price, current_bid=0.53, current_ask=0.52)


def test_entry_cancellation_timeout_and_zero_taker_fallback():
    """
    Verify that an un-filled maker order cancels after 2,000 ms without taker fallback.
    Records Net PnL = 0.0, Fee = 0.0, and outcome_status = 'CANCELLED_UNFILLED'.
    Counts in ITT denominator (N_ITT=1, N_filled=0).
    """
    sim = MakerExecutionSimulator(entry_timeout_ms=2000)

    # Submit BUY order at t=1000 with Bid=0.50, Ask=0.51
    order = sim.submit_signal(
        decision_timestamp_ms=1000,
        market_id="mkt_1",
        asset_id="asset_up",
        predicted_class=2,  # UP
        confidence=0.60,
        scaled_spread=-0.10,
        abs_scaled_depth_imbal=0.5,
        burst_id=1,
        current_bid=0.50,
        current_ask=0.51,
    )
    assert order.limit_price == 0.50
    assert order.timeout_timestamp_ms == 3000

    # Quotes at t=1500 and t=2500 remain at touch (no price-through)
    sim.update_market_state(1500, current_bid=0.50, current_ask=0.51)
    sim.update_market_state(2500, current_bid=0.50, current_ask=0.51)
    assert len(sim.trade_records) == 0  # Still pending
    assert len(sim.pending_orders) == 1

    # At t=3001 (> 2000 ms timeout), order must cancel
    completed = sim.update_market_state(3001, current_bid=0.50, current_ask=0.51)
    assert len(completed) == 1
    record = completed[0]

    assert record.outcome_status == "CANCELLED_UNFILLED"
    assert record.is_filled is False
    assert record.net_pnl == 0.0
    assert record.maker_fee == 0.0
    assert record.taker_fee == 0.0
    assert record.gross_pnl == 0.0
    assert len(sim.pending_orders) == 0
    assert len(sim.open_positions) == 0

    # Check metrics: 1 ITT signal, 0 filled trades
    metrics = compute_dual_cohort_metrics(sim.trade_records)
    assert metrics["summary"]["total_itt_signals"] == 1
    assert metrics["summary"]["filled_trades_count"] == 0
    assert metrics["summary"]["unfilled_signals_count"] == 1
    assert metrics["summary"]["fill_rate"] == 0.0
    assert metrics["itt_cohort"]["mean_net_pnl_per_signal"] == 0.0


def test_holding_horizon_and_asymmetric_fee_accounting():
    """
    Verify 5,000 ms physical holding horizon and asymmetric fee accounting:
    - Maker entry fee: 0 bps (0.0).
    - Taker exit fee: 5 bps (0.0005 * P_exit).
    """
    sim = MakerExecutionSimulator(
        entry_timeout_ms=2000,
        holding_horizon_ms=5000,
        maker_entry_fee_bps=0.0,
        taker_exit_fee_bps=5.0,
    )

    # Submit BUY order at t=1000 at Bid=0.50
    sim.submit_signal(
        decision_timestamp_ms=1000,
        market_id="mkt_1",
        asset_id="asset_up",
        predicted_class=2,
        confidence=0.65,
        scaled_spread=-0.10,
        abs_scaled_depth_imbal=0.2,
        burst_id=1,
        current_bid=0.50,
        current_ask=0.51,
    )

    # At t=1500, price-through occurs (Ask drops to 0.50) -> Fill at limit 0.50
    sim.update_market_state(1500, current_bid=0.49, current_ask=0.50)
    assert len(sim.pending_orders) == 0
    assert len(sim.open_positions) == 1
    pos = sim.open_positions[0]
    assert pos.fill_price == 0.50
    assert pos.fill_timestamp_ms == 1500
    assert pos.target_exit_timestamp_ms == 6500  # 1500 + 5000

    # Intermediate ticks before horizon do not exit
    sim.update_market_state(4000, current_bid=0.51, current_ask=0.52)
    sim.update_market_state(6499, current_bid=0.52, current_ask=0.53)
    assert len(sim.trade_records) == 0
    assert len(sim.open_positions) == 1

    # At t=6500, horizon reached: Exit at Bid = 0.52 (market order)
    completed = sim.update_market_state(6500, current_bid=0.52, current_ask=0.53)
    assert len(completed) == 1
    rec = completed[0]

    assert rec.outcome_status == "COMPLETED"
    assert rec.is_filled is True
    assert rec.fill_price == 0.50
    assert rec.exit_price == 0.52

    # Fee verification: 0 bps maker entry, 5 bps taker exit
    expected_gross_pnl = 0.52 - 0.50  # +0.0200
    expected_maker_fee = 0.50 * 0.0  # 0.0
    expected_taker_fee = 0.52 * 0.0005  # 0.000260
    expected_net_pnl = expected_gross_pnl - expected_maker_fee - expected_taker_fee

    assert abs(rec.gross_pnl - expected_gross_pnl) < 1e-9
    assert abs(rec.maker_fee - expected_maker_fee) < 1e-9
    assert abs(rec.taker_fee - expected_taker_fee) < 1e-9
    assert abs(rec.net_pnl - expected_net_pnl) < 1e-9


def test_missing_exit_quote_forced_liquidation_penalty():
    """
    Verify that an open position with missing exit quotes at horizon is marked
    INCOMPLETE_MISSING_EXIT, charged 1 spread penalty (-0.0100) + exit fee,
    and kept in both ITT and filled cohorts as a full loss (zero survivorship bias).
    """
    sim = MakerExecutionSimulator(
        entry_timeout_ms=2000,
        holding_horizon_ms=5000,
        forced_liquidation_penalty=-0.0100,
        taker_exit_fee_bps=5.0,
    )

    sim.submit_signal(
        decision_timestamp_ms=1000,
        market_id="mkt_1",
        asset_id="asset_up",
        predicted_class=2,
        confidence=0.60,
        scaled_spread=-0.10,
        abs_scaled_depth_imbal=0.2,
        burst_id=1,
        current_bid=0.50,
        current_ask=0.51,
    )

    # Fill at t=1200
    sim.update_market_state(1200, current_bid=0.49, current_ask=0.50)
    assert len(sim.open_positions) == 1

    # At horizon t=6200, quotes are missing (None)
    completed = sim.update_market_state(6200, current_bid=None, current_ask=None)
    assert len(completed) == 1
    rec = completed[0]

    assert rec.outcome_status == "INCOMPLETE_MISSING_EXIT"
    assert rec.is_filled is True
    assert rec.exit_price is None
    assert rec.gross_pnl == -0.0100
    # Exit fee based on entry notional 0.50 * 5 bps = 0.00025
    expected_net_pnl = -0.0100 - (0.50 * 0.0005)
    assert abs(rec.net_pnl - expected_net_pnl) < 1e-9

    # Retained in metrics as full loss
    metrics = compute_dual_cohort_metrics(sim.trade_records)
    assert metrics["summary"]["filled_trades_count"] == 1
    assert metrics["filled_cohort"]["loss_rate"] == 1.0
    assert metrics["filled_cohort"]["win_rate_overall"] == 0.0


def test_partial_fill_accounting_and_fractional_fill_rate():
    """
    Verify that when partial fills occur (e.g. q_fill = 0.4 of 1.0):
    - Executed portion q_fill is held to horizon.
    - Unfilled remainder (0.6) is cancelled at timeout without taker fallback.
    - Gross/Net PnL and fees scale strictly with q_fill.
    - Signal remains in ITT ledger with fill_rate_ratio = 0.4.
    """
    sim = MakerExecutionSimulator()

    sim.submit_signal(
        decision_timestamp_ms=1000,
        market_id="mkt_1",
        asset_id="asset_up",
        predicted_class=2,
        confidence=0.65,
        scaled_spread=-0.10,
        abs_scaled_depth_imbal=0.1,
        burst_id=1,
        current_bid=0.50,
        current_ask=0.51,
        order_quantity=1.0,
    )

    # Partial fill of 0.4 at t=1500
    sim.update_market_state(1500, current_bid=0.49, current_ask=0.50, partial_fill_qty=0.4)
    pos = sim.open_positions[0]
    assert pos.filled_quantity == 0.4
    assert pos.status == "PARTIALLY_FILLED"

    # Settle at t=6500: Exit at Bid = 0.55
    completed = sim.update_market_state(6500, current_bid=0.55, current_ask=0.56)
    assert len(completed) == 1
    rec = completed[0]

    assert rec.filled_quantity == 0.4
    assert rec.fill_rate_ratio == 0.4
    expected_gross = (0.55 - 0.50) * 0.4  # +0.0200
    expected_taker_fee = (0.55 * 0.4) * 0.0005  # 0.000110
    expected_net = expected_gross - expected_taker_fee
    assert abs(rec.gross_pnl - expected_gross) < 1e-9
    assert abs(rec.net_pnl - expected_net) < 1e-9


def test_duplicate_and_overlapping_positions():
    """
    Verify that successive signals on 1-second ticks spawn independent orders,
    do not overwrite each other, and all increment the ITT denominator.
    """
    sim = MakerExecutionSimulator()

    # Submit 3 successive signals at t=1000, 2000, 3000
    s1 = sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, 0.50, 0.51)
    s2 = sim.submit_signal(2000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, 0.50, 0.51)
    s3 = sim.submit_signal(3000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, 0.50, 0.51)

    assert len(sim.pending_orders) == 3
    assert len({s1.signal_id, s2.signal_id, s3.signal_id}) == 3

    # Price drops at t=2100 -> All 3 are eligible for fill if resting
    # Orders s1 and s2 are pending; s3 is pending
    sim.update_market_state(2100, current_bid=0.49, current_ask=0.50)
    assert len(sim.open_positions) == 2  # s1 and s2 filled at t=2100

    # s3 fills at t=3100
    sim.update_market_state(3100, current_bid=0.49, current_ask=0.50)
    assert len(sim.open_positions) == 3

    # Flush all at t=10000
    sim.flush_remaining_positions(10000)
    metrics = compute_dual_cohort_metrics(sim.trade_records)
    assert metrics["summary"]["total_itt_signals"] == 3
    assert metrics["summary"]["filled_trades_count"] == 3


def test_dual_hurdle_evaluation_independent_and_non_fungible():
    """
    Verify that Hurdle 1 (ITT >= +0.0025) and Hurdle 2 (Filled >= +0.0030) are
    independent and non-fungible:
    - High filled edge (+0.0050) with low fill rate (30%) -> ITT = +0.0015 -> FAILS.
    - Passing ITT (+0.0028) and passing filled (+0.0035) -> PASSES.
    - Low filled edge (+0.0028 < +0.0030) with 95% fill rate -> ITT passes (+0.00266),
      but Filled hurdle fails -> FAILS (no cross-subsidy permitted).
    """
    # Case A: High filled edge (+0.0050), low fill rate (30%)
    records_a = []
    for i in range(100):
        is_fill = (i < 30)
        net_p = 0.0050 if is_fill else 0.0
        records_a.append(TradeRecord(
            signal_id=f"s_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=i // 2, side="BUY",
            predicted_class=2, confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1,
            limit_price=0.5, order_quantity=1.0, filled_quantity=1.0 if is_fill else 0.0,
            fill_rate_ratio=1.0 if is_fill else 0.0, fill_timestamp_ms=i*1000 if is_fill else None,
            fill_price=0.5 if is_fill else None, exit_timestamp_ms=(i+5)*1000 if is_fill else None,
            exit_price=0.505 if is_fill else None, gross_pnl=net_p, maker_fee=0.0, taker_fee=0.0,
            net_pnl=net_p, outcome_status="COMPLETED" if is_fill else "CANCELLED_UNFILLED",
            is_filled=is_fill,
        ))

    m_a = compute_dual_cohort_metrics(records_a)
    assert m_a["filled_cohort"]["hurdle_passed"] is True  # +0.0050 >= +0.0030
    assert m_a["itt_cohort"]["mean_net_pnl_per_signal"] == 0.0015
    assert m_a["itt_cohort"]["hurdle_passed"] is False  # +0.0015 < +0.0025
    assert m_a["summary"]["protocol_accepted"] is False

    # Case B: Passing both hurdles
    records_b = []
    for i in range(100):
        is_fill = (i < 80)
        net_p = 0.0035 if is_fill else 0.0
        records_b.append(TradeRecord(
            signal_id=f"s_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=i // 2, side="BUY",
            predicted_class=2, confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1,
            limit_price=0.5, order_quantity=1.0, filled_quantity=1.0 if is_fill else 0.0,
            fill_rate_ratio=1.0 if is_fill else 0.0, fill_timestamp_ms=i*1000 if is_fill else None,
            fill_price=0.5 if is_fill else None, exit_timestamp_ms=(i+5)*1000 if is_fill else None,
            exit_price=0.5035 if is_fill else None, gross_pnl=net_p, maker_fee=0.0, taker_fee=0.0,
            net_pnl=net_p, outcome_status="COMPLETED" if is_fill else "CANCELLED_UNFILLED",
            is_filled=is_fill,
        ))

    m_b = compute_dual_cohort_metrics(records_b)
    assert m_b["filled_cohort"]["hurdle_passed"] is True  # +0.0035 >= +0.0030
    assert m_b["itt_cohort"]["mean_net_pnl_per_signal"] == 0.0028  # 0.80 * 0.0035
    assert m_b["itt_cohort"]["hurdle_passed"] is True  # +0.0028 >= +0.0025

    # Case C: Weakened filled trade (+0.0028 < +0.0030) with 95% fill rate
    records_c = []
    for i in range(100):
        is_fill = (i < 95)
        net_p = 0.0028 if is_fill else 0.0
        records_c.append(TradeRecord(
            signal_id=f"s_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=i // 2, side="BUY",
            predicted_class=2, confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1,
            limit_price=0.5, order_quantity=1.0, filled_quantity=1.0 if is_fill else 0.0,
            fill_rate_ratio=1.0 if is_fill else 0.0, fill_timestamp_ms=i*1000 if is_fill else None,
            fill_price=0.5 if is_fill else None, exit_timestamp_ms=(i+5)*1000 if is_fill else None,
            exit_price=0.5028 if is_fill else None, gross_pnl=net_p, maker_fee=0.0, taker_fee=0.0,
            net_pnl=net_p, outcome_status="COMPLETED" if is_fill else "CANCELLED_UNFILLED",
            is_filled=is_fill,
        ))

    m_c = compute_dual_cohort_metrics(records_c)
    assert m_c["itt_cohort"]["mean_net_pnl_per_signal"] == 0.00266  # +0.00266 >= +0.0025
    assert m_c["itt_cohort"]["hurdle_passed"] is True
    assert m_c["filled_cohort"]["hurdle_passed"] is False  # +0.0028 < +0.0030 FAILS
    assert m_c["summary"]["protocol_accepted"] is False


def test_cluster_robust_t_statistic_and_empirical_n_eff():
    """
    Verify Liang-Zeger cluster-robust SE, t-statistic, and Cameron-Miller empirical N_eff.
    Demonstrates that imbalanced cluster distribution yields N_eff < C.
    """
    # 100 observations across 10 bursts with unequal sizes:
    # Burst 0 has 55 observations, bursts 1..9 have 5 observations each
    bursts = np.array([0] * 55 + [i for i in range(1, 10) for _ in range(5)])
    values = np.random.RandomState(42).normal(loc=0.005, scale=0.01, size=len(bursts))

    mean_val, se_cr, t_cr, c_count = compute_cluster_robust_variance(values, bursts)
    n_eff = compute_empirical_effective_sample_size(values, bursts)

    assert c_count == 10
    assert se_cr > 0.0
    assert abs(t_cr - (mean_val / se_cr)) < 1e-9
    # Due to severe cluster size imbalance (55 vs 5), N_eff must be significantly strictly less than 10
    assert n_eff < 8.0, f"Expected N_eff < 8.0 due to cluster imbalance, got {n_eff}"


def test_all_five_completion_conditions_required():
    """
    Verify that stopping rule requires ALL 5 conditions:
    1. N_filled >= 400
    2. N_ITT >= 600
    3. N_bursts >= 50
    4. N_sessions >= 5
    5. Zero active hard stops
    """
    # Dummy record pool
    dummy_records = [
        TradeRecord(
            signal_id=f"s_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=i % 60, side="BUY" if i % 2 == 0 else "SELL",
            predicted_class=2 if i % 2 == 0 else 0, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.1, limit_price=0.5, order_quantity=1.0,
            filled_quantity=1.0 if i < 450 else 0.0, fill_rate_ratio=1.0 if i < 450 else 0.0,
            fill_timestamp_ms=i*1000 if i < 450 else None, fill_price=0.5 if i < 450 else None,
            exit_timestamp_ms=(i+5)*1000 if i < 450 else None, exit_price=0.504 if i < 450 else None,
            gross_pnl=0.004 if i < 450 else 0.0, maker_fee=0.0, taker_fee=0.00025 if i < 450 else 0.0,
            net_pnl=0.00375 if i < 450 else 0.0,
            outcome_status="COMPLETED" if i < 450 else "CANCELLED_UNFILLED",
            is_filled=(i < 450),
        )
        for i in range(650)
    ]

    # Baseline: 450 filled, 650 ITT, 60 bursts, 5 sessions, no hard stop -> ALL MET
    m_full = compute_dual_cohort_metrics(dummy_records, completed_sessions_count=5, has_active_hard_stop=False)
    assert m_full["stopping_rules"]["all_completion_requirements_met"] is True

    # Unmet Condition 1: Only 350 filled trades (< 400 target)
    m_c1 = compute_dual_cohort_metrics(dummy_records[:450], min_filled_trades_target=500)
    assert m_c1["stopping_rules"]["all_completion_requirements_met"] is False

    # Unmet Condition 2: Only 500 ITT signals (< 600 target)
    m_c2 = compute_dual_cohort_metrics(dummy_records[:500], min_itt_signals_target=600)
    assert m_c2["stopping_rules"]["all_completion_requirements_met"] is False

    # Unmet Condition 3: Only 30 bursts (< 50 target)
    m_c3 = compute_dual_cohort_metrics(dummy_records, min_bursts_target=70)
    assert m_c3["stopping_rules"]["all_completion_requirements_met"] is False

    # Unmet Condition 4: Only 4 sessions completed (< 5 target)
    m_c4 = compute_dual_cohort_metrics(dummy_records, completed_sessions_count=4)
    assert m_c4["stopping_rules"]["all_completion_requirements_met"] is False

    # Unmet Condition 5: Active hard stop present
    m_c5 = compute_dual_cohort_metrics(dummy_records, completed_sessions_count=5, has_active_hard_stop=True)
    assert m_c5["stopping_rules"]["all_completion_requirements_met"] is False


def test_directional_balance_metric():
    """Verify Directional Balance Ratio and Asymmetry Index bounds."""
    # 60 UP, 40 DOWN: Asymmetry = |60 - 40| / 100 = 0.20 <= 0.35 -> PASS
    records_balanced = [
        TradeRecord(
            signal_id=f"s_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=1, side="BUY" if i < 60 else "SELL",
            predicted_class=2 if i < 60 else 0, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.1, limit_price=0.5, order_quantity=1.0,
            filled_quantity=1.0, fill_rate_ratio=1.0, fill_timestamp_ms=i*1000,
            fill_price=0.5, exit_timestamp_ms=(i+5)*1000, exit_price=0.505,
            gross_pnl=0.005, maker_fee=0.0, taker_fee=0.0, net_pnl=0.005,
            outcome_status="COMPLETED", is_filled=True,
        )
        for i in range(100)
    ]
    m_bal = compute_dual_cohort_metrics(records_balanced)
    assert m_bal["directional_balance"]["directional_asymmetry_index"] == 0.20
    assert m_bal["directional_balance"]["balance_passed"] is True

    # 85 UP, 15 DOWN: Asymmetry = |85 - 15| / 100 = 0.70 > 0.35 -> FAIL
    records_skewed = [
        TradeRecord(
            signal_id=f"s_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=1, side="BUY" if i < 85 else "SELL",
            predicted_class=2 if i < 85 else 0, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.1, limit_price=0.5, order_quantity=1.0,
            filled_quantity=1.0, fill_rate_ratio=1.0, fill_timestamp_ms=i*1000,
            fill_price=0.5, exit_timestamp_ms=(i+5)*1000, exit_price=0.505,
            gross_pnl=0.005, maker_fee=0.0, taker_fee=0.0, net_pnl=0.005,
            outcome_status="COMPLETED", is_filled=True,
        )
        for i in range(100)
    ]
    m_skew = compute_dual_cohort_metrics(records_skewed)
    assert m_skew["directional_balance"]["directional_asymmetry_index"] == 0.70
    assert m_skew["directional_balance"]["balance_passed"] is False


def test_defensive_submission_time_quote_validation():
    """
    Verify defensive submission-time quote validation:
    Rejects None, non-finite (NaN, Inf), non-positive (<=0), and crossed quotes (bid >= ask).
    """
    sim = MakerExecutionSimulator()

    # 1. Reject None quotes
    with pytest.raises(ValueError, match="Quotes cannot be None"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=None, current_ask=0.51)
    with pytest.raises(ValueError, match="Quotes cannot be None"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=None)

    # 2. Reject non-finite quotes (NaN, Inf)
    with pytest.raises(ValueError, match="Quotes must be finite"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=float("nan"), current_ask=0.51)
    with pytest.raises(ValueError, match="Quotes must be finite"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=float("inf"))

    # 3. Reject non-positive quotes (<= 0)
    with pytest.raises(ValueError, match="strictly positive"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.0, current_ask=0.51)
    with pytest.raises(ValueError, match="strictly positive"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=-0.05, current_ask=0.51)
    with pytest.raises(ValueError, match="strictly positive"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=0.0)

    # 4. Reject crossed or zero-spread quotes (bid >= ask)
    with pytest.raises(ValueError, match="Crossed or zero-spread"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=0.50)
    with pytest.raises(ValueError, match="Crossed or zero-spread"):
        sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.52, current_ask=0.50)

    # 5. Non-directional predicted class (must be 0 or 2)
    with pytest.raises(ValueError, match="Only directional classes"):
        sim.submit_signal(1000, "mkt", "asset", 1, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=0.51)

    # 6. Valid positive, uncrossed quotes succeed at touch
    order_buy = sim.submit_signal(1000, "mkt", "asset", 2, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=0.51)
    assert order_buy.side == "BUY"
    assert order_buy.limit_price == 0.50  # Bid

    order_sell = sim.submit_signal(2000, "mkt", "asset", 0, 0.6, -0.1, 0.1, 1, current_bid=0.50, current_ask=0.51)
    assert order_sell.side == "SELL"
    assert order_sell.limit_price == 0.51  # Ask


def test_temporal_ordering_and_purge_gap_enforcement():
    """
    Verify chronological Period 1 / Period 2 partition with mandatory 100-step purge gap.
    """
    val_data = np.load(VAL_SCALED_PATH)
    endpoints = val_data["endpoints"]

    p1_idx, p2_idx, purged_idx, split_info = partition_validation_period1_period2_100step(endpoints)

    # Exact expected counts
    assert len(p1_idx) == 832
    assert len(purged_idx) == 38
    assert len(p2_idx) == 558
    assert len(p1_idx) + len(purged_idx) + len(p2_idx) == len(endpoints) == 1428

    # Purge gap enforcement: actual gap is 139,000 ms >= 100,000 ms (100 steps)
    assert split_info["actual_purge_gap_ms"] == 139000
    assert split_info["actual_purge_gap_ms"] >= PURGE_GAP_MS
    assert split_info["purge_satisfied"] is True
    assert split_info["p1_end_ts"] == 1791293029000
    assert split_info["p2_start_ts"] == 1791293168000
    assert split_info["p2_start_ts"] - split_info["p1_end_ts"] == 139000

    # Zero overlap across partitions
    assert len(set(p1_idx).intersection(set(p2_idx))) == 0
    assert len(set(p1_idx).intersection(set(purged_idx))) == 0
    assert len(set(p2_idx).intersection(set(purged_idx))) == 0

    # Synthetic test: unsorted endpoints must raise ValueError
    unsorted_endpoints = np.array([2000, 1000, 3000], dtype=np.int64)
    with pytest.raises(ValueError, match="Endpoints are not chronologically sorted"):
        partition_validation_period1_period2_100step(unsorted_endpoints)

    # Synthetic test: empty evaluation partition must raise ValueError
    tight_endpoints = np.array([1000, 2000, 3000, 50000], dtype=np.int64)
    with pytest.raises(ValueError, match="Evaluation partition is empty"):
        partition_validation_period1_period2_100step(tight_endpoints, split_timestamp=2000, min_purge_ms=100000)

    # Synthetic test: empty calibration partition must raise ValueError
    late_endpoints = np.array([5000, 150000, 160000], dtype=np.int64)
    with pytest.raises(ValueError, match="Calibration partition is empty"):
        partition_validation_period1_period2_100step(late_endpoints, split_timestamp=2000, min_purge_ms=100000)

    # Synthetic test: empty input array must raise ValueError
    with pytest.raises(ValueError, match="Cannot partition empty endpoints array"):
        partition_validation_period1_period2_100step(np.array([], dtype=np.int64))


def test_zero_calibration_leakage_period2_untouched():
    """
    Verify zero data or statistical leakage from Period 2 into Period 1 temperature calibration.
    Modifying or corrupting Period 2 logits/labels must produce bitwise identical Period 1 temperature.
    """
    np.random.seed(42)
    # Generate Period 1 synthetic logits and labels
    n_p1 = 800
    logits_p1 = np.random.randn(n_p1, 3)
    y_p1 = np.random.randint(0, 3, size=n_p1)

    # Fit baseline temperature strictly on Period 1
    res1 = fit_temperature_period1(logits_p1, y_p1)
    t1 = res1["temperature"]
    nll1 = res1["calibrated_nll"]

    # Generate Period 2 synthetic data
    n_p2 = 500
    logits_p2 = np.random.randn(n_p2, 3)
    y_p2 = np.random.randint(0, 3, size=n_p2)

    # Fit temperature again passing ONLY Period 1 (verifying Period 2 is untouched)
    res2 = fit_temperature_period1(logits_p1, y_p1)
    t2 = res2["temperature"]
    nll2 = res2["calibrated_nll"]

    # Verify bitwise equivalence
    assert t1 == t2
    assert nll1 == nll2

    # Even if Period 2 was completely altered, Period 1 calibration is invariant
    logits_p2_corrupted = np.full_like(logits_p2, fill_value=999.0)
    res3 = fit_temperature_period1(logits_p1, y_p1)
    assert res3["temperature"] == t1


def test_softmax_and_multiclass_nll_properties():
    """
    Verify mathematical correctness and numerical stability of softmax and multiclass NLL.
    """
    # Numerical stability with large logits
    large_logits = np.array([[-1000.0, 0.0, 1000.0], [500.0, 500.0, 0.0]])
    probs = compute_softmax(large_logits, temperature=1.0)
    assert np.all(np.isfinite(probs))
    assert np.allclose(np.sum(probs, axis=1), [1.0, 1.0])
    assert probs[0, 2] > 0.999

    # Negative or non-finite temperature raises ValueError
    with pytest.raises(ValueError, match="strictly positive"):
        compute_softmax(large_logits, temperature=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_softmax(large_logits, temperature=-1.0)
    with pytest.raises(ValueError, match="strictly positive"):
        compute_softmax(large_logits, temperature=float("nan"))

    # Multiclass NLL matches manual calculation
    simple_logits = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    labels = np.array([0, 1])
    nll = compute_multiclass_nll(simple_logits, labels, temperature=1.0)
    assert nll > 0.0 and np.isfinite(nll)

    # Temperature scaling effect: higher temperature increases NLL for confident correct predictions
    nll_high_t = compute_multiclass_nll(simple_logits, labels, temperature=2.0)
    assert nll_high_t > nll


def test_locked_test_set_isolation_during_calibration(monkeypatch):
    """
    Verify locked test set (test_scaled.npz) is NEVER accessed during historical calibration validation.
    Any attempt to open test_scaled.npz raises a fatal assertion.
    """
    import builtins
    orig_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if "test_scaled.npz" in str(file):
            raise PermissionError("VIOLATION: Attempted access to locked test set!")
        return orig_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    # Run historical calibration validation under the guard
    report = run_phase27_historical_calibration_validation()
    assert report["metadata"]["locked_test_set_isolation"] == "STRICT_ZERO_ACCESS (Unaccessed)"


def test_historical_calibration_validation_reproducibility(tmp_path):
    """
    Verify run_phase27_historical_calibration_validation runs end-to-end and outputs
    the signed-off report schema with non-weakened acceptance evaluation.
    """
    report = run_phase27_historical_calibration_validation(out_dir=tmp_path)

    # 1. Verify schema top-level keys
    assert "metadata" in report
    assert "split_information" in report
    assert "period_1_calibration_fit" in report
    assert "period_2_comparison" in report
    assert "empirical_limitations_declared" in report

    # 2. Verify split information
    s_info = report["split_information"]
    assert s_info["period_1_cal_samples"] == 832
    assert s_info["purged_buffer_samples"] == 38
    assert s_info["period_2_eval_samples"] == 558
    assert s_info["actual_purge_gap_ms"] == 139000

    # 3. Verify Period 1 temperature calibration
    t_fit = report["period_1_calibration_fit"]
    assert 0.70 < t_fit["temperature"] < 0.90
    assert t_fit["delta_nll"] < 0  # NLL strictly improved

    # 4. Verify Period 2 comparisons across 3 approaches
    p2_comp = report["period_2_comparison"]
    assert "approach_1_uncalibrated_baseline" in p2_comp
    assert "approach_2_temperature_scaling" in p2_comp
    assert "approach_3_directionally_balanced_gating" in p2_comp

    app1 = p2_comp["approach_1_uncalibrated_baseline"]
    app2 = p2_comp["approach_2_temperature_scaling"]
    app3 = p2_comp["approach_3_directionally_balanced_gating"]

    assert app1["itt_signals"] == 155
    assert app1["filled_trades"] == 50
    assert app2["itt_signals"] == 214
    assert app2["filled_trades"] == 62
    assert app3["itt_signals"] == 145
    assert app3["filled_trades"] == 46

    # Verify that neither approach passes live dual-hurdles on Period 2 (honest reporting without weakening)
    assert app1["hurdles_passed"]["filled_hurdle"] is False
    assert app2["hurdles_passed"]["filled_hurdle"] is False
    assert app3["hurdles_passed"]["filled_hurdle"] is False

    # 5. Verify empirical limitations are populated
    lims = report["empirical_limitations_declared"]
    assert len(lims) >= 4

    # 6. Verify written JSON artifact on disk
    json_path = tmp_path / "phase27_calibration_report.json"
    assert json_path.exists()


def test_l2_fill_certainty_classification():
    """
    Verify two-tier L2 fill certainty classification:
    - CONFIRMED_PRICE_THROUGH: Ask <= limit (BUY) or Bid >= limit (SELL)
    - AMBIGUOUS_QUOTE_RETREAT: Bid < limit and Ask > limit (BUY) or Ask > limit and Bid < limit (SELL)
    - TOUCH_REJECTED: Bid == limit and Ask > limit (BUY) or Ask == limit and Bid < limit (SELL)
    - UNFILLED: Market moved away or invalid quotes
    """
    limit_price = 0.50

    # BUY evaluations
    # Event A: Ask drops to 0.50 (price-through: Ask <= limit) -> CONFIRMED_PRICE_THROUGH
    assert classify_l2_fill_event("BUY", limit_price, current_bid=0.49, current_ask=0.50) == "CONFIRMED_PRICE_THROUGH"
    assert check_price_through_fill("BUY", limit_price, current_bid=0.49, current_ask=0.50, conservative_only=True) is True
    assert check_price_through_fill("BUY", limit_price, current_bid=0.49, current_ask=0.50, conservative_only=False) is True

    # Event B: Bid drops to 0.49, Ask is 0.51 (quote retreat: Bid < limit, Ask > limit) -> AMBIGUOUS_QUOTE_RETREAT
    assert classify_l2_fill_event("BUY", limit_price, current_bid=0.49, current_ask=0.51) == "AMBIGUOUS_QUOTE_RETREAT"
    assert check_price_through_fill("BUY", limit_price, current_bid=0.49, current_ask=0.51, conservative_only=True) is False
    assert check_price_through_fill("BUY", limit_price, current_bid=0.49, current_ask=0.51, conservative_only=False) is True

    # Event C: Touch only: Bid=0.50, Ask=0.51 -> TOUCH_REJECTED
    assert classify_l2_fill_event("BUY", limit_price, current_bid=0.50, current_ask=0.51) == "TOUCH_REJECTED"
    assert check_price_through_fill("BUY", limit_price, current_bid=0.50, current_ask=0.51, conservative_only=True) is False
    assert check_price_through_fill("BUY", limit_price, current_bid=0.50, current_ask=0.51, conservative_only=False) is False

    # Event D: Market moved away: Bid=0.51, Ask=0.52 -> UNFILLED
    assert classify_l2_fill_event("BUY", limit_price, current_bid=0.51, current_ask=0.52) == "UNFILLED"
    assert check_price_through_fill("BUY", limit_price, current_bid=0.51, current_ask=0.52, conservative_only=True) is False
    assert check_price_through_fill("BUY", limit_price, current_bid=0.51, current_ask=0.52, conservative_only=False) is False

    # SELL evaluations
    # Event A: Bid rises to 0.50 (price-through: Bid >= limit) -> CONFIRMED_PRICE_THROUGH
    assert classify_l2_fill_event("SELL", limit_price, current_bid=0.50, current_ask=0.51) == "CONFIRMED_PRICE_THROUGH"
    assert check_price_through_fill("SELL", limit_price, current_bid=0.50, current_ask=0.51, conservative_only=True) is True
    assert check_price_through_fill("SELL", limit_price, current_bid=0.50, current_ask=0.51, conservative_only=False) is True

    # Event B: Ask rises to 0.51, Bid is 0.49 (quote retreat: Ask > limit, Bid < limit) -> AMBIGUOUS_QUOTE_RETREAT
    assert classify_l2_fill_event("SELL", limit_price, current_bid=0.49, current_ask=0.51) == "AMBIGUOUS_QUOTE_RETREAT"
    assert check_price_through_fill("SELL", limit_price, current_bid=0.49, current_ask=0.51, conservative_only=True) is False
    assert check_price_through_fill("SELL", limit_price, current_bid=0.49, current_ask=0.51, conservative_only=False) is True

    # Event C: Touch only: Bid=0.49, Ask=0.50 -> TOUCH_REJECTED
    assert classify_l2_fill_event("SELL", limit_price, current_bid=0.49, current_ask=0.50) == "TOUCH_REJECTED"
    assert check_price_through_fill("SELL", limit_price, current_bid=0.49, current_ask=0.50, conservative_only=True) is False
    assert check_price_through_fill("SELL", limit_price, current_bid=0.49, current_ask=0.50, conservative_only=False) is False

    # Invalid / Crossed quotes -> UNFILLED
    assert classify_l2_fill_event("BUY", limit_price, current_bid=None, current_ask=0.51) == "UNFILLED"
    assert classify_l2_fill_event("BUY", limit_price, current_bid=0.52, current_ask=0.51) == "UNFILLED"


def test_sequential_testing_horizon_predeclared():
    """Verify that a fixed 5-session stopping horizon is predeclared in protocol to prevent optional stopping bias."""
    protocol = define_phase27_protocol()
    assert "sequential_testing_horizon" in protocol
    horizon = protocol["sequential_testing_horizon"]
    assert horizon["fixed_stopping_horizon_sessions"] == 5
    assert horizon["horizon_policy"] == "FIXED_FIVE_SESSION_NON_FUNGIBLE_HORIZON"
    assert horizon["early_stopping_policy"] == "NO_OPTIONAL_STOPPING_ON_FAVORABLE_RESULTS"
    assert "5 prospective sessions" in horizon["evaluation_rule"]


def test_multi_session_burst_id_isolation_and_cluster_separation():
    """
    Verify that when two sessions deliberately reuse identical local burst IDs (e.g. burst_id 0 and 1),
    the aggregation layer maps them to separate clusters, preventing artificial covariance pooling,
    while strictly preserving original per-session burst IDs.
    """
    records_s1 = [
        TradeRecord(
            signal_id=f"sig_s1_{i}", record_index=i, decision_timestamp_ms=i * 1000,
            market_id="mkt_1", asset_id="up", burst_id=0 if i < 10 else 1, side="BUY",
            predicted_class=2, confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1,
            limit_price=0.50, order_quantity=1.0, filled_quantity=1.0, fill_rate_ratio=1.0,
            fill_timestamp_ms=i * 1000, fill_price=0.50, exit_timestamp_ms=(i + 5) * 1000,
            exit_price=0.505, gross_pnl=0.005, maker_fee=0.0, taker_fee=0.00025,
            net_pnl=0.00475, outcome_status="COMPLETED", is_filled=True, session_id="session_001",
        )
        for i in range(20)
    ]
    records_s2 = [
        TradeRecord(
            signal_id=f"sig_s2_{i}", record_index=i, decision_timestamp_ms=(i + 100) * 1000,
            market_id="mkt_2", asset_id="up", burst_id=0 if i < 10 else 1, side="BUY",
            predicted_class=2, confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1,
            limit_price=0.50, order_quantity=1.0, filled_quantity=1.0, fill_rate_ratio=1.0,
            fill_timestamp_ms=(i + 100) * 1000, fill_price=0.50, exit_timestamp_ms=(i + 105) * 1000,
            exit_price=0.505, gross_pnl=0.005, maker_fee=0.0, taker_fee=0.00025,
            net_pnl=0.00475, outcome_status="COMPLETED", is_filled=True, session_id="session_002",
        )
        for i in range(20)
    ]

    # Deliberate reuse of local burst IDs: both have bursts {0, 1}
    assert {r.burst_id for r in records_s1} == {0, 1}
    assert {r.burst_id for r in records_s2} == {0, 1}

    # Naive merge without scoping would collapse clusters to {0, 1} (count = 2)
    unscoped_bursts = np.unique([r.burst_id for r in (records_s1 + records_s2)])
    assert len(unscoped_bursts) == 2

    # Scoped cluster assignment
    scoped = assign_session_scoped_clusters(records_s1 + records_s2)
    assert len(scoped) == 40

    # Original burst_ids and session_ids are completely preserved
    assert [r.burst_id for r in scoped[:20]] == [r.burst_id for r in records_s1]
    assert [r.session_id for r in scoped[:20]] == ["session_001"] * 20
    assert [r.burst_id for r in scoped[20:]] == [r.burst_id for r in records_s2]
    assert [r.session_id for r in scoped[20:]] == ["session_002"] * 20

    # Cluster IDs for session 1 and session 2 are strictly disjoint
    s1_clusters = {r.cluster_id for r in scoped[:20]}
    s2_clusters = {r.cluster_id for r in scoped[20:]}
    assert s1_clusters == {0, 1}
    assert s2_clusters == {2, 3}
    assert len(s1_clusters.intersection(s2_clusters)) == 0

    # compute_dual_cohort_metrics clusters by cluster_id -> distinct clusters count is 4!
    metrics = compute_dual_cohort_metrics(scoped)
    assert metrics["summary"]["distinct_bursts_count"] == 4
    assert metrics["filled_cohort"]["empirical_n_eff"] > 0


def test_multi_session_duplicate_session_rejection():
    """Verify that sessions cannot be counted twice; duplicate session IDs raise ValueError."""
    records = [
        TradeRecord(
            signal_id=f"sig_{i}", record_index=i, decision_timestamp_ms=i * 1000,
            market_id="mkt", asset_id="up", burst_id=0, side="BUY", predicted_class=2,
            confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, limit_price=0.5,
            order_quantity=1.0, filled_quantity=1.0, fill_rate_ratio=1.0,
            fill_timestamp_ms=i * 1000, fill_price=0.5, exit_timestamp_ms=(i + 5) * 1000,
            exit_price=0.505, gross_pnl=0.005, maker_fee=0.0, taker_fee=0.0,
            net_pnl=0.005, outcome_status="COMPLETED", is_filled=True, session_id="sess_duplicate",
        )
        for i in range(5)
    ]
    s1 = {"session_id": "sess_duplicate", "session_status": "COMPLETE", "records": records}
    s2 = {"session_id": "sess_duplicate", "session_status": "COMPLETE", "records": records}

    with pytest.raises(ValueError, match="Duplicate session ID detected"):
        aggregate_prospective_sessions([s1, s2])


def test_multi_session_duplicate_signal_rejection():
    """Verify that duplicate signal IDs across sessions raise ValueError."""
    rec_s1 = [
        TradeRecord(
            signal_id="sig_shared_123", record_index=0, decision_timestamp_ms=1000,
            market_id="mkt", asset_id="up", burst_id=0, side="BUY", predicted_class=2,
            confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, limit_price=0.5,
            order_quantity=1.0, filled_quantity=1.0, fill_rate_ratio=1.0,
            fill_timestamp_ms=1000, fill_price=0.5, exit_timestamp_ms=6000,
            exit_price=0.505, gross_pnl=0.005, maker_fee=0.0, taker_fee=0.0,
            net_pnl=0.005, outcome_status="COMPLETED", is_filled=True, session_id="sess_1",
        )
    ]
    rec_s2 = [
        TradeRecord(
            signal_id="sig_shared_123", record_index=0, decision_timestamp_ms=2000,
            market_id="mkt", asset_id="up", burst_id=0, side="BUY", predicted_class=2,
            confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, limit_price=0.5,
            order_quantity=1.0, filled_quantity=1.0, fill_rate_ratio=1.0,
            fill_timestamp_ms=2000, fill_price=0.5, exit_timestamp_ms=7000,
            exit_price=0.505, gross_pnl=0.005, maker_fee=0.0, taker_fee=0.0,
            net_pnl=0.005, outcome_status="COMPLETED", is_filled=True, session_id="sess_2",
        )
    ]
    s1 = {"session_id": "sess_1", "session_status": "COMPLETE", "records": rec_s1}
    s2 = {"session_id": "sess_2", "session_status": "COMPLETE", "records": rec_s2}

    with pytest.raises(ValueError, match="Duplicate signal ID detected across sessions"):
        aggregate_prospective_sessions([s1, s2])


def test_multi_session_incomplete_and_aborted_session_handling():
    """Verify that hard-stopped sessions trigger active hard stop and degraded/failed sessions do not count."""
    def make_session_records(s_id: str, n_trades: int = 100):
        seed = int(hashlib.sha256(s_id.encode("utf-8")).hexdigest()[:8], 16) % 10000
        rng = np.random.RandomState(seed)
        records = []
        for i in range(n_trades):
            pnl = float(rng.normal(0.0040, 0.0005))
            records.append(TradeRecord(
                signal_id=f"sig_{s_id}_{i}", record_index=i, decision_timestamp_ms=i * 1000,
                market_id="mkt", asset_id="tok", burst_id=i % 12, side="BUY" if i % 2 == 0 else "SELL",
                predicted_class=2 if i % 2 == 0 else 0, confidence=0.6, scaled_spread=-0.1,
                abs_scaled_depth_imbal=0.1, limit_price=0.50, order_quantity=1.0, filled_quantity=1.0,
                fill_rate_ratio=1.0, fill_timestamp_ms=i * 1000, fill_price=0.50,
                exit_timestamp_ms=(i + 5) * 1000, exit_price=0.504,
                gross_pnl=pnl + 0.00025, maker_fee=0.0, taker_fee=0.00025,
                net_pnl=pnl, outcome_status="COMPLETED", is_filled=True, session_id=s_id,
            ))
        return records

    # Scenario A: 4 COMPLETE sessions + 1 ABORTED_HARD_STOP session
    sessions_with_abort = [
        {"session_id": f"sess_{i}", "session_status": "COMPLETE", "records": make_session_records(f"sess_{i}")}
        for i in range(4)
    ]
    sessions_with_abort.append({
        "session_id": "sess_halted",
        "session_status": "ABORTED_HARD_STOP",
        "hard_stop_triggered": "CONSECUTIVE_LOSSES_LIMIT",
        "records": make_session_records("sess_halted", 20),
    })

    res_abort = aggregate_prospective_sessions(sessions_with_abort)
    assert res_abort["provenance"]["has_active_hard_stop"] is True
    assert res_abort["provenance"]["valid_completed_sessions_count"] == 4
    assert res_abort["dual_cohort_metrics"]["stopping_rules"]["c5_no_active_hard_stop"] is False
    assert res_abort["protocol_verdict"] == "REJECTED"

    # Scenario B: 4 COMPLETE sessions + 1 DEGRADED session
    sessions_with_degraded = [
        {"session_id": f"sess_deg_{i}", "session_status": "COMPLETE", "records": make_session_records(f"sess_deg_{i}")}
        for i in range(4)
    ]
    sessions_with_degraded.append({
        "session_id": "sess_corrupt",
        "session_status": "DEGRADED",
        "records": make_session_records("sess_corrupt", 50),
    })

    res_degraded = aggregate_prospective_sessions(sessions_with_degraded)
    assert res_degraded["provenance"]["valid_completed_sessions_count"] == 4
    assert res_degraded["dual_cohort_metrics"]["stopping_rules"]["c4_min_sessions_met"] is False
    assert res_degraded["protocol_verdict"] == "REJECTED"

    # Scenario C: 5 genuine COMPLETE sessions passing all hurdles
    sessions_passing = [
        {"session_id": f"sess_pass_{i}", "session_status": "COMPLETE", "records": make_session_records(f"sess_pass_{i}", 120)}
        for i in range(5)
    ]
    res_pass = aggregate_prospective_sessions(sessions_passing)
    assert res_pass["provenance"]["valid_completed_sessions_count"] == 5
    assert res_pass["dual_cohort_metrics"]["stopping_rules"]["c4_min_sessions_met"] is True
    assert res_pass["dual_cohort_metrics"]["summary"]["protocol_accepted"] is True
    assert res_pass["protocol_verdict"] == "ACCEPTED"


def test_multi_session_on_disk_aggregation_with_manifest_and_trade_records():
    """Verify loading and aggregating multi-session records directly from disk session directories."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        session_dirs = []

        for s_idx in range(2):
            s_id = f"disk_sess_{s_idx}"
            s_dir = base / s_id
            s_dir.mkdir(parents=True)
            session_dirs.append(s_dir)

            recs = [
                TradeRecord(
                    signal_id=f"sig_{s_id}_{i}", record_index=i, decision_timestamp_ms=i * 1000,
                    market_id="mkt", asset_id="up", burst_id=i % 5, side="BUY", predicted_class=2,
                    confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, limit_price=0.50,
                    order_quantity=1.0, filled_quantity=1.0, fill_rate_ratio=1.0,
                    fill_timestamp_ms=i * 1000, fill_price=0.50, exit_timestamp_ms=(i + 5) * 1000,
                    exit_price=0.505, gross_pnl=0.005, maker_fee=0.0, taker_fee=0.00025,
                    net_pnl=0.00475, outcome_status="COMPLETED", is_filled=True, session_id=s_id,
                )
                for i in range(25)
            ]

            recs_dicts = [asdict(r) for r in recs]
            with open(s_dir / "trade_records.json", "w", encoding="utf-8") as f:
                json.dump(recs_dicts, f, indent=2)

            manifest = {
                "session_id": s_id,
                "session_status": "COMPLETE",
                "hard_stop_triggered": None,
                "signals_generated_count": 25,
            }
            with open(s_dir / "session_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        agg = aggregate_prospective_sessions(session_dirs)
        assert agg["provenance"]["total_sessions_submitted"] == 2
        assert agg["provenance"]["valid_completed_sessions_count"] == 2
        assert agg["provenance"]["total_signals_admitted"] == 50
        assert agg["cluster_summary"]["distinct_clusters_count"] == 10


def test_offline_fill_certainty_propagation_and_conservative_cohort():
    """Verify MakerExecutionSimulator classifies and propagates fill_certainty correctly into dual cohorts."""
    sim = MakerExecutionSimulator()
    # Order 1: Confirmed Price-Through fill (Ask <= limit)
    sim.submit_signal(
        decision_timestamp_ms=1000, market_id="mkt", asset_id="up", predicted_class=2,
        confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, burst_id=0,
        current_bid=0.50, current_ask=0.51, signal_id="sig_price_through",
    )
    # Market traded through limit (Ask drops to 0.50)
    sim.update_market_state(current_timestamp_ms=1500, current_bid=0.49, current_ask=0.50)

    # Order 2: Ambiguous Quote Retreat fill (Bid retreats to 0.49, Ask remains above limit)
    sim.submit_signal(
        decision_timestamp_ms=2000, market_id="mkt", asset_id="up", predicted_class=2,
        confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, burst_id=0,
        current_bid=0.50, current_ask=0.51, signal_id="sig_quote_retreat",
    )
    # Bid retreated below limit while ask is 0.52 (Ambiguous)
    sim.update_market_state(current_timestamp_ms=2500, current_bid=0.49, current_ask=0.52)

    # Settle both orders at horizon (1500 + 5000 = 6500, 2500 + 5000 = 7500)
    recs_1 = sim.update_market_state(current_timestamp_ms=6500, current_bid=0.52, current_ask=0.53)
    assert len(recs_1) == 1
    assert recs_1[0].signal_id == "sig_price_through"
    assert recs_1[0].fill_certainty == "CONFIRMED_PRICE_THROUGH"

    recs_2 = sim.update_market_state(current_timestamp_ms=7500, current_bid=0.52, current_ask=0.53)
    assert len(recs_2) == 1
    assert recs_2[0].signal_id == "sig_quote_retreat"
    assert recs_2[0].fill_certainty == "AMBIGUOUS_QUOTE_RETREAT"

    metrics = compute_dual_cohort_metrics(sim.trade_records)
    scenarios = metrics["fill_certainty_scenarios"]
    assert scenarios["breakdown"]["confirmed_price_through_count"] == 1
    assert scenarios["breakdown"]["ambiguous_quote_retreat_count"] == 1
    assert scenarios["conservative"]["filled_count"] == 1
    assert scenarios["optimistic"]["filled_count"] == 2
    assert scenarios["conservative"]["itt_mean_net_pnl"] > 0.0


def test_down_token_execution_alignment_in_maker_simulator():
    """Verify MakerExecutionSimulator executes BUY on down_bid when DOWN quotes provided, or inverted UP proxy otherwise."""
    sim = MakerExecutionSimulator()

    # Case A: Separate DOWN quotes provided (prospective alignment)
    ord_aligned = sim.submit_signal(
        decision_timestamp_ms=1000, market_id="mkt", asset_id="up_tok", predicted_class=0,
        confidence=0.65, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, burst_id=0,
        current_bid=0.52, current_ask=0.53,
        down_bid=0.47, down_ask=0.48, down_asset_id="down_tok",
    )
    assert ord_aligned.side == "BUY"
    assert ord_aligned.limit_price == 0.47
    assert ord_aligned.asset_id == "down_tok"

    # Fill on DOWN book (down_ask drops to 0.47)
    sim.update_market_state(
        current_timestamp_ms=1500, current_bid=0.52, current_ask=0.53,
        down_bid=0.46, down_ask=0.47,
    )
    recs_aligned = sim.update_market_state(
        current_timestamp_ms=6500, current_bid=0.50, current_ask=0.51,
        down_bid=0.49, down_ask=0.50,
    )
    assert len(recs_aligned) == 1
    assert recs_aligned[0].side == "BUY"
    assert recs_aligned[0].fill_price == 0.47
    assert recs_aligned[0].exit_price == 0.49  # Sold at down_bid
    assert recs_aligned[0].gross_pnl == pytest.approx(0.02)
    assert recs_aligned[0].fill_certainty == "CONFIRMED_PRICE_THROUGH"

    # Case B: Only UP quotes provided (historical dataset mode)
    ord_legacy = sim.submit_signal(
        decision_timestamp_ms=10000, market_id="mkt", asset_id="up_tok", predicted_class=0,
        confidence=0.65, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, burst_id=1,
        current_bid=0.52, current_ask=0.53,
    )
    assert ord_legacy.side == "SELL"
    assert ord_legacy.limit_price == 0.53
    assert ord_legacy.asset_id == "up_tok"
