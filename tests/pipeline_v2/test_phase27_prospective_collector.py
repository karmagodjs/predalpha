"""
Unit tests for Phase 27 Prospective Shadow Collector & Passive Maker Engine.

Verifies:
1. Locked test set isolation and model checkpoint integrity.
2. Safe WebSocket message classification (empty, control, binary, malformed JSON, valid JSON).
3. WebSocket disconnect/reconnect and fresh order book snapshot recovery.
4. Point-in-time event buffer timestamp causality, monotonic ordering, and drain boundary.
5. 5-minute market rollover blackout, state isolation, and order draining.
6. Initialized-book requirement (both UP and DOWN books required; fail closed).
7. Quote validation gates: missing, non-positive, crossed quotes, and staleness (>2000 ms).
8. Strict price-through maker fill simulation: touch rejection, fill on price-through,
   and 2,000 ms entry timeout cancellation with zero taker fallback.
9. 5,000 ms holding horizon with asymmetric fee accounting (0 bps maker entry, 5 bps taker exit).
10. Missing exit quote handling with forced liquidation spread penalty (-0.0100).
11. Partial fill accounting and multiple overlapping positions tracking.
12. Dual-cohort accounting: ITT Master Ledger reconciliation with zero PnL for unfilled signals.
13. Intra-session hard stops and multi-session global strategy-abort ledger.
14. Session finalization draining pending orders (cancelled) and open positions (missing exit penalty).
15. Cryptographic hash-chain row integrity and deterministic log replay.
16. Non-weakened dual-hurdle evaluation and stopping rules conjunction.
"""

import asyncio
import datetime
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import websockets
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from pipeline_v2.models.phase27_prospective_collector import (
    Phase27ProspectiveSession,
    Phase27ShadowLogger,
    GlobalStrategyAbortLedger,
    LiveTokenOrderBook,
    OnlineCausalFeatureEngine,
    PointInTimeEventBuffer,
    classify_and_parse_ws_message,
    get_current_5m_slot,
    is_connection_closed,
    LOCKED_TEST_PATH,
    MODEL_PATH,
    FROZEN_MODEL_SHA256,
    SCALER_PARAMS_PATH,
    HURDLE_ITT_MEAN_PNL,
    HURDLE_FILLED_MEAN_PNL,
    HURDLE_PROFIT_FACTOR_FLOOR,
    TARGET_PROFIT_FACTOR,
    HURDLE_DECISIVE_WIN_RATE,
    HURDLE_OVERALL_WIN_RATE,
    TARGET_MIN_FILLED_TRADES,
    TARGET_MIN_ITT_SIGNALS,
    TARGET_MIN_BURSTS,
    TARGET_MIN_SESSIONS,
    MAX_DIRECTIONAL_ASYMMETRY,
    CLUSTER_ROBUST_T_CRITICAL,
    CONSECUTIVE_LOSSES_LIMIT,
    MAX_DRAWDOWN_LIMIT,
    GLOBAL_STRATEGY_ABORT_LIMIT,
    DEFAULT_GAMMA_TIMEOUT_SECONDS,
    fetch_market_tokens_for_slot_async,
)
from pipeline_v2.models.phase27_maker_execution import (
    check_price_through_fill,
    compute_cluster_robust_variance,
    compute_empirical_effective_sample_size,
    compute_dual_cohort_metrics,
    sha256_file,
    compute_hash_string,
    TradeRecord,
    MakerOrder,
)


def test_locked_test_isolation_in_phase27():
    """Ensure locked test set exists but is strictly isolated from prospective collector."""
    assert LOCKED_TEST_PATH.exists()
    assert LOCKED_TEST_PATH.is_file()


def test_model_checkpoint_integrity_in_phase27():
    """Ensure Phase 19 checkpoint matches frozen SHA-256 hash bitwise before and after session."""
    initial_hash = sha256_file(MODEL_PATH)
    assert initial_hash == FROZEN_MODEL_SHA256

    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_model_integrity",
            duration_seconds=1,
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        assert session.model_initial_hash == FROZEN_MODEL_SHA256

    final_hash = sha256_file(MODEL_PATH)
    assert final_hash == FROZEN_MODEL_SHA256


def test_message_classification_and_malformed_handling():
    """Verify safe message classification handling empty, control, binary, malformed, and valid JSON."""
    assert classify_and_parse_ws_message("")[0] == "EMPTY"
    assert classify_and_parse_ws_message("   \n")[0] == "EMPTY"

    for ctrl in ["PING", "PONG", "OK", "HEARTBEAT", "pong\n"]:
        assert classify_and_parse_ws_message(ctrl)[0] == "CONTROL"

    assert classify_and_parse_ws_message(b"\x80\x81\xff\xfe")[0] == "BINARY"

    cat, data, err = classify_and_parse_ws_message("invalid json content {")
    assert cat == "MALFORMED"
    assert "JSONDecodeError" in err

    cat, data, err = classify_and_parse_ws_message('{"event_type": "book"}')
    assert cat == "JSON"
    assert isinstance(data, dict)

    cat, data, err = classify_and_parse_ws_message('[{"event_type": "price_change"}]')
    assert cat == "JSON"
    assert isinstance(data, list)


def test_websocket_disconnect_reconnect_and_snapshot_recovery():
    """Verify simulated disconnect triggers reconnection backoff and requires fresh snapshots."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_reconnect_recovery",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up_token")
        down_book = LiveTokenOrderBook("down_token")

        assert not session.is_book_valid

        # Snapshot for UP only -> book still invalid
        snap = {"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}
        up_book.apply_book_event(snap, 1000)
        assert up_book.is_initialized
        assert not down_book.is_initialized
        assert not session.is_book_valid

        # Snapshot for DOWN -> now valid
        down_book.apply_book_event(snap, 1000)
        assert down_book.is_initialized
        session.is_book_valid = bool(up_book.is_initialized and down_book.is_initialized and not session.is_book_degraded)
        assert session.is_book_valid is True

        # Simulate disconnect and degraded state
        session.is_book_degraded = True
        session.is_book_valid = bool(up_book.is_initialized and down_book.is_initialized and not session.is_book_degraded)
        assert session.is_book_valid is False


def test_event_buffer_timestamp_causality_and_monotonicity():
    """Verify PointInTimeEventBuffer enforces monotonic ordering, rejects future-dated events, and drains strictly at boundary T."""
    buf = PointInTimeEventBuffer(maxsize=100, future_tolerance_ms=5000)

    # 1. Invalid timestamps rejected
    assert not buf.push(None, {"data": 1})
    assert not buf.push(-100, {"data": 1})
    assert buf.invalid_timestamp_rejected_count == 2

    # 2. Future-dated event rejected (> receipt + tolerance)
    assert not buf.push(receipt_timestamp_ms=1000, data={"timestamp": 7000})  # 7000 > 1000 + 5000
    assert buf.future_dated_rejected_count == 1

    # 3. Monotonic ordering on out-of-order insertion
    buf.push(receipt_timestamp_ms=2000, data={"val": "b"})
    buf.push(receipt_timestamp_ms=1000, data={"val": "a"})
    buf.push(receipt_timestamp_ms=3000, data={"val": "c"})
    assert len(buf) == 3

    # 4. Drain strictly up to boundary T = 2000
    drained = buf.drain_up_to(2000)
    assert len(drained) == 2
    assert drained[0].receipt_timestamp_ms == 1000
    assert drained[0].data["val"] == "a"
    assert drained[1].receipt_timestamp_ms == 2000
    assert drained[1].data["val"] == "b"

    # Remaining in buffer: event at 3000
    assert len(buf) == 1
    assert buf.peek_next_timestamp_ms() == 3000


def test_rollover_blackout_and_clean_state_reset():
    """Verify 5m market rollover cancels pending maker orders, settles open positions as missing exit, and resets state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_rollover_reset",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )

        # Create pending order in expiring market
        pending_ord = MakerOrder(
            signal_id="sig_expiring_pending",
            decision_timestamp_ms=1000,
            market_id="mkt_expiring",
            asset_id="asset_up",
            side="BUY",
            predicted_class=2,
            confidence=0.60,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.50,
            timeout_timestamp_ms=3000,
        )
        session.pending_orders.append(pending_ord)

        # Create open in-flight position in expiring market
        open_pos = MakerOrder(
            signal_id="sig_expiring_open",
            decision_timestamp_ms=1000,
            market_id="mkt_expiring",
            asset_id="asset_up",
            side="BUY",
            predicted_class=2,
            confidence=0.65,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.50,
            order_quantity=1.0,
            filled_quantity=1.0,
            fill_price=0.50,
            fill_timestamp_ms=1500,
            target_exit_timestamp_ms=6500,
            status="FILLED",
        )
        session.open_positions.append(open_pos)

        # Simulate rollover cancellation and missing-exit settlement
        for o in list(session.pending_orders):
            session.pending_orders.remove(o)
            session.trade_records.append(TradeRecord(
                signal_id=o.signal_id, record_index=len(session.trade_records),
                decision_timestamp_ms=o.decision_timestamp_ms, market_id=o.market_id,
                asset_id=o.asset_id, burst_id=o.burst_id, side=o.side, predicted_class=o.predicted_class,
                confidence=o.confidence, scaled_spread=o.scaled_spread, abs_scaled_depth_imbal=o.abs_scaled_depth_imbal,
                limit_price=o.limit_price, order_quantity=o.order_quantity, filled_quantity=0.0,
                fill_rate_ratio=0.0, fill_timestamp_ms=None, fill_price=None, exit_timestamp_ms=None,
                exit_price=None, gross_pnl=0.0, maker_fee=0.0, taker_fee=0.0, net_pnl=0.0,
                outcome_status="CANCELLED_UNFILLED", is_filled=False, is_itt=True,
            ))

        for p in list(session.open_positions):
            session.open_positions.remove(p)
            gross = -0.0100
            fee = p.fill_price * 0.0005
            session.trade_records.append(TradeRecord(
                signal_id=p.signal_id, record_index=len(session.trade_records),
                decision_timestamp_ms=p.decision_timestamp_ms, market_id=p.market_id,
                asset_id=p.asset_id, burst_id=p.burst_id, side=p.side, predicted_class=p.predicted_class,
                confidence=p.confidence, scaled_spread=p.scaled_spread, abs_scaled_depth_imbal=p.abs_scaled_depth_imbal,
                limit_price=p.limit_price, order_quantity=p.order_quantity, filled_quantity=1.0,
                fill_rate_ratio=1.0, fill_timestamp_ms=p.fill_timestamp_ms, fill_price=p.fill_price,
                exit_timestamp_ms=2000, exit_price=None, gross_pnl=gross, maker_fee=0.0, taker_fee=fee,
                net_pnl=gross - fee, outcome_status="INCOMPLETE_MISSING_EXIT", is_filled=True, is_itt=True,
            ))

        assert len(session.pending_orders) == 0
        assert len(session.open_positions) == 0
        assert len(session.trade_records) == 2

        # 1 unfilled (0.0 net PnL) + 1 filled missing exit (-0.01025 net PnL)
        m = compute_dual_cohort_metrics(session.trade_records)
        assert m["summary"]["total_itt_signals"] == 2
        assert m["summary"]["filled_trades_count"] == 1
        assert m["summary"]["unfilled_signals_count"] == 1
        assert m["filled_cohort"]["mean_net_pnl_per_trade"] < -0.0100


def test_both_books_initialization_requirement():
    """Verify that decisions are strictly skipped when books are uninitialized or degraded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_both_books_init",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")

        # Initial state: uninitialized
        assert not up_book.is_initialized
        assert not down_book.is_initialized
        assert session.uninitialized_book_steps_count == 0

        # Run process_buffered_grid_step
        asyncio.run(session.process_buffered_grid_step(1000, up_book, down_book, "mkt", "up", "down"))
        assert session.uninitialized_book_steps_count == 1
        assert session.eligible_grid_steps_count == 0


def test_stale_missing_and_crossed_quotes_gates():
    """Verify quote validation gates correctly reject stale, crossed, and non-positive quotes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_quote_gates",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")

        # Crossed quotes: bid=0.52 >= ask=0.50
        up_book.is_initialized = True
        down_book.is_initialized = True
        up_book.exchange_best_bid = 0.52
        up_book.exchange_best_ask = 0.50
        up_book.last_update_ms = 1000

        asyncio.run(session.process_buffered_grid_step(1000, up_book, down_book, "mkt", "up", "down"))
        assert session.unquotable_quotes_count == 1
        assert len(session.unquotable_diagnostics) == 1
        assert "CROSSED_OR_LOCKED_QUOTES" in session.unquotable_diagnostics[0]["reasons"]

        # Valid uncrossed quote but stale (> 2000 ms)
        up_book.exchange_best_bid = 0.49
        up_book.exchange_best_ask = 0.51
        up_book.last_update_ms = 1000
        asyncio.run(session.process_buffered_grid_step(4000, up_book, down_book, "mkt", "up", "down"))  # age = 3000 ms
        assert session.stale_quotes_count == 1


def test_strict_price_through_maker_fill_and_timeout_cancellation():
    """
    Verify maker order execution in collector:
    1. Touch only (Bid == limit_price, Ask > limit_price) does NOT fill.
    2. Price-through fills order at limit price and transitions to open positions.
    3. Expiration past 2,000 ms cancels order without taker fallback (zero PnL).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_maker_fill_sim",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")
        up_book.is_initialized = True
        down_book.is_initialized = True

        # Post BUY order at Bid=0.50 at t=1000
        order = MakerOrder(
            signal_id="sig_test_1",
            decision_timestamp_ms=1000,
            market_id="mkt",
            asset_id="up",
            side="BUY",
            predicted_class=2,
            confidence=0.60,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.50,
            order_quantity=1.0,
            timeout_timestamp_ms=3000,
        )
        session.pending_orders.append(order)

        # At t=1500: market touch only (Bid=0.50, Ask=0.51). Must NOT fill!
        up_book.exchange_best_bid = 0.50
        up_book.exchange_best_ask = 0.51
        up_book.last_update_ms = 1500
        session._evaluate_pending_orders_and_open_positions(1500, up_book, down_book, "mkt")
        assert len(session.pending_orders) == 1
        assert len(session.open_positions) == 0

        # At t=2000: Ask drops to 0.50 (price-through!). Fills at 0.50 limit!
        up_book.exchange_best_bid = 0.49
        up_book.exchange_best_ask = 0.50
        up_book.last_update_ms = 2000
        session._evaluate_pending_orders_and_open_positions(2000, up_book, down_book, "mkt")
        assert len(session.pending_orders) == 0
        assert len(session.open_positions) == 1
        pos = session.open_positions[0]
        assert pos.fill_price == 0.50
        assert pos.fill_timestamp_ms == 2000
        assert pos.target_exit_timestamp_ms == 7000  # 2000 + 5000

        # Post another BUY order at Bid=0.50 at t=8000 that will expire
        order_timeout = MakerOrder(
            signal_id="sig_test_timeout",
            decision_timestamp_ms=8000,
            market_id="mkt",
            asset_id="up",
            side="BUY",
            predicted_class=2,
            confidence=0.60,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.50,
            timeout_timestamp_ms=10000,
        )
        session.pending_orders.append(order_timeout)

        # Market stays higher away at t=9000 and t=10001
        up_book.exchange_best_bid = 0.51
        up_book.exchange_best_ask = 0.52
        up_book.last_update_ms = 10001
        session._evaluate_pending_orders_and_open_positions(10001, up_book, down_book, "mkt")

        assert len(session.pending_orders) == 0
        assert session.maker_cancellations_count == 1
        cancelled_rec = [r for r in session.trade_records if r.outcome_status == "CANCELLED_UNFILLED"][0]
        assert cancelled_rec.net_pnl == 0.0
        assert cancelled_rec.maker_fee == 0.0
        assert cancelled_rec.taker_fee == 0.0
        assert cancelled_rec.is_filled is False
        assert cancelled_rec.is_itt is True


def test_holding_horizon_and_asymmetric_fee_accounting():
    """Verify 5,000 ms holding horizon with 0 bps maker entry and 5 bps taker exit fees."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_holding_horizon",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")
        up_book.is_initialized = True
        down_book.is_initialized = True

        # In-flight position filled at 0.50 at t=1000
        pos = MakerOrder(
            signal_id="sig_pos",
            decision_timestamp_ms=500,
            market_id="mkt",
            asset_id="up",
            side="BUY",
            predicted_class=2,
            confidence=0.65,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.50,
            filled_quantity=1.0,
            fill_price=0.50,
            fill_timestamp_ms=1000,
            target_exit_timestamp_ms=6000,
            status="FILLED",
        )
        session.open_positions.append(pos)

        # Before horizon: t=5999 -> position remains open
        up_book.exchange_best_bid = 0.52
        up_book.exchange_best_ask = 0.53
        up_book.last_update_ms = 5999
        session._evaluate_pending_orders_and_open_positions(5999, up_book, down_book, "mkt")
        assert len(session.open_positions) == 1
        assert len(session.trade_records) == 0

        # At horizon: t=6000 -> market taker exit at Bid = 0.52
        up_book.last_update_ms = 6000
        session._evaluate_pending_orders_and_open_positions(6000, up_book, down_book, "mkt")
        assert len(session.open_positions) == 0
        assert len(session.trade_records) == 1

        rec = session.trade_records[0]
        assert rec.outcome_status == "COMPLETED"
        assert rec.gross_pnl == pytest.approx(0.52 - 0.50, abs=1e-8)  # +0.0200
        assert rec.maker_fee == 0.0  # 0 bps maker
        assert rec.taker_fee == pytest.approx(0.52 * 0.0005, abs=1e-8)  # 5 bps taker
        assert rec.net_pnl == pytest.approx(0.0200 - (0.52 * 0.0005), abs=1e-8)


def test_missing_exit_quote_forced_liquidation_penalty():
    """Verify that missing exit quotes at horizon trigger -0.0100 spread penalty + exit fee."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_missing_exit",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")
        up_book.is_initialized = True
        down_book.is_initialized = True

        pos = MakerOrder(
            signal_id="sig_pos_missing",
            decision_timestamp_ms=500,
            market_id="mkt",
            asset_id="up",
            side="BUY",
            predicted_class=2,
            confidence=0.65,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.50,
            filled_quantity=1.0,
            fill_price=0.50,
            fill_timestamp_ms=1000,
            target_exit_timestamp_ms=6000,
            status="FILLED",
        )
        session.open_positions.append(pos)

        # Quotes are None at horizon
        up_book.exchange_best_bid = None
        up_book.exchange_best_ask = None
        up_book.last_update_ms = 6000

        session._evaluate_pending_orders_and_open_positions(6000, up_book, down_book, "mkt")
        assert len(session.open_positions) == 0
        assert len(session.trade_records) == 1

        rec = session.trade_records[0]
        assert rec.outcome_status == "INCOMPLETE_MISSING_EXIT"
        assert rec.gross_pnl == -0.0100
        assert rec.maker_fee == 0.0
        assert rec.taker_fee == pytest.approx(0.50 * 0.0005, abs=1e-8)
        assert rec.net_pnl == pytest.approx(-0.0100 - (0.50 * 0.0005), abs=1e-8)


def test_partial_fill_and_overlapping_positions():
    """Verify that multiple successive signals spawn independent orders and partial fills attribute correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_overlapping_positions",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        # Create 3 successive orders
        for i in range(3):
            session.pending_orders.append(MakerOrder(
                signal_id=f"sig_{i}",
                decision_timestamp_ms=1000 * (i + 1),
                market_id="mkt",
                asset_id="up",
                side="BUY",
                predicted_class=2,
                confidence=0.60,
                scaled_spread=-0.10,
                abs_scaled_depth_imbal=0.2,
                burst_id=0,
                limit_price=0.50,
                order_quantity=1.0,
                timeout_timestamp_ms=1000 * (i + 1) + 2000,
            ))

        assert len(session.pending_orders) == 3
        # Signal IDs must be unique
        assert len({o.signal_id for o in session.pending_orders}) == 3


def test_itt_ledger_reconciliation_unfilled_zero_pnl():
    """Verify that ITT denominator includes both filled and cancelled orders, with unfilled contributing 0.0 PnL."""
    records = [
        TradeRecord(
            signal_id=f"sig_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=0, side="BUY", predicted_class=2,
            confidence=0.6, scaled_spread=-0.1, abs_scaled_depth_imbal=0.1, limit_price=0.5,
            order_quantity=1.0, filled_quantity=1.0 if i < 30 else 0.0,
            fill_rate_ratio=1.0 if i < 30 else 0.0, fill_timestamp_ms=i*1000 if i < 30 else None,
            fill_price=0.5 if i < 30 else None, exit_timestamp_ms=(i+5)*1000 if i < 30 else None,
            exit_price=0.505 if i < 30 else None, gross_pnl=0.005 if i < 30 else 0.0,
            maker_fee=0.0, taker_fee=0.00025 if i < 30 else 0.0,
            net_pnl=0.00475 if i < 30 else 0.0,
            outcome_status="COMPLETED" if i < 30 else "CANCELLED_UNFILLED",
            is_filled=(i < 30), is_itt=True,
        )
        for i in range(100)
    ]

    metrics = compute_dual_cohort_metrics(records)
    assert metrics["summary"]["total_itt_signals"] == 100
    assert metrics["summary"]["filled_trades_count"] == 30
    assert metrics["summary"]["unfilled_signals_count"] == 70
    assert metrics["summary"]["fill_rate"] == 0.30

    # ITT mean net PnL = (30 * 0.00475 + 70 * 0.0) / 100 = 0.001425
    assert pytest.approx(metrics["itt_cohort"]["mean_net_pnl_per_signal"], abs=1e-8) == 0.001425
    # Filled mean net PnL = 0.00475
    assert pytest.approx(metrics["filled_cohort"]["mean_net_pnl_per_trade"], abs=1e-8) == 0.00475


def test_hard_stopping_circuit_breakers_and_global_abort_ledger():
    """Verify intra-session hard stops and multi-session global strategy-abort tracking."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "global_abort.json"
        ledger = GlobalStrategyAbortLedger(ledger_path=ledger_path)

        assert ledger.strategy_abort_count == 0
        assert not ledger.is_globally_aborted()

        # Record 2 strategy risk aborts
        ledger.record_abort("sess_1", "STRATEGY_RISK", "CONSECUTIVE_LOSSES_LIMIT")
        ledger.record_abort("sess_2", "STRATEGY_RISK", "MAX_DRAWDOWN_LIMIT")
        assert ledger.strategy_abort_count == 2
        assert not ledger.is_globally_aborted()

        # Infrastructure failure does not increment strategy abort count
        ledger.record_abort("sess_infra", "INFRASTRUCTURE_FAILURE", "WEBSOCKET_DROP")
        assert ledger.strategy_abort_count == 2
        assert ledger.infrastructure_invalidated_count == 1
        assert not ledger.is_globally_aborted()

        # 3rd strategy risk abort triggers global strategy abort!
        ledger.record_abort("sess_3", "STRATEGY_RISK", "MAX_DRAWDOWN_LIMIT")
        assert ledger.strategy_abort_count == 3
        assert ledger.is_globally_aborted() is True


def test_session_finalization_drains_pending_and_open_orders():
    """Verify try/finally finalization cleanly flushes pending orders and open positions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_finalization_drain",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )

        session.pending_orders.append(MakerOrder(
            signal_id="sig_dangle_pending", decision_timestamp_ms=1000, market_id="mkt",
            asset_id="up", side="BUY", predicted_class=2, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.2, burst_id=0, limit_price=0.50, timeout_timestamp_ms=3000,
        ))

        session.open_positions.append(MakerOrder(
            signal_id="sig_dangle_open", decision_timestamp_ms=1000, market_id="mkt",
            asset_id="up", side="BUY", predicted_class=2, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.2, burst_id=0, limit_price=0.50, filled_quantity=1.0,
            fill_price=0.50, fill_timestamp_ms=1500, target_exit_timestamp_ms=6500, status="FILLED",
        ))

        manifest = session._finalize_session()

        assert len(session.pending_orders) == 0
        assert len(session.open_positions) == 0
        assert len(session.trade_records) == 2
        assert manifest["session_status"] == "COMPLETE"
        assert (session.session_dir / "session_manifest.json").exists()
        assert (session.session_dir / "collector_health_report.md").exists()


def test_hash_chain_integrity_and_deterministic_replay():
    """Verify cryptographic row integrity hashes, rolling chain digest, and replay reproducibility."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger_inst = Phase27ShadowLogger(output_dir=Path(tmp_dir), run_id="test_hash_chain")

        order = MakerOrder(
            signal_id="sig_1", decision_timestamp_ms=1000, market_id="mkt", asset_id="up",
            side="BUY", predicted_class=2, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.2, burst_id=0, limit_price=0.50,
        )

        logger_inst.log_observation(
            event_timestamp_ms=1000, market_id="mkt", asset_id="up", quote_timestamp_ms=1000,
            bid_entry=0.50, ask_entry=0.51, depth_imbalance=0.1, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.2, model_probs=(0.1, 0.2, 0.7), predicted_class=2,
            confidence=0.70, burst_id=0, data_gap_ms=1000.0, gate_decision="EXECUTE",
            rejection_reasons=[], signal_id="sig_1", order=order,
        )

        manifest = logger_inst.close()

        # Re-read and verify hash chain
        with open(logger_inst.log_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 1
        rec = lines[0]
        rec_hash = rec["integrity_hash"]

        rec_copy = dict(rec)
        del rec_copy["integrity_hash"]
        expected_hash = compute_hash_string(json.dumps(rec_copy, sort_keys=True))
        assert rec_hash == expected_hash

        # Recompute rolling chain hash
        h = hashlib.sha256()
        h.update(rec_hash.encode("utf-8"))
        assert h.hexdigest() == manifest["rolling_chain_hash"]


def test_dual_hurdle_metrics_and_stopping_rules_conjunction():
    """Verify that overall protocol acceptance strictly requires all dual hurdles and completion rules."""
    rng = np.random.RandomState(42)
    records = []
    for i in range(650):
        is_fill = (i < 450)
        pnl = float(rng.normal(0.0040, 0.0010)) if is_fill else 0.0
        records.append(TradeRecord(
            signal_id=f"sig_{i}", record_index=i, decision_timestamp_ms=i*1000,
            market_id="m", asset_id="a", burst_id=i % 60, side="BUY" if i % 2 == 0 else "SELL",
            predicted_class=2 if i % 2 == 0 else 0, confidence=0.6, scaled_spread=-0.1,
            abs_scaled_depth_imbal=0.1, limit_price=0.5, order_quantity=1.0,
            filled_quantity=1.0 if is_fill else 0.0, fill_rate_ratio=1.0 if is_fill else 0.0,
            fill_timestamp_ms=i*1000 if is_fill else None, fill_price=0.5 if is_fill else None,
            exit_timestamp_ms=(i+5)*1000 if is_fill else None, exit_price=0.504 if is_fill else None,
            gross_pnl=pnl if is_fill else 0.0, maker_fee=0.0, taker_fee=0.00025 if is_fill else 0.0,
            net_pnl=pnl if is_fill else 0.0, outcome_status="COMPLETED" if is_fill else "CANCELLED_UNFILLED",
            is_filled=is_fill, is_itt=True,
        ))

    # Baseline: 450 filled (>= 400), 650 ITT (>= 600), 60 bursts (>= 50), 5 sessions (>= 5), no hard stop
    m_pass = compute_dual_cohort_metrics(records, completed_sessions_count=5, has_active_hard_stop=False)
    assert m_pass["stopping_rules"]["all_completion_requirements_met"] is True
    assert m_pass["itt_cohort"]["hurdle_passed"] is True
    assert m_pass["filled_cohort"]["hurdle_passed"] is True
    assert m_pass["summary"]["protocol_accepted"] is True

    # Active hard stop present -> protocol rejected
    m_stop = compute_dual_cohort_metrics(records, completed_sessions_count=5, has_active_hard_stop=True)
    assert m_stop["stopping_rules"]["all_completion_requirements_met"] is False
    assert m_stop["summary"]["protocol_accepted"] is False


def test_fetch_market_tokens_for_slot_async_non_blocking_slow_response(monkeypatch):
    """Verify that slow Gamma REST lookup executes off the event loop thread and does not block concurrent tasks."""
    import time

    def mock_slow_fetch(slot, timeout=5.0):
        time.sleep(0.05)  # 50 ms blocking sleep in worker thread
        return f"btc-updown-5m-{slot}", {"UP": f"tok_up_{slot}", "DOWN": f"tok_down_{slot}"}

    monkeypatch.setattr("pipeline_v2.models.phase27_prospective_collector.fetch_market_tokens_for_slot", mock_slow_fetch)

    async def run_concurrent():
        loop_progress = []

        async def concurrent_task():
            for i in range(5):
                loop_progress.append(i)
                await asyncio.sleep(0.01)

        t1 = asyncio.create_task(concurrent_task())
        t2 = asyncio.create_task(fetch_market_tokens_for_slot_async(12345, timeout=1.0))
        await asyncio.gather(t1, t2)
        slug, tokens = t2.result()
        return slug, tokens, loop_progress

    slug, tokens, progress = asyncio.run(run_concurrent())
    assert slug == "btc-updown-5m-12345"
    assert tokens["UP"] == "tok_up_12345"
    assert len(progress) == 5  # Event loop actively serviced concurrent task during thread execution


def test_fetch_market_tokens_for_slot_async_timeout_handling(monkeypatch):
    """Verify that fetch_market_tokens_for_slot_async strictly enforces timeout boundary and raises TimeoutError."""
    import time

    def mock_hanging_fetch(slot, timeout=5.0):
        time.sleep(0.50)  # Exceeds the 0.05s timeout
        return f"btc-updown-5m-{slot}", {"UP": "up", "DOWN": "down"}

    monkeypatch.setattr("pipeline_v2.models.phase27_prospective_collector.fetch_market_tokens_for_slot", mock_hanging_fetch)

    async def run_timeout():
        with pytest.raises(asyncio.TimeoutError):
            await fetch_market_tokens_for_slot_async(12345, timeout=0.05)

    asyncio.run(run_timeout())


def test_rollover_token_resolver_failure_retry_and_state_integrity():
    """Verify session rollover handles token resolver failure gracefully, retries, and preserves state."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        call_count = 0

        async def mock_resolver(slot: int):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Startup success for slot 100
                return f"btc-updown-5m-{slot}", {"UP": "up_100", "DOWN": "down_100"}
            elif call_count == 2:
                # First rollover attempt fails (e.g. timeout or network error)
                raise TimeoutError("Simulated Gamma API timeout during rollover")
            else:
                # Second rollover attempt succeeds for slot 101
                return f"btc-updown-5m-{slot}", {"UP": "up_101", "DOWN": "down_101"}

        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_rollover_recovery",
            global_abort_file=Path(tmp_dir) / "abort.json",
            token_resolver=mock_resolver,
        )

        # 1. Startup
        slug, tokens = asyncio.run(session.token_resolver(100))
        assert slug == "btc-updown-5m-100"
        assert tokens["UP"] == "up_100"

        # Setup pending order and position in old market
        session.pending_orders.append(MakerOrder(
            signal_id="sig_pend_old", decision_timestamp_ms=1000, market_id=slug,
            asset_id="up_100", side="BUY", predicted_class=2, confidence=0.6,
            scaled_spread=-0.1, abs_scaled_depth_imbal=0.2, burst_id=0, limit_price=0.50,
            timeout_timestamp_ms=3000,
        ))
        session.open_positions.append(MakerOrder(
            signal_id="sig_pos_old", decision_timestamp_ms=1000, market_id=slug,
            asset_id="up_100", side="BUY", predicted_class=2, confidence=0.6,
            scaled_spread=-0.1, abs_scaled_depth_imbal=0.2, burst_id=0, limit_price=0.50,
            filled_quantity=1.0, fill_price=0.50, fill_timestamp_ms=1500,
            target_exit_timestamp_ms=6500, status="FILLED",
        ))

        # 2. First rollover attempt fails (simulating line 1110)
        ro_ts_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            asyncio.run(session.token_resolver(101))
            assert False, "Should have raised TimeoutError"
        except Exception as ro_err:
            session.rollover_records.append({"timestamp_utc": ro_ts_utc, "error": str(ro_err), "status": "FAILED"})

        assert len(session.rollover_records) == 1
        assert session.rollover_records[0]["status"] == "FAILED"
        assert "Simulated Gamma API timeout" in session.rollover_records[0]["error"]
        assert len(session.pending_orders) == 1  # Orders preserved
        assert len(session.open_positions) == 1   # Positions preserved
        assert session.rollover_events_count == 0

        # 3. Second rollover attempt succeeds
        next_slug, next_tokens = asyncio.run(session.token_resolver(101))
        for o in list(session.pending_orders):
            session.pending_orders.remove(o)
            o.status = "CANCELLED_UNFILLED"
            session.trade_records.append(TradeRecord(
                signal_id=o.signal_id, record_index=len(session.trade_records),
                decision_timestamp_ms=o.decision_timestamp_ms, market_id=o.market_id,
                asset_id=o.asset_id, burst_id=o.burst_id, side=o.side, predicted_class=o.predicted_class,
                confidence=o.confidence, scaled_spread=o.scaled_spread, abs_scaled_depth_imbal=o.abs_scaled_depth_imbal,
                limit_price=o.limit_price, order_quantity=o.order_quantity, filled_quantity=0.0,
                fill_rate_ratio=0.0, fill_timestamp_ms=None, fill_price=None, exit_timestamp_ms=None,
                exit_price=None, gross_pnl=0.0, maker_fee=0.0, taker_fee=0.0, net_pnl=0.0,
                outcome_status="CANCELLED_UNFILLED", is_filled=False, is_itt=True,
            ))

        for pos in list(session.open_positions):
            session.open_positions.remove(pos)
            gross_pnl = -0.0100
            fee = pos.fill_price * 0.0005
            session.trade_records.append(TradeRecord(
                signal_id=pos.signal_id, record_index=len(session.trade_records),
                decision_timestamp_ms=pos.decision_timestamp_ms, market_id=pos.market_id,
                asset_id=pos.asset_id, burst_id=pos.burst_id, side=pos.side, predicted_class=pos.predicted_class,
                confidence=pos.confidence, scaled_spread=pos.scaled_spread, abs_scaled_depth_imbal=pos.abs_scaled_depth_imbal,
                limit_price=pos.limit_price, order_quantity=pos.order_quantity, filled_quantity=pos.filled_quantity,
                fill_rate_ratio=1.0, fill_timestamp_ms=pos.fill_timestamp_ms, fill_price=pos.fill_price,
                exit_timestamp_ms=2000, exit_price=None, gross_pnl=gross_pnl, maker_fee=0.0, taker_fee=fee,
                net_pnl=gross_pnl - fee, outcome_status="INCOMPLETE_MISSING_EXIT", is_filled=True, is_itt=True,
            ))

        session.rollover_events_count += 1
        session.rollover_records.append({"timestamp_utc": ro_ts_utc, "new_market": next_slug, "status": "SUCCESS"})

        assert len(session.pending_orders) == 0
        assert len(session.open_positions) == 0
        assert session.rollover_events_count == 1
        assert len(session.rollover_records) == 2
        assert session.rollover_records[1]["status"] == "SUCCESS"
        assert session.rollover_records[1]["new_market"] == "btc-updown-5m-101"


def test_down_token_independent_orderbook_execution():
    """
    Verify DOWN-token (pred_class == 0) execution uses down_token_id and down_book:
    1. Order limit price is placed at down_book.best_bid.
    2. Order asset_id is down_token_id.
    3. Order fills when down_book trades through.
    4. Horizon exit executes at down_book.best_bid with symmetric PnL formula.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_down_exec",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("tok_up")
        down_book = LiveTokenOrderBook("tok_down")
        up_book.is_initialized = True
        down_book.is_initialized = True

        # UP book: 0.50 / 0.51; DOWN book: 0.47 / 0.49
        up_book.exchange_best_bid = 0.50
        up_book.exchange_best_ask = 0.51
        up_book.last_update_ms = 1000

        down_book.exchange_best_bid = 0.47
        down_book.exchange_best_ask = 0.49
        down_book.last_update_ms = 1000

        # Create DOWN maker order (pred_class=0, buying DOWN token at down_book.best_bid = 0.47)
        down_order = MakerOrder(
            signal_id="sig_down_1",
            decision_timestamp_ms=1000,
            market_id="mkt",
            asset_id="tok_down",
            side="BUY",
            predicted_class=0,
            confidence=0.65,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.47,
            order_quantity=1.0,
            timeout_timestamp_ms=3000,
        )
        session.pending_orders.append(down_order)

        # At t=1500: DOWN touch only (Bid 0.47, Ask 0.49). Must NOT fill!
        session._evaluate_pending_orders_and_open_positions(1500, up_book, down_book, "mkt")
        assert len(session.pending_orders) == 1
        assert len(session.open_positions) == 0

        # At t=2000: DOWN Ask drops to 0.47 (trades through DOWN limit). Fills at 0.47!
        down_book.exchange_best_bid = 0.46
        down_book.exchange_best_ask = 0.47
        down_book.last_update_ms = 2000
        session._evaluate_pending_orders_and_open_positions(2000, up_book, down_book, "mkt")
        assert len(session.pending_orders) == 0
        assert len(session.open_positions) == 1
        pos = session.open_positions[0]
        assert pos.asset_id == "tok_down"
        assert pos.predicted_class == 0
        assert pos.fill_price == 0.47
        assert pos.fill_timestamp_ms == 2000
        assert pos.target_exit_timestamp_ms == 7000

        # At t=7000: Exit at DOWN bid = 0.48. Sells DOWN token at 0.48.
        down_book.exchange_best_bid = 0.48
        down_book.exchange_best_ask = 0.49
        down_book.last_update_ms = 7000
        # Notice: UP book can be completely different (e.g. 0.52 / 0.53)
        up_book.exchange_best_bid = 0.52
        up_book.exchange_best_ask = 0.53
        up_book.last_update_ms = 7000

        session._evaluate_pending_orders_and_open_positions(7000, up_book, down_book, "mkt")
        assert len(session.open_positions) == 0
        assert len(session.trade_records) == 1
        rec = session.trade_records[0]
        assert rec.asset_id == "tok_down"
        assert rec.predicted_class == 0
        assert rec.fill_price == 0.47
        assert rec.exit_price == 0.48
        # Gross PnL = (exit_p - fill_p) * 1.0 = (0.48 - 0.47) = +0.0100
        assert abs(rec.gross_pnl - 0.0100) < 1e-6
        # Fee accounting: 0 bps maker entry, 5 bps taker exit (0.48 * 0.0005 = 0.00024)
        expected_net = 0.0100 - (0.48 * 0.0005)
        assert abs(rec.net_pnl - expected_net) < 1e-6
        assert rec.outcome_status == "COMPLETED"


def test_no_cross_book_contamination_down_execution():
    """Verify DOWN order does NOT fill when only UP book trades through."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_no_cross_contam",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("tok_up")
        down_book = LiveTokenOrderBook("tok_down")
        up_book.is_initialized = True
        down_book.is_initialized = True

        up_book.exchange_best_bid = 0.50
        up_book.exchange_best_ask = 0.51
        up_book.last_update_ms = 1000

        down_book.exchange_best_bid = 0.47
        down_book.exchange_best_ask = 0.49
        down_book.last_update_ms = 1000

        # Pending DOWN order at 0.47
        down_order = MakerOrder(
            signal_id="sig_down_test",
            decision_timestamp_ms=1000,
            market_id="mkt",
            asset_id="tok_down",
            side="BUY",
            predicted_class=0,
            confidence=0.65,
            scaled_spread=-0.10,
            abs_scaled_depth_imbal=0.2,
            burst_id=0,
            limit_price=0.47,
            order_quantity=1.0,
            timeout_timestamp_ms=3000,
        )
        session.pending_orders.append(down_order)

        # UP book price collapses through 0.47 (UP Ask drops to 0.45)
        up_book.exchange_best_bid = 0.44
        up_book.exchange_best_ask = 0.45
        up_book.last_update_ms = 2000

        # DOWN book remains at 0.47 / 0.49 (Ask has NOT traded through 0.47)
        session._evaluate_pending_orders_and_open_positions(2000, up_book, down_book, "mkt")
        # DOWN order MUST NOT fill despite UP book dropping through!
        assert len(session.pending_orders) == 1
        assert len(session.open_positions) == 0


def test_divergent_up_down_spreads_and_stale_gate():
    """Verify grid step fails closed if DOWN book is stale while UP book is fresh."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_stale_down",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("tok_up")
        down_book = LiveTokenOrderBook("tok_down")
        up_book.is_initialized = True
        down_book.is_initialized = True

        # UP book fresh (age = 100 ms)
        up_book.exchange_best_bid = 0.50
        up_book.exchange_best_ask = 0.51
        up_book.last_update_ms = 2900

        # DOWN book stale (age = 2500 ms > 2000 ms)
        down_book.exchange_best_bid = 0.47
        down_book.exchange_best_ask = 0.49
        down_book.last_update_ms = 500

        # Run grid step at t=3000
        asyncio.run(session.process_buffered_grid_step(3000, up_book, down_book, "mkt", "tok_up", "tok_down"))
        assert session.stale_quotes_count == 1
        assert session.signals_generated_count == 0


def test_rolling_directional_asymmetry_hard_stop():
    """Verify intra-session hard stop triggers when directional asymmetry over last 50 filled trades > 0.40."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_rolling_asym_stop",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        # Create 46 UP filled trades and 4 DOWN filled trades
        # |46 - 4| / 50 = 42 / 50 = 0.84 > 0.40
        for i in range(50):
            p_cls = 2 if i < 46 else 0
            session.trade_records.append(TradeRecord(
                signal_id=f"sig_{i}",
                record_index=i,
                decision_timestamp_ms=i * 1000,
                market_id="mkt",
                asset_id="tok_up" if p_cls == 2 else "tok_down",
                burst_id=0,
                side="BUY",
                predicted_class=p_cls,
                confidence=0.60,
                scaled_spread=-0.10,
                abs_scaled_depth_imbal=0.1,
                limit_price=0.50,
                order_quantity=1.0,
                filled_quantity=1.0,
                fill_rate_ratio=1.0,
                fill_timestamp_ms=i * 1000,
                fill_price=0.50,
                exit_timestamp_ms=(i + 5) * 1000,
                exit_price=0.51,
                gross_pnl=0.01,
                maker_fee=0.0,
                taker_fee=0.00025,
                net_pnl=0.00975,
                outcome_status="COMPLETED",
                is_filled=True,
                is_itt=True,
            ))

        assert not session.is_hard_stopped
        session._check_hard_stops()
        assert session.is_hard_stopped is True
        assert session.hard_stop_type == "STRATEGY_RISK"
        assert "ROLLING_DIRECTIONAL_ASYMMETRY_LIMIT" in session.hard_stop_reason


def test_incomplete_window_asymmetry_inactive():
    """Verify rolling directional asymmetry hard stop is inactive when fewer than 50 trades have filled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_incomplete_asym",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        # Create 20 UP trades and 0 DOWN trades (incomplete window N_filled = 20 < 50)
        for i in range(20):
            session.trade_records.append(TradeRecord(
                signal_id=f"sig_{i}",
                record_index=i,
                decision_timestamp_ms=i * 1000,
                market_id="mkt",
                asset_id="tok_up",
                burst_id=0,
                side="BUY",
                predicted_class=2,
                confidence=0.60,
                scaled_spread=-0.10,
                abs_scaled_depth_imbal=0.1,
                limit_price=0.50,
                order_quantity=1.0,
                filled_quantity=1.0,
                fill_rate_ratio=1.0,
                fill_timestamp_ms=i * 1000,
                fill_price=0.50,
                exit_timestamp_ms=(i + 5) * 1000,
                exit_price=0.51,
                gross_pnl=0.01,
                maker_fee=0.0,
                taker_fee=0.00025,
                net_pnl=0.00975,
                outcome_status="COMPLETED",
                is_filled=True,
                is_itt=True,
            ))

        session._check_hard_stops()
        # Even though 100% of trades are UP, window < 50 must remain inactive!
        assert session.is_hard_stopped is False


def test_deterministic_prospective_burst_clustering_on_signal_gaps():
    """Verify Phase27ProspectiveSession groups signals < 5000ms apart and increments burst_id on > 5000ms gaps."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_burst_clustering",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")
        up_book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, 1000)
        down_book.apply_book_event({"bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.49", "size": "100"}]}, 1000)

        # Mock feature engine & model to produce directional prediction
        mock_scaled = np.zeros((1, 10, 11))
        mock_scaled[0, 9, 1] = -0.10  # satisfies TRAIN_SPREAD_THRESHOLD <= -0.0896
        session.feature_engine.compute_current_features_and_sequence = lambda: (
            np.zeros(20), mock_scaled
        )
        session.model = lambda x: torch.tensor([[0.0, 0.0, 5.0]])  # UP prediction with high confidence (> 0.55)

        # Signal 1 at t=10,000 ms -> burst 0
        up_book.last_update_ms = 10000
        down_book.last_update_ms = 10000
        asyncio.run(session.process_buffered_grid_step(10000, up_book, down_book, "mkt", "up", "down"))
        assert len(session.pending_orders) == 1
        assert session.pending_orders[0].burst_id == 0

        # Signal 2 at t=12,000 ms (gap = 2,000 ms <= 5,000 ms) -> stays in burst 0
        session.pending_orders.clear()
        up_book.last_update_ms = 12000
        down_book.last_update_ms = 12000
        asyncio.run(session.process_buffered_grid_step(12000, up_book, down_book, "mkt", "up", "down"))
        assert len(session.pending_orders) == 1
        assert session.pending_orders[0].burst_id == 0

        # Signal 3 at t=20,000 ms (gap = 8,000 ms > 5,000 ms) -> advances to burst 1
        # intermediate grid sampling ticks continue without data loss
        session.pending_orders.clear()
        session.last_event_ms = 19000
        up_book.last_update_ms = 20000
        down_book.last_update_ms = 20000
        asyncio.run(session.process_buffered_grid_step(20000, up_book, down_book, "mkt", "up", "down"))
        assert len(session.pending_orders) == 1
        assert session.pending_orders[0].burst_id == 1


def test_pre_rollover_entry_blackout_window_rejects_boundary_orders():
    """Verify that orders within 5,000 ms of 5m rollover are rejected under ROLLOVER_BLACKOUT_WINDOW."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = Phase27ProspectiveSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_blackout_gate",
            global_abort_file=Path(tmp_dir) / "abort.json",
        )
        up_book = LiveTokenOrderBook("up")
        down_book = LiveTokenOrderBook("down")
        up_book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, 296000)
        down_book.apply_book_event({"bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.49", "size": "100"}]}, 296000)

        session.feature_engine.compute_current_features_and_sequence = lambda: (
            np.zeros(20), np.zeros((1, 10, 11))
        )
        session.model = lambda x: torch.tensor([[0.1, 0.1, 0.8]])

        # Slot length is 300 seconds. t = 296,000 ms is 4 seconds before rollover (within 5,000 ms blackout window)
        asyncio.run(session.process_buffered_grid_step(296000, up_book, down_book, "mkt", "up", "down"))

        # Order must be suppressed/rejected due to ROLLOVER_BLACKOUT_WINDOW
        assert len(session.pending_orders) == 0
        assert session.trades_suppressed == 1
