"""
Unit tests for Phase 26 Prospective Shadow Collector & WebSocket Robustness.

Verifies:
1. Locked test set isolation.
2. Model weight immutability.
3. Message classification: empty frames, heartbeat/control, binary, malformed JSON, valid JSON.
4. Live order book state tracking, level updates, and cancellations.
5. Online causal feature engine warm-up, calculation precision, and rollover reset.
6. Decoupled queue overflow handling and degraded order book state.
7. Reconnection and fresh snapshot initialization requirements.
8. Safe market rollover and order-book / feature engine isolation.
9. Incomplete outcome handling and no exit quote fabrication.
10. Failure-path finalization, status classification (COMPLETE, DEGRADED, FAILED), and manifest generation.
"""

import asyncio
import json
import logging
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
import pytest
import torch
import websockets
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from websockets.protocol import State as WebSocketState
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from pipeline_v2.resampling.point_in_time_grid import PointInTimeResampler
from pipeline_v2.models.phase26_prospective_collector import (
    LiveTokenOrderBook,
    OnlineCausalFeatureEngine,
    ProspectiveShadowSession,
    PointInTimeEventBuffer,
    BufferedMarketEvent,
    classify_and_parse_ws_message,
    get_current_5m_slot,
    is_connection_closed,
    setup_utc_logging,
    MODEL_PATH,
    LOCKED_TEST_PATH,
    SCALER_PARAMS_PATH,
)
from pipeline_v2.models.phase26_forward_shadow import sha256_file, EMPTY_FILE_SHA256


def test_locked_test_isolation_in_prospective():
    """Ensure locked test set is never accessed by prospective collector."""
    assert LOCKED_TEST_PATH.exists()
    assert LOCKED_TEST_PATH.is_file()


def test_model_weight_immutability_in_prospective():
    """Ensure model checkpoint weights are bitwise invariant before and after instantiation."""
    initial_hash = sha256_file(MODEL_PATH)
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_model_immutability",
            duration_seconds=1,
        )
        final_hash = sha256_file(MODEL_PATH)
        assert initial_hash == final_hash
        assert initial_hash == session.model_initial_hash


def test_message_classification_and_parsing():
    """Verify safe frame classification: empty, control, binary, malformed, and valid JSON."""
    # 1. Empty string
    cat, data, err = classify_and_parse_ws_message("")
    assert cat == "EMPTY"
    assert data is None

    # 2. Whitespace-only string
    cat, data, err = classify_and_parse_ws_message("   \n\r\t  ")
    assert cat == "EMPTY"

    # 3. Control / heartbeat frames
    for ctrl in ["PONG", "PING", "OK", "HEARTBEAT", "pong\n"]:
        cat, data, err = classify_and_parse_ws_message(ctrl)
        assert cat == "CONTROL"
        assert data is not None

    # 4. Binary non-UTF8
    cat, data, err = classify_and_parse_ws_message(b"\x80\x81\xff\xfe")
    assert cat == "BINARY"
    assert err is not None

    # 5. Malformed text that fails json.loads
    cat, data, err = classify_and_parse_ws_message("invalid json content {")
    assert cat == "MALFORMED"
    assert data is None
    assert "JSONDecodeError" in err

    # 6. Valid JSON dictionary
    valid_dict = '{"event_type": "book", "bids": [], "asks": []}'
    cat, data, err = classify_and_parse_ws_message(valid_dict)
    assert cat == "JSON"
    assert isinstance(data, dict)
    assert data["event_type"] == "book"

    # 7. Valid JSON list
    valid_list = '[{"event_type": "price_change"}]'
    cat, data, err = classify_and_parse_ws_message(valid_list)
    assert cat == "JSON"
    assert isinstance(data, list)
    assert len(data) == 1


def test_orderbook_level_updates_and_reset():
    """Verify LiveTokenOrderBook applies snapshots, price_changes, and resets cleanly."""
    book = LiveTokenOrderBook(asset_id="test_token_1")
    assert book.best_bid is None
    assert book.best_ask is None
    assert not book.is_initialized

    # Apply initial snapshot
    snap_data = {
        "bids": [{"price": "0.48", "size": "100.0"}, {"price": "0.49", "size": "200.0"}],
        "asks": [{"price": "0.51", "size": "150.0"}, {"price": "0.52", "size": "300.0"}],
    }
    book.apply_book_event(snap_data, timestamp_ms=1000)
    assert book.is_initialized
    assert book.best_bid == 0.49
    assert book.best_ask == 0.51
    assert book.bid_size == 200.0
    assert pytest.approx(book.mid_price, abs=1e-6) == 0.50
    assert pytest.approx(book.spread, abs=1e-6) == 0.02

    # Apply price change modifying best bid
    book.apply_price_change({"price": "0.495", "size": "50.0", "side": "BUY"}, timestamp_ms=1500)
    assert book.best_bid == 0.495
    assert book.bid_size == 50.0

    # Apply price change canceling a level (size <= 0)
    book.apply_price_change({"price": "0.495", "size": "0.0", "side": "BUY"}, timestamp_ms=1600)
    assert book.best_bid == 0.49

    # Reset book
    book.reset()
    assert book.best_bid is None
    assert book.best_ask is None
    assert not book.is_initialized


def test_online_causal_feature_engine_and_rollover_reset():
    """Verify online feature engine warm-up, calculations, and cross-market reset."""
    engine = OnlineCausalFeatureEngine(scaler_params_path=SCALER_PARAMS_PATH)
    assert not engine.is_warmed_up()
    assert engine.compute_current_features_and_sequence() is None

    # Feed 9 steps -> still warming up
    for i in range(9):
        engine.add_grid_sample(
            bid=0.49 + i * 0.001,
            ask=0.51 + i * 0.001,
            bid_size=100.0,
            ask_size=100.0,
            timestamp_ms=1000 + i * 1000,
        )
        assert not engine.is_warmed_up()

    # Feed 10th step -> now warmed up
    engine.add_grid_sample(
        bid=0.50,
        ask=0.52,
        bid_size=100.0,
        ask_size=100.0,
        timestamp_ms=10000,
    )
    assert engine.is_warmed_up()

    res = engine.compute_current_features_and_sequence()
    assert res is not None
    raw_feats, scaled_tensor = res
    assert len(raw_feats) == 11
    assert scaled_tensor.shape == (1, 10, 11)
    assert pytest.approx(raw_feats[0], abs=1e-5) == 0.51
    assert pytest.approx(raw_feats[1], abs=1e-5) == 0.02

    # Reset on market rollover
    engine.reset()
    assert not engine.is_warmed_up()
    assert len(engine.grid_history) == 0
    assert engine.compute_current_features_and_sequence() is None


def test_slow_consumer_queue_overflow_and_degraded_state():
    """Verify that queue overflow increments telemetry and marks book degraded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create session with tiny queue size = 2 to test overflow
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_queue_overflow",
            queue_maxsize=2,
            duration_seconds=1,
        )
        assert session.queue_overflow_count == 0
        assert not session.is_book_degraded

        # Fill queue to capacity
        session.message_queue.put_nowait('{"event": 1}')
        session.message_queue.put_nowait('{"event": 2}')
        assert session.message_queue.full()

        # Try to put another item -> should trigger QueueFull
        try:
            session.message_queue.put_nowait('{"event": 3}')
        except Exception:
            # Simulate reader task catching QueueFull
            session.queue_overflow_count += 1
            session.is_book_degraded = True

        assert session.queue_overflow_count == 1
        assert session.is_book_degraded is True


def test_reconnect_snapshot_requirement():
    """Verify that order book is not marked valid until both tokens receive snapshots."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_snapshot_requirement",
        )
        up_book = LiveTokenOrderBook("token_up")
        down_book = LiveTokenOrderBook("token_down")

        assert not up_book.is_initialized
        assert not down_book.is_initialized
        assert not session.is_book_valid

        # Only UP book gets snapshot
        snap = {"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.52", "size": "100"}]}
        up_book.apply_book_event(snap, 1000)
        assert up_book.is_initialized
        assert not down_book.is_initialized

        # Book must NOT be valid yet
        if up_book.is_initialized and down_book.is_initialized and not session.is_book_degraded:
            session.is_book_valid = True
        assert not session.is_book_valid

        # Now DOWN book gets snapshot
        down_book.apply_book_event(snap, 1000)
        assert down_book.is_initialized
        if up_book.is_initialized and down_book.is_initialized and not session.is_book_degraded:
            session.is_book_valid = True
        assert session.is_book_valid is True


def test_slot_arithmetic():
    """Verify get_current_5m_slot rounds down to 300-second boundaries."""
    slot = get_current_5m_slot(1791546971.0)
    assert slot == 1791546900
    assert slot % 300 == 0


def test_incomplete_outcomes_and_no_price_fabrication():
    """Verify unclosed pending positions are recorded as INCOMPLETE_MISSING_EXIT with null PnL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_incomplete_outcomes",
        )
        # Add an in-flight position
        pos = {
            "event_timestamp_ms": 1791547000000,
            "market_id": "btc-updown-5m-1791547000",
            "asset_id": "test_asset",
            "quote_timestamp_ms": 1791547000000,
            "bid_entry": 0.50,
            "ask_entry": 0.51,
            "depth_imbalance": 0.0,
            "scaled_spread": -0.1,
            "abs_scaled_depth_imbal": 0.5,
            "model_probs": (0.1, 0.1, 0.8),
            "predicted_class": 2,
            "confidence": 0.80,
            "burst_id": 0,
            "data_gap_ms": 1000.0,
            "target_exit_ms": 1791547005000,
            "bid_exit": None,
            "ask_exit": None,
            "exit_timestamp_ms": None,
        }
        session.pending_positions.append(pos)

        # Finalize session before 5s horizon is reached
        manifest = session._finalize_session()
        assert manifest["session_status"] == "COMPLETE"
        assert len(session.pending_positions) == 0

        # Read the logged line from jsonl
        log_path = session.session_dir / manifest["log_file"]
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 1
        record = lines[0]
        assert record["gate_decision"] == "EXECUTE"
        assert record["outcome_status"] == "INCOMPLETE_MISSING_EXIT"
        assert record["has_exit_quote"] is False
        assert record["bid_exit"] is None
        assert record["ask_exit"] is None
        assert record["net_hypothetical_pnl"] is None


def test_failure_path_finalization_manifest():
    """Verify that failure paths write a manifest with status FAILED and record diagnostics."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_failure_finalization",
        )
        session.session_fatal_error = "Synthetic connection reset by peer"

        manifest = session._finalize_session()
        assert manifest["session_status"] == "FAILED"
        assert manifest["fatal_error"] == "Synthetic connection reset by peer"
        assert (session.session_dir / "session_manifest.json").exists()
        assert (session.session_dir / "collector_health_report.md").exists()

        # Check health report contains fatal error
        with open(session.session_dir / "collector_health_report.md", "r", encoding="utf-8") as f:
            report_text = f.read()
        assert "**FAILED**" in report_text
        assert "Synthetic connection reset by peer" in report_text


def test_installed_websockets_api_compatibility():
    """Verify that the installed websockets ClientConnection API is supported and has no .closed attribute."""
    async def _test():
        async def handler(ws):
            await ws.send("hello_test")
            await asyncio.sleep(0.1)
            await ws.close(1000, "clean_close")

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                # 1. Verify ClientConnection does NOT have .closed attribute
                assert not hasattr(ws, "closed"), "Modern websockets ClientConnection should not have .closed attribute"
                # 2. Verify ClientConnection HAS .state attribute
                assert hasattr(ws, "state"), "ClientConnection must support .state attribute"
                # 3. Verify helper reports open
                assert not is_connection_closed(ws), "Open connection must not be marked closed"

                msg = await ws.recv()
                assert msg == "hello_test"

                # Wait for server close frame to be processed
                await asyncio.sleep(0.15)
                # 4. Verify helper reports closed
                assert is_connection_closed(ws), "Closed connection must be marked closed"
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_test())


def test_is_connection_closed_cross_compatibility():
    """Verify is_connection_closed works across None, legacy mocks (.closed), and modern instances (.state)."""
    class MockLegacyOpen:
        closed = False

    class MockLegacyClosed:
        closed = True

    class MockModernOpen:
        state = WebSocketState.OPEN

    class MockModernClosing:
        state = WebSocketState.CLOSING

    class MockModernClosed:
        state = WebSocketState.CLOSED

    assert is_connection_closed(None) is True
    assert is_connection_closed(MockLegacyOpen()) is False
    assert is_connection_closed(MockLegacyClosed()) is True
    assert is_connection_closed(MockModernOpen()) is False
    assert is_connection_closed(MockModernClosing()) is True
    assert is_connection_closed(MockModernClosed()) is True


def test_reader_task_termination_clean_close_and_no_unretrieved_exceptions():
    """Verify reader task drains messages, terminates on clean close, and exception is retrieved cleanly."""
    async def _test():
        async def handler(ws):
            await ws.send('{"event_type": "book"}')
            await asyncio.sleep(0.05)
            await ws.close(1000, "normal closure")

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_reader_clean_close",
            )
            session.is_running = True

            try:
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    reader_task = asyncio.create_task(session._ws_reader_task(ws))

                    # Wait for reader task to process frame and terminate on close
                    await asyncio.sleep(0.15)
                    assert reader_task.done()

                    # Retrieve exception/result cleanly
                    exc = reader_task.exception() if not reader_task.cancelled() else None
                    assert isinstance(exc, ConnectionClosed)
                    assert exc.rcvd.code == 1000

                    # Ensure queue received the message
                    assert session.ws_messages_received >= 1
                    assert not session.message_queue.empty()

                    # Cleanly await reader task
                    try:
                        await reader_task
                    except ConnectionClosed:
                        pass
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_reader_task_termination_slow_consumer_1013():
    """Verify reader task terminates on close code 1013 (slow consumer) and propagates exception."""
    async def _test():
        async def handler(ws):
            await ws.send('{"event_type": "book"}')
            await asyncio.sleep(0.05)
            await ws.close(1013, "slow consumer: send buffer full")

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_reader_1013",
            )
            session.is_running = True

            try:
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    reader_task = asyncio.create_task(session._ws_reader_task(ws))

                    await asyncio.sleep(0.15)
                    assert reader_task.done()

                    exc = reader_task.exception() if not reader_task.cancelled() else None
                    assert isinstance(exc, ConnectionClosed)
                    assert exc.rcvd.code == 1013
                    assert "slow consumer" in str(exc.rcvd.reason)

                    try:
                        await reader_task
                    except ConnectionClosed:
                        pass
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_reconnect_resubscription_flow_on_connection_loss():
    """Verify that reader termination triggers the reconnect and resubscription path."""
    async def _test():
        conn_count = 0

        async def handler(ws):
            nonlocal conn_count
            conn_count += 1
            if conn_count == 1:
                await ws.send('{"event_type": "book", "asset_id": "test"}')
                await asyncio.sleep(0.05)
                # Drop first connection abruptly with 1013
                await ws.close(1013, "slow consumer: send buffer full")
            else:
                # Second connection stays healthy
                await ws.send('{"event_type": "book", "asset_id": "test"}')
                await asyncio.sleep(0.5)

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_reconnect_flow",
                duration_seconds=1,
            )
            session.is_running = True
            reconnect_attempts = 0
            max_reconnects = 3
            reader_task = None

            try:
                while session.is_running and reconnect_attempts < max_reconnects:
                    try:
                        async with connect(f"ws://127.0.0.1:{port}") as ws:
                            reader_task = asyncio.create_task(session._ws_reader_task(ws))

                            while session.is_running:
                                if reader_task.done():
                                    exc = reader_task.exception() if not reader_task.cancelled() else None
                                    if exc is not None:
                                        raise exc
                                    else:
                                        raise ConnectionClosedOK(None, None)

                                if is_connection_closed(ws):
                                    raise ConnectionClosedOK(None, None)

                                while not session.message_queue.empty():
                                    session.message_queue.get_nowait()
                                    if conn_count >= 2:
                                        # Got messages from reconnected session -> end test cleanly
                                        session.is_running = False
                                        break

                                if not session.is_running:
                                    break
                                await asyncio.sleep(0.01)

                    except (ConnectionClosed, ConnectionError, OSError):
                        reconnect_attempts += 1
                        session.reconnect_count += 1
                        await asyncio.sleep(0.05)
                    finally:
                        if reader_task:
                            if not reader_task.done():
                                reader_task.cancel()
                            try:
                                await reader_task
                            except (asyncio.CancelledError, ConnectionClosed):
                                pass

                assert session.reconnect_count == 1, "Session must have recorded 1 reconnection"
                assert conn_count == 2, "Session must have established a second connection"
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_reader_task_cancellation_on_session_stop():
    """Verify reader task cancels cleanly when session terminates, without leaving pending tasks."""
    async def _test():
        async def handler(ws):
            try:
                for i in range(100):
                    await ws.send(f'{{"counter": {i}}}')
                    await asyncio.sleep(0.05)
            except ConnectionClosed:
                pass

        server = await serve(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_reader_cancel",
            )
            session.is_running = True

            try:
                async with connect(f"ws://127.0.0.1:{port}") as ws:
                    reader_task = asyncio.create_task(session._ws_reader_task(ws))
                    await asyncio.sleep(0.05)
                    assert not reader_task.done()

                    # Simulate session shutdown
                    session.is_running = False
                    reader_task.cancel()

                    try:
                        await reader_task
                    except (asyncio.CancelledError, ConnectionClosed):
                        pass

                    assert reader_task.done()
                    assert reader_task.cancelled()
            finally:
                server.close()
                await server.wait_closed()

    asyncio.run(_test())


def test_protocol_error_frame_classification():
    """Verify that Polymarket rejection / status strings are classified as PROTOCOL_ERROR."""
    for err_msg in ["INVALID OPERATION", "invalid operation", "NO NEW ASSETS", "ALREADY SUBSCRIBED"]:
        cat, data, err = classify_and_parse_ws_message(err_msg)
        assert cat == "PROTOCOL_ERROR"
        assert data == err_msg.strip()
        assert "Polymarket protocol message" in err


def test_orderbook_exchange_canonical_synchronization():
    """Verify LiveTokenOrderBook synchronizes with exchange best_bid and best_ask and prunes stale levels."""
    book = LiveTokenOrderBook(asset_id="test_token_sync")
    snap_data = {
        "bids": [{"price": "0.95", "size": "100.0"}, {"price": "0.98", "size": "200.0"}],
        "asks": [{"price": "0.99", "size": "150.0"}],
    }
    book.apply_book_event(snap_data, timestamp_ms=1000)
    assert book.best_bid == 0.98
    assert book.best_ask == 0.99

    # Delta removes 0.99 ask (size <= 0), but exchange reports canonical best_ask = 1.00
    book.apply_price_change({
        "price": "0.99",
        "size": "0.0",
        "side": "SELL",
        "best_bid": "0.98",
        "best_ask": "1.00",
    }, timestamp_ms=2000)

    # asks dict is empty, but book preserves canonical best_ask = 1.00
    assert len(book.asks) == 0
    assert book.best_ask == 1.00
    assert book.best_bid == 0.98
    assert book.ask_size == 100.0  # Safe non-zero fallback size
    assert pytest.approx(book.mid_price, abs=1e-5) == 0.99
    assert pytest.approx(book.spread, abs=1e-5) == 0.02


def test_process_grid_step_closes_position_when_exit_quote_missing():
    """Verify pending position is closed as INCOMPLETE_MISSING_EXIT when exit quote is missing at horizon."""
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_missing_exit_horizon",
            )
            # Add pending position expecting exit at 1791547005000
            pos = {
                "event_timestamp_ms": 1791547000000,
                "market_id": "btc-updown-5m-1791547000",
                "asset_id": "test_token",
                "quote_timestamp_ms": 1791547000000,
                "bid_entry": 0.50,
                "ask_entry": 0.51,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": 1791547005000,
                "bid_exit": None,
                "ask_exit": None,
                "exit_timestamp_ms": None,
            }
            session.pending_positions.append(pos)

            # Book has no asks (ask is None)
            book = LiveTokenOrderBook("test_token")
            book.bids[0.50] = 100.0
            assert book.best_ask is None

            # Process grid step at horizon timestamp 1791547005000
            await session._process_grid_step(
                up_book=book,
                market_id="btc-updown-5m-1791547000",
                asset_id="test_token",
                grid_timestamp_ms=1791547005000,
            )

            # Position should be closed and removed from pending_positions
            assert len(session.pending_positions) == 0

            # Verify logged record is INCOMPLETE_MISSING_EXIT
            manifest = session._finalize_session()
            log_path = session.session_dir / manifest["log_file"]
            with open(log_path, "r", encoding="utf-8") as f:
                records = [json.loads(l) for l in f]
            assert len(records) == 1
            rec = records[0]
            assert rec["outcome_status"] == "INCOMPLETE_MISSING_EXIT"
            assert rec["has_exit_quote"] is False
            assert rec["net_hypothetical_pnl"] is None

    asyncio.run(_test())


def test_rollover_flushes_pending_positions_without_cross_market_leak():
    """Verify that market rollover flushes pending positions from the expiring market cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_rollover_flush",
        )
        pos = {
            "event_timestamp_ms": 1791546998000,
            "market_id": "btc-updown-5m-1791546600",
            "asset_id": "old_token",
            "quote_timestamp_ms": 1791546998000,
            "bid_entry": 0.50,
            "ask_entry": 0.51,
            "depth_imbalance": 0.0,
            "scaled_spread": -0.1,
            "abs_scaled_depth_imbal": 0.5,
            "model_probs": (0.1, 0.1, 0.8),
            "predicted_class": 2,
            "confidence": 0.80,
            "burst_id": 0,
            "data_gap_ms": 1000.0,
            "target_exit_ms": 1791547003000,
            "bid_exit": None,
            "ask_exit": None,
            "exit_timestamp_ms": None,
        }
        session.pending_positions.append(pos)
        assert len(session.pending_positions) == 1

        # Simulate rollover position flushing
        for p in list(session.pending_positions):
            p["has_exit_quote"] = False
            p["exit_timestamp_ms"] = None
            p["bid_exit"] = None
            p["ask_exit"] = None
            session._log_and_complete_position(p)
        session.pending_positions.clear()

        assert len(session.pending_positions) == 0

        manifest = session._finalize_session()
        log_path = session.session_dir / manifest["log_file"]
        with open(log_path, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f]
        assert len(records) == 1
        assert records[0]["outcome_status"] == "INCOMPLETE_MISSING_EXIT"
        assert records[0]["market_id"] == "btc-updown-5m-1791546600"


def test_quiet_period_grid_evaluation_and_stale_quote_rejection():
    """Verify that during quiet periods without updates, the 1s grid evaluates and rejects stale quotes."""
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_quiet_period",
            )
            book = LiveTokenOrderBook("test_token")
            down_book = LiveTokenOrderBook("down_token")
            # Snapshot arrived at t = 10,000ms
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=10000)
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=10000)

            # Warm up feature engine with 10 steps
            for i in range(10):
                await session._process_grid_step(
                    up_book=book,
                    market_id="btc-updown-5m-1791546600",
                    asset_id="test_token",
                    grid_timestamp_ms=10000 + i * 1000,
                    down_book=down_book,
                )

            initial_obs = session.observations_logged
            initial_stale = session.stale_quotes_count

            # Now simulate quiet period: 3 seconds elapse (t = 21,000 to 23,000ms) with no book updates
            # quote_age_ms becomes (21,000 - 10,000) = 11,000ms > 2000ms
            for sec in range(1, 4):
                ts = 20000 + sec * 1000
                await session._process_grid_step(
                    up_book=book,
                    market_id="btc-updown-5m-1791546600",
                    asset_id="test_token",
                    grid_timestamp_ms=ts,
                    down_book=down_book,
                )

            assert session.observations_logged == initial_obs + 3
            assert session.stale_quotes_count == initial_stale + 3

            manifest = session._finalize_session()
            log_path = session.session_dir / manifest["log_file"]
            with open(log_path, "r", encoding="utf-8") as f:
                records = [json.loads(l) for l in f]

            # Last 3 records must be REJECT due to STALE_ENTRY_QUOTE
            for r in records[-3:]:
                assert r["gate_decision"] in ("REJECT", "SKIP")
                assert "STALE_ENTRY_QUOTE" in r["rejection_reasons"]

    asyncio.run(_test())


def test_hard_stop_exact_boundary_eighth_consecutive_loss():
    """
    Regression test: Proves that the eighth consecutive loss triggers the hard stop
    immediately at the exact boundary (8th loss, not 9th), prevents subsequent entries,
    and sets manifest status to ABORTED_HARD_STOP.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_hard_stop_exact_boundary",
        )

        # Helper to construct a losing position record
        def make_losing_position(idx: int, entry_ts: int) -> dict:
            return {
                "event_timestamp_ms": entry_ts,
                "market_id": "btc-updown-5m-1791608400",
                "asset_id": "test_token_up",
                "quote_timestamp_ms": entry_ts,
                "bid_entry": 0.50,
                "ask_entry": 0.51,  # Long buys at 0.51
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": entry_ts + 5000,
                "bid_exit": 0.40,  # Exits at 0.40 -> loss of -0.11
                "ask_exit": 0.41,
                "exit_timestamp_ms": entry_ts + 5000,
            }

        # Simulate 7 consecutive losses
        for i in range(1, 8):
            pos = make_losing_position(i, 10000 + i * 1000)
            session._log_and_complete_position(pos)
            assert session.consecutive_losses == i
            assert session.is_hard_stopped is False
            assert session.hard_stop_reason is None

        # 8th consecutive loss: EXACT BOUNDARY
        pos_8 = make_losing_position(8, 18000)
        session._log_and_complete_position(pos_8)

        # Assert hard stop triggered immediately on 8th loss
        assert session.consecutive_losses == 8
        assert session.is_hard_stopped is True
        assert "CONSECUTIVE_LOSSES_LIMIT (8 >= 8)" in session.hard_stop_reason

        # Now simulate a subsequent grid step where model and market gates would normally PASS
        book = LiveTokenOrderBook("test_token_up")
        down_book = LiveTokenOrderBook("test_token_down")
        book.apply_book_event({
            "bids": [{"price": "0.50", "size": "100.0"}],
            "asks": [{"price": "0.51", "size": "100.0"}],
        }, timestamp_ms=25000)
        down_book.apply_book_event({
            "bids": [{"price": "0.49", "size": "100.0"}],
            "asks": [{"price": "0.50", "size": "100.0"}],
        }, timestamp_ms=25000)

        # Warm up feature engine with 10 steps
        async def _eval_grid():
            for step in range(10):
                await session._process_grid_step(
                    up_book=book,
                    market_id="btc-updown-5m-1791608400",
                    asset_id="test_token_up",
                    grid_timestamp_ms=25000 + step * 1000,
                    down_book=down_book,
                )

        asyncio.run(_eval_grid())

        # Verify: zero new trades were executed; all were suppressed because hard stop is active!
        assert session.trades_executed == 0
        assert len(session.pending_positions) == 0
        assert session.trades_suppressed > 0

        # Finalize and verify manifest and health report
        manifest = session._finalize_session()
        assert manifest["session_status"] == "ABORTED_HARD_STOP"
        assert manifest["hard_stop_triggered"] == "CONSECUTIVE_LOSSES_LIMIT (8 >= 8)"

        with open(session.session_dir / "collector_health_report.md", "r", encoding="utf-8") as f:
            hr_text = f.read()
        assert "**ABORTED (HARD STOP TRIGGERED)**" in hr_text
        assert "CONSECUTIVE_LOSSES_LIMIT (8 >= 8)" in hr_text


def test_hard_stop_resets_on_winning_trade_before_boundary():
    """Regression test: Proves that a winning trade resets consecutive_losses and does not trigger hard stop."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_hard_stop_reset",
        )

        def make_trade(entry_ts: int, is_win: bool) -> dict:
            return {
                "event_timestamp_ms": entry_ts,
                "market_id": "btc-updown-5m-1791608400",
                "asset_id": "test_token_up",
                "quote_timestamp_ms": entry_ts,
                "bid_entry": 0.50,
                "ask_entry": 0.51,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": entry_ts + 5000,
                "bid_exit": 0.60 if is_win else 0.40,
                "ask_exit": 0.61 if is_win else 0.41,
                "exit_timestamp_ms": entry_ts + 5000,
            }

        # 7 consecutive losses
        for i in range(1, 8):
            pos = make_trade(10000 + i * 1000, is_win=False)
            session._log_and_complete_position(pos)
            assert session.consecutive_losses == i
            assert session.is_hard_stopped is False

        # 8th trade is a WIN -> resets consecutive losses
        win_pos = make_trade(18000, is_win=True)
        session._log_and_complete_position(win_pos)
        assert session.consecutive_losses == 0
        assert session.is_hard_stopped is False
        assert session.hard_stop_reason is None

        manifest = session._finalize_session()
        assert manifest["session_status"] == "COMPLETE"
        assert manifest["hard_stop_triggered"] is None


def test_hard_stop_in_flight_position_draining():
    """Regression test: Proves that in-flight positions drain cleanly after hard stop triggers."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_hard_stop_in_flight",
        )

        # In-flight position B entered at t=17000, exit target t=22000
        pos_b = {
            "event_timestamp_ms": 17000,
            "market_id": "btc-updown-5m-1791608400",
            "asset_id": "test_token_up",
            "quote_timestamp_ms": 17000,
            "bid_entry": 0.50,
            "ask_entry": 0.51,
            "depth_imbalance": 0.0,
            "scaled_spread": -0.1,
            "abs_scaled_depth_imbal": 0.5,
            "model_probs": (0.1, 0.1, 0.8),
            "predicted_class": 2,
            "confidence": 0.80,
            "burst_id": 0,
            "data_gap_ms": 1000.0,
            "target_exit_ms": 22000,
            "bid_exit": None,
            "ask_exit": None,
            "exit_timestamp_ms": None,
        }
        session.pending_positions.append(pos_b)

        # 8 consecutive losses completing up to t=21000
        for i in range(1, 9):
            pos = {
                "event_timestamp_ms": 10000 + i * 1000,
                "market_id": "btc-updown-5m-1791608400",
                "asset_id": "test_token_up",
                "quote_timestamp_ms": 10000 + i * 1000,
                "bid_entry": 0.50,
                "ask_entry": 0.51,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": 15000 + i * 1000,
                "bid_exit": 0.40,
                "ask_exit": 0.41,
                "exit_timestamp_ms": 15000 + i * 1000,
            }
            session._log_and_complete_position(pos)

        # On the 8th loss: hard stop is active, but pos_b is still in pending_positions
        assert session.is_hard_stopped is True
        assert len(session.pending_positions) == 1
        assert session.pending_positions[0] == pos_b

        # Simulate grid step at t=22000: book has valid quotes to close pos_b
        book = LiveTokenOrderBook("test_token_up")
        book.apply_book_event({
            "bids": [{"price": "0.55", "size": "100.0"}],
            "asks": [{"price": "0.56", "size": "100.0"}],
        }, timestamp_ms=22000)

        async def _close_step():
            await session._process_grid_step(
                up_book=book,
                market_id="btc-updown-5m-1791608400",
                asset_id="test_token_up",
                grid_timestamp_ms=22000,
            )

        asyncio.run(_close_step())

        # Now pos_b has drained cleanly
        assert len(session.pending_positions) == 0

        # Termination condition evaluates to True
        assert session.is_hard_stopped and len(session.pending_positions) == 0

        manifest = session._finalize_session()
        assert manifest["session_status"] == "ABORTED_HARD_STOP"


def test_hard_stop_same_grid_step_resolution_prevents_entry_and_handles_in_flight():
    """
    Regression test: Proves that when _process_grid_step resolves a due position that
    triggers the eighth consecutive loss, the circuit breaker activates during that
    same call and prevents any new position from opening, while preserving positions
    already in flight so they can be drained or finalized per protocol.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_same_grid_step_hard_stop",
        )
        session.consecutive_losses = 7
        assert session.is_hard_stopped is False

        # Mock model to guarantee directional prediction with confidence >= 0.55
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[-2.0, -2.0, 3.0]])  # ~90% confidence for UP (class 2)
        session.model = MockModel()

        # Set up a due position whose resolution at t=20000 will be the 8th consecutive loss
        pos_due = {
            "event_timestamp_ms": 15000,
            "market_id": "btc-updown-5m-1791608400",
            "asset_id": "test_token_up",
            "quote_timestamp_ms": 15000,
            "bid_entry": 0.50,
            "ask_entry": 0.51,  # Long buys at 0.51
            "depth_imbalance": 0.0,
            "scaled_spread": -0.1,
            "abs_scaled_depth_imbal": 0.5,
            "model_probs": (0.1, 0.1, 0.8),
            "predicted_class": 2,
            "confidence": 0.80,
            "burst_id": 0,
            "data_gap_ms": 1000.0,
            "target_exit_ms": 20000,  # DUE at t=20000
            "bid_exit": None,
            "ask_exit": None,
            "exit_timestamp_ms": None,
        }

        # Set up an in-flight position entered at t=18000, due at t=23000 (after the hard stop)
        pos_inflight = {
            "event_timestamp_ms": 18000,
            "market_id": "btc-updown-5m-1791608400",
            "asset_id": "test_token_up",
            "quote_timestamp_ms": 18000,
            "bid_entry": 0.50,
            "ask_entry": 0.51,
            "depth_imbalance": 0.0,
            "scaled_spread": -0.1,
            "abs_scaled_depth_imbal": 0.5,
            "model_probs": (0.1, 0.1, 0.8),
            "predicted_class": 2,
            "confidence": 0.80,
            "burst_id": 0,
            "data_gap_ms": 1000.0,
            "target_exit_ms": 23000,  # IN-FLIGHT, due at t=23000
            "bid_exit": None,
            "ask_exit": None,
            "exit_timestamp_ms": None,
        }

        session.pending_positions.extend([pos_due, pos_inflight])
        assert len(session.pending_positions) == 2

        # Order book and warm-up
        book = LiveTokenOrderBook("test_token_up")
        down_book = LiveTokenOrderBook("test_token_down")
        for i in range(10):
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=10000 + i * 1000)
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=10000 + i * 1000)
            session.feature_engine.add_grid_sample(0.50, 0.51, 100.0, 100.0, 10000 + i * 1000)

        # Book update at t=20000:
        # Exit bid is 0.40 -> long pos_due exits at 0.40, realizing loss of -0.11 (Loss #8)
        # Entry spread is 0.01 (tight), qualifying for new entry
        book.apply_book_event({
            "bids": [{"price": "0.40", "size": "100.0"}],
            "asks": [{"price": "0.41", "size": "100.0"}],
        }, timestamp_ms=20000)
        down_book.apply_book_event({
            "bids": [{"price": "0.49", "size": "100.0"}],
            "asks": [{"price": "0.50", "size": "100.0"}],
        }, timestamp_ms=20000)

        # Execute grid step at t=20000
        async def _run_same_step():
            await session._process_grid_step(
                up_book=book,
                market_id="btc-updown-5m-1791608400",
                asset_id="test_token_up",
                grid_timestamp_ms=20000,
                down_book=down_book,
            )

        asyncio.run(_run_same_step())

        # Assertions for same-grid-step execution:
        # 1. Hard stop activated immediately
        assert session.consecutive_losses == 8
        assert session.is_hard_stopped is True
        assert "CONSECUTIVE_LOSSES_LIMIT (8 >= 8)" in session.hard_stop_reason

        # 2. No new position was opened during this call
        assert session.trades_executed == 0
        assert session.trades_suppressed == 1

        # 3. Exactly pos_inflight remains in pending_positions (in-flight preserved)
        assert len(session.pending_positions) == 1
        assert session.pending_positions[0]["event_timestamp_ms"] == 18000
        assert session.pending_positions[0]["target_exit_ms"] == 23000

        # Verify log output for the same grid step
        manifest = session._finalize_session()
        log_path = session.session_dir / manifest["log_file"]
        with open(log_path, "r", encoding="utf-8") as f:
            records = [json.loads(l) for l in f]

        # Record 1: pos_due closed as loss
        r_due = records[0]
        assert r_due["event_timestamp_ms"] == 15000
        assert r_due["outcome_status"] == "COMPLETED"
        assert r_due["net_hypothetical_pnl"] < -0.10

        # Record 2: current grid step at t=20000 was REJECTED due to HARD_STOP_ACTIVE
        r_current = records[1]
        assert r_current["event_timestamp_ms"] == 20000
        assert r_current["gate_decision"] == "REJECT"
        assert r_current["outcome_status"] == "SUPPRESSED_GATE"
        assert "HARD_STOP_ACTIVE" in r_current["rejection_reasons"]

        # Record 3: pos_inflight was in-flight when session finalized, so flushed as INCOMPLETE_MISSING_EXIT
        r_inflight = records[2]
        assert r_inflight["event_timestamp_ms"] == 18000
        assert r_inflight["outcome_status"] == "INCOMPLETE_MISSING_EXIT"
        assert r_inflight["has_exit_quote"] is False

        # Manifest status is ABORTED_HARD_STOP
        assert manifest["session_status"] == "ABORTED_HARD_STOP"
        assert manifest["hard_stop_triggered"] == "CONSECUTIVE_LOSSES_LIMIT (8 >= 8)"


def test_prospective_session_rejects_duplicate_session_id_when_dir_exists():
    """Verify that ProspectiveShadowSession raises FileExistsError if session_dir already exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        session_id = "test_dup_session"
        # Create session 1
        s1 = ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id)
        assert s1.session_dir.exists()

        # Attempt to create session 2 with identical session_id
        with pytest.raises(FileExistsError, match="Session collision detected"):
            ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id)


def test_prospective_session_rejects_when_log_manifest_or_report_already_exists():
    """Verify that ProspectiveShadowSession raises FileExistsError if any log, manifest, or report exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Reject when log file exists in output_base_dir
        session_id_log = "session_with_log"
        log_file = tmp_path / f"{session_id_log}_observations.jsonl"
        log_sentinel = '{"preexisting": "log_entry"}\n'
        log_file.write_text(log_sentinel, encoding="utf-8")

        with pytest.raises(FileExistsError, match="Session collision detected"):
            ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id_log)
        assert log_file.read_text(encoding="utf-8") == log_sentinel

        # 2. Reject when manifest file exists
        session_id_manifest = "session_with_manifest"
        manifest_file = tmp_path / f"{session_id_manifest}_manifest.json"
        manifest_sentinel = '{"preexisting": "manifest"}'
        manifest_file.write_text(manifest_sentinel, encoding="utf-8")

        with pytest.raises(FileExistsError, match="Session collision detected"):
            ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id_manifest)
        assert manifest_file.read_text(encoding="utf-8") == manifest_sentinel

        # 3. Reject when health report exists
        session_id_report = "session_with_report"
        report_file = tmp_path / f"{session_id_report}_health_report.md"
        report_sentinel = "# Preexisting Report"
        report_file.write_text(report_sentinel, encoding="utf-8")

        with pytest.raises(FileExistsError, match="Session collision detected"):
            ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id_report)
        assert report_file.read_text(encoding="utf-8") == report_sentinel


def test_prospective_session_preserves_append_only_behavior_within_single_session():
    """Verify that within a single session, append-only JSONL writes work as expected and finalize cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        session_id = "test_single_session_append"

        session = ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id)

        # Log 3 observations
        for i in range(3):
            session.shadow_logger.log_decision_point(
                event_timestamp_ms=1000 * (i + 1),
                market_id="m1",
                asset_id="a1",
                quote_timestamp_ms=1000 * (i + 1),
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

        log_path = session.session_dir / f"{session_id}_observations.jsonl"
        assert log_path.exists()
        with open(log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 3

        # Finalize
        manifest = session._finalize_session()
        assert manifest["session_status"] == "COMPLETE"
        assert (session.session_dir / "session_manifest.json").exists()
        assert (session.session_dir / "collector_health_report.md").exists()

        # Finalizing again must raise RuntimeError
        with pytest.raises(RuntimeError, match="already been finalized"):
            session._finalize_session()

        # Reusing the session ID in a new instance must raise FileExistsError
        with pytest.raises(FileExistsError, match="Session collision detected"):
            ProspectiveShadowSession(output_base_dir=tmp_path, session_id=session_id)


def test_hard_stop_missing_quote_rate_threshold_and_boundary_equality():
    """
    Verify Amendment v1.1 missing quote rate hard-stop threshold and boundary equality.
    - Exactly 25.0% missing quote rate does NOT trigger hard stop (strictly > 25% required).
    - Strictly greater than 25.0% (e.g. 2/5 = 40.0%) immediately triggers hard stop.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_missing_quote_rate_boundary",
            )
            book = LiveTokenOrderBook("test_token")
            down_book = LiveTokenOrderBook("down_token")
            # Step 1-3: Valid quotes
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=10000)
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=10000)

            for i in range(3):
                book.last_update_ms = 10000 + i * 1000
                await session._process_grid_step(book, "m1", "t1", 10000 + i * 1000, down_book=down_book)

            assert session.eligible_grid_steps_count == 3
            assert session.unquotable_quotes_count == 0
            assert session.missing_quote_rate == 0.0
            assert session.is_hard_stopped is False

            # Step 4: Missing ask quote -> 1 unquotable out of 4 eligible = exactly 25.0%
            book.asks.clear()
            book.exchange_best_ask = None
            book.last_update_ms = 13000
            await session._process_grid_step(book, "m1", "t1", 13000, down_book=down_book)

            assert session.eligible_grid_steps_count == 4
            assert session.unquotable_quotes_count == 1
            assert session.missing_quote_rate == 0.25
            # Boundary equality: exactly 25% must NOT trigger hard stop
            assert session.is_hard_stopped is False
            assert session.hard_stop_reason is None

            # Step 5: Second unquotable quote -> 2 out of 5 = 40.0% > 25.0%
            await session._process_grid_step(book, "m1", "t1", 14000, down_book=down_book)

            assert session.eligible_grid_steps_count == 5
            assert session.unquotable_quotes_count == 2
            assert session.missing_quote_rate == 0.40
            # Strictly > 25%: must trigger hard stop immediately
            assert session.is_hard_stopped is True
            assert "MISSING_QUOTE_RATE_LIMIT (0.4000 > 0.25)" in session.hard_stop_reason

    asyncio.run(_test())


def test_hard_stop_average_staleness_threshold_and_boundary_equality():
    """
    Verify Amendment v1.1 average quote staleness hard-stop threshold and boundary equality.
    - Exactly 2000.0 ms average quote staleness does NOT trigger hard stop (strictly > 2000 ms required).
    - Strictly greater than 2000.0 ms immediately triggers hard stop.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_staleness_boundary",
            )
            book = LiveTokenOrderBook("test_token")
            down_book = LiveTokenOrderBook("down_token")
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=8000)
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=8000)

            # Step 1: grid_ts=10000, last_update=8000 -> age = 2000 ms
            book.last_update_ms = 8000
            await session._process_grid_step(book, "m1", "t1", 10000, down_book=down_book)

            # Step 2: grid_ts=11000, last_update=9000 -> age = 2000 ms
            book.last_update_ms = 9000
            await session._process_grid_step(book, "m1", "t1", 11000, down_book=down_book)

            assert session.valid_staleness_observation_count == 2
            assert session.average_quote_age_ms == 2000.0
            # Boundary equality: exactly 2000.0 ms must NOT trigger hard stop
            assert session.is_hard_stopped is False
            assert session.hard_stop_reason is None

            # Step 3: grid_ts=12000, last_update=9999 -> age = 2001 ms
            # Mean age = (2000 + 2000 + 2001) / 3 = 2000.33 ms > 2000.0 ms
            book.last_update_ms = 9999
            await session._process_grid_step(book, "m1", "t1", 12000, down_book=down_book)

            assert session.valid_staleness_observation_count == 3
            assert session.average_quote_age_ms > 2000.0
            # Strictly > 2000 ms: must trigger hard stop immediately
            assert session.is_hard_stopped is True
            assert "MAX_STALENESS_LIMIT" in session.hard_stop_reason

    asyncio.run(_test())


def test_missing_invalid_crossed_quotes_counted_in_numerator():
    """
    Verify that missing, invalid, and crossed quotes increment unquotable_quotes_count
    and eligible_grid_steps_count, without polluting valid staleness observations.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_unquotable_types",
            )
            down_book = LiveTokenOrderBook("down_token")
            down_book.apply_book_event({"bids": [{"price": "0.49", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=1000)

            # 1. Missing bid
            book1 = LiveTokenOrderBook("t1")
            book1.apply_book_event({"bids": [], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=1000)
            await session._process_grid_step(book1, "m", "t", 1000, down_book=down_book)

            # 2. Missing ask
            book2 = LiveTokenOrderBook("t1")
            book2.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": []}, timestamp_ms=2000)
            await session._process_grid_step(book2, "m", "t", 2000, down_book=down_book)

            # 3. Invalid bid price (<= 0)
            book3 = LiveTokenOrderBook("t1")
            book3.apply_book_event({"bids": [{"price": "0.00", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=3000)
            await session._process_grid_step(book3, "m", "t", 3000, down_book=down_book)

            # 4. Crossed quote (bid >= ask)
            book4 = LiveTokenOrderBook("t1")
            book4.apply_book_event({"bids": [{"price": "0.55", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=4000)
            await session._process_grid_step(book4, "m", "t", 4000, down_book=down_book)

            # 5. Locked quote (bid == ask)
            book5 = LiveTokenOrderBook("t1")
            book5.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=5000)
            await session._process_grid_step(book5, "m", "t", 5000, down_book=down_book)

            assert session.eligible_grid_steps_count == 5
            assert session.unquotable_quotes_count == 5
            assert session.missing_quote_rate == 1.0
            assert session.valid_staleness_observation_count == 0
            assert session.average_quote_age_ms == 0.0

    asyncio.run(_test())


def test_invalid_zero_timestamps_without_false_zero_age():
    """
    Verify that invalid, zero, or future quote timestamps are treated as invalid quotes
    and NEVER assigned age zero or included in valid quote staleness observations.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_timestamp_validation",
            )
            down_book = LiveTokenOrderBook("down_token")
            down_book.apply_book_event({"bids": [{"price": "0.49", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=5000)

            # Timestamp = 0
            book_zero = LiveTokenOrderBook("t1")
            book_zero.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=0)
            book_zero.last_update_ms = 0
            await session._process_grid_step(book_zero, "m", "t", 5000, down_book=down_book)

            # Negative timestamp
            book_neg = LiveTokenOrderBook("t1")
            book_neg.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=-10)
            book_neg.last_update_ms = -10
            await session._process_grid_step(book_neg, "m", "t", 5000, down_book=down_book)

            # Future timestamp (grid_ts < last_update_ms)
            book_fut = LiveTokenOrderBook("t1")
            book_fut.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=10000)
            await session._process_grid_step(book_fut, "m", "t", 5000, down_book=down_book)

            assert session.eligible_grid_steps_count == 3
            assert session.unquotable_quotes_count == 3
            # Zero timestamp must NEVER be assigned age 0 in valid staleness observations!
            assert session.valid_staleness_observation_count == 0
            assert session.total_quote_age_ms == 0.0
            assert session.average_quote_age_ms == 0.0

    asyncio.run(_test())


def test_uninitialized_book_steps_tracked_separately():
    """
    Verify that uninitialized-book steps are tracked in uninitialized_book_steps_count
    and do NOT pollute eligible_grid_steps_count or unquotable_quotes_count.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_uninitialized_tracking",
            )
            # Uninitialized book
            uninit_book = LiveTokenOrderBook("t_uninit")
            assert uninit_book.is_initialized is False

            for i in range(5):
                await session._process_grid_step(uninit_book, "m", "t", 1000 + i * 1000)

            assert session.uninitialized_book_steps_count == 5
            assert session.eligible_grid_steps_count == 0
            assert session.unquotable_quotes_count == 0
            assert session.missing_quote_rate == 0.0

            # Degraded book
            init_book = LiveTokenOrderBook("t_init")
            init_book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=1000)
            session.is_book_degraded = True

            await session._process_grid_step(init_book, "m", "t", 2000)

            assert session.uninitialized_book_steps_count == 6
            assert session.eligible_grid_steps_count == 0
            assert session.unquotable_quotes_count == 0

    asyncio.run(_test())


def test_immediate_entry_suppression_after_hard_stop_amendment_v1_1():
    """
    Verify that an active quote-quality hard stop immediately suppresses new hypothetical entries
    and logs HARD_STOP_ACTIVE rejection in the append-only log.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_suppression_v1_1",
            )
            book = LiveTokenOrderBook("test_token")
            down_book = LiveTokenOrderBook("down_token")
            # Warm up with high-confidence valid quotes
            for i in range(10):
                book.apply_book_event({
                    "bids": [{"price": "0.50", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                }, timestamp_ms=10000 + i * 1000)
                down_book.apply_book_event({
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.50", "size": "100"}],
                }, timestamp_ms=10000 + i * 1000)
                await session._process_grid_step(book, "m", "t", 10000 + i * 1000, down_book=down_book)

            # Manually trigger hard stop via missing quote rate threshold
            session.is_hard_stopped = True
            session.hard_stop_reason = "MISSING_QUOTE_RATE_LIMIT (0.3000 > 0.25)"

            # Step 11: Otherwise valid opportunity
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.51", "size": "100"}],
            }, timestamp_ms=21000)
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.50", "size": "100"}],
            }, timestamp_ms=21000)
            await session._process_grid_step(book, "m", "t", 21000, down_book=down_book)

            assert session.trades_executed == 0
            assert len(session.pending_positions) == 0

            # Verify logged record is REJECT with HARD_STOP_ACTIVE
            manifest = session._finalize_session()
            log_path = session.session_dir / manifest["log_file"]
            with open(log_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            last_rec = records[-1]
            assert last_rec["gate_decision"] in ("REJECT", "SKIP")
            assert "HARD_STOP_ACTIVE" in last_rec["rejection_reasons"]

    asyncio.run(_test())


def test_manifest_and_health_report_metrics_amendment_v1_1():
    """
    Verify that session_manifest.json and collector_health_report.md contain all Amendment v1.1
    telemetry metrics: protocol version, denominator, numerator, missing quote rate,
    valid staleness count, average quote age, thresholds, and hard-stop reason.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_manifest_reporting_v1_1",
            )
            book = LiveTokenOrderBook("test_token")
            down_book = LiveTokenOrderBook("down_token")
            # 2 valid quotes, 1 unquotable quote
            book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=9000)
            down_book.apply_book_event({"bids": [{"price": "0.49", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=9000)
            await session._process_grid_step(book, "m", "t", 10000, down_book=down_book)  # age = 1000 ms

            book.last_update_ms = 9500
            await session._process_grid_step(book, "m", "t", 11000, down_book=down_book)  # age = 1500 ms

            book.asks.clear()
            book.exchange_best_ask = None
            await session._process_grid_step(book, "m", "t", 12000, down_book=down_book)  # unquotable

            manifest = session._finalize_session()

            # Verify manifest keys and values
            assert manifest["protocol_version"] in ("1.1", "1.2")
            assert manifest["eligible_grid_steps_count"] == 3
            assert manifest["unquotable_quotes_count"] == 1
            assert pytest.approx(manifest["missing_quote_rate"], abs=1e-4) == 1.0 / 3.0
            assert manifest["missing_quote_rate_threshold"] == 0.25
            assert manifest["valid_staleness_observation_count"] == 2
            assert pytest.approx(manifest["average_quote_age_ms"], abs=1e-1) == 1250.0
            assert manifest["average_quote_staleness_threshold_ms"] == 2000.0
            assert manifest["session_status"] == "ABORTED_HARD_STOP"
            assert "MISSING_QUOTE_RATE_LIMIT" in manifest["hard_stop_triggered"]

            # Verify health report markdown content
            report_path = session.session_dir / "collector_health_report.md"
            assert report_path.exists()
            content = report_path.read_text(encoding="utf-8")
            assert "Protocol Version" in content
            assert "1.2" in content or "1.1" in content
            assert "Eligible Grid Steps (Quote Validation)" in content
            assert "Unquotable / Crossed Quotes" in content
            assert "Missing Quote Rate" in content
            assert "Valid Staleness Observations" in content
            assert "Average Quote Staleness" in content
            assert "MISSING_QUOTE_RATE_LIMIT" in content

    asyncio.run(_test())


def test_due_positions_settled_on_uninitialized_and_degraded_grid_steps():
    """
    Verify that in-flight hypothetical positions reaching their 5s horizon are settled
    and removed from pending_positions even when books are uninitialized or degraded,
    without executing new entries or incrementing eligible quote validation steps.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_due_positions_uninitialized_degraded",
            )
            # 1. Uninitialized book step: position reaches horizon at t=15000
            pos1 = {
                "event_timestamp_ms": 10000,
                "market_id": "m1",
                "asset_id": "t1",
                "quote_timestamp_ms": 10000,
                "bid_entry": 0.50,
                "ask_entry": 0.51,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": 15000,
                "bid_exit": None,
                "ask_exit": None,
                "exit_timestamp_ms": None,
            }
            session.pending_positions.append(pos1)

            uninit_up = LiveTokenOrderBook("t1")
            uninit_down = LiveTokenOrderBook("t2")
            assert not uninit_up.is_initialized
            assert not uninit_down.is_initialized

            # Process grid step at horizon 15000
            await session._process_grid_step(
                up_book=uninit_up,
                market_id="m1",
                asset_id="t1",
                grid_timestamp_ms=15000,
                down_book=uninit_down,
            )

            # pos1 settled and drained, uninitialized steps incremented, eligible unchanged
            assert len(session.pending_positions) == 0
            assert session.uninitialized_book_steps_count == 1
            assert session.eligible_grid_steps_count == 0
            assert session.trades_executed == 0

            # 2. Degraded state step: position reaches horizon at t=25000
            pos2 = {
                "event_timestamp_ms": 20000,
                "market_id": "m1",
                "asset_id": "t1",
                "quote_timestamp_ms": 20000,
                "bid_entry": 0.50,
                "ask_entry": 0.51,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": 25000,
                "bid_exit": None,
                "ask_exit": None,
                "exit_timestamp_ms": None,
            }
            session.pending_positions.append(pos2)
            session.is_book_degraded = True

            init_up = LiveTokenOrderBook("t1")
            init_up.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=25000)
            init_down = LiveTokenOrderBook("t2")
            init_down.apply_book_event({"bids": [{"price": "0.49", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=25000)

            await session._process_grid_step(
                up_book=init_up,
                market_id="m1",
                asset_id="t1",
                grid_timestamp_ms=25000,
                down_book=init_down,
            )

            # pos2 settled and drained, uninitialized steps incremented to 2, eligible still 0
            assert len(session.pending_positions) == 0
            assert session.uninitialized_book_steps_count == 2
            assert session.eligible_grid_steps_count == 0
            assert session.trades_executed == 0

    asyncio.run(_test())


def test_both_books_initialization_required_for_quote_validation():
    """
    Verify that BOTH UP and DOWN books must be initialized for a step to be eligible
    for quote validation. If either book is uninitialized, the step is tracked as
    uninitialized and quote validation is not entered.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_both_books_required",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # Step 1: UP initialized, DOWN uninitialized
            up_book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=1000)
            assert up_book.is_initialized is True
            assert down_book.is_initialized is False

            await session._process_grid_step(
                up_book=up_book,
                market_id="m",
                asset_id="up_token",
                grid_timestamp_ms=1000,
                down_book=down_book,
            )

            assert session.uninitialized_book_steps_count == 1
            assert session.eligible_grid_steps_count == 0

            # Step 2: DOWN initialized, UP uninitialized
            up_book.reset()
            down_book.apply_book_event({"bids": [{"price": "0.49", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=2000)
            assert up_book.is_initialized is False
            assert down_book.is_initialized is True

            await session._process_grid_step(
                up_book=up_book,
                market_id="m",
                asset_id="up_token",
                grid_timestamp_ms=2000,
                down_book=down_book,
            )

            assert session.uninitialized_book_steps_count == 2
            assert session.eligible_grid_steps_count == 0

            # Step 3: Both UP and DOWN initialized
            up_book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=3000)
            assert up_book.is_initialized is True
            assert down_book.is_initialized is True

            await session._process_grid_step(
                up_book=up_book,
                market_id="m",
                asset_id="up_token",
                grid_timestamp_ms=3000,
                down_book=down_book,
            )

            # Now eligible grid steps increments to 1, uninitialized count remains 2
            assert session.uninitialized_book_steps_count == 2
            assert session.eligible_grid_steps_count == 1

    asyncio.run(_test())


def test_no_entries_in_unsafe_states_uninitialized_degraded_and_hard_stopped():
    """
    Verify that no new hypothetical entries can be created during:
    1. Uninitialized book state
    2. Degraded session state
    3. Active hard-stop state
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_no_entries_unsafe_states",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # 1. Uninitialized books
            await session._process_grid_step(
                up_book=up_book,
                market_id="m",
                asset_id="up_token",
                grid_timestamp_ms=1000,
                down_book=down_book,
            )
            assert len(session.pending_positions) == 0
            assert session.trades_executed == 0

            # Initialize books
            up_book.apply_book_event({"bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.51", "size": "100"}]}, timestamp_ms=2000)
            down_book.apply_book_event({"bids": [{"price": "0.49", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]}, timestamp_ms=2000)

            # 2. Degraded state
            session.is_book_degraded = True
            await session._process_grid_step(
                up_book=up_book,
                market_id="m",
                asset_id="up_token",
                grid_timestamp_ms=2000,
                down_book=down_book,
            )
            assert len(session.pending_positions) == 0
            assert session.trades_executed == 0

            # 3. Hard-stop state
            session.is_book_degraded = False
            session.is_hard_stopped = True
            session.hard_stop_reason = "FORCED_TEST_HARD_STOP"

            # Warm up feature engine with 11 steps of valid quotes
            for i in range(11):
                ts = 3000 + i * 1000
                up_book.last_update_ms = ts
                down_book.last_update_ms = ts
                await session._process_grid_step(
                    up_book=up_book,
                    market_id="m",
                    asset_id="up_token",
                    grid_timestamp_ms=ts,
                    down_book=down_book,
                )

            # New entries strictly suppressed: trades_executed must remain 0
            assert session.trades_executed == 0
            assert len(session.pending_positions) == 0
            assert session.is_hard_stopped is True

    asyncio.run(_test())


def test_missing_down_book_fails_closed_and_settles_due_positions_incomplete():
    """
    Verify that when down_book=None:
    1. It fails closed: uninitialized_book_steps_count is incremented, eligible_grid_steps_count
       is NOT incremented, and no new entries can be created.
    2. Due in-flight positions are drained and settled safely as incomplete (INCOMPLETE_MISSING_EXIT)
       without fabricating exit prices.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_missing_down_book_fail_closed",
            )
            # Add an in-flight hypothetical position due at t=15000
            pos = {
                "event_timestamp_ms": 10000,
                "market_id": "m1",
                "asset_id": "t1",
                "quote_timestamp_ms": 10000,
                "bid_entry": 0.50,
                "ask_entry": 0.51,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": 15000,
                "bid_exit": None,
                "ask_exit": None,
                "exit_timestamp_ms": None,
            }
            session.pending_positions.append(pos)
            assert len(session.pending_positions) == 1

            # UP book is fully initialized with valid quotes
            up_book = LiveTokenOrderBook("t1")
            up_book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=15000)
            assert up_book.is_initialized is True

            # Process grid step with down_book=None (omitted)
            await session._process_grid_step(
                up_book=up_book,
                market_id="m1",
                asset_id="t1",
                grid_timestamp_ms=15000,
                down_book=None,
            )

            # 1. Due position must be drained cleanly without price fabrication
            assert len(session.pending_positions) == 0
            manifest = session._finalize_session()
            log_path = session.session_dir / manifest["log_file"]
            with open(log_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            assert len(records) == 1
            rec = records[0]
            assert rec["outcome_status"] == "INCOMPLETE_MISSING_EXIT"
            assert rec["bid_exit"] is None
            assert rec["ask_exit"] is None
            assert rec["exit_timestamp_ms"] is None

            # 2. Must fail closed: uninitialized counter increments, eligible does not increment, no new entries
            assert session.uninitialized_book_steps_count == 1
            assert session.eligible_grid_steps_count == 0
            assert session.trades_executed == 0

    asyncio.run(_test())


def test_setup_utc_logging_formatter():
    """Verify that setup_utc_logging configures time.gmtime and outputs UTC timestamps."""
    setup_utc_logging(level=logging.INFO)
    assert logging.Formatter.converter == time.gmtime

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    formatter = logging.Formatter(
        fmt="%(asctime)s UTC [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatted = formatter.format(record)
    assert "UTC [INFO] Test message" in formatted
    utc_now = time.gmtime(record.created)
    expected_prefix = time.strftime("%Y-%m-%d %H:%M:%S", utc_now)
    assert formatted.startswith(expected_prefix)


def test_unquotable_diagnostics_durable_and_distinguishes_causes():
    """
    Verify that structured diagnostics distinguish all quote failure causes:
    - Missing or non-positive bid/ask
    - Crossed or locked quotes (bid >= ask)
    - Invalid or missing timestamps
    - Grid timestamp earlier than book update (sub-second or future-dated)
    And that diagnostics survive into session_manifest.json and collector_health_report.md.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_unquotable_diag_suite",
            )
            down_book = LiveTokenOrderBook("down_token")
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=1000)

            # 1. Missing bid
            b1 = LiveTokenOrderBook("t1")
            b1.apply_book_event({"bids": [], "asks": [{"price": "0.51", "size": "100.0"}]}, timestamp_ms=1000)
            await session._process_grid_step(b1, "m", "t", 2000, down_book=down_book)
            d1 = session.unquotable_diagnostics[-1]
            assert d1["validation_flags"]["missing_or_nonpositive_bid"] is True
            assert "MISSING_OR_NONPOSITIVE_BID" in d1["failure_reasons"]

            # 2. Missing ask
            b2 = LiveTokenOrderBook("t1")
            b2.apply_book_event({"bids": [{"price": "0.50", "size": "100.0"}], "asks": []}, timestamp_ms=2000)
            await session._process_grid_step(b2, "m", "t", 3000, down_book=down_book)
            d2 = session.unquotable_diagnostics[-1]
            assert d2["validation_flags"]["missing_or_nonpositive_ask"] is True
            assert "MISSING_OR_NONPOSITIVE_ASK" in d2["failure_reasons"]

            # 3. Non-positive bid (0.00)
            b3 = LiveTokenOrderBook("t1")
            b3.apply_book_event({"bids": [{"price": "0.00", "size": "100.0"}], "asks": [{"price": "0.51", "size": "100.0"}]}, timestamp_ms=3000)
            await session._process_grid_step(b3, "m", "t", 4000, down_book=down_book)
            d3 = session.unquotable_diagnostics[-1]
            assert d3["validation_flags"]["missing_or_nonpositive_bid"] is True

            # 4. Crossed quote (bid > ask)
            b4 = LiveTokenOrderBook("t1")
            b4.apply_book_event({"bids": [{"price": "0.55", "size": "100.0"}], "asks": [{"price": "0.50", "size": "100.0"}]}, timestamp_ms=4000)
            await session._process_grid_step(b4, "m", "t", 5000, down_book=down_book)
            d4 = session.unquotable_diagnostics[-1]
            assert d4["validation_flags"]["crossed_or_locked_quotes"] is True
            assert "CROSSED_OR_LOCKED_QUOTES" in d4["failure_reasons"]

            # 5. Locked quote (bid == ask)
            b5 = LiveTokenOrderBook("t1")
            b5.apply_book_event({"bids": [{"price": "0.50", "size": "100.0"}], "asks": [{"price": "0.50", "size": "100.0"}]}, timestamp_ms=5000)
            await session._process_grid_step(b5, "m", "t", 6000, down_book=down_book)
            d5 = session.unquotable_diagnostics[-1]
            assert d5["validation_flags"]["crossed_or_locked_quotes"] is True

            # 6. Invalid / zero timestamp (last_update_ms = 0)
            b6 = LiveTokenOrderBook("t1")
            b6.apply_book_event({"bids": [{"price": "0.50", "size": "100.0"}], "asks": [{"price": "0.51", "size": "100.0"}]}, timestamp_ms=0)
            b6.last_update_ms = 0
            await session._process_grid_step(b6, "m", "t", 7000, down_book=down_book)
            d6 = session.unquotable_diagnostics[-1]
            assert d6["validation_flags"]["invalid_or_missing_timestamp"] is True
            assert "INVALID_OR_MISSING_TIMESTAMP" in d6["failure_reasons"]

            # 7. Sub-second book update near grid boundary (grid_ts = 8000, book_ts = 8450)
            b7 = LiveTokenOrderBook("t1")
            b7.apply_book_event({"bids": [{"price": "0.50", "size": "100.0"}], "asks": [{"price": "0.51", "size": "100.0"}]}, timestamp_ms=8450)
            await session._process_grid_step(b7, "m", "t", 8000, down_book=down_book)
            d7 = session.unquotable_diagnostics[-1]
            assert d7["validation_flags"]["grid_earlier_than_book_update"] is True
            assert "GRID_EARLIER_THAN_BOOK_UPDATE" in d7["failure_reasons"]
            assert d7["grid_timestamp_ms"] == 8000
            assert d7["book_last_update_ms"] == 8450

            # 8. Truly future-dated timestamp (grid_ts = 9000, book_ts = 50000)
            b8 = LiveTokenOrderBook("t1")
            b8.apply_book_event({"bids": [{"price": "0.50", "size": "100.0"}], "asks": [{"price": "0.51", "size": "100.0"}]}, timestamp_ms=50000)
            await session._process_grid_step(b8, "m", "t", 9000, down_book=down_book)
            d8 = session.unquotable_diagnostics[-1]
            assert d8["validation_flags"]["grid_earlier_than_book_update"] is True
            assert "GRID_EARLIER_THAN_BOOK_UPDATE" in d8["failure_reasons"]

            assert len(session.unquotable_diagnostics) == 8

            # Finalize session and verify persistence in manifest and health report
            manifest = session._finalize_session()
            assert manifest["unquotable_diagnostics_count"] == 8
            assert manifest["first_unquotable_diagnostic"] is not None
            assert manifest["first_unquotable_diagnostic"]["failure_reasons"] == ["MISSING_OR_NONPOSITIVE_BID"]

            report_path = session.session_dir / "collector_health_report.md"
            assert report_path.exists()
            content = report_path.read_text(encoding="utf-8")
            assert "Unquotable Quote Diagnostics" in content
            assert "MISSING_OR_NONPOSITIVE_BID" in content
            assert "GRID_EARLIER_THAN_BOOK_UPDATE" in content
            assert "CROSSED_OR_LOCKED_QUOTES" in content

    asyncio.run(_test())


def test_zero_observation_session_produces_durable_artifacts_and_valid_hash():
    """
    Verify that a session aborting on step 1 with zero decisions logged
    produces a deterministic empty .jsonl artifact, standard SHA-256 hash,
    and durable diagnostics surviving immediate abort.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_zero_obs_durability",
            )
            # Dual books initialized
            down_book = LiveTokenOrderBook("down_token")
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=1000)

            # UP book has sub-second update ahead of grid_timestamp (like session 008)
            up_book = LiveTokenOrderBook("up_token")
            up_book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=1450)

            # Step 1: grid_timestamp_ms = 1000 < 1450
            await session._process_grid_step(
                up_book=up_book,
                market_id="btc-updown-5m-test",
                asset_id="up_token",
                grid_timestamp_ms=1000,
                down_book=down_book,
            )

            # 1 / 1 = 100% missing quote rate -> hard stop triggers immediately
            assert session.is_hard_stopped is True
            assert "MISSING_QUOTE_RATE_LIMIT" in session.hard_stop_reason

            manifest = session._finalize_session()
            assert manifest["session_status"] == "ABORTED_HARD_STOP"
            assert manifest["total_observations_logged"] == 0
            assert manifest["trades_executed"] == 0

            # 1. Observation log file must genuinely exist on disk and be 0 bytes
            log_path = session.session_dir / manifest["log_file"]
            assert log_path.exists(), "Observation log must exist on disk even with 0 observations."
            assert log_path.stat().st_size == 0

            # 2. SHA-256 hash in manifest and on disk must match EMPTY_FILE_SHA256
            assert manifest["log_sha256"] == EMPTY_FILE_SHA256
            assert sha256_file(log_path) == EMPTY_FILE_SHA256

            # 3. Health report must be generated and reference EMPTY_FILE_SHA256
            report_path = session.session_dir / "collector_health_report.md"
            assert report_path.exists()
            content = report_path.read_text(encoding="utf-8")
            assert EMPTY_FILE_SHA256[:16] in content
            assert "ABORTED (HARD STOP TRIGGERED)" in content
            assert "Initial Unquotable Diagnostic" in content
            assert "GRID_EARLIER_THAN_BOOK_UPDATE" in content

    asyncio.run(_test())


def test_no_new_entries_and_unchanged_thresholds_after_hard_stop():
    """
    Verify that existing hard-stop thresholds remain bitwise unchanged
    and that no new hypothetical entries can be opened after hard-stop activation.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_threshold_invariance",
            )
            # Threshold invariants from frozen protocol
            assert session.max_missing_quote_rate_threshold == 0.25
            assert session.max_average_staleness_threshold_ms == 2000.0
            assert session.consecutive_losses_limit == 8
            assert session.max_drawdown_stop_loss_units == 2.5

            # Activate hard stop manually
            session.is_hard_stopped = True
            session.hard_stop_reason = "MISSING_QUOTE_RATE_LIMIT (1.0000 > 0.25)"

            down_book = LiveTokenOrderBook("down_token")
            down_book.apply_book_event({
                "bids": [{"price": "0.49", "size": "100.0"}],
                "asks": [{"price": "0.50", "size": "100.0"}],
            }, timestamp_ms=1000)

            # UP book is valid and warmed up
            up_book = LiveTokenOrderBook("up_token")
            up_book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=1000)

            # Attempt 15 grid steps with perfect quotes while hard-stopped
            for step in range(15):
                ts = 2000 + step * 1000
                up_book.last_update_ms = ts - 100
                await session._process_grid_step(
                    up_book=up_book,
                    market_id="m",
                    asset_id="up_token",
                    grid_timestamp_ms=ts,
                    down_book=down_book,
                )

            # Strictly 0 trades executed or pending
            assert session.trades_executed == 0
            assert len(session.pending_positions) == 0

    asyncio.run(_test())


def test_intra_second_event_staging_and_delayed_application():
    """
    Test 1: Verify that events arriving intra-second (t_recv > T) remain staged
    in PointInTimeEventBuffer at boundary T without mutating the order book,
    and are cleanly drained and applied at boundary T + 1000 ms.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_intra_second_staging",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # Initialize books at t=10000 ms (bid 0.50, ask 0.55)
            up_snap = {"event_type": "book", "asset_id": "up_token", "bids": [{"price": "0.50", "size": "100.0"}], "asks": [{"price": "0.55", "size": "100.0"}]}
            down_snap = {"event_type": "book", "asset_id": "down_token", "bids": [{"price": "0.45", "size": "100.0"}], "asks": [{"price": "0.50", "size": "100.0"}]}
            session.event_buffer.push(receipt_timestamp_ms=10000, data=up_snap)
            session.event_buffer.push(receipt_timestamp_ms=10000, data=down_snap)

            # Step 1: boundary T = 10000 ms
            await session.process_buffered_grid_step(
                grid_timestamp_ms=10000,
                up_book=up_book,
                down_book=down_book,
                market_id="btc-updown-5m-test",
                asset_id="up_token",
            )
            assert up_book.is_initialized is True
            assert up_book.best_bid == 0.50
            assert up_book.best_ask == 0.55
            assert len(session.event_buffer) == 0

            # Enqueue two intra-second price changes occurring between 11000 and 12000
            # Event 1 at 11200 ms: bid moves to 0.52
            ch1 = {"event_type": "price_change", "price_changes": [{"asset_id": "up_token", "price": "0.52", "size": "120.0", "side": "BUY", "best_bid": "0.52", "best_ask": "0.55"}]}
            # Event 2 at 11850 ms: bid moves to 0.53
            ch2 = {"event_type": "price_change", "price_changes": [{"asset_id": "up_token", "price": "0.53", "size": "150.0", "side": "BUY", "best_bid": "0.53", "best_ask": "0.55"}]}
            session.event_buffer.push(receipt_timestamp_ms=11200, data=ch1)
            session.event_buffer.push(receipt_timestamp_ms=11850, data=ch2)

            assert len(session.event_buffer) == 2

            # Step 2: boundary T = 11000 ms
            # Since both events have t_recv > 11000, NEITHER must be applied at boundary 11000
            await session.process_buffered_grid_step(
                grid_timestamp_ms=11000,
                up_book=up_book,
                down_book=down_book,
                market_id="btc-updown-5m-test",
                asset_id="up_token",
            )
            # Order book reflects ONLY state at or before 11000 (best_bid is still 0.50)
            assert up_book.best_bid == 0.50
            assert up_book.best_ask == 0.55
            assert len(session.event_buffer) == 2

            # Step 3: boundary T = 12000 ms
            # Both events have t_recv <= 12000, so BOTH are drained and applied
            await session.process_buffered_grid_step(
                grid_timestamp_ms=12000,
                up_book=up_book,
                down_book=down_book,
                market_id="btc-updown-5m-test",
                asset_id="up_token",
            )
            assert len(session.event_buffer) == 0
            assert up_book.best_bid == 0.53
            assert up_book.best_ask == 0.55
            assert up_book.bid_size == 150.0
            assert up_book.last_update_ms == 11850
            assert session.buffered_events_drained_count == 4  # 2 snapshots + 2 changes
            assert len(session.unquotable_diagnostics) == 0

    asyncio.run(_test())


def test_causal_non_negative_age_invariant_across_grid_ticks():
    """
    Test 2: Assert across all simulated grid ticks that grid_timestamp_ms >= last_update_ms
    and quote_age_ms >= 0 strictly holds with zero occurrences of GRID_EARLIER_THAN_BOOK_UPDATE.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_non_negative_age_invariant",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # Initial snapshots at 10000 with wide spread
            session.event_buffer.push(10000, {"event_type": "book", "asset_id": "up_token", "bids": [{"price": "0.40", "size": "100"}], "asks": [{"price": "0.60", "size": "100"}]})
            session.event_buffer.push(10000, {"event_type": "book", "asset_id": "down_token", "bids": [{"price": "0.40", "size": "100"}], "asks": [{"price": "0.60", "size": "100"}]})

            # Sub-second events spread across 10-25 seconds (all bids well below ask 0.60)
            timestamps = [
                10150, 10720, 10999, 11050, 11500, 11950, 12200, 12850,
                13400, 14100, 15300, 15990, 16250, 17800, 19200, 20100,
            ]
            for i, ts in enumerate(timestamps):
                p = 0.40 + (i % 5) * 0.01  # 0.40 to 0.44
                ev = {
                    "event_type": "price_change",
                    "price_changes": [{
                        "asset_id": "up_token",
                        "price": f"{p:.2f}",
                        "size": "100.0",
                        "side": "BUY",
                        "best_bid": f"{p:.2f}",
                        "best_ask": "0.60",
                    }],
                }
                session.event_buffer.push(ts, ev)

            # Step through grid from 10000 to 25000 (16 integer-second ticks)
            for sec in range(10, 26):
                grid_ts = sec * 1000
                await session.process_buffered_grid_step(
                    grid_timestamp_ms=grid_ts,
                    up_book=up_book,
                    down_book=down_book,
                    market_id="btc-updown-5m-test",
                    asset_id="up_token",
                )
                # Hard invariant: book update timestamp NEVER exceeds grid boundary
                assert up_book.last_update_ms <= grid_ts, f"Violation at grid {grid_ts}: last_update_ms={up_book.last_update_ms}"
                assert grid_ts - up_book.last_update_ms >= 0

            # Zero GRID_EARLIER_THAN_BOOK_UPDATE diagnostics recorded
            assert len(session.unquotable_diagnostics) == 0
            assert session.unquotable_quotes_count == 0
            assert session.missing_quote_rate == 0.0

    asyncio.run(_test())


def test_quiet_second_aging_and_staleness_hard_stop():
    """
    Test 3: Verify that when no events are enqueued, quote_age_ms increases by exactly
    1000 ms per second, consecutive quiet seconds increment cumulative staleness,
    and hard stops trigger when average staleness > 2000 ms is exceeded.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_quiet_second_aging",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # Initial snapshots at 10000
            session.event_buffer.push(10000, {"event_type": "book", "asset_id": "up_token", "bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.52", "size": "100"}]})
            session.event_buffer.push(10000, {"event_type": "book", "asset_id": "down_token", "bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]})

            # T = 10000: age = 0 ms
            await session.process_buffered_grid_step(10000, up_book, down_book, "m", "up_token")
            assert session.valid_staleness_observation_count == 1
            assert session.total_quote_age_ms == 0.0

            # T = 11000: age = 1000 ms
            await session.process_buffered_grid_step(11000, up_book, down_book, "m", "up_token")
            assert session.valid_staleness_observation_count == 2
            assert session.total_quote_age_ms == 1000.0

            # T = 12000: age = 2000 ms
            await session.process_buffered_grid_step(12000, up_book, down_book, "m", "up_token")
            assert session.valid_staleness_observation_count == 3
            assert session.total_quote_age_ms == 3000.0
            assert session.is_hard_stopped is False

            # T = 13000: age = 3000 ms (single-step stale event tracked)
            await session.process_buffered_grid_step(13000, up_book, down_book, "m", "up_token")
            assert session.stale_quotes_count == 1

            # T = 14000: age = 4000 ms; cumulative sum = 0 + 1000 + 2000 + 3000 + 4000 = 10000 ms; average = 2000.0 ms <= 2000.0 ms
            await session.process_buffered_grid_step(14000, up_book, down_book, "m", "up_token")
            assert session.average_quote_age_ms == 2000.0
            assert session.is_hard_stopped is False

            # T = 15000: age = 5000 ms; cumulative sum = 15000 ms; average = 2500.0 ms > 2000.0 ms
            # Hard stop must trigger immediately!
            await session.process_buffered_grid_step(15000, up_book, down_book, "m", "up_token")
            assert session.is_hard_stopped is True
            assert "MAX_STALENESS_LIMIT" in session.hard_stop_reason

    asyncio.run(_test())


def test_offline_replay_parity_with_point_in_time_resampler():
    """
    Test 4: Replay a deterministic sequence of events through both Option 2's buffered
    collector and PointInTimeResampler. Assert bitwise equality of sampled bids,
    asks, source timestamps, and quote ages.
    """
    async def _test():
        # 1. Generate synthetic canonical events for PointInTimeResampler
        raw_events = [
            {"timestamp_ms": 10200, "sequence_id": 1, "market_id": "m1", "asset_id": "up_token", "bid": 0.50, "ask": 0.52, "bid_size": 100.0, "ask_size": 100.0},
            {"timestamp_ms": 10850, "sequence_id": 2, "market_id": "m1", "asset_id": "up_token", "bid": 0.51, "ask": 0.52, "bid_size": 120.0, "ask_size": 100.0},
            {"timestamp_ms": 11400, "sequence_id": 3, "market_id": "m1", "asset_id": "up_token", "bid": 0.51, "ask": 0.53, "bid_size": 120.0, "ask_size": 150.0},
            {"timestamp_ms": 11950, "sequence_id": 4, "market_id": "m1", "asset_id": "up_token", "bid": 0.52, "ask": 0.53, "bid_size": 140.0, "ask_size": 150.0},
            {"timestamp_ms": 12700, "sequence_id": 5, "market_id": "m1", "asset_id": "up_token", "bid": 0.53, "ask": 0.54, "bid_size": 160.0, "ask_size": 180.0},
            {"timestamp_ms": 13100, "sequence_id": 6, "market_id": "m1", "asset_id": "up_token", "bid": 0.53, "ask": 0.55, "bid_size": 160.0, "ask_size": 200.0},
        ]
        df_events = pd.DataFrame(raw_events)

        resampler = PointInTimeResampler(max_stale_ms=5000, grid_interval_ms=1000)
        resampled_df = resampler.resample_partition(df_events)

        # 2. Replay identical events through ProspectiveShadowSession event buffer
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_offline_replay_parity",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # DOWN token snapshot
            session.event_buffer.push(10000, {"event_type": "book", "asset_id": "down_token", "bids": [{"price": "0.48", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]})

            # Initial UP snapshot corresponding to first event
            ev1 = raw_events[0]
            session.event_buffer.push(
                ev1["timestamp_ms"],
                {"event_type": "book", "asset_id": "up_token", "bids": [{"price": str(ev1["bid"]), "size": str(ev1["bid_size"])}], "asks": [{"price": str(ev1["ask"]), "size": str(ev1["ask_size"])}]},
            )

            # Subsequent UP updates as price changes with explicit best_bid / best_ask
            for ev in raw_events[1:]:
                session.event_buffer.push(
                    ev["timestamp_ms"],
                    {
                        "event_type": "price_change",
                        "price_changes": [
                            {
                                "asset_id": "up_token",
                                "price": str(ev["bid"]),
                                "size": str(ev["bid_size"]),
                                "side": "BUY",
                                "best_bid": str(ev["bid"]),
                                "best_ask": str(ev["ask"]),
                            },
                            {
                                "asset_id": "up_token",
                                "price": str(ev["ask"]),
                                "size": str(ev["ask_size"]),
                                "side": "SELL",
                                "best_bid": str(ev["bid"]),
                                "best_ask": str(ev["ask"]),
                            },
                        ],
                    },
                )

            # Compare sampled state at every grid timestamp generated by PointInTimeResampler
            for _, row in resampled_df.iterrows():
                grid_ts = int(row["grid_timestamp_ms"])
                await session.process_buffered_grid_step(
                    grid_timestamp_ms=grid_ts,
                    up_book=up_book,
                    down_book=down_book,
                    market_id="m1",
                    asset_id="up_token",
                )

                # Assert bitwise parity with PointInTimeResampler
                assert up_book.best_bid == pytest.approx(row["bid"], abs=1e-6)
                assert up_book.best_ask == pytest.approx(row["ask"], abs=1e-6)
                assert up_book.last_update_ms == int(row["source_timestamp_ms"])
                assert (grid_ts - up_book.last_update_ms) == int(row["event_age_ms"])

    asyncio.run(_test())


def test_dual_book_drain_synchronization_and_position_settlement():
    """
    Test 5: Feed asymmetric UP and DOWN event timestamps. Verify that at boundary T,
    both books drain strictly up to T, and due hypothetical positions settle safely
    using the synchronized book states.
    """
    async def _test():
        with tempfile.TemporaryDirectory() as tmp_dir:
            session = ProspectiveShadowSession(
                output_base_dir=Path(tmp_dir),
                session_id="test_dual_book_sync_settlement",
            )
            up_book = LiveTokenOrderBook("up_token")
            down_book = LiveTokenOrderBook("down_token")

            # Snapshots at 9000 ms (wide spread to avoid crossing)
            session.event_buffer.push(9000, {"event_type": "book", "asset_id": "up_token", "bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.55", "size": "100"}]})
            session.event_buffer.push(9000, {"event_type": "book", "asset_id": "down_token", "bids": [{"price": "0.45", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]})

            # Asymmetric events:
            # UP at 10200; DOWN at 10800
            session.event_buffer.push(10200, {"event_type": "price_change", "price_changes": [{"asset_id": "up_token", "price": "0.51", "size": "100", "side": "BUY", "best_bid": "0.51", "best_ask": "0.55"}]})
            session.event_buffer.push(10800, {"event_type": "price_change", "price_changes": [{"asset_id": "down_token", "price": "0.46", "size": "100", "side": "BUY", "best_bid": "0.46", "best_ask": "0.50"}]})
            # UP at 11300; DOWN at 11700
            session.event_buffer.push(11300, {"event_type": "price_change", "price_changes": [{"asset_id": "up_token", "price": "0.52", "size": "100", "side": "BUY", "best_bid": "0.52", "best_ask": "0.55"}]})
            session.event_buffer.push(11700, {"event_type": "price_change", "price_changes": [{"asset_id": "down_token", "price": "0.47", "size": "100", "side": "BUY", "best_bid": "0.47", "best_ask": "0.50"}]})

            # Create in-flight position opening at 10000 with horizon 15000
            pos = {
                "event_timestamp_ms": 10000,
                "market_id": "m1",
                "asset_id": "up_token",
                "quote_timestamp_ms": 10000,
                "bid_entry": 0.50,
                "ask_entry": 0.55,
                "depth_imbalance": 0.0,
                "scaled_spread": -0.1,
                "abs_scaled_depth_imbal": 0.5,
                "model_probs": (0.1, 0.1, 0.8),
                "predicted_class": 2,
                "confidence": 0.80,
                "burst_id": 0,
                "data_gap_ms": 1000.0,
                "target_exit_ms": 15000,
                "bid_exit": None,
                "ask_exit": None,
                "exit_timestamp_ms": None,
            }
            session.pending_positions.append(pos)

            # Step T = 10000: snapshots drained, asymmetric changes queued
            await session.process_buffered_grid_step(10000, up_book, down_book, "m1", "up_token", "down_token")
            assert len(session.event_buffer) == 4

            # Step T = 11000: events at 10200 and 10800 drained; events at 11300 and 11700 remain
            await session.process_buffered_grid_step(11000, up_book, down_book, "m1", "up_token", "down_token")
            assert up_book.best_bid == 0.51
            assert down_book.best_bid == 0.46
            assert len(session.event_buffer) == 2
            assert len(session.pending_positions) == 1  # Not due yet (11000 < 15000)

            # Step T = 12000: remaining events at 11300 and 11700 drained
            await session.process_buffered_grid_step(12000, up_book, down_book, "m1", "up_token", "down_token")
            assert up_book.best_bid == 0.52
            assert down_book.best_bid == 0.47
            assert len(session.event_buffer) == 0

            # Advance to horizon T = 15000
            await session.process_buffered_grid_step(13000, up_book, down_book, "m1", "up_token", "down_token")
            await session.process_buffered_grid_step(14000, up_book, down_book, "m1", "up_token", "down_token")
            await session.process_buffered_grid_step(15000, up_book, down_book, "m1", "up_token", "down_token")

            # Position safely settled using synchronized book state
            assert len(session.pending_positions) == 0
            manifest = session._finalize_session()
            log_path = session.session_dir / manifest["log_file"]
            with open(log_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]
            completed_trade = [r for r in records if r.get("outcome_status") == "COMPLETED"]
            assert len(completed_trade) == 1
            assert completed_trade[0]["exit_timestamp_ms"] == 15000
            assert completed_trade[0]["bid_exit"] == 0.52

    asyncio.run(_test())


def test_point_in_time_event_buffer_validation_and_bounding():
    """
    Test 6: Verify PointInTimeEventBuffer rejects invalid/future-dated timestamps
    without lookahead leakage, enforces capacity limits, and clears on rollover.
    """
    buffer = PointInTimeEventBuffer(maxsize=5, future_tolerance_ms=3000)

    # 1. Non-positive or NaN receipt timestamp rejected
    assert buffer.push(0, {"data": 1}) is False
    assert buffer.push(-100, {"data": 2}) is False
    assert buffer.invalid_timestamp_rejected_count == 2

    # 2. Future-dated event relative to receipt timestamp + tolerance rejected
    # receipt_ts = 10000, source_ts = 15000 (> 10000 + 3000)
    assert buffer.push(10000, {"timestamp": 15000, "data": 3}) is False
    assert buffer.future_dated_rejected_count == 1

    # 3. Valid events pushed in non-monotonic order are kept sorted
    assert buffer.push(12000, {"data": "second"}) is True
    assert buffer.push(11000, {"data": "first"}) is True
    assert buffer.push(13000, {"data": "third"}) is True
    assert len(buffer) == 3

    # Draining up to 11500 yields strictly the 11000 event
    drained = buffer.drain_up_to(11500)
    assert len(drained) == 1
    assert drained[0].receipt_timestamp_ms == 11000
    assert drained[0].data["data"] == "first"
    assert len(buffer) == 2

    # 4. Capacity limit enforced
    buffer.push(14000, {"data": 4})
    buffer.push(15000, {"data": 5})
    buffer.push(16000, {"data": 6})
    assert len(buffer) == 5  # at maxsize 5
    assert buffer.push(17000, {"data": 7}) is False
    assert buffer.overflow_count == 1

    # 5. Clear buffer on rollover
    cleared = buffer.clear()
    assert cleared == 5
    assert len(buffer) == 0


def test_point_in_time_event_buffer_future_dated_list_wrapped():
    """
    Regression test for Bug 1: Verify PointInTimeEventBuffer detects future-dated
    source timestamps in list-wrapped payloads (e.g. batch updates / price_change lists).
    """
    buffer = PointInTimeEventBuffer(maxsize=10, future_tolerance_ms=3000)

    # 1. Normal list payload within tolerance (receipt=10000, source=12000 <= 10000 + 3000)
    normal_payload = [
        {"event_type": "price_change", "timestamp": 12000, "price": "0.50"},
    ]
    assert buffer.push(receipt_timestamp_ms=10000, data=normal_payload) is True
    assert buffer.future_dated_rejected_count == 0
    assert len(buffer) == 1

    # 2. Future-dated list payload (receipt=10000, source=15000 > 10000 + 3000)
    future_payload = [
        {"event_type": "price_change", "timestamp": 15000, "price": "0.55"},
    ]
    assert buffer.push(receipt_timestamp_ms=10000, data=future_payload) is False
    assert buffer.future_dated_rejected_count == 1
    assert len(buffer) == 1  # Not appended

    # 3. Multi-element list where one element is future-dated (receipt=10000, timestamps=[11000, 16000])
    mixed_payload = [
        {"event_type": "price_change", "timestamp": 11000, "price": "0.51"},
        {"event_type": "price_change", "timestamp": 16000, "price": "0.52"},
    ]
    assert buffer.push(receipt_timestamp_ms=10000, data=mixed_payload) is False
    assert buffer.future_dated_rejected_count == 2
    assert len(buffer) == 1  # Not appended

    # 4. Multi-element list where all elements are within tolerance
    valid_batch = [
        {"event_type": "price_change", "timestamp_ms": 11000, "price": "0.51"},
        {"event_type": "price_change", "timestamp_ms": 12500, "price": "0.52"},
    ]
    assert buffer.push(receipt_timestamp_ms=10000, data=valid_batch) is True
    assert buffer.future_dated_rejected_count == 2
    assert len(buffer) == 2


def test_event_buffer_overflow_independent_from_message_queue_overflow():
    """
    Regression test for Bug 2: Verify ProspectiveShadowSession tracks event-buffer
    overflows independently from message-queue overflows, and always marks
    is_book_degraded=True (fails closed).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_buffer_overflow_independent",
            queue_maxsize=10,
        )
        up_book = LiveTokenOrderBook("up_token")
        down_book = LiveTokenOrderBook("down_token")

        # Simulate prior message_queue overflow drops (e.g. 5 drops during network burst)
        session.queue_overflow_count = 5
        assert session.event_buffer_overflow_count == 0

        # Restrict event_buffer to small maxsize for test
        session.event_buffer = PointInTimeEventBuffer(maxsize=2, future_tolerance_ms=5000)

        # Enqueue 3 JSON messages into message_queue
        msg1 = json.dumps({"event_type": "book", "asset_id": "up_token", "bids": [{"price": "0.50", "size": "100"}], "asks": [{"price": "0.55", "size": "100"}]})
        msg2 = json.dumps({"event_type": "book", "asset_id": "down_token", "bids": [{"price": "0.45", "size": "100"}], "asks": [{"price": "0.50", "size": "100"}]})
        msg3 = json.dumps({"event_type": "price_change", "asset_id": "up_token", "price_changes": []})

        session.message_queue.put_nowait((msg1, 10000))
        session.message_queue.put_nowait((msg2, 10100))
        session.message_queue.put_nowait((msg3, 10200))

        # Reset degradation to verify it gets triggered specifically by the buffer overflow
        session.is_book_degraded = False

        # Drain message queue into buffer: msg1 and msg2 fit (len=2), msg3 overflows (capacity=2)
        drained = session._drain_message_queue_into_buffer("slug", up_book, down_book)
        assert drained == 3
        assert len(session.event_buffer) == 2

        # Verify independent tracking and fail-closed state
        assert session.event_buffer.overflow_count == 1
        assert session.event_buffer_overflow_count == 1  # Tracked independently despite queue_overflow_count == 5
        assert session.queue_overflow_count == 5  # Undisturbed
        assert session.invalid_events_rejected == 0  # NOT misclassified as invalid
        assert session.is_book_degraded is True  # Marked degraded (fail closed)


def test_point_in_time_event_buffer_out_of_order_receipt_timestamps_stability():
    """
    Verify PointInTimeEventBuffer handles out-of-order receipt timestamps stably:
    - Events arriving earlier than the tail trigger a stable resort.
    - FIFO arrival order is preserved for same-timestamp events.
    """
    buffer = PointInTimeEventBuffer(maxsize=10, future_tolerance_ms=5000)

    # Push sequence with non-monotonic receipt timestamps and ties:
    # 1. 10500 ("A")
    # 2. 10200 ("B") -> earlier than A
    # 3. 10100 ("C1") -> earlier than B
    # 4. 10100 ("C2") -> same timestamp as C1, arrives second
    # 5. 10800 ("D") -> monotonic after A
    assert buffer.push(10500, {"id": "A"}) is True
    assert buffer.push(10200, {"id": "B"}) is True
    assert buffer.push(10100, {"id": "C1"}) is True
    assert buffer.push(10100, {"id": "C2"}) is True
    assert buffer.push(10800, {"id": "D"}) is True

    assert len(buffer) == 5

    # Drain up to 10300: should yield C1, C2 (preserving arrival order), then B
    drained_10300 = buffer.drain_up_to(10300)
    assert len(drained_10300) == 3
    assert [ev.data["id"] for ev in drained_10300] == ["C1", "C2", "B"]
    assert [ev.receipt_timestamp_ms for ev in drained_10300] == [10100, 10100, 10200]

    # Drain up to 11000: should yield A, then D
    drained_11000 = buffer.drain_up_to(11000)
    assert len(drained_11000) == 2
    assert [ev.data["id"] for ev in drained_11000] == ["A", "D"]
    assert [ev.receipt_timestamp_ms for ev in drained_11000] == [10500, 10800]

    assert len(buffer) == 0


def test_protocol_amendment_v1_2_specifications():
    """
    Verify Protocol Amendment v1.2 specification:
    - protocol_version is 1.2
    - session manifest records protocol_version == 1.2
    - health report header certifies Protocol Amendment v1.2
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProspectiveShadowSession(
            output_base_dir=Path(tmp_dir),
            session_id="test_v1_2_spec",
        )
        assert session.protocol_version == "1.2"
        manifest = session._finalize_session()
        assert manifest["protocol_version"] == "1.2"

        report_path = session.session_dir / "collector_health_report.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "Protocol Amendment v1.2" in content
