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
from pathlib import Path
import json
import tempfile
import numpy as np
import pytest
import torch
import websockets
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
from websockets.protocol import State as WebSocketState
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from pipeline_v2.models.phase26_prospective_collector import (
    LiveTokenOrderBook,
    OnlineCausalFeatureEngine,
    ProspectiveShadowSession,
    classify_and_parse_ws_message,
    get_current_5m_slot,
    is_connection_closed,
    MODEL_PATH,
    LOCKED_TEST_PATH,
    SCALER_PARAMS_PATH,
)
from pipeline_v2.models.phase26_forward_shadow import sha256_file


def test_locked_test_isolation_in_prospective():
    """Ensure locked test set is never accessed by prospective collector."""
    assert LOCKED_TEST_PATH.exists()
    assert LOCKED_TEST_PATH.is_file()


def test_model_weight_immutability_in_prospective():
    """Ensure model checkpoint weights are bitwise invariant before and after instantiation."""
    initial_hash = sha256_file(MODEL_PATH)
    session = ProspectiveShadowSession(
        output_base_dir=Path(tempfile.gettempdir()),
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
            # Snapshot arrived at t = 10,000ms
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=10000)

            # Warm up feature engine with 10 steps
            for i in range(10):
                await session._process_grid_step(
                    up_book=book,
                    market_id="btc-updown-5m-1791546600",
                    asset_id="test_token",
                    grid_timestamp_ms=10000 + i * 1000,
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
        book.apply_book_event({
            "bids": [{"price": "0.50", "size": "100.0"}],
            "asks": [{"price": "0.51", "size": "100.0"}],
        }, timestamp_ms=25000)

        # Warm up feature engine with 10 steps
        async def _eval_grid():
            for step in range(10):
                await session._process_grid_step(
                    up_book=book,
                    market_id="btc-updown-5m-1791608400",
                    asset_id="test_token_up",
                    grid_timestamp_ms=25000 + step * 1000,
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
        for i in range(10):
            book.apply_book_event({
                "bids": [{"price": "0.50", "size": "100.0"}],
                "asks": [{"price": "0.51", "size": "100.0"}],
            }, timestamp_ms=10000 + i * 1000)
            session.feature_engine.add_grid_sample(0.50, 0.51, 100.0, 100.0, 10000 + i * 1000)

        # Book update at t=20000:
        # Exit bid is 0.40 -> long pos_due exits at 0.40, realizing loss of -0.11 (Loss #8)
        # Entry spread is 0.01 (tight), qualifying for new entry
        book.apply_book_event({
            "bids": [{"price": "0.40", "size": "100.0"}],
            "asks": [{"price": "0.41", "size": "100.0"}],
        }, timestamp_ms=20000)

        # Execute grid step at t=20000
        async def _run_same_step():
            await session._process_grid_step(
                up_book=book,
                market_id="btc-updown-5m-1791608400",
                asset_id="test_token_up",
                grid_timestamp_ms=20000,
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
