"""Unit and integration tests for Phase 3 Data Expansion & Orderbook Capture.

Covers:
- Multi-session creation, directory isolation, and metadata lifecycle
- Restartable collection and duplicate prevention
- Graceful shutdown handling and programmatic shutdown request
- Network failure retry loops and error logging with exponential backoff
- Genuine Level 2 depth snapshot capture with sequence numbers and timestamps
- Level 2 incremental update sequence gap detection and resynchronization
- Invalid orderbook detection (crossed book, negative price/size)
- Session integrity validation (duplicates, timestamp ordering, gaps, malformed records, price/volume checks)
- Offline session processing, parquet creation, and raw data immutability
- Polymarket L2 adapter behavior and documented sequence limitations

All network calls are strictly mocked. No live exchange access is required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_data.orderbook_collector import (
    BinanceL2OrderBookRecorder,
    InvalidOrderBookError,
    L2DeltaUpdate,
    L2OrderBook,
    L2Snapshot,
    OrderBookSequenceGapError,
    PolymarketL2Adapter,
)
from market_data.track_a_collector import (
    TrackACollectionConfig,
    TrackACollector,
    process_session_data,
    validate_session_data,
)


class MockResponse:
    def __init__(self, payload: Any, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {"Date": "Sun, 04 Oct 2026 12:00:00 GMT"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP_{self.status_code}")

    def json(self):
        return self._payload


class MockSession:
    def __init__(self, responses: list[MockResponse]):
        self._responses = iter(responses)

    def get(self, *args, **kwargs):
        try:
            return next(self._responses)
        except StopIteration:
            raise RuntimeError("No more mocked responses available.")


def make_candle(ts_ms: int, open_p: str = "60000.0", close_p: str = "60050.0") -> list:
    return [[ts_ms, open_p, "60100.0", "59950.0", close_p, "12.5", ts_ms + 59999]]


def make_depth_payload(last_update_id: int = 1000) -> dict[str, Any]:
    return {
        "lastUpdateId": last_update_id,
        "bids": [["60000.00", "1.50"], ["59990.00", "2.25"]],
        "asks": [["60010.00", "0.80"], ["60020.00", "3.10"]],
    }


# =========================================================================
# 1. Multi-Session Management & Lifecycle
# =========================================================================

def test_multi_session_creation_and_metadata(tmp_path):
    """Verify that a session directory and session_metadata.json are correctly created and updated."""
    session_id = "session_btc_test_01"
    config = TrackACollectionConfig(
        session_id=session_id,
        symbol="BTCUSDT",
        interval_seconds=1.0,
        output_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        capture_l2_depth=False,
    )

    collector = TrackACollector(config)
    session_dir = tmp_path / "raw" / "sessions" / session_id
    assert session_dir.exists(), f"Session directory not created at {session_dir}"

    meta_file = session_dir / "session_metadata.json"
    assert meta_file.exists(), "session_metadata.json was not created"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    assert meta["session_id"] == session_id
    assert meta["symbol"] == "BTCUSDT"
    assert meta["status"] == "RUNNING"
    assert meta["collector_version"] == "3.0"


# =========================================================================
# 2. Restart / Resume & Duplicate Prevention
# =========================================================================

def test_restart_resume_duplicate_prevention(tmp_path):
    """Verify collector restarts cleanly on the same session without duplicating records."""
    session_id = "session_resume_test"
    candle1 = make_candle(100000, "60000.0", "60050.0")
    candle2 = make_candle(160000, "60050.0", "60100.0")
    candle3 = make_candle(220000, "60100.0", "60080.0")

    config = TrackACollectionConfig(
        session_id=session_id,
        output_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        max_retries=1,
    )

    # Run 1: collect candle1 and candle2
    session1 = MockSession([MockResponse(candle1), MockResponse(candle2)])
    c1 = TrackACollector(config, session=session1)
    assert c1.collect_once() is True
    assert c1.collect_once() is True

    # Run 2: resume on same session, provide candle2 again (duplicate) and new candle3
    session2 = MockSession([MockResponse(candle2), MockResponse(candle3)])
    c2 = TrackACollector(config, session=session2)

    # candle2 should be skipped as duplicate
    assert c2.collect_once() is False
    # candle3 should be appended
    assert c2.collect_once() is True

    # Verify observations file has exactly 3 records
    obs_file = tmp_path / "raw" / "sessions" / session_id / "observations.jsonl"
    lines = [line for line in obs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3, f"Expected exactly 3 records, got {len(lines)}"

    timestamps = [json.loads(line)["open_time"] for line in lines]
    assert len(set(timestamps)) == 3, "Found duplicate timestamps in resumed session"


# =========================================================================
# 3. Graceful Shutdown Handling
# =========================================================================

def test_graceful_shutdown_request(tmp_path):
    """Verify that programmatic or signal shutdown terminates the collection loop cleanly."""
    session_id = "session_shutdown_test"
    config = TrackACollectionConfig(
        session_id=session_id,
        output_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        interval_seconds=0.01,
        max_retries=1,
    )

    collector = TrackACollector(config, session=MockSession([MockResponse(make_candle(1000))]))
    collector.request_shutdown()
    assert collector.is_shutdown_requested is True

    # run() must exit immediately without executing iterations
    collected = collector.run()
    assert collected == 0

    meta_file = tmp_path / "raw" / "sessions" / session_id / "session_metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "STOPPED_BY_SIGNAL"
    assert meta["end_time"] is not None


# =========================================================================
# 4. Network Failure, Reconnection & Backoff
# =========================================================================

def test_reconnection_and_http_retry_exponential_backoff(tmp_path):
    """Verify retry loop survives HTTP 500 and 429 rate limit errors before succeeding."""
    session_id = "session_retry_test"
    config = TrackACollectionConfig(
        session_id=session_id,
        output_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        max_retries=3,
        timeout_seconds=1.0,
    )

    valid_candle = make_candle(500000)
    mock_responses = [
        MockResponse({}, status_code=500),  # Attempt 1: server error
        MockResponse({}, status_code=429),  # Attempt 2: rate limited
        MockResponse(valid_candle, status_code=200),  # Attempt 3: success
    ]

    collector = TrackACollector(config, session=MockSession(mock_responses))
    assert collector.collect_once() is True

    # Verify errors were logged to collection_errors.jsonl
    error_file = tmp_path / "raw" / "sessions" / session_id / "collection_errors.jsonl"
    assert error_file.exists()
    error_lines = [json.loads(line) for line in error_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(error_lines) == 2
    assert "HTTP_500" in error_lines[0]["error"]
    assert "rate_limited_http_429" in error_lines[1]["error"]


# =========================================================================
# 5. Genuine Level 2 Depth Capture
# =========================================================================

def test_genuine_l2_depth_snapshot_recording(tmp_path):
    """Verify genuine L2 orderbook depth snapshot capture, JSONL recording, and metrics."""
    session_id = "session_l2_test"
    config = TrackACollectionConfig(
        session_id=session_id,
        output_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        capture_l2_depth=True,
        depth_limit=20,
    )

    depth_payload = make_depth_payload(last_update_id=50001)
    candle_payload = make_candle(600000)

    # 1st request is kline, 2nd request is depth
    session = MockSession([MockResponse(candle_payload), MockResponse(depth_payload)])
    collector = TrackACollector(config, session=session)

    assert collector.collect_once() is True

    # Verify depth snapshot file
    depth_file = tmp_path / "raw" / "sessions" / session_id / "orderbook_snapshots.jsonl"
    assert depth_file.exists()

    records = [json.loads(line) for line in depth_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 1
    rec = records[0]

    assert rec["symbol"] == "BTCUSDT"
    assert rec["last_update_id"] == 50001
    assert rec["bids"][0] == [60000.0, 1.5]
    assert rec["asks"][0] == [60010.0, 0.8]
    assert rec["source"] == "binance_l2_rest"

    # Verify in-memory book state
    book = collector.depth_recorder.book
    assert book.best_bid == 60000.0
    assert book.best_ask == 60010.0
    assert book.mid_price == 60005.0
    assert book.spread == 10.0
    assert book.order_book_imbalance(5) == pytest.approx((3.75 - 3.90) / (3.75 + 3.90))


# =========================================================================
# 6. Orderbook Sequence Gap Handling & Resynchronization
# =========================================================================

def test_orderbook_sequence_handling_and_gap_detection():
    """Verify L2OrderBook enforces sequence continuity and flags packet gaps."""
    book = L2OrderBook(symbol="BTCUSDT")

    # Initial snapshot with last_update_id = 100
    snap = L2Snapshot(
        symbol="BTCUSDT",
        last_update_id=100,
        bids=((50000.0, 1.0), (49990.0, 2.0)),
        asks=((50010.0, 1.0), (50020.0, 2.0)),
    )
    book.apply_snapshot(snap)
    assert book.last_update_id == 100

    # 1. Valid contiguous delta: U=101, u=105
    delta1 = L2DeltaUpdate(
        symbol="BTCUSDT",
        first_update_id=101,
        final_update_id=105,
        bids=((50005.0, 1.5),),  # new higher bid
        asks=(),
    )
    assert book.apply_delta(delta1) is True
    assert book.last_update_id == 105
    assert book.best_bid == 50005.0

    # 2. Outdated delta (in the past): U=95, u=102 -> ignored safely
    delta_past = L2DeltaUpdate(
        symbol="BTCUSDT",
        first_update_id=95,
        final_update_id=102,
        bids=(),
        asks=(),
    )
    assert book.apply_delta(delta_past) is False
    assert book.last_update_id == 105

    # 3. Sequence gap: expected U <= 106, but got U=115 -> raises OrderBookSequenceGapError
    delta_broken = L2DeltaUpdate(
        symbol="BTCUSDT",
        first_update_id=115,
        final_update_id=120,
        bids=((50006.0, 1.0),),
        asks=(),
    )
    with pytest.raises(OrderBookSequenceGapError) as exc_info:
        book.apply_delta(delta_broken)

    assert exc_info.value.expected_start == 106
    assert exc_info.value.actual_start == 115


def test_orderbook_resynchronization_on_gap(tmp_path):
    """Verify BinanceL2OrderBookRecorder catches sequence gaps and re-fetches a fresh snapshot."""
    recorder = BinanceL2OrderBookRecorder(output_dir=tmp_path, symbol="BTCUSDT")

    # Initial snapshot last_update_id = 50
    initial_snap = L2Snapshot(
        symbol="BTCUSDT",
        last_update_id=50,
        bids=((50000.0, 1.0),),
        asks=((50010.0, 1.0),),
    )
    recorder.record_snapshot(initial_snap)

    # Provide fresh snapshot response for the resynchronization call
    resync_payload = make_depth_payload(last_update_id=200)
    recorder.session = MockSession([MockResponse(resync_payload)])

    # Send a gapped delta (U=100 > 51)
    gapped_delta = L2DeltaUpdate(
        symbol="BTCUSDT",
        first_update_id=100,
        final_update_id=105,
        bids=((50005.0, 1.0),),
        asks=(),
    )
    # record_delta should catch gap, log to errors, and resync
    assert recorder.record_delta(gapped_delta) is False
    # Book must have resynchronized to the fresh snapshot lastUpdateId (200)
    assert recorder.book.last_update_id == 200

    # Error file must record the gap event
    err_file = tmp_path / "collection_errors.jsonl"
    errs = [json.loads(l) for l in err_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert any(e["stage"] == "orderbook_sequence_gap" for e in errs)


# =========================================================================
# 7. Invalid Orderbook Rejection
# =========================================================================

def test_invalid_orderbook_detection():
    """Verify crossed books and negative quantities trigger InvalidOrderBookError."""
    book = L2OrderBook(symbol="BTCUSDT")

    # Crossed book (bid 50010 > ask 50000)
    crossed_snap = L2Snapshot(
        symbol="BTCUSDT",
        last_update_id=1,
        bids=((50010.0, 1.0),),
        asks=((50000.0, 1.0),),
    )
    with pytest.raises(InvalidOrderBookError, match="Crossed orderbook"):
        book.apply_snapshot(crossed_snap)

    # Negative price
    neg_price_snap = L2Snapshot(
        symbol="BTCUSDT",
        last_update_id=2,
        bids=((-50.0, 1.0),),
        asks=((50000.0, 1.0),),
    )
    with pytest.raises(InvalidOrderBookError, match="strictly positive"):
        book.apply_snapshot(neg_price_snap)

    # Negative size
    neg_size_snap = L2Snapshot(
        symbol="BTCUSDT",
        last_update_id=3,
        bids=((50000.0, -1.0),),
        asks=((50010.0, 1.0),),
    )
    with pytest.raises(InvalidOrderBookError, match="cannot be negative"):
        book.apply_snapshot(neg_size_snap)


# =========================================================================
# 8. Session Integrity Validation
# =========================================================================

def test_session_integrity_validation_passes_valid_session(tmp_path):
    """Verify validation passes cleanly on a valid session with observations and depth."""
    session_dir = tmp_path / "valid_session"
    session_dir.mkdir(parents=True)

    # Write valid metadata
    (session_dir / "session_metadata.json").write_text(
        json.dumps({"session_id": "valid_session", "status": "COMPLETED"}), encoding="utf-8"
    )

    # Write 5 valid observations at 60s intervals
    obs_file = session_dir / "observations.jsonl"
    lines = []
    base_ms = 1760000000000
    for i in range(5):
        t = base_ms + i * 60000
        rec = {
            "source": "binance",
            "symbol": "BTCUSDT",
            "source_timestamp": pd.Timestamp(t, unit="ms", tz="UTC").isoformat(),
            "collected_at": pd.Timestamp(t, unit="ms", tz="UTC").isoformat(),
            "open": "60000.0",
            "high": "60100.0",
            "low": "59900.0",
            "close": "60050.0",
            "volume": "10.0",
            "record_id": f"rec_{i}",
        }
        lines.append(json.dumps(rec))
    obs_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write 3 valid depth snapshots
    depth_file = session_dir / "orderbook_snapshots.jsonl"
    d_lines = []
    for i in range(3):
        d_rec = {
            "source": "binance_l2_rest",
            "symbol": "BTCUSDT",
            "last_update_id": 1000 + i,
            "bids": [[60000.0, 1.0], [59990.0, 2.0]],
            "asks": [[60010.0, 1.0], [60020.0, 2.0]],
            "received_at": pd.Timestamp(base_ms + i * 60000, unit="ms", tz="UTC").isoformat(),
            "record_id": f"depth_{i}",
        }
        d_lines.append(json.dumps(d_rec))
    depth_file.write_text("\n".join(d_lines) + "\n", encoding="utf-8")

    report = validate_session_data(session_dir, expected_interval_seconds=60.0)
    assert report["passed"] is True
    assert all(c["passed"] for c in report["checks"].values())
    assert report["summary"]["valid_observations"] == 5
    assert report["summary"]["total_depth_snapshots"] == 3


def test_session_integrity_validation_flags_corrupt_data(tmp_path):
    """Verify validation detects out-of-order timestamps, duplicate timestamps, and invalid prices."""
    session_dir = tmp_path / "corrupt_session"
    session_dir.mkdir(parents=True)

    obs_file = session_dir / "observations.jsonl"
    t1 = "2026-10-04T12:00:00+00:00"
    t2 = "2026-10-04T12:00:00+00:00"  # duplicate timestamp
    t3 = "2026-10-04T11:50:00+00:00"  # out of order

    records = [
        {"source": "binance", "symbol": "BTCUSDT", "source_timestamp": t1, "open": "60000", "high": "60100", "low": "59900", "close": "60050", "volume": "5"},
        {"source": "binance", "symbol": "BTCUSDT", "source_timestamp": t2, "open": "-10", "high": "60100", "low": "59900", "close": "60050", "volume": "5"},  # negative price
        {"source": "binance", "symbol": "BTCUSDT", "source_timestamp": t3, "open": "60000", "high": "59000", "low": "60000", "close": "59500", "volume": "5"},  # high < low
    ]
    obs_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    report = validate_session_data(session_dir, expected_interval_seconds=60.0)
    assert report["passed"] is False
    assert report["checks"]["timestamp_ordering"]["passed"] is False
    assert report["checks"]["duplicate_timestamps"]["passed"] is False
    assert report["checks"]["positive_prices"]["passed"] is False
    assert report["checks"]["high_low_validity"]["passed"] is False


# =========================================================================
# 9. Offline Session Processing & Raw Immutability
# =========================================================================

def test_offline_session_processing_and_raw_immutability(tmp_path):
    """Verify session processing creates clean parquets while leaving raw JSONL bit-for-bit unchanged."""
    raw_dir = tmp_path / "raw_session"
    proc_dir = tmp_path / "proc_session"
    raw_dir.mkdir(parents=True)

    obs_file = raw_dir / "observations.jsonl"
    obs_file.write_text(
        json.dumps({
            "source": "binance", "symbol": "BTCUSDT", "source_timestamp": "2026-10-04T12:00:00Z",
            "collected_at": "2026-10-04T12:00:01Z", "open": "60000.0", "high": "60100.0",
            "low": "59900.0", "close": "60050.0", "volume": "10.0", "record_id": "r1"
        }) + "\n",
        encoding="utf-8"
    )

    depth_file = raw_dir / "orderbook_snapshots.jsonl"
    depth_file.write_text(
        json.dumps({
            "source": "binance_l2_rest", "symbol": "BTCUSDT", "last_update_id": 100,
            "bids": [[60000.0, 1.0]], "asks": [[60010.0, 1.0]],
            "received_at": "2026-10-04T12:00:01Z", "record_id": "d1"
        }) + "\n",
        encoding="utf-8"
    )

    # Hash raw files before processing
    obs_hash_before = hashlib.sha256(obs_file.read_bytes()).hexdigest()
    depth_hash_before = hashlib.sha256(depth_file.read_bytes()).hexdigest()

    # Process
    res = process_session_data(raw_dir, proc_dir, expected_interval_seconds=60.0)
    assert res["passed"] is True

    # Verify Parquet outputs
    assert (proc_dir / "observations.parquet").exists()
    assert (proc_dir / "orderbook_snapshots.parquet").exists()
    assert (proc_dir / "session_report.json").exists()
    assert (proc_dir / "session_report.md").exists()

    obs_df = pd.read_parquet(proc_dir / "observations.parquet")
    assert len(obs_df) == 1
    assert obs_df["close"].iloc[0] == 60050.0

    depth_df = pd.read_parquet(proc_dir / "orderbook_snapshots.parquet")
    assert len(depth_df) == 1
    assert depth_df["mid_price"].iloc[0] == 60005.0
    assert depth_df["spread"].iloc[0] == 10.0

    # Verify raw files are bit-for-bit unchanged
    obs_hash_after = hashlib.sha256(obs_file.read_bytes()).hexdigest()
    depth_hash_after = hashlib.sha256(depth_file.read_bytes()).hexdigest()
    assert obs_hash_before == obs_hash_after, "Raw observations file was modified during processing!"
    assert depth_hash_before == depth_hash_after, "Raw depth file was modified during processing!"


# =========================================================================
# 10. Polymarket L2 Adapter & Documented Limitations
# =========================================================================

def test_polymarket_l2_adapter_and_blocker_documentation():
    """Verify PolymarketL2Adapter functions and clearly documents the sequence tracking limitation."""
    adapter = PolymarketL2Adapter(asset_id="token_yes", symbol="BTC-UP")

    # Documented limitation check
    assert "Polymarket" in PolymarketL2Adapter.__doc__
    assert "sequence" in PolymarketL2Adapter.__doc__.lower()
    assert "monotonic sequence numbers" in PolymarketL2Adapter.__doc__.lower()

    # Initial book event
    book_event = {
        "event_type": "book",
        "market": "c1",
        "bids": [{"price": "0.45", "size": "100"}],
        "asks": [{"price": "0.55", "size": "150"}],
    }
    snap = adapter.process_book_event(book_event)
    assert snap is not None
    assert adapter.book.best_bid == 0.45
    assert adapter.book.best_ask == 0.55

    # Price change delta event
    price_change_event = {
        "event_type": "price_change",
        "price_changes": [
            {"asset_id": "token_yes", "price": "0.48", "size": "200", "side": "BUY"},
            {"asset_id": "token_other", "price": "0.99", "size": "10", "side": "BUY"},  # ignored
        ]
    }
    assert adapter.process_price_change(price_change_event) is True
    assert adapter.book.best_bid == 0.48


# =========================================================================
# 11. CLI Argument Parser & Routing Tests
# =========================================================================

def test_cli_argument_parser_flags():
    """Verify build_parser accepts all required Phase 3 arguments and assigns correct types."""
    from scripts.collect_track_a import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--session-id", "session_cli_test_01",
        "--symbol", "ETHUSDT",
        "--interval-seconds", "5",
        "--duration-seconds", "120",
        "--capture-l2",
        "--depth-limit", "50",
        "--dry-run",
        "--output-dir", "custom/raw",
        "--processed-dir", "custom/proc",
    ])

    assert args.session_id == "session_cli_test_01"
    assert args.symbol == "ETHUSDT"
    assert args.interval_seconds == 5.0
    assert args.duration_seconds == 120.0
    assert args.capture_l2 is True
    assert args.depth_limit == 50
    assert args.dry_run is True
    assert args.output_dir == "custom/raw"
    assert args.processed_dir == "custom/proc"

    # Test validate-only and process-only flags
    args_val = parser.parse_args(["--session-id", "s1", "--validate-only"])
    assert args_val.validate_only is True
    assert args_val.process_only is False

    args_proc = parser.parse_args(["--session-id", "s1", "--process-only"])
    assert args_proc.process_only is True
    assert args_proc.validate_only is False


def test_cli_main_dry_run_routing(tmp_path):
    """Verify executing main() with CLI flags routes correctly to session directory and creates artifacts."""
    from scripts.collect_track_a import main

    session_id = "cli_smoke_session"
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"

    exit_code = main([
        "--session-id", session_id,
        "--symbol", "BTCUSDT",
        "--interval-seconds", "1",
        "--output-dir", str(raw_dir),
        "--processed-dir", str(proc_dir),
        "--capture-l2",
        "--depth-limit", "20",
        "--dry-run",
    ])

    assert exit_code == 0
    session_raw = raw_dir / "sessions" / session_id
    assert session_raw.exists()
    assert (session_raw / "session_metadata.json").exists()
    assert (session_raw / "observations.jsonl").exists()
    assert (session_raw / "orderbook_snapshots.jsonl").exists()


def test_cli_main_validate_and_process_only_isolation(tmp_path):
    """Verify --validate-only and --process-only operate strictly on the targeted session without touching unrelated data."""
    from scripts.collect_track_a import main

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"

    target_sess = raw_dir / "sessions" / "target_session"
    unrelated_sess = raw_dir / "sessions" / "unrelated_session"
    target_sess.mkdir(parents=True)
    unrelated_sess.mkdir(parents=True)

    # Put valid record in target_session
    obs_file = target_sess / "observations.jsonl"
    obs_file.write_text(
        json.dumps({
            "source": "binance", "symbol": "BTCUSDT", "source_timestamp": "2026-10-04T12:00:00Z",
            "open": "60000.0", "high": "60100.0", "low": "59900.0", "close": "60050.0",
            "volume": "10.0", "record_id": "r_target"
        }) + "\n",
        encoding="utf-8"
    )

    # 1. Test validate-only on target session succeeds
    exit_val = main([
        "--session-id", "target_session",
        "--output-dir", str(raw_dir),
        "--validate-only",
    ])
    assert exit_val == 0

    # 2. Test validate-only on non-existent session fails safely with FileNotFoundError
    with pytest.raises(FileNotFoundError, match="raw directory does not exist"):
        main([
            "--session-id", "ghost_session",
            "--output-dir", str(raw_dir),
            "--validate-only",
        ])

    # 3. Test process-only on target session strictly produces processed target without polluting unrelated
    exit_proc = main([
        "--session-id", "target_session",
        "--output-dir", str(raw_dir),
        "--processed-dir", str(proc_dir),
        "--process-only",
    ])
    assert exit_proc == 0

    target_proc_dir = proc_dir / "sessions" / "target_session"
    assert (target_proc_dir / "observations.parquet").exists()
    assert (target_proc_dir / "session_report.json").exists()

    # Unrelated session must NOT have been processed
    unrelated_proc_dir = proc_dir / "sessions" / "unrelated_session"
    assert not unrelated_proc_dir.exists(), "Unrelated session was processed unexpectedly!"


# =========================================================================
# 11. Regression Tests: Timestamp Semantics & Live Session Revalidation
# =========================================================================

def test_validation_distinguishes_intra_candle_updates_from_duplicates(tmp_path):
    """Verify validation passes when multiple 1s observations share the same 1m candle source timestamp."""
    session_dir = tmp_path / "intra_candle_session"
    session_dir.mkdir(parents=True)

    candle_close = "2026-10-04T12:00:59.999000+00:00"
    records = []
    for i in range(5):
        records.append({
            "source": "binance",
            "symbol": "BTCUSDT",
            "source_timestamp": candle_close,
            "open_time": "2026-10-04T12:00:00+00:00",
            "collected_at": f"2026-10-04T12:00:0{i}.100000+00:00",
            "open": "60000.0",
            "high": "60050.0",
            "low": "59950.0",
            "close": str(60010.0 + i),
            "volume": str(1.0 + i * 0.5),
            "record_id": f"candle_update_{i}",
        })

    obs_file = session_dir / "observations.jsonl"
    obs_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    report = validate_session_data(session_dir, expected_interval_seconds=1.0)
    assert report["passed"] is True
    assert report["checks"]["duplicate_records"]["passed"] is True
    assert report["checks"]["duplicate_timestamps"]["passed"] is True
    assert report["checks"]["timestamp_ordering"]["passed"] is True
    assert report["checks"]["candle_source_timestamps"]["passed"] is True
    assert report["checks"]["candle_source_timestamps"]["intra_candle_updates"] == 4
    assert report["checks"]["candle_source_timestamps"]["unique_candle_count"] == 1

    # Verify Parquet processing retains all 5 distinct intra-candle observations
    proc_dir = tmp_path / "proc_intra"
    process_session_data(session_dir, proc_dir, expected_interval_seconds=1.0)
    df = pd.read_parquet(proc_dir / "observations.parquet")
    assert len(df) == 5


def test_validation_detects_genuine_duplicate_records(tmp_path):
    """Verify validation flags genuine duplicate records with identical payload/record_id."""
    session_dir = tmp_path / "duplicate_session"
    session_dir.mkdir(parents=True)

    rec = {
        "source": "binance",
        "symbol": "BTCUSDT",
        "source_timestamp": "2026-10-04T12:00:59.999000+00:00",
        "collected_at": "2026-10-04T12:00:01.000000+00:00",
        "open": "60000.0",
        "high": "60050.0",
        "low": "59950.0",
        "close": "60010.0",
        "volume": "1.0",
        "record_id": "duplicate_id_001",
    }
    # Write identical record twice
    obs_file = session_dir / "observations.jsonl"
    obs_file.write_text(f"{json.dumps(rec)}\n{json.dumps(rec)}\n", encoding="utf-8")

    report = validate_session_data(session_dir, expected_interval_seconds=1.0)
    assert report["passed"] is False
    assert report["checks"]["duplicate_records"]["passed"] is False
    assert report["checks"]["duplicate_records"]["duplicate_count"] == 1
    assert report["checks"]["duplicate_timestamps"]["passed"] is False


@pytest.mark.parametrize(
    "session_id,expected_snapshots",
    [
        ("session_live_001", 70),
        ("session_live_002", 61),
    ],
)
def test_live_sessions_revalidation_and_raw_immutability(session_id: str, expected_snapshots: int):
    """Verify that actual live sessions pass validation and preserve raw file immutability."""
    raw_dir = Path(f"data/raw/track_a/sessions/{session_id}")
    if not raw_dir.exists():
        pytest.skip(f"{session_id} not present in this checkout.")

    # 1. Compute SHA-256 hashes of all raw files before validation/processing
    hashes_before = {}
    for f in sorted(raw_dir.glob("*.json*")):
        hashes_before[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    # 2. Run validation on live session
    report = validate_session_data(raw_dir)
    assert report["passed"] is True, f"Validation failed with checks: {report['checks']}"
    assert report["summary"]["valid_observations"] == 60
    assert report["summary"]["total_depth_snapshots"] == expected_snapshots
    assert report["checks"]["candle_source_timestamps"]["intra_candle_updates"] == 58
    assert report["checks"]["candle_source_timestamps"]["unique_candle_count"] == 2
    assert report["checks"]["duplicate_records"]["passed"] is True
    assert report["checks"]["duplicate_timestamps"]["passed"] is True
    assert report["checks"]["timestamp_ordering"]["passed"] is True
    assert report["checks"]["missing_intervals"]["passed"] is True

    # 3. Process session to parquet
    proc_dir = Path(f"data/processed/track_a/sessions/{session_id}")
    process_session_data(raw_dir, proc_dir)

    df_obs = pd.read_parquet(proc_dir / "observations.parquet")
    df_depth = pd.read_parquet(proc_dir / "orderbook_snapshots.parquet")
    assert len(df_obs) == 60
    assert len(df_depth) == expected_snapshots

    # 4. Verify raw files were not modified during validation or parquet conversion
    hashes_after = {}
    for f in sorted(raw_dir.glob("*.json*")):
        hashes_after[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    assert hashes_before == hashes_after, f"Raw {session_id} files were modified!"



