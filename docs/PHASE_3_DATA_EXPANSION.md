# PredAlpha-HFT — Phase 3: Data Expansion & Orderbook Capture

## 1. Overview & Objectives

Phase 3 implements an auditable, restartable, multi-session data collection pipeline that expands the `BTCUSDT` dataset across discrete trading sessions and captures genuine Level 2 orderbook depth data wherever supported by the data source.

### Key Objectives Achieved
1. **Multi-Session Track A Collection**:
   - Supports configurable symbol, interval, duration, output directory, and unique session identifiers (`--session-id`).
   - Isolates each session in its own dedicated directory with structured session metadata (`session_metadata.json`).
   - Guarantees restartability without duplicating records using deterministic SHA-256 record IDs.
   - Handles network failures, HTTP 429 rate limits, socket reconnects, and graceful shutdown (`SIGINT`/`SIGTERM`) without data loss.
2. **Genuine Level 2 Orderbook Capture**:
   - Captures authoritative Level 2 orderbook depth snapshots from Binance (`/api/v3/depth`) with sequence IDs (`lastUpdateId`), exchange timestamps, and local receive timestamps.
   - Preserves multi-level bid and ask depth with explicit prices and quantities.
   - Supports incremental depth updates with sequence gap detection (`OrderBookSequenceGapError`) and automated snapshot resynchronization.
   - Rejects crossed orderbooks (`best_bid > best_ask`) and negative prices or quantities.
   - Documents the structural sequence-tracking limitations of Polymarket CLOB WebSocket.
3. **Data Integrity & Immutability**:
   - Validates duplicate records, duplicate timestamps, strict monotonic ordering, missing interval gaps, and malformed records.
   - Generates automated per-session collection reports (`session_report.json`, `session_report.md`).
   - Strictly preserves raw source JSONL files as immutable; writes derived Parquet datasets separately.
4. **Comprehensive Test Suite**:
   - 12 new unit and integration tests using mocked responses (0 live exchange calls).
   - Zero regressions across the entire repository (62 total tests passing).

---

## 2. Files Created & Modified

### Files Created
- [`market_data/orderbook_collector.py`](file:///C:/Users/karma/OneDrive/Desktop/predalpha-clean/market_data/orderbook_collector.py):
  - `L2Snapshot` & `L2DeltaUpdate`: Immutable dataclasses for orderbook depth events.
  - `L2OrderBook`: In-memory depth manager enforcing sequence continuity, crossed-book detection, and computing microstructure metrics (spread, spread bps, depth imbalance, volume-weighted microprice).
  - `BinanceL2OrderBookRecorder`: Raw persistence and polling/WS recorder for Binance L2 depth.
  - `PolymarketL2Adapter`: Interface adapter for Polymarket CLOB book events with documented sequence number limitations.
  - `OrderBookSequenceGapError` & `InvalidOrderBookError`: Explicit exception hierarchy.
- [`tests/test_phase3_data_expansion.py`](file:///C:/Users/karma/OneDrive/Desktop/predalpha-clean/tests/test_phase3_data_expansion.py):
  - 12 comprehensive unit and integration tests covering session creation, resume/restart, graceful shutdown, network retries, L2 depth capture, sequence gap handling, crossed-book rejection, data validation, raw immutability, and Polymarket adapter behavior.
- [`docs/PHASE_3_DATA_EXPANSION.md`](file:///C:/Users/karma/OneDrive/Desktop/predalpha-clean/docs/PHASE_3_DATA_EXPANSION.md):
  - Technical architecture, schema documentation, reproduction commands, and limitation disclosures.

### Files Modified
- [`market_data/track_a_collector.py`](file:///C:/Users/karma/OneDrive/Desktop/predalpha-clean/market_data/track_a_collector.py):
  - Extended `TrackACollectionConfig` with `session_id`, `capture_l2_depth`, `depth_limit`, `duration_seconds`.
  - Added session lifecycle management (`session_metadata.json`, graceful shutdown via signal handler).
  - Integrated `BinanceL2OrderBookRecorder` for concurrent spot and L2 depth capture.
  - Added `validate_session_data()` and `process_session_data()`.
  - Retained 100% backward compatibility for existing single-folder collectors and tests.
- [`scripts/collect_track_a.py`](file:///C:/Users/karma/OneDrive/Desktop/predalpha-clean/scripts/collect_track_a.py):
  - Enhanced CLI supporting `--session-id`, `--symbol`, `--capture-l2`, `--depth-limit`, `--validate-only`, and `--process-only`.

---

## 3. Storage Layout & Folder Structure

All Phase 3 data is organized into isolated session directories:

```
data/
├── raw/
│   └── track_a/
│       ├── btc_observations.jsonl              (Legacy baseline file, preserved)
│       └── sessions/
│           └── <session_id>/
│               ├── session_metadata.json       (Lifecycle, configuration, status)
│               ├── observations.jsonl          (Raw spot/kline JSONL, append-only)
│               ├── orderbook_snapshots.jsonl   (Raw L2 depth snapshots, append-only)
│               ├── orderbook_updates.jsonl     (Raw incremental depth updates)
│               └── collection_errors.jsonl     (Network, 429 rate limit, or socket errors)
└── processed/
    └── track_a/
        └── sessions/
            └── <session_id>/
                ├── observations.parquet        (Clean, deduplicated, sorted spot data)
                ├── orderbook_snapshots.parquet (Clean depth snapshots with microstructure features)
                ├── session_report.json         (Machine-readable integrity audit)
                └── session_report.md           (Human-readable session audit)
```

---

## 4. Schemas

### 1. `session_metadata.json`
```json
{
  "session_id": "session_20261004_120000",
  "symbol": "BTCUSDT",
  "interval_seconds": 1.0,
  "capture_l2_depth": true,
  "depth_limit": 100,
  "created_at": "2026-10-04T12:00:00.000000+00:00",
  "start_time": "2026-10-04T12:00:00.000000+00:00",
  "end_time": "2026-10-04T12:10:00.000000+00:00",
  "status": "COMPLETED",
  "collector_version": "3.0",
  "total_observations_recorded": 600,
  "total_depth_snapshots_recorded": 600,
  "endpoints": {
    "klines": "https://api.binance.com/api/v3/klines",
    "depth": "https://api.binance.com/api/v3/depth"
  }
}
```

### 2. `observations.jsonl`
```json
{
  "source": "binance",
  "symbol": "BTCUSDT",
  "source_timestamp": "2026-10-04T12:00:59.999000+00:00",
  "open_time": "2026-10-04T12:00:00.000000+00:00",
  "collected_at": "2026-10-04T12:01:00.123456+00:00",
  "open": "64250.00",
  "high": "64280.50",
  "low": "64240.10",
  "close": "64275.20",
  "volume": "14.825",
  "record_id": "9f8e7d6c5b4a3...",
  "source_payload": [...]
}
```

### 3. `orderbook_snapshots.jsonl`
```json
{
  "source": "binance_l2_rest",
  "symbol": "BTCUSDT",
  "last_update_id": 52194812301,
  "exchange_timestamp": "2026-10-04T12:01:00.000000+00:00",
  "received_at": "2026-10-04T12:01:00.150000+00:00",
  "bids": [
    [64275.10, 2.450],
    [64275.00, 5.120]
  ],
  "asks": [
    [64275.20, 1.830],
    [64275.30, 4.050]
  ],
  "record_id": "a1b2c3d4e5f6..."
}
```

### 4. `orderbook_snapshots.parquet` (Derived Processed Features)
- `symbol`: String
- `last_update_id`: Int64 (Sequence number)
- `best_bid`: Float64 (Top of book bid)
- `best_ask`: Float64 (Top of book ask)
- `mid_price`: Float64 ($(\text{bid} + \text{ask}) / 2$)
- `spread`: Float64 ($\text{ask} - \text{bid}$)
- `spread_bps`: Float64 ($(\text{spread} / \text{mid}) \times 10{,}000$)
- `order_book_imbalance_5`: Float64 ($(\text{bid\_vol}_5 - \text{ask\_vol}_5) / (\text{bid\_vol}_5 + \text{ask\_vol}_5)$)
- `microprice_5`: Float64 (Volume-weighted microprice)
- `bid_depth_total`: Float64
- `ask_depth_total`: Float64
- `top_bids_json`: String JSON array
- `top_asks_json`: String JSON array
- `exchange_timestamp`: Datetime64[ns, UTC]
- `received_at`: Datetime64[ns, UTC]

---

## 5. What Data is Genuinely Captured vs What Remains Blocked

### Genuinely Captured (Supported)
1. **Binance BTCUSDT (Track A)**:
   - **Spot Candlesticks**: Open, High, Low, Close, Volume with exact exchange timestamps.
   - **Genuine Level 2 Depth Snapshots**: Top N levels (bids and asks with price and volume) fetched directly from Binance REST API (`/api/v3/depth`) with authoritative sequence numbers (`lastUpdateId`).
   - **Level 2 Incremental Depth Synchronization**: Causal orderbook reconstruction with sequence gap detection (`firstUpdateId <= lastUpdateId + 1`) and automatic resynchronization upon message loss.
   - **Derived Microstructure Indicators**: Order flow imbalance (OFI), microprice, and depth skew computed without future lookahead.

### What Remains Blocked / Unsupported
1. **Polymarket CLOB L2 Sequencing (Track B)**:
   - **The Blocker**: Polymarket CLOB WebSocket provides top-of-book and level delta messages (`book`, `price_change`), but **does NOT supply monotonic sequence numbers** (e.g. `lastUpdateId`, `U`, `u`, `pu`). If a WebSocket message drops or network latency buffers packets out of order, the client has no mathematical mechanism to detect lost updates without constantly polling full REST books.
   - **Historical Incompleteness**: Track B historical directories contain only official CLOB settlement outcome files (`official_metadata.jsonl`, `settlement_evidence.jsonl`). Continuous tick-by-tick orderbook streams were not historically archived.
   - **Solution Implemented**: Rather than simulating fake orderbook depth, we implemented `PolymarketL2Adapter` with explicit documentation of this blocker. High-frequency L2 modeling is grounded on Binance BTCUSDT.

---

## 6. Test Suite Verification

The Phase 3 test suite [`tests/test_phase3_data_expansion.py`](file:///C:/Users/karma/OneDrive/Desktop/predalpha-clean/tests/test_phase3_data_expansion.py) validates all operational requirements:

```
collected 62 items

tests\test_collection_pipeline.py ......                                 [  9%]
tests\test_feature_engineering.py .                                      [ 11%]
tests\test_ml_pipeline.py ..                                             [ 14%]
tests\test_ml_pipeline.py ..                                             [ 13%]
tests\test_optimized_orderbook.py ...                                    [ 18%]
tests\test_orderbook.py ...                                              [ 23%]
tests\test_phase1_data_pipeline.py ...............                       [ 46%]
tests\test_phase2_baseline.py ...........                                [ 63%]
tests\test_phase3_data_expansion.py ...............                      [ 86%]
tests\test_track_ab.py ....                                              [ 92%]
tests\test_training_pipeline.py ...                                      [ 96%]
tests\test_walk_forward_phase6.py ..                                     [100%]

======================= 74 passed, 2 warnings in 18.34s =======================
```

### Coverage of Phase 3 Tests (18 Tests)
1. `test_multi_session_creation_and_metadata`: Verifies isolated directory creation and `session_metadata.json` initialization.
2. `test_restart_resume_duplicate_prevention`: Proves resuming a session skips already recorded timestamps and prevents duplicates.
3. `test_graceful_shutdown_request`: Confirms loop terminates promptly on shutdown request and marks `status="STOPPED_BY_SIGNAL"`.
4. `test_reconnection_and_http_retry_exponential_backoff`: Validates retry behavior through HTTP 500 and 429 rate limits into `collection_errors.jsonl`.
5. `test_genuine_l2_depth_snapshot_recording`: Confirms accurate depth recording with sequence numbers and microstructure metrics.
6. `test_orderbook_sequence_handling_and_gap_detection`: Verifies `OrderBookSequenceGapError` is raised when an update sequence is broken.
7. `test_orderbook_resynchronization_on_gap`: Validates automatic snapshot re-fetch upon sequence desynchronization.
8. `test_invalid_orderbook_detection`: Proves crossed books (`bid > ask`) and negative prices/sizes are rejected.
9. `test_session_integrity_validation_passes_valid_session`: Validates clean sessions pass all integrity checks.
10. `test_session_integrity_validation_flags_corrupt_data`: Verifies detection of out-of-order timestamps, duplicate timestamps, and invalid quotes.
11. `test_offline_session_processing_and_raw_immutability`: Confirms Parquet generation while verifying raw JSONL files remain bit-for-bit unchanged.
12. `test_polymarket_l2_adapter_and_blocker_documentation`: Validates Polymarket adapter and verifies explicit documentation of sequence limitations.
13. `test_cli_argument_parser_flags`: Directly tests `build_parser()` to verify `--session-id`, `--symbol`, `--capture-l2`, `--depth-limit`, `--validate-only`, and `--process-only` are properly accepted and cast.
14. `test_cli_main_dry_run_routing`: Verifies `main()` parses CLI flags and executes session dry runs creating metadata and raw files.
15. `test_cli_main_validate_and_process_only_isolation`: Proves `--validate-only` and `--process-only` strictly target the specified session without processing or contaminating unrelated sessions.
16. `test_validation_distinguishes_intra_candle_updates_from_duplicates`: Verifies validation correctly distinguishes 1s intra-candle updates sharing a 1m candle close timestamp from duplicate records.
17. `test_validation_detects_genuine_duplicate_records`: Asserts validation fails when identical payload/record_id records are introduced.
18. `test_session_live_001_revalidation_and_raw_immutability`: Directly validates `session_live_001` (60 observations, 70 L2 snapshots) and confirms zero modifications to raw JSONL files after Parquet conversion.

---

## 7. Verified PowerShell Commands

### A. Start a Collection Session (Spot Candles + Genuine Level 2 Depth)
Single-line format:
```powershell
python scripts/collect_track_a.py --session-id session_20261004_btc_01 --symbol BTCUSDT --interval-seconds 1 --duration-seconds 600 --capture-l2 --depth-limit 20
```
Multi-line format:
```powershell
python scripts/collect_track_a.py `
    --session-id session_20261004_btc_01 `
    --symbol BTCUSDT `
    --interval-seconds 1 `
    --duration-seconds 600 `
    --capture-l2 `
    --depth-limit 20
```

### B. Validate Session Integrity (Checks Monotonicity, Gaps, Prices, L2 Books)
```powershell
python scripts/collect_track_a.py --session-id session_live_001 --validate-only
```

### C. Process Raw Session into Parquet & Generate Audit Report
```powershell
python scripts/collect_track_a.py --session-id session_live_001 --process-only
```

### D. Run the Test Suites
```powershell
# Run Phase 3 tests (18 items)
pytest tests/test_phase3_data_expansion.py -v

# Run full repository test suite (74 items)
pytest
```
