"""Multi-session Track A collector with genuine Level 2 orderbook capture and data integrity validation.

Features:
- Configurable symbol, interval, duration, output directory, and session identifier.
- Isolated per-session storage with immutable AppendOnly JSONL files.
- Restartable collection without duplicating records.
- Graceful shutdown handling (SIGINT/SIGTERM) with updated session metadata.
- Resilient retry loops with exponential backoff on HTTP 429 rate limits and connection drops.
- Genuine Level 2 depth snapshot and delta recording via BinanceL2OrderBookRecorder.
- Offline session processing into immutable Parquet artifacts and human-readable validation reports.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from market_data.collection_common import (
    AppendOnlyJsonl,
    append_error,
    jsonl_records,
    timestamp_quality,
    utc_now,
)
from market_data.orderbook_collector import (
    BinanceL2OrderBookRecorder,
    InvalidOrderBookError,
    L2OrderBook,
    L2Snapshot,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackACollectionConfig:
    """Configuration for Track A collection (spot/candles and genuine L2 depth)."""

    symbol: str = "BTCUSDT"
    interval_seconds: float = 60.0
    output_dir: Path = Path("data/raw/track_a")
    processed_dir: Path = Path("data/processed/track_a")
    endpoint: str = "https://api.binance.com/api/v3/klines"
    depth_endpoint: str = "https://api.binance.com/api/v3/depth"
    timeout_seconds: float = 10.0
    max_retries: int = 3
    session_id: str | None = None
    capture_l2_depth: bool = False
    depth_limit: int = 100
    duration_seconds: float = 0.0


class TrackACollector:
    """Restartable, multi-session Track A collector supporting candles and L2 depth."""

    def __init__(
        self,
        config: TrackACollectionConfig,
        session: requests.Session | Any | None = None,
    ):
        self.config = config
        self.session = session or requests.Session()
        self._shutdown_requested = False

        # Determine target session directory
        if config.session_id:
            if config.output_dir.name == "sessions":
                self.session_raw_dir = config.output_dir / config.session_id
            else:
                self.session_raw_dir = config.output_dir / "sessions" / config.session_id
            self.session_processed_dir = config.processed_dir / "sessions" / config.session_id
            obs_file = self.session_raw_dir / "observations.jsonl"
            error_file = self.session_raw_dir / "collection_errors.jsonl"
        else:
            # Backward-compatible default path
            self.session_raw_dir = config.output_dir
            self.session_processed_dir = config.processed_dir
            obs_file = config.output_dir / "btc_observations.jsonl"
            error_file = config.output_dir / "collection_errors.jsonl"

        self.session_raw_dir.mkdir(parents=True, exist_ok=True)
        self.raw_writer = AppendOnlyJsonl(obs_file)
        self.error_writer = AppendOnlyJsonl(error_file)

        # L2 Depth Recorder
        self.depth_recorder: BinanceL2OrderBookRecorder | None = None
        if config.capture_l2_depth:
            self.depth_recorder = BinanceL2OrderBookRecorder(
                output_dir=self.session_raw_dir,
                symbol=config.symbol,
                endpoint=config.depth_endpoint,
                session=self.session,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
            )

        # Initialize session metadata if in session mode
        if config.session_id:
            self._init_session_metadata()

        # Register graceful shutdown signals if in main thread
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown on SIGINT and SIGTERM."""
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, AttributeError):
            # Not in main thread or signal unsupported on current runtime
            pass

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info("Shutdown signal %d received. Finishing current collection step...", signum)
        self._shutdown_requested = True

    def request_shutdown(self) -> None:
        """Programmatically request graceful shutdown."""
        self._shutdown_requested = True

    @property
    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def _init_session_metadata(self) -> None:
        """Initialize or update session_metadata.json."""
        meta_path = self.session_raw_dir / "session_metadata.json"
        if not meta_path.exists():
            meta = {
                "session_id": self.config.session_id,
                "symbol": self.config.symbol,
                "interval_seconds": self.config.interval_seconds,
                "capture_l2_depth": self.config.capture_l2_depth,
                "depth_limit": self.config.depth_limit,
                "created_at": utc_now(),
                "start_time": utc_now(),
                "end_time": None,
                "status": "RUNNING",
                "collector_version": "3.0",
                "endpoints": {
                    "klines": self.config.endpoint,
                    "depth": self.config.depth_endpoint if self.config.capture_l2_depth else None,
                },
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        else:
            # Resuming an existing session
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["resumed_at"] = utc_now()
                meta["status"] = "RESUMED"
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as e:
                append_error(self.error_writer, "resume_metadata", e)

    def _finalize_session_metadata(self, status: str = "COMPLETED", collected_count: int = 0) -> None:
        """Update session_metadata.json on shutdown or loop completion."""
        if not self.config.session_id:
            return
        meta_path = self.session_raw_dir / "session_metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta["end_time"] = utc_now()
            meta["status"] = status
            meta["total_observations_recorded"] = len(self.raw_writer._ids)
            if self.depth_recorder:
                meta["total_depth_snapshots_recorded"] = len(self.depth_recorder.snapshot_writer._ids)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as e:
            append_error(self.error_writer, "finalize_metadata", e)

    def fetch_once(self) -> dict[str, Any] | None:
        """Fetch one completed candle; retry transient HTTP/network errors."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.get(
                    self.config.endpoint,
                    params={"symbol": self.config.symbol, "interval": "1m", "limit": 1},
                    timeout=self.config.timeout_seconds,
                )
                if getattr(response, "status_code", 200) == 429:
                    raise RuntimeError("rate_limited_http_429")
                response.raise_for_status()
                rows = response.json()
                if not isinstance(rows, list) or not rows or len(rows[0]) < 7:
                    raise ValueError("missing_or_malformed_kline")
                candle = rows[0]
                return {
                    "source": "binance",
                    "symbol": self.config.symbol,
                    "source_timestamp": pd.Timestamp(int(candle[6]), unit="ms", tz="UTC").isoformat(),
                    "open_time": pd.Timestamp(int(candle[0]), unit="ms", tz="UTC").isoformat(),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "source_payload": candle,
                }
            except (requests.RequestException, ValueError, RuntimeError) as error:
                append_error(self.error_writer, "fetch_btc", error, attempt=attempt, symbol=self.config.symbol)
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
        return None

    def collect_once(self, dry_run: bool = False) -> bool:
        """Execute one collection step for candles and optional L2 depth."""
        if dry_run:
            appended = self.raw_writer.append({
                "source": "dry_run",
                "symbol": self.config.symbol,
                "source_timestamp": utc_now(),
                "open": "0",
                "high": "0",
                "low": "0",
                "close": "0",
                "volume": "0",
                "dry_run": True,
            })
            if self.depth_recorder:
                dummy_snap = L2Snapshot(
                    symbol=self.config.symbol,
                    last_update_id=1,
                    bids=((0.08, 1.0),),
                    asks=((0.09, 1.0),),
                    exchange_timestamp=utc_now(),
                    source="dry_run",
                )
                self.depth_recorder.record_snapshot(dummy_snap)
            return appended

        record = self.fetch_once()
        appended_obs = record is not None and self.raw_writer.append(record)

        # Collect genuine L2 depth if configured
        if self.depth_recorder:
            snapshot = self.depth_recorder.fetch_snapshot_rest(limit=self.config.depth_limit)
            if snapshot:
                self.depth_recorder.record_snapshot(snapshot)

        return appended_obs

    def run(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        dry_run: bool = False,
    ) -> int:
        """Run collection loop with timing control, graceful shutdown, and metadata tracking."""
        start_time = time.time() if start_time is None else start_time
        if start_time > time.time() and not dry_run:
            time.sleep(start_time - time.time())

        collected = 0
        final_status = "COMPLETED"

        try:
            while not self._shutdown_requested:
                if end_time is not None and time.time() >= end_time:
                    break

                if self.collect_once(dry_run=dry_run):
                    collected += 1

                if dry_run:
                    break

                next_run = start_time + (collected * self.config.interval_seconds)
                sleep_time = max(0.0, next_run - time.time())

                # Sleep in small slices to respond promptly to shutdown signals
                step_sleep = 0.2
                elapsed = 0.0
                while elapsed < sleep_time and not self._shutdown_requested:
                    time.sleep(min(step_sleep, sleep_time - elapsed))
                    elapsed += step_sleep

            if self._shutdown_requested:
                final_status = "STOPPED_BY_SIGNAL"

        except KeyboardInterrupt:
            logger.info("Collector interrupted by user.")
            final_status = "STOPPED_BY_USER"
        except Exception as e:
            logger.error("Collector terminated with error: %s", e)
            append_error(self.error_writer, "run_loop_fatal", e)
            final_status = "FAILED"
        finally:
            self._finalize_session_metadata(status=final_status, collected_count=collected)

        return collected


def validate_session_data(
    raw_session_dir: Path,
    expected_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    """Validate data integrity for a single session directory.

    Checks:
    - Duplicate records and duplicate timestamps.
    - Strictly monotonic timestamp ordering.
    - Interval gaps exceeding 1.5x expected interval.
    - Malformed records (missing keys, invalid JSON).
    - Price and quantity validity (> 0 prices, >= 0 volumes, high >= low).
    - Level 2 orderbook integrity (uncrossed book, monotonic last_update_id).
    """
    raw_session_dir = Path(raw_session_dir)
    obs_path = raw_session_dir / "observations.jsonl"
    if not obs_path.exists():
        obs_path = raw_session_dir / "btc_observations.jsonl"

    depth_path = raw_session_dir / "orderbook_snapshots.jsonl"
    meta_path = raw_session_dir / "session_metadata.json"

    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    report: dict[str, Any] = {
        "session_dir": str(raw_session_dir),
        "validation_timestamp": utc_now(),
        "passed": True,
        "observations_present": obs_path.exists(),
        "depth_present": depth_path.exists(),
        "checks": {},
        "summary": {},
    }

    # 1. Validate Observations
    obs_records = jsonl_records(obs_path) if obs_path.exists() else []
    obs_valid_records = [r for r in obs_records if "_parse_error" not in r and not r.get("dry_run")]

    # Check parse errors
    parse_errors = [r for r in obs_records if "_parse_error" in r]
    report["checks"]["malformed_json"] = {
        "passed": len(parse_errors) == 0,
        "malformed_count": len(parse_errors),
    }

    # Check required fields and numerical validity
    required_obs = ["source", "symbol", "source_timestamp", "open", "high", "low", "close", "volume"]
    missing_fields_count = 0
    invalid_price_count = 0
    invalid_hl_count = 0
    invalid_vol_count = 0

    for r in obs_valid_records:
        if not all(r.get(f) is not None for f in required_obs):
            missing_fields_count += 1
            continue
        try:
            op = float(r["open"])
            hi = float(r["high"])
            lo = float(r["low"])
            cl = float(r["close"])
            vol = float(r["volume"])

            if op <= 0 or hi <= 0 or lo <= 0 or cl <= 0:
                invalid_price_count += 1
            if hi < lo or hi < op or hi < cl:
                invalid_hl_count += 1
            if vol < 0:
                invalid_vol_count += 1
        except (ValueError, TypeError):
            missing_fields_count += 1

    report["checks"]["required_fields"] = {
        "passed": missing_fields_count == 0,
        "missing_count": missing_fields_count,
    }
    report["checks"]["positive_prices"] = {
        "passed": invalid_price_count == 0,
        "invalid_count": invalid_price_count,
    }
    report["checks"]["high_low_validity"] = {
        "passed": invalid_hl_count == 0,
        "invalid_count": invalid_hl_count,
    }
    report["checks"]["non_negative_volume"] = {
        "passed": invalid_vol_count == 0,
        "invalid_count": invalid_vol_count,
    }

    # Duplicate records check (payload uniqueness)
    record_ids = [r.get("record_id") for r in obs_valid_records if r.get("record_id")]
    if len(record_ids) == len(obs_valid_records) and len(record_ids) > 0:
        dup_records = len(record_ids) - len(set(record_ids))
    else:
        basis_tuples = [
            (r.get("source"), r.get("symbol"), r.get("source_timestamp"), r.get("close"), r.get("volume"))
            for r in obs_valid_records
        ]
        dup_records = len(basis_tuples) - len(set(basis_tuples)) if not any(r.get("collected_at") for r in obs_valid_records) else 0

    report["checks"]["duplicate_records"] = {
        "passed": dup_records == 0,
        "duplicate_count": dup_records,
    }

    # Timestamp quality
    has_collected_at = any(r.get("collected_at") is not None for r in obs_valid_records)

    if has_collected_at:
        # Physical sampling timestamps: strictly monotonic, no duplicates
        ts_quality_sampling = timestamp_quality(obs_valid_records, "collected_at", expected_interval_seconds)
        report["checks"]["timestamp_ordering"] = {
            "passed": ts_quality_sampling["out_of_order_records"] == 0,
            "out_of_order_count": ts_quality_sampling["out_of_order_records"],
        }
        report["checks"]["duplicate_timestamps"] = {
            "passed": ts_quality_sampling["duplicate_timestamps"] == 0,
            "duplicate_count": ts_quality_sampling["duplicate_timestamps"],
        }
        # Allow reasonable network latency jitter (e.g. 1s poll might have 1.5s round trip)
        effective_gap = max(expected_interval_seconds * 2.0, expected_interval_seconds + 2.0)
        gap_quality = timestamp_quality(obs_valid_records, "collected_at", effective_gap)
        report["checks"]["missing_intervals"] = {
            "passed": gap_quality["gaps_over_expected"] == 0,
            "gaps_over_expected": gap_quality["gaps_over_expected"],
            "max_gap_seconds": gap_quality["max_gap_seconds"],
        }

        # Check candle bucket timestamps (source_timestamp)
        src_quality = timestamp_quality(obs_valid_records, "source_timestamp", None)
        candle_out_of_order = src_quality["out_of_order_records"]
        intra_candle_updates = src_quality["duplicate_timestamps"]
        unique_candles = len(obs_valid_records) - intra_candle_updates
        report["checks"]["candle_source_timestamps"] = {
            "passed": candle_out_of_order == 0,
            "out_of_order_count": candle_out_of_order,
            "intra_candle_updates": intra_candle_updates,
            "unique_candle_count": unique_candles,
        }
        primary_ts = ts_quality_sampling
    else:
        # Fallback when only source_timestamp is available
        ts_quality = timestamp_quality(obs_valid_records, "source_timestamp", expected_interval_seconds)
        report["checks"]["timestamp_ordering"] = {
            "passed": ts_quality["out_of_order_records"] == 0,
            "out_of_order_count": ts_quality["out_of_order_records"],
        }
        report["checks"]["duplicate_timestamps"] = {
            "passed": ts_quality["duplicate_timestamps"] == 0,
            "duplicate_count": ts_quality["duplicate_timestamps"],
        }
        report["checks"]["missing_intervals"] = {
            "passed": ts_quality["gaps_over_expected"] == 0,
            "gaps_over_expected": ts_quality["gaps_over_expected"],
            "max_gap_seconds": ts_quality["max_gap_seconds"],
        }
        primary_ts = ts_quality

    # 2. Validate Level 2 Depth Snapshots if present
    depth_records = jsonl_records(depth_path) if depth_path.exists() else []
    depth_valid = [r for r in depth_records if "_parse_error" not in r and not r.get("dry_run")]

    if depth_valid:
        crossed_count = 0
        non_monotonic_id = 0
        last_id = -1
        invalid_levels_count = 0

        for r in depth_valid:
            cur_id = r.get("last_update_id", 0)
            if cur_id <= last_id:
                non_monotonic_id += 1
            last_id = cur_id

            bids = r.get("bids", [])
            asks = r.get("asks", [])

            # Check prices and sizes
            for p, s in bids + asks:
                if float(p) <= 0 or float(s) < 0:
                    invalid_levels_count += 1

            if bids and asks:
                best_b = max(float(p) for p, _ in bids)
                best_a = min(float(p) for p, _ in asks)
                if best_b > best_a:
                    crossed_count += 1

        report["checks"]["l2_uncrossed_books"] = {
            "passed": crossed_count == 0,
            "crossed_count": crossed_count,
        }
        report["checks"]["l2_monotonic_sequence"] = {
            "passed": non_monotonic_id == 0,
            "non_monotonic_count": non_monotonic_id,
        }
        report["checks"]["l2_valid_levels"] = {
            "passed": invalid_levels_count == 0,
            "invalid_count": invalid_levels_count,
        }

    # Overall pass determination
    all_passed = all(check["passed"] for check in report["checks"].values())
    report["passed"] = all_passed
    report["summary"] = {
        "total_observations_raw": len(obs_records),
        "valid_observations": len(obs_valid_records),
        "total_depth_snapshots": len(depth_valid),
        "first_timestamp": primary_ts.get("first_timestamp"),
        "last_timestamp": primary_ts.get("last_timestamp"),
        "session_status": metadata.get("status", "UNKNOWN"),
    }
    return report


def process_session_data(
    raw_session_dir: Path,
    processed_session_dir: Path,
    expected_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    """Clean and export raw session data to Parquet without modifying raw inputs.

    Generates:
    - observations.parquet
    - orderbook_snapshots.parquet (if L2 depth exists)
    - session_report.json
    - session_report.md
    """
    raw_session_dir = Path(raw_session_dir)
    processed_session_dir = Path(processed_session_dir)
    processed_session_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Data Validation
    validation_report = validate_session_data(raw_session_dir, expected_interval_seconds)

    # 2. Process Observations
    obs_path = raw_session_dir / "observations.jsonl"
    if not obs_path.exists():
        obs_path = raw_session_dir / "btc_observations.jsonl"

    obs_records = jsonl_records(obs_path) if obs_path.exists() else []
    obs_valid = [r for r in obs_records if "_parse_error" not in r and not r.get("dry_run")]
    for r in obs_valid:
        r.setdefault("collected_at", r.get("source_timestamp") or utc_now())

    required_obs = ["source", "symbol", "source_timestamp", "collected_at", "open", "high", "low", "close", "volume"]
    clean_obs = [r for r in obs_valid if all(r.get(f) is not None for f in required_obs)]

    obs_df = pd.DataFrame(clean_obs)
    if not obs_df.empty:
        obs_df["source_timestamp"] = pd.to_datetime(obs_df["source_timestamp"], utc=True)
        if "open_time" in obs_df.columns:
            obs_df["open_time"] = pd.to_datetime(obs_df["open_time"], utc=True)
        if "collected_at" in obs_df.columns:
            obs_df["collected_at"] = pd.to_datetime(obs_df["collected_at"], utc=True)

        for col in ["open", "high", "low", "close", "volume"]:
            obs_df[col] = pd.to_numeric(obs_df[col], errors="coerce")

        if "source_payload" in obs_df.columns:
            obs_df["source_payload"] = obs_df["source_payload"].map(
                lambda v: json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else str(v)
            )

        # Sort chronologically by arrival time if present, else exchange timestamp
        sort_col = "collected_at" if "collected_at" in obs_df.columns else "source_timestamp"
        obs_df = obs_df.sort_values(sort_col)

        # Deduplicate genuine duplicate observations without dropping valid intra-candle updates:
        if "record_id" in obs_df.columns and obs_df["record_id"].notna().all():
            obs_df = obs_df.drop_duplicates(subset=["record_id"], keep="first")
        elif "collected_at" in obs_df.columns and obs_df["collected_at"].notna().all():
            obs_df = obs_df.drop_duplicates(subset=["source", "symbol", "collected_at"], keep="first")
        else:
            obs_df = obs_df.drop_duplicates(subset=["source", "symbol", "source_timestamp"], keep="first")

        obs_df = obs_df.reset_index(drop=True)
        obs_df.to_parquet(processed_session_dir / "observations.parquet", index=False)

    # 3. Process L2 Orderbook Depth if present
    depth_path = raw_session_dir / "orderbook_snapshots.jsonl"
    depth_records = jsonl_records(depth_path) if depth_path.exists() else []
    depth_valid = [r for r in depth_records if "_parse_error" not in r and not r.get("dry_run")]

    if depth_valid:
        depth_rows = []
        book = L2OrderBook(symbol=depth_valid[0].get("symbol", "BTCUSDT"))

        for r in depth_valid:
            try:
                snap = L2Snapshot(
                    symbol=r["symbol"],
                    last_update_id=int(r["last_update_id"]),
                    bids=tuple((float(p), float(s)) for p, s in r.get("bids", [])),
                    asks=tuple((float(p), float(s)) for p, s in r.get("asks", [])),
                    exchange_timestamp=r.get("exchange_timestamp"),
                    received_at=r.get("received_at", utc_now()),
                    source=r.get("source", "binance_l2_rest"),
                )
                book.apply_snapshot(snap)
                state = book.snapshot_state(levels=10)
                depth_rows.append({
                    "symbol": state["symbol"],
                    "last_update_id": state["last_update_id"],
                    "best_bid": state["best_bid"],
                    "best_ask": state["best_ask"],
                    "mid_price": state["mid_price"],
                    "spread": state["spread"],
                    "spread_bps": state["spread_bps"],
                    "order_book_imbalance_5": state["order_book_imbalance_5"],
                    "microprice_5": state["microprice_5"],
                    "bid_depth_total": state["bid_depth_total"],
                    "ask_depth_total": state["ask_depth_total"],
                    "top_bids_json": json.dumps(state["top_bids"]),
                    "top_asks_json": json.dumps(state["top_asks"]),
                    "exchange_timestamp": state["exchange_timestamp"],
                    "received_at": state["received_at"],
                })
            except Exception as e:
                logger.warning("Failed to process depth snapshot row: %s", e)

        if depth_rows:
            depth_df = pd.DataFrame(depth_rows)
            depth_df["received_at"] = pd.to_datetime(depth_df["received_at"], utc=True)
            if depth_df["exchange_timestamp"].notna().any():
                depth_df["exchange_timestamp"] = pd.to_datetime(depth_df["exchange_timestamp"], utc=True)
            depth_df = depth_df.sort_values("received_at").drop_duplicates(
                subset=["symbol", "last_update_id"], keep="first"
            ).reset_index(drop=True)
            depth_df.to_parquet(processed_session_dir / "orderbook_snapshots.parquet", index=False)

    # 4. Save Session Validation Reports
    report_json_path = processed_session_dir / "session_report.json"
    report_json_path.write_text(json.dumps(validation_report, indent=2), encoding="utf-8")

    report_md_path = processed_session_dir / "session_report.md"
    md_lines = [
        f"# PredAlpha Phase 3 — Session Validation Report: `{raw_session_dir.name}`\n",
        f"- **Validation Time**: {validation_report['validation_timestamp']}",
        f"- **Overall Status**: {'✅ PASSED' if validation_report['passed'] else '❌ FAILED'}",
        f"- **Raw Observations**: {validation_report['summary']['total_observations_raw']}",
        f"- **Valid Observations**: {validation_report['summary']['valid_observations']}",
        f"- **Depth Snapshots**: {validation_report['summary']['total_depth_snapshots']}",
        f"- **Time Range**: {validation_report['summary']['first_timestamp']} to {validation_report['summary']['last_timestamp']}\n",
        "## Integrity Checks Breakdown\n",
        "| Check Name | Status | Details |",
        "|:-----------|:-------|:--------|",
    ]
    for name, c in validation_report["checks"].items():
        status_str = "✅ PASS" if c["passed"] else "❌ FAIL"
        details_str = ", ".join(f"{k}={v}" for k, v in c.items() if k != "passed")
        md_lines.append(f"| `{name}` | {status_str} | {details_str} |")

    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return validation_report


# Backward-compatible helper function
def process_track_a_raw(raw_dir: Path, processed_dir: Path, expected_gap_seconds: float) -> dict[str, Any]:
    """Backward-compatible wrapper for processing single raw directory."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    records = [
        record
        for record in jsonl_records(raw_dir / "btc_observations.jsonl")
        if "_parse_error" not in record and not record.get("dry_run")
    ]
    required = ["source", "symbol", "source_timestamp", "collected_at", "open", "high", "low", "close", "volume"]
    valid = [record for record in records if all(record.get(field) is not None for field in required)]
    frame = pd.DataFrame(valid)
    if not frame.empty:
        frame["source_timestamp"] = pd.to_datetime(frame["source_timestamp"], utc=True)
        if "source_payload" in frame:
            frame["source_payload"] = frame["source_payload"].map(
                lambda value: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
            )
        frame = frame.sort_values("source_timestamp").drop_duplicates(
            ["source", "symbol", "source_timestamp"], keep="first"
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_dir / "btc_observations.parquet", index=False)
    quality = timestamp_quality(valid, "source_timestamp", expected_gap_seconds)
    quality.update({
        "raw_records": len(records),
        "valid_records": len(valid),
        "missing_required_fields": len(records) - len(valid),
        "processed_rows": len(frame),
        "labels_created": False,
    })
    (processed_dir / "status.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality
