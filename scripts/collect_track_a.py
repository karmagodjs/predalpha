"""CLI for multi-session Track A collection, Level 2 depth capture, and offline session processing."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.track_a_collector import (
    TrackACollectionConfig,
    TrackACollector,
    process_session_data,
    process_track_a_raw,
    validate_session_data,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for Track A collector."""
    parser = argparse.ArgumentParser(description="Track A Multi-Session and L2 Orderbook Collector")
    parser.add_argument(
        "--session-id",
        help="Session identifier (e.g. session_20261004_01). If omitted, runs in legacy single-folder mode.",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Market symbol to collect (default: BTCUSDT)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("PREDALPHA_TRACK_A_RAW_DIR", "data/raw/track_a"),
        help="Base output directory for raw data",
    )
    parser.add_argument(
        "--processed-dir",
        default=os.getenv("PREDALPHA_TRACK_A_PROCESSED_DIR", "data/processed/track_a"),
        help="Base directory for processed data",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("PREDALPHA_TRACK_A_INTERVAL_SECONDS", "60")),
        help="Collection interval in seconds",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Duration to collect in seconds (0 = run indefinitely until stopped)",
    )
    parser.add_argument(
        "--start-time",
        help="UTC ISO-8601 start time, e.g. 2026-10-04T12:00:00Z",
    )
    parser.add_argument(
        "--end-time",
        help="UTC ISO-8601 end time; overrides duration when supplied",
    )
    parser.add_argument(
        "--capture-l2",
        action="store_true",
        help="Enable genuine Level 2 orderbook depth snapshot capture",
    )
    parser.add_argument(
        "--depth-limit",
        type=int,
        default=100,
        help="Number of L2 depth levels to capture (5, 10, 20, 50, 100, 500, 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a dry run collection step without network requests",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Process raw data into parquet without running collection",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate session integrity without running collection or processing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI workflow based on parsed arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_base = Path(args.output_dir)
    proc_base = Path(args.processed_dir)

    # Resolve session directory strictly without touching unrelated data
    if args.session_id:
        if raw_base.name == "sessions":
            session_raw = raw_base / args.session_id
            session_proc = proc_base / args.session_id
        else:
            session_raw = raw_base / "sessions" / args.session_id
            session_proc = proc_base / "sessions" / args.session_id
    else:
        session_raw = raw_base
        session_proc = proc_base

    config = TrackACollectionConfig(
        symbol=args.symbol,
        interval_seconds=args.interval_seconds,
        output_dir=raw_base,
        processed_dir=proc_base,
        session_id=args.session_id,
        capture_l2_depth=args.capture_l2,
        depth_limit=args.depth_limit,
        duration_seconds=args.duration_seconds,
    )

    if args.validate_only:
        if not session_raw.exists():
            raise FileNotFoundError(f"Cannot validate session: raw directory does not exist: {session_raw}")
        logging.info("Validating session integrity at: %s", session_raw)
        report = validate_session_data(session_raw, expected_interval_seconds=config.interval_seconds)
        logging.info("Validation report:\n%s", json.dumps(report, indent=2))
        return 0 if report.get("passed", False) else 1

    elif args.process_only:
        if not session_raw.exists():
            raise FileNotFoundError(f"Cannot process session: raw directory does not exist: {session_raw}")
        logging.info("Processing session '%s' from %s to %s", args.session_id or "default", session_raw, session_proc)
        if args.session_id:
            res = process_session_data(session_raw, session_proc, expected_interval_seconds=config.interval_seconds)
            logging.info("Session processed successfully: %s", json.dumps(res.get("summary", {}), indent=2))
            return 0 if res.get("passed", False) else 1
        else:
            quality = process_track_a_raw(session_raw, session_proc, config.interval_seconds)
            logging.info("Legacy processing completed: %s", quality)
            return 0

    else:
        parse_time = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None
        start = parse_time(args.start_time)
        end = parse_time(args.end_time) or (time.time() + args.duration_seconds if args.duration_seconds else None)
        if start and end and end <= start:
            parser.error("--end-time must be later than --start-time")

        collector = TrackACollector(config)
        logging.info(
            "Starting Track A Collector (Session: %s, Symbol: %s, Interval: %.1fs, L2 Depth: %s)",
            args.session_id or "default",
            args.symbol,
            args.interval_seconds,
            args.capture_l2,
        )
        collected = collector.run(start_time=start, end_time=end, dry_run=args.dry_run)
        logging.info("Collection finished. Recorded %d observations.", collected)

        # Automatically process and validate if a session ID was given
        if args.session_id and not args.dry_run:
            logging.info("Auto-processing session to: %s", session_proc)
            process_session_data(session_raw, session_proc, expected_interval_seconds=config.interval_seconds)

        return 0


if __name__ == "__main__":
    sys.exit(main())
