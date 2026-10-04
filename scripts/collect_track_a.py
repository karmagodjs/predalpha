"""CLI for Track A collection and separate raw processing."""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.track_a_collector import TrackACollectionConfig, TrackACollector, process_track_a_raw

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default=os.getenv("PREDALPHA_TRACK_A_RAW_DIR", "data/raw/track_a"))
parser.add_argument("--processed-dir", default=os.getenv("PREDALPHA_TRACK_A_PROCESSED_DIR", "data/processed/track_a"))
parser.add_argument("--interval-seconds", type=float, default=float(os.getenv("PREDALPHA_TRACK_A_INTERVAL_SECONDS", "60")))
parser.add_argument("--duration-seconds", type=float, default=0)
parser.add_argument("--start-time", help="UTC ISO-8601 start time, e.g. 2026-10-04T12:00:00Z")
parser.add_argument("--end-time", help="UTC ISO-8601 end time; overrides duration when supplied")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--process-only", action="store_true")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
config = TrackACollectionConfig(interval_seconds=args.interval_seconds, output_dir=Path(args.output_dir), processed_dir=Path(args.processed_dir))
if args.process_only:
    logging.info("%s", process_track_a_raw(config.output_dir, config.processed_dir, config.interval_seconds))
else:
    parse_time = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else None
    start = parse_time(args.start_time)
    end = parse_time(args.end_time) or (time.time() + args.duration_seconds if args.duration_seconds else None)
    if start and end and end <= start:
        parser.error("--end-time must be later than --start-time")
    logging.info("Collected %s observations", TrackACollector(config).run(start_time=start, end_time=end, dry_run=args.dry_run))
