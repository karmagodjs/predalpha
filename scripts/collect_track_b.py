"""CLI for metadata-first Track B market capture and settlement refresh."""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.track_b_collector import TrackBCollectionConfig, TrackBCollector, capture_live_websocket, track_b_status, verified_mapping

parser = argparse.ArgumentParser()
parser.add_argument("condition_ids", nargs="*")
parser.add_argument("--output-dir", default=os.getenv("PREDALPHA_TRACK_B_RAW_DIR", "data/raw/track_b"))
parser.add_argument("--discover", action="store_true")
parser.add_argument("--settlement-only", action="store_true")
parser.add_argument("--max-messages", type=int, default=0, help="Record this many live websocket messages after metadata verification (0 = metadata only).")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
collector = TrackBCollector(TrackBCollectionConfig(output_dir=Path(args.output_dir)))
ids = args.condition_ids
if args.discover:
    ids += [str(m.get("conditionId")) for m in collector.discover() if m.get("conditionId")]
for condition_id in dict.fromkeys(ids):
    metadata = collector.initialize_market(condition_id)
    logging.info("Verified mapping for %s: %s", condition_id, verified_mapping(metadata))
    if args.max_messages:
        logging.info("Recorded %s events", asyncio.run(capture_live_websocket(collector, condition_id, verified_mapping(metadata), args.max_messages)))
    if args.settlement_only:
        logging.info("Settlement: %s", collector.capture_settlement(condition_id))
print(json.dumps(track_b_status(Path(args.output_dir)), indent=2))
