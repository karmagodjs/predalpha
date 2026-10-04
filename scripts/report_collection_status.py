"""Create a human-readable collection health report from immutable raw logs."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.collection_common import jsonl_records, timestamp_quality
from market_data.track_b_collector import track_b_status

root = ROOT
track_a = root / "data/raw/track_a"
track_b = root / "data/raw/track_b"
a_records = jsonl_records(track_a / "btc_observations.jsonl")
a_errors = jsonl_records(track_a / "collection_errors.jsonl")
a_quality = timestamp_quality(a_records, "collected_at")
b_status = track_b_status(track_b)
report = "# Data collection status\n\n"
report += "## Track A\n\n" + json.dumps({**a_quality, "collection_errors": len(a_errors)}, indent=2) + "\n\n"
report += "## Track B\n\n" + json.dumps(b_status, indent=2) + "\n\n"
report += "Repeated snapshots are events, not independent market samples. A market counts as resolved only when official CLOB settlement evidence has exactly one token marked `winner: true`.\n"
(root / "reports/data_collection_status.md").write_text(report, encoding="utf-8")
print(root / "reports/data_collection_status.md")
