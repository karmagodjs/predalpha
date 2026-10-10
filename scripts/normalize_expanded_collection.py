"""
Normalize and audit raw Polymarket data for the expanded collection (Phase 11B).

Processes any raw JSONL files in data/raw that do not yet have canonical parquets
in data/clean_v2/01_canonical_events/expanded_collection/.
Computes full provenance, immutability SHA-256 verification, and audit metadata.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pipeline_v2.ingestion.ingest_new_collection import (
    compute_file_sha256,
    format_ms_to_iso,
    generate_markdown_report,
    process_single_market,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("normalize_expanded_collection")


def main() -> None:
    raw_dir = Path("data/raw")
    output_dir = Path("data/clean_v2/01_canonical_events/expanded_collection")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_files = sorted(raw_dir.glob("btc-updown-5m-*.jsonl"))
    logger.info(f"Discovered {len(all_raw_files)} total raw JSONL files in {raw_dir}")

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # Exclude any market still in progress
    closed_files = []
    for rf in all_raw_files:
        meta_p = rf.with_suffix(".meta.json")
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            slot_end = meta.get("slot_end_utc")
            if slot_end:
                end_dt = datetime.datetime.fromisoformat(slot_end)
                if end_dt > now_utc:
                    logger.info(f"Skipping active in-progress market: {rf.name}")
                    continue
        except Exception:
            pass
        if rf.stat().st_size == 0:
            continue
        closed_files.append(rf)

    logger.info(f"Total completed markets available: {len(closed_files)}")

    start_total = time.time()
    per_market_records = []

    for idx, raw_file in enumerate(closed_files, start=1):
        out_parquet = output_dir / f"{raw_file.stem}_canonical.parquet"
        if out_parquet.exists() and out_parquet.stat().st_size > 0:
            logger.info(f"[{idx}/{len(closed_files)}] Already normalized: {raw_file.name}")
            continue

        logger.info(f"[{idx}/{len(closed_files)}] Normalizing {raw_file.name}...")
        rec = process_single_market(raw_file, output_dir)
        per_market_records.append(rec)

    elapsed_total = time.time() - start_total
    logger.info(f"Normalized {len(per_market_records)} new markets in {elapsed_total:.2f}s")


if __name__ == "__main__":
    main()
