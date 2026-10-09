"""
Unit and integration tests for Phase 11B New Polymarket Collection Ingestion.

Verifies:
1. Session isolation: No orderbook state or events leak across market session boundaries.
2. Deterministic reproducibility: Normalizing a raw session yields identical canonical events.
3. Raw file immutability: Raw files in data/raw/ are bit-for-bit unchanged.
4. Schema & Data Quality: Parquet outputs conform strictly to Phase 11 canonical schema,
   strict monotonicity, bid < ask spread enforcement, and positive depth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline_v2.ingestion.event_normalizer import EventNormalizer


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(512 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_session_isolation():
    """
    Verify that normalizing sequential markets maintains strict session isolation
    and order books are cleanly reset without cross-market contamination.
    """
    raw_files = sorted(Path("data/raw").glob("btc-updown-5m-*.jsonl"))
    if len(raw_files) < 2:
        pytest.skip("Requires at least 2 raw market files.")

    file_a = raw_files[1]  # 1791204600
    file_b = raw_files[2]  # 1791204900

    with open(file_a, "r", encoding="utf-8") as f:
        lines_a = [f.readline() for _ in range(500)]
    with open(file_b, "r", encoding="utf-8") as f:
        lines_b = [f.readline() for _ in range(500)]

    norm = EventNormalizer()

    # Pass 1: Run file A
    events_a_pass1 = norm.normalize_stream(lines_a, source_file=file_a.name)
    mkt_a_assets = set(e.asset_id for e in events_a_pass1)

    # Pass 2: Run file B with state reset
    norm.reset_state()
    events_b = norm.normalize_stream(lines_b, source_file=file_b.name)
    mkt_b_assets = set(e.asset_id for e in events_b)

    # Assets in market A and market B must be completely disjoint
    assert len(mkt_a_assets.intersection(mkt_b_assets)) == 0
    assert all(e.source_file == file_b.name for e in events_b)

    # Pass 3: Run file A again in a fresh instance vs reused instance
    norm_fresh = EventNormalizer()
    events_a_fresh = norm_fresh.normalize_stream(lines_a, source_file=file_a.name)

    norm.reset_state()
    events_a_pass2 = norm.normalize_stream(lines_a, source_file=file_a.name)

    assert len(events_a_pass1) == len(events_a_pass2) == len(events_a_fresh)
    for e1, e2 in zip(events_a_pass1, events_a_pass2):
        assert e1 == e2


def test_deterministic_reproducibility():
    """Verify that normalizing the same input stream produces identical outputs."""
    raw_files = sorted(Path("data/raw").glob("btc-updown-5m-*.jsonl"))
    if not raw_files:
        pytest.skip("No raw files found.")

    sample_file = raw_files[1]
    with open(sample_file, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(1000)]

    norm1 = EventNormalizer()
    events1 = norm1.normalize_stream(lines, source_file="rep_test")

    norm2 = EventNormalizer()
    events2 = norm2.normalize_stream(lines, source_file="rep_test")

    assert len(events1) == len(events2)
    for i in range(len(events1)):
        assert events1[i] == events2[i]
        assert events1[i].timestamp_ms == events2[i].timestamp_ms
        assert events1[i].sequence_id == events2[i].sequence_id
        assert events1[i].bid == events2[i].bid
        assert events1[i].ask == events2[i].ask
        assert events1[i].bid_size == events2[i].bid_size
        assert events1[i].ask_size == events2[i].ask_size


def test_raw_file_immutability():
    """Verify that every raw JSONL file in data/raw/ remains untouched."""
    meta_path = Path("data/clean_v2/01_canonical_events/new_collection/new_collection_metadata.json")
    if not meta_path.exists():
        pytest.skip("Metadata file not yet generated.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for market_rec in meta["markets"]:
        raw_file = Path("data/raw") / market_rec["raw_file"]
        assert raw_file.exists(), f"Raw file {raw_file} missing!"
        expected_sha = market_rec["raw_sha256"]
        current_sha = compute_sha256(raw_file)
        assert current_sha == expected_sha, f"Raw file {raw_file.name} was altered!"


def test_canonical_parquet_outputs_integrity():
    """Verify schema, bid < ask enforcement, and monotonicity across output parquet files."""
    output_dir = Path("data/clean_v2/01_canonical_events/new_collection")
    parquet_files = sorted(output_dir.glob("*.parquet"))

    if not parquet_files:
        pytest.skip("No canonical parquet files found.")

    expected_cols = [
        "timestamp_ms",
        "sequence_id",
        "market_id",
        "asset_id",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "source_file",
        "source_line",
        "source_record_idx",
    ]

    for pq_file in parquet_files:
        df = pd.read_parquet(pq_file)
        assert list(df.columns) == expected_cols
        assert df["timestamp_ms"].dtype == "int64"
        assert df["sequence_id"].dtype == "int64"

        if len(df) > 0:
            # Monotonic non-decreasing timestamps
            assert (df["timestamp_ms"].diff().dropna() >= 0).all(), (
                f"Non-monotonic timestamps in {pq_file.name}"
            )
            # Bid < Ask strictly enforced
            assert (df["bid"] < df["ask"]).all(), f"Crossed or locked quotes in {pq_file.name}"
            # Positive depths strictly enforced
            assert (df["bid_size"] > 0).all(), f"Non-positive bid size in {pq_file.name}"
            assert (df["ask_size"] > 0).all(), f"Non-positive ask size in {pq_file.name}"
            # Prices strictly between 0 and 1
            assert (df["bid"] > 0.0).all() and (df["bid"] < 1.0).all()
            assert (df["ask"] > 0.0).all() and (df["ask"] < 1.0).all()
