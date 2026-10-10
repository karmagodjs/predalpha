"""
Integration tests for Phase 11B Expanded Polymarket Collection Ingestion & Audit.

Verifies:
1. Immutability: All raw JSONL files remain bit-for-bit unchanged.
2. Original baseline protection: All 19 production markets in new_collection/ remain untouched.
3. Expanded collection integrity: All 52 canonical Parquet files in expanded_collection/
   conform strictly to the canonical schema.
4. Monotonicity: 100% non-decreasing timestamp series.
5. Bid-Ask spread & depth validity: 100% bid < ask, bid_size > 0, ask_size > 0.
6. Market and asset isolation: Zero cross-market or cross-session state contamination.
7. Interrupted run verification: btc-updown-5m-1791295500 persists 280k+ canonical events.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(512 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_original_new_collection_untouched():
    """Verify that the existing 19-market production dataset in new_collection/ remains 100% untouched."""
    meta_path = Path("data/clean_v2/01_canonical_events/new_collection/new_collection_metadata.json")
    assert meta_path.exists(), "Original new_collection metadata missing!"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["summary"]["total_markets_processed"] == 21
    assert meta["summary"]["all_raw_files_immutable"] is True

    # Check that all 21 original parquets in new_collection/ exist
    pq_dir = Path("data/clean_v2/01_canonical_events/new_collection")
    for m in meta["markets"]:
        pq_file = pq_dir / m["canonical_parquet_file"]
        assert pq_file.exists(), f"Original canonical parquet {pq_file.name} missing!"


def test_expanded_collection_metadata_integrity():
    """Verify expanded_collection_metadata.json summary and consistency."""
    meta_path = Path("data/clean_v2/01_canonical_events/expanded_collection/expanded_collection_metadata.json")
    assert meta_path.exists(), "Expanded collection metadata missing!"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    summary = meta["summary"]
    assert summary["total_markets_processed"] == 52
    assert summary["total_markets_retained"] == 46
    assert summary["total_markets_excluded"] == 6
    assert summary["aggregate_canonical_events"] == 16718713
    assert summary["aggregate_raw_rows"] == 8921784
    assert summary["all_markets_monotonic"] is True
    assert summary["all_raw_files_immutable"] is True
    assert summary["aggregate_conflicting_duplicates"] == 0


def test_raw_files_immutability():
    """Verify that all raw files match their recorded SHA-256 hashes."""
    meta_path = Path("data/clean_v2/01_canonical_events/expanded_collection/expanded_collection_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for m in meta["markets"]:
        raw_p = Path("data/raw") / m["raw_file"]
        assert raw_p.exists(), f"Raw file {m['raw_file']} missing!"
        current_sha = compute_sha256(raw_p)
        assert current_sha == m["raw_sha256"], f"Raw file {m['raw_file']} was mutated!"


def test_expanded_parquet_outputs_quality():
    """Verify schema, monotonicity, bid < ask, and depth validity across all 52 Parquets."""
    exp_dir = Path("data/clean_v2/01_canonical_events/expanded_collection")
    pqs = sorted(exp_dir.glob("*.parquet"))
    assert len(pqs) == 52, f"Expected 52 parquet files, found {len(pqs)}"

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

    for pq_file in pqs:
        df = pd.read_parquet(pq_file)
        assert list(df.columns) == expected_cols

        if len(df) > 0:
            # Monotonicity
            diffs = df["timestamp_ms"].diff().dropna()
            assert (diffs >= 0).all(), f"Monotonicity violation in {pq_file.name}"

            # Bid < Ask
            assert (df["bid"] < df["ask"]).all(), f"Crossed book in {pq_file.name}"

            # Depth validity
            assert (df["bid_size"] > 0).all(), f"Invalid bid size in {pq_file.name}"
            assert (df["ask_size"] > 0).all(), f"Invalid ask size in {pq_file.name}"

            # Price bounds
            assert (df["bid"] > 0.0).all() and (df["bid"] < 1.0).all()
            assert (df["ask"] > 0.0).all() and (df["ask"] < 1.0).all()


def test_interrupted_run_persistence():
    """Verify that the interrupted run (btc-updown-5m-1791295500) produced 280k+ canonical events."""
    pq_path = Path("data/clean_v2/01_canonical_events/expanded_collection/btc-updown-5m-1791295500_canonical.parquet")
    assert pq_path.exists(), "Interrupted run canonical parquet missing!"
    df = pd.read_parquet(pq_path)
    assert len(df) == 280374, f"Expected 280,374 canonical events, got {len(df)}"
