"""
Unit tests for Pipeline V2 Canonical Event Normalizer.

Covers all Phase 11 requirements:
- Malformed JSON handling
- Invalid bid/ask prices
- Crossed book detection (bid >= ask)
- Non-positive depth rejection
- Deduplication (identical vs conflicting payload)
- Deterministic ordering (timestamp_ms ASC, sequence_id ASC)
- Exact integer timestamp preservation (int64)
- Provenance tracking (source_file, source_line, source_record_idx)
- Real dataset integration smoke test
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from pipeline_v2.ingestion.event_normalizer import (
    CanonicalEvent,
    EventNormalizer,
    NormalizationReport,
    RejectReason,
)


@pytest.fixture
def normalizer():
    return EventNormalizer(
        target_asset_id=None,
        min_price=0.0,
        max_price=1.0,
    )


def test_malformed_json(normalizer):
    """FAIL malformed records: unparseable json, empty lines, non-dict payloads."""
    raw_lines = [
        "",  # Empty
        "   ",  # Whitespace
        "{not valid json",  # Syntax error
        "12345",  # Non-dict JSON
        '["a", "b"]',  # JSON array at top level with non-dict
    ]
    events = normalizer.normalize_stream(raw_lines, source_file="test_malformed.jsonl")

    assert len(events) == 0
    assert normalizer.report.valid_canonical_events == 0
    assert normalizer.report.rejected_by_reason.get(RejectReason.MALFORMED_JSON.value, 0) == 1
    assert normalizer.report.rejected_by_reason.get(RejectReason.EMPTY_RECORD.value, 0) >= 2


def test_invalid_bid_ask(normalizer):
    """FAIL records with missing, non-numeric, or impossible prices."""
    records = [
        # Missing bid
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": None, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        # Non-numeric ask
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": "not_a_num", "bid_size": 10.0, "ask_size": 10.0},
        # Impossible negative price
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": -0.10, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        # Impossible price 0.0
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.0, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        # Impossible price >= 1.0 in prediction markets
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 1.00, "bid_size": 10.0, "ask_size": 10.0},
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 1.25, "bid_size": 10.0, "ask_size": 10.0},
        # NaN / Inf prices
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": float("nan"), "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": float("inf"), "bid_size": 10.0, "ask_size": 10.0},
    ]
    raw_lines = [json.dumps(r) for r in records]
    events = normalizer.normalize_stream(raw_lines, source_file="test_invalid_prices.jsonl")

    assert len(events) == 0
    assert normalizer.report.rejected_by_reason.get(RejectReason.MISSING_PRICE.value, 0) == 1
    assert normalizer.report.rejected_by_reason.get(RejectReason.INVALID_PRICE.value, 0) >= 3
    assert normalizer.report.rejected_by_reason.get(RejectReason.IMPOSSIBLE_PRICE.value, 0) >= 4


def test_crossed_book(normalizer):
    """FAIL records where bid >= ask (crossed or locked book)."""
    records = [
        # Crossed: bid > ask
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.60, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        # Locked: bid == ask
        {"timestamp_ms": 1700000000001, "market_id": "m1", "asset_id": "a1", "bid": 0.50, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        # Valid quote (bid < ask)
        {"timestamp_ms": 1700000000002, "market_id": "m1", "asset_id": "a1", "bid": 0.49, "ask": 0.51, "bid_size": 10.0, "ask_size": 10.0},
    ]
    raw_lines = [json.dumps(r) for r in records]
    events = normalizer.normalize_stream(raw_lines, source_file="test_crossed.jsonl")

    assert len(events) == 1
    assert events[0].bid == 0.49
    assert events[0].ask == 0.51
    assert normalizer.report.rejected_by_reason.get(RejectReason.CROSSED_BOOK.value, 0) == 2


def test_non_positive_depth(normalizer):
    """FAIL records where bid_size <= 0 or ask_size <= 0 or sizes are missing/invalid."""
    records = [
        # Zero bid size
        {"timestamp_ms": 1700000000000, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 0.0, "ask_size": 10.0},
        # Negative ask size
        {"timestamp_ms": 1700000000001, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 10.0, "ask_size": -5.0},
        # Missing bid size
        {"timestamp_ms": 1700000000002, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": None, "ask_size": 10.0},
        # Invalid ask size string
        {"timestamp_ms": 1700000000003, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 10.0, "ask_size": "bad"},
        # Valid sizes
        {"timestamp_ms": 1700000000004, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 100.5, "ask_size": 50.2},
    ]
    raw_lines = [json.dumps(r) for r in records]
    events = normalizer.normalize_stream(raw_lines, source_file="test_depth.jsonl")

    assert len(events) == 1
    assert events[0].bid_size == 100.5
    assert events[0].ask_size == 50.2
    assert normalizer.report.rejected_by_reason.get(RejectReason.NON_POSITIVE_DEPTH.value, 0) == 2
    assert normalizer.report.rejected_by_reason.get(RejectReason.MISSING_SIZE.value, 0) == 1
    assert normalizer.report.rejected_by_reason.get(RejectReason.INVALID_SIZE.value, 0) == 1


def test_duplicate_events(normalizer):
    """
    Test deduplication policy:
    duplicate key: (market_id, asset_id, timestamp_ms, sequence_id)
    - Keep first received event.
    - If duplicate payloads match -> increment identical duplicate count.
    - If duplicate payloads differ -> increment conflicting duplicate count and log audit entry.
    """
    base = {
        "timestamp_ms": 1700000000000,
        "sequence_id": 101,
        "market_id": "m1",
        "asset_id": "a1",
        "bid": 0.45,
        "ask": 0.55,
        "bid_size": 100.0,
        "ask_size": 200.0,
    }
    identical_dup = dict(base)
    conflicting_dup = dict(base)
    conflicting_dup["bid"] = 0.46  # Conflicting price for same key!

    raw_lines = [
        json.dumps(base),
        json.dumps(identical_dup),
        json.dumps(conflicting_dup),
    ]

    events = normalizer.normalize_stream(raw_lines, source_file="test_duplicates.jsonl")

    # Only first event kept
    assert len(events) == 1
    assert events[0].bid == 0.45
    assert normalizer.report.duplicates_dropped_identical == 1
    assert normalizer.report.duplicates_dropped_conflicting == 1
    assert normalizer.report.total_duplicates_dropped == 2

    # Audit conflict entry verified
    assert len(normalizer.report.conflicting_duplicates) == 1
    conflict = normalizer.report.conflicting_duplicates[0]
    assert conflict.key == ("m1", "a1", 1700000000000, 101)
    assert conflict.first_payload[0] == 0.45
    assert conflict.conflicting_payload[0] == 0.46


def test_deterministic_ordering(normalizer):
    """
    Test sorting by:
        timestamp_ms ASC
        sequence_id ASC
    Regardless of raw line arrival sequence.
    """
    records = [
        {"timestamp_ms": 1700000003000, "sequence_id": 1, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0},
        {"timestamp_ms": 1700000001000, "sequence_id": 5, "market_id": "m1", "asset_id": "a1", "bid": 0.41, "ask": 0.51, "bid_size": 10.0, "ask_size": 10.0},
        {"timestamp_ms": 1700000001000, "sequence_id": 2, "market_id": "m1", "asset_id": "a1", "bid": 0.42, "ask": 0.52, "bid_size": 10.0, "ask_size": 10.0},
        {"timestamp_ms": 1700000002000, "sequence_id": 1, "market_id": "m1", "asset_id": "a1", "bid": 0.43, "ask": 0.53, "bid_size": 10.0, "ask_size": 10.0},
    ]
    raw_lines = [json.dumps(r) for r in records]
    events = normalizer.normalize_stream(raw_lines, source_file="test_ordering.jsonl")

    assert len(events) == 4
    expected_order = [
        (1700000001000, 2),
        (1700000001000, 5),
        (1700000002000, 1),
        (1700000003000, 1),
    ]
    actual_order = [(e.timestamp_ms, e.sequence_id) for e in events]
    assert actual_order == expected_order
    assert normalizer.report.is_monotonic_increasing is True


def test_exact_integer_timestamp_preservation(normalizer, tmp_path):
    """
    Explicitly test exact int64 preservation without float conversion loss,
    rounding, or flooring.
    """
    ts_values = [
        1790916232243,
        1790916232244,
        1790916232999,
        1790916233000,
    ]
    records = [
        {"timestamp_ms": ts, "sequence_id": idx, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0}
        for idx, ts in enumerate(ts_values, start=1)
    ]

    test_file = tmp_path / "timestamps_test.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out_parquet = tmp_path / "timestamps_canonical.parquet"
    df, report = normalizer.normalize_file(test_file, output_parquet=out_parquet)

    assert len(df) == 4
    assert df["timestamp_ms"].dtype == "int64"
    assert df["timestamp_ms"].tolist() == ts_values

    # Reload from parquet to confirm bit-level roundtrip integrity
    reloaded = pd.read_parquet(out_parquet)
    assert reloaded["timestamp_ms"].dtype == "int64"
    assert reloaded["timestamp_ms"].tolist() == ts_values


def test_provenance_tracking(normalizer, tmp_path):
    """
    Verify that every canonical row is traceable to its source file,
    source line (1-indexed), and source record index (0-indexed).
    """
    test_file = tmp_path / "provenance_test.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        # Line 1: valid
        f.write(json.dumps({"timestamp_ms": 1700000001000, "market_id": "m1", "asset_id": "a1", "bid": 0.40, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0}) + "\n")
        # Line 2: invalid (crossed)
        f.write(json.dumps({"timestamp_ms": 1700000002000, "market_id": "m1", "asset_id": "a1", "bid": 0.60, "ask": 0.50, "bid_size": 10.0, "ask_size": 10.0}) + "\n")
        # Line 3: valid
        f.write(json.dumps({"timestamp_ms": 1700000003000, "market_id": "m1", "asset_id": "a1", "bid": 0.42, "ask": 0.52, "bid_size": 10.0, "ask_size": 10.0}) + "\n")

    df, report = normalizer.normalize_file(test_file)

    assert len(df) == 2
    assert df["source_file"].tolist() == ["provenance_test.jsonl", "provenance_test.jsonl"]
    assert df["source_line"].tolist() == [1, 3]  # Line 2 skipped
    assert df["source_record_idx"].tolist() == [0, 0]


def test_orderbook_delta_tracking(normalizer):
    """
    Test L2 book snapshot followed by price_change delta updates.
    Verifies top of book tracking and depth updates.
    """
    lines = [
        # 1. Book snapshot: best bid 0.20 (size 10), best ask 0.30 (size 20)
        json.dumps({
            "event": {
                "event_type": "book",
                "market": "mkt1",
                "asset_id": "ast1",
                "timestamp": "1700000001000",
                "bids": [{"price": "0.20", "size": "10.0"}, {"price": "0.19", "size": "5.0"}],
                "asks": [{"price": "0.30", "size": "20.0"}, {"price": "0.31", "size": "15.0"}],
            }
        }),
        # 2. Price change: new higher bid 0.22 (size 8.0)
        json.dumps({
            "event": {
                "event_type": "price_change",
                "market": "mkt1",
                "timestamp": "1700000002000",
                "price_changes": [
                    {"asset_id": "ast1", "price": "0.22", "size": "8.0", "side": "BUY"}
                ],
            }
        }),
        # 3. Price change: cancel the 0.22 bid (size 0.0) -> best bid returns to 0.20
        json.dumps({
            "event": {
                "event_type": "price_change",
                "market": "mkt1",
                "timestamp": "1700000003000",
                "price_changes": [
                    {"asset_id": "ast1", "price": "0.22", "size": "0.0", "side": "BUY"}
                ],
            }
        }),
    ]

    events = normalizer.normalize_stream(lines, source_file="test_delta.jsonl")

    assert len(events) == 3
    # Step 1
    assert events[0].bid == 0.20
    assert events[0].bid_size == 10.0
    assert events[0].ask == 0.30
    assert events[0].ask_size == 20.0

    # Step 2: higher bid arrived
    assert events[1].bid == 0.22
    assert events[1].bid_size == 8.0
    assert events[1].ask == 0.30

    # Step 3: higher bid cancelled -> reverted to 0.20
    assert events[2].bid == 0.20
    assert events[2].bid_size == 10.0
    assert events[2].ask == 0.30


def test_real_smoke_test_file_normalization():
    """
    Integration smoke test on real backup Polymarket raw jsonl data.
    Verifies that the canonical dataset is produced under data/clean_v2/01_canonical_events/.
    """
    raw_path = Path("data/backup/polymarket_extended_smoke_test.jsonl")
    if not raw_path.exists():
        pytest.skip(f"Backup raw file {raw_path} not found.")

    output_dir = Path("data/clean_v2/01_canonical_events")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = output_dir / "smoke_test_canonical.parquet"

    normalizer = EventNormalizer(
        target_asset_id="32338220190071351435772801779725302244575775216413325951443816017994629993401"
    )
    df, report = normalizer.normalize_file(raw_path, output_parquet=out_parquet)

    assert out_parquet.exists()
    assert len(df) == 57
    assert report.valid_canonical_events == 57
    assert report.is_monotonic_increasing is True
    assert report.min_timestamp_ms == 1790916232243
    assert report.max_timestamp_ms == 1790916512467

    # Schema integrity check
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
    assert list(df.columns) == expected_cols
    assert df["timestamp_ms"].dtype == "int64"
    assert (df["bid"] < df["ask"]).all()
    assert (df["bid_size"] > 0).all()
    assert (df["ask_size"] > 0).all()
