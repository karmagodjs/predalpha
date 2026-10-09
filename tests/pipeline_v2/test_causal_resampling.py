"""
Unit tests for Pipeline V2 Causal 1-Second Point-In-Time Grid Resampler.

Covers all Phase 12 requirements:
A. Future-event prevention:
   Events: 12.1s (12100ms), 12.85s (12850ms), 13.2s (13200ms)
   Grid: 13.0s (13000ms) => 12.85s selected. Must NOT select 13.2s.
B. Exact timestamp:
   Event: 13.0s (13000ms) -> Grid: 13.0s => event_age_ms == 0.
C. Multiple events before T:
   Several events prior to grid timestamp => latest one selected.
D. Missing seconds:
   Gaps between events => grid rows generated continuously with increasing event_age_ms.
E. Stale quote:
   event_age_ms > 5000 ms => is_stale == True; <= 5000 ms => is_stale == False.
F. No future event ever selected:
   Assert source_timestamp_ms <= grid_timestamp_ms across all rows.
G. Timestamp monotonicity:
   grid_timestamp_ms is strictly monotonically increasing.
H. Session boundary isolation:
   Events from session 1 never match grid timestamps in session 2.
I. Provenance preservation:
   source_sequence_id, source_timestamp_ms, market_id, asset_id, quotes preserved.
J. Event-age calculation:
   event_age_ms = grid_timestamp_ms - source_timestamp_ms exactly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.resampling.point_in_time_grid import (
    PointInTimeResampler,
    ResamplingValidationReport,
)


@pytest.fixture
def resampler():
    return PointInTimeResampler(max_stale_ms=5000, grid_interval_ms=1000)


def create_canonical_df(rows, session_id=None):
    """Helper to construct a valid canonical DataFrame for testing."""
    records = []
    for r in rows:
        rec = {
            "timestamp_ms": int(r["timestamp_ms"]),
            "sequence_id": int(r.get("sequence_id", 1)),
            "market_id": str(r.get("market_id", "mkt_test")),
            "asset_id": str(r.get("asset_id", "ast_test")),
            "bid": float(r.get("bid", 0.45)),
            "ask": float(r.get("ask", 0.55)),
            "bid_size": float(r.get("bid_size", 100.0)),
            "ask_size": float(r.get("ask_size", 200.0)),
            "source_file": str(r.get("source_file", "test.jsonl")),
            "source_line": int(r.get("source_line", 1)),
            "source_record_idx": int(r.get("source_record_idx", 0)),
        }
        if session_id is not None:
            rec["session_id"] = str(session_id)
        elif "session_id" in r:
            rec["session_id"] = str(r["session_id"])
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.astype({
        "timestamp_ms": "int64",
        "sequence_id": "int64",
        "market_id": "string",
        "asset_id": "string",
        "bid": "float64",
        "ask": "float64",
        "bid_size": "float64",
        "ask_size": "float64",
        "source_file": "string",
        "source_line": "int64",
        "source_record_idx": "int64",
    })
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].astype("string")
    return df


def test_mandatory_future_event_prevention(resampler):
    """
    Mandatory Prompt Test:
    Events: 12.100s, 12.850s, 13.200s
    Grid: 13.000s
    Expected: 13.000s snapshot uses 12.850s event. It MUST NOT use 13.200s.
    """
    events_df = create_canonical_df([
        {"timestamp_ms": 12100, "sequence_id": 1, "bid": 0.41, "ask": 0.51},
        {"timestamp_ms": 12850, "sequence_id": 2, "bid": 0.42, "ask": 0.52},
        {"timestamp_ms": 13200, "sequence_id": 3, "bid": 0.43, "ask": 0.53},
    ])

    resampled_df, report = resampler.resample_events(events_df)

    # Grid start is ceil(12100/1000)*1000 = 13000ms. Grid end is (13200//1000)*1000 = 13000ms.
    assert len(resampled_df) == 1
    row = resampled_df.iloc[0]

    assert row["grid_timestamp_ms"] == 13000
    assert row["source_timestamp_ms"] == 12850
    assert row["source_sequence_id"] == 2
    assert row["bid"] == 0.42
    assert row["ask"] == 0.52
    assert row["event_age_ms"] == 150  # 13000 - 12850
    assert bool(row["is_stale"]) is False

    # Hard assertion: Must NOT use 13.2s
    assert row["source_timestamp_ms"] != 13200
    assert row["bid"] != 0.43


def test_exact_timestamp_matching(resampler):
    """
    Test exact timestamp:
    If event_timestamp == grid_timestamp:
        event_age_ms = 0
        that event is valid and is_stale is False.
    """
    events_df = create_canonical_df([
        {"timestamp_ms": 12500, "sequence_id": 1, "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 13000, "sequence_id": 2, "bid": 0.45, "ask": 0.55},
        {"timestamp_ms": 13500, "sequence_id": 3, "bid": 0.48, "ask": 0.58},
    ])

    resampled_df, report = resampler.resample_events(events_df)

    # Grid points: 13000
    row_13 = resampled_df[resampled_df["grid_timestamp_ms"] == 13000].iloc[0]
    assert row_13["source_timestamp_ms"] == 13000
    assert row_13["source_sequence_id"] == 2
    assert row_13["bid"] == 0.45
    assert row_13["event_age_ms"] == 0
    assert bool(row_13["is_stale"]) is False


def test_multiple_events_before_grid(resampler):
    """
    Test multiple events before a grid timestamp:
    Select ONLY the latest event (tie-broken by sequence_id if timestamps match).
    """
    events_df = create_canonical_df([
        {"timestamp_ms": 12100, "sequence_id": 1, "bid": 0.41, "ask": 0.51},
        {"timestamp_ms": 12500, "sequence_id": 2, "bid": 0.42, "ask": 0.52},
        {"timestamp_ms": 12900, "sequence_id": 3, "bid": 0.43, "ask": 0.53},
        {"timestamp_ms": 12900, "sequence_id": 4, "bid": 0.44, "ask": 0.54},  # duplicate ms, higher sequence
        {"timestamp_ms": 13100, "sequence_id": 5, "bid": 0.46, "ask": 0.56},
    ])

    resampled_df, report = resampler.resample_events(events_df)
    row_13 = resampled_df[resampled_df["grid_timestamp_ms"] == 13000].iloc[0]

    assert row_13["source_timestamp_ms"] == 12900
    assert row_13["source_sequence_id"] == 4
    assert row_13["bid"] == 0.44
    assert row_13["event_age_ms"] == 100


def test_missing_seconds_and_continuous_grid(resampler):
    """
    Test missing seconds:
    An event at 10.0s and next event at 15.5s.
    Grid timestamps 10s, 11s, 12s, 13s, 14s, 15s must be produced,
    with event_age_ms increasing monotonically.
    """
    events_df = create_canonical_df([
        {"timestamp_ms": 10000, "sequence_id": 1, "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 15500, "sequence_id": 2, "bid": 0.45, "ask": 0.55},
    ])

    resampled_df, report = resampler.resample_events(events_df)

    assert len(resampled_df) == 6  # 10s, 11s, 12s, 13s, 14s, 15s
    assert resampled_df["grid_timestamp_ms"].tolist() == [10000, 11000, 12000, 13000, 14000, 15000]

    # All grid points from 10s to 15s reference the 10000ms event
    assert (resampled_df["source_timestamp_ms"] == 10000).all()
    assert resampled_df["event_age_ms"].tolist() == [0, 1000, 2000, 3000, 4000, 5000]
    assert (resampled_df["is_stale"] == False).all()  # 5000ms is not > 5000ms


def test_staleness_threshold(resampler):
    """
    Test staleness rule:
    Maximum allowed event age: 5000 ms.
    If event_age_ms > 5000:
        mark is_stale = True
    """
    events_df = create_canonical_df([
        {"timestamp_ms": 10000, "sequence_id": 1, "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 17000, "sequence_id": 2, "bid": 0.45, "ask": 0.55},
    ])

    resampled_df, report = resampler.resample_events(events_df)

    # Grid: 10s, 11s, 12s, 13s, 14s, 15s, 16s, 17s
    expected_ages = [0, 1000, 2000, 3000, 4000, 5000, 6000, 0]
    expected_stale = [False, False, False, False, False, False, True, False]

    assert resampled_df["event_age_ms"].tolist() == expected_ages
    assert resampled_df["is_stale"].tolist() == expected_stale
    assert report.stale_row_count == 1
    assert report.stale_percentage == pytest.approx(12.5)


def test_no_future_event_ever_selected(resampler):
    """
    Hard assertion across dataset:
    source_timestamp_ms <= grid_timestamp_ms for 100% of rows.
    """
    np.random.seed(42)
    timestamps = np.sort(np.random.randint(1_000_000, 1_100_000, size=50))
    events_df = create_canonical_df([
        {"timestamp_ms": ts, "sequence_id": i}
        for i, ts in enumerate(timestamps)
    ])

    resampled_df, report = resampler.resample_events(events_df)

    assert report.future_event_violations == 0
    assert (resampled_df["source_timestamp_ms"] <= resampled_df["grid_timestamp_ms"]).all()
    assert (resampled_df["event_age_ms"] >= 0).all()


def test_timestamp_monotonicity(resampler):
    """
    Verify grid_timestamp_ms is strictly monotonic increasing.
    """
    events_df = create_canonical_df([
        {"timestamp_ms": 10000, "sequence_id": 1},
        {"timestamp_ms": 12000, "sequence_id": 2},
        {"timestamp_ms": 15000, "sequence_id": 3},
    ])

    resampled_df, report = resampler.resample_events(events_df)

    assert report.monotonicity is True
    diffs = np.diff(resampled_df["grid_timestamp_ms"].to_numpy())
    assert (diffs == 1000).all()  # exact 1-second grid increments


def test_session_boundary_isolation(resampler):
    """
    Session boundaries: Never match an event from another session.
    Session 1: events at 10000ms, 12000ms
    Session 2: events at 20000ms, 22000ms
    Grid points in session 2 must never match events from session 1.
    """
    events_session1 = [
        {"timestamp_ms": 10000, "sequence_id": 1, "session_id": "sess_1", "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 12000, "sequence_id": 2, "session_id": "sess_1", "bid": 0.42, "ask": 0.52},
    ]
    events_session2 = [
        {"timestamp_ms": 20000, "sequence_id": 3, "session_id": "sess_2", "bid": 0.60, "ask": 0.70},
        {"timestamp_ms": 22000, "sequence_id": 4, "session_id": "sess_2", "bid": 0.62, "ask": 0.72},
    ]

    events_df = create_canonical_df(events_session1 + events_session2)
    resampled_df, report = resampler.resample_events(events_df)

    assert report.session_boundary_violations == 0

    s1_rows = resampled_df[resampled_df["session_id"] == "sess_1"]
    s2_rows = resampled_df[resampled_df["session_id"] == "sess_2"]

    assert len(s1_rows) == 3  # 10s, 11s, 12s
    assert len(s2_rows) == 3  # 20s, 21s, 22s

    assert (s1_rows["source_timestamp_ms"] <= 12000).all()
    assert (s2_rows["source_timestamp_ms"] >= 20000).all()


def test_provenance_preservation(resampler):
    """
    Verify that each resampled row retains:
    - grid_timestamp_ms
    - source_timestamp_ms
    - source_sequence_id
    - market_id
    - asset_id
    - bid
    - ask
    - bid_size
    - ask_size
    - source_file, source_line, source_record_idx
    """
    events_df = create_canonical_df([
        {
            "timestamp_ms": 10500,
            "sequence_id": 888,
            "market_id": "mkt_poly_1",
            "asset_id": "ast_yes_1",
            "bid": 0.45,
            "ask": 0.55,
            "bid_size": 123.4,
            "ask_size": 567.8,
            "source_file": "poly_smoke.jsonl",
            "source_line": 42,
            "source_record_idx": 3,
        },
        {
            "timestamp_ms": 11500,
            "sequence_id": 999,
            "market_id": "mkt_poly_1",
            "asset_id": "ast_yes_1",
            "bid": 0.46,
            "ask": 0.56,
            "bid_size": 222.0,
            "ask_size": 333.0,
            "source_file": "poly_smoke.jsonl",
            "source_line": 45,
            "source_record_idx": 0,
        },
    ])

    resampled_df, report = resampler.resample_events(events_df)
    row_11 = resampled_df[resampled_df["grid_timestamp_ms"] == 11000].iloc[0]

    assert row_11["grid_timestamp_ms"] == 11000
    assert row_11["source_timestamp_ms"] == 10500
    assert row_11["source_sequence_id"] == 888
    assert row_11["market_id"] == "mkt_poly_1"
    assert row_11["asset_id"] == "ast_yes_1"
    assert row_11["bid"] == 0.45
    assert row_11["ask"] == 0.55
    assert row_11["bid_size"] == 123.4
    assert row_11["ask_size"] == 567.8
    assert row_11["source_file"] == "poly_smoke.jsonl"
    assert row_11["source_line"] == 42
    assert row_11["source_record_idx"] == 3


def test_input_validation_errors(resampler):
    """
    Verify input validation enforces:
    - canonical schema
    - int64 timestamps
    - monotonicity
    - bid < ask
    - positive depth
    """
    # 1. Missing columns
    bad_df1 = pd.DataFrame({"timestamp_ms": [10000], "bid": [0.5]})
    with pytest.raises(ValueError, match="missing required canonical columns"):
        resampler.resample_events(bad_df1)

    # 2. Float timestamp
    bad_df2 = create_canonical_df([{"timestamp_ms": 10000}])
    bad_df2["timestamp_ms"] = bad_df2["timestamp_ms"].astype("float64")
    with pytest.raises(TypeError, match="must be integer dtype"):
        resampler.resample_events(bad_df2)

    # 3. Non-monotonic timestamps
    bad_df3 = create_canonical_df([
        {"timestamp_ms": 12000},
        {"timestamp_ms": 11000},
    ])
    with pytest.raises(ValueError, match="not monotonically ordered"):
        resampler.resample_events(bad_df3)

    # 4. Crossed book
    bad_df4 = create_canonical_df([
        {"timestamp_ms": 10000, "bid": 0.60, "ask": 0.50},
    ])
    with pytest.raises(ValueError, match="crossed book"):
        resampler.resample_events(bad_df4)

    # 5. Non-positive depth
    bad_df5 = create_canonical_df([
        {"timestamp_ms": 10000, "bid_size": 0.0},
    ])
    with pytest.raises(ValueError, match="non-positive depth"):
        resampler.resample_events(bad_df5)


def test_real_smoke_test_canonical_resampling(resampler, tmp_path):
    """
    Integration test on real Phase 11 canonical events dataset.
    """
    input_path = Path("data/clean_v2/01_canonical_events/smoke_test_canonical.parquet")
    if not input_path.exists():
        pytest.skip(f"Input file {input_path} not found.")

    out_parquet = tmp_path / "smoke_test_resampled_1s.parquet"
    resampled_df, report = resampler.resample_file(input_path, output_parquet=out_parquet)

    assert out_parquet.exists()
    assert report.input_events == 57
    assert report.output_grid_rows > 0
    assert report.future_event_violations == 0
    assert report.duplicate_grid_timestamps == 0
    assert report.monotonicity is True
    assert (resampled_df["grid_timestamp_ms"] % 1000 == 0).all()
    assert (resampled_df["event_age_ms"] >= 0).all()
