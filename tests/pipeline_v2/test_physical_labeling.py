"""
Unit tests for Pipeline V2 Physical Elapsed-Time 5-Second Horizon Labeler.

Covers all 17 Phase 13 requirements:
1. Irregular timestamps: verify target is based on physical milliseconds, not row position.
2. Exact 5-second target: 5000ms => valid.
3. 5001ms target: valid.
4. 6999ms target: valid.
5. 7000ms target: valid.
6. 7001ms target: invalid / dropped.
7. 4999ms target: must NOT be selected.
8. Multiple future events: select FIRST event >= T+5000ms.
9. Stale current observation: no label.
10. Session boundary: target must never come from another session.
11. Constant price: should correctly produce FLAT.
12. UP classification.
13. DOWN classification.
14. Future columns absent from exported labeled dataset.
15. No shift(-5) implementation.
16. Physical horizon assertion: 5000 <= horizon_ms <= 7000.
17. Provenance: current and target source timestamps remain auditable.
"""

import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline_v2.labeling.physical_horizon_labeler import (
    LabelingValidationReport,
    PhysicalHorizonLabeler,
)


@pytest.fixture
def labeler():
    return PhysicalHorizonLabeler(
        min_horizon_ms=5000,
        max_horizon_ms=7000,
        min_spread_fraction=0.5,
        min_threshold_abs=0.001,
    )


def create_resampled_df(rows):
    """Helper to construct 1s resampled DataFrame fixture."""
    records = []
    for r in rows:
        records.append({
            "grid_timestamp_ms": int(r["timestamp_ms"]),
            "source_timestamp_ms": int(r.get("source_ts", r["timestamp_ms"])),
            "source_sequence_id": int(r.get("seq", 1)),
            "market_id": str(r.get("market_id", "mkt1")),
            "asset_id": str(r.get("asset_id", "ast1")),
            "bid": float(r.get("bid", 0.40)),
            "ask": float(r.get("ask", 0.50)),
            "bid_size": float(r.get("bid_size", 100.0)),
            "ask_size": float(r.get("ask_size", 200.0)),
            "event_age_ms": int(r.get("event_age_ms", 0)),
            "is_stale": bool(r.get("is_stale", False)),
            "session_id": str(r.get("session_id", "sess1")),
        })
    df = pd.DataFrame(records)
    return df.astype({
        "grid_timestamp_ms": "int64",
        "source_timestamp_ms": "int64",
        "source_sequence_id": "int64",
        "market_id": "string",
        "asset_id": "string",
        "bid": "float64",
        "ask": "float64",
        "bid_size": "float64",
        "ask_size": "float64",
        "event_age_ms": "int64",
        "is_stale": "bool",
        "session_id": "string",
    })


def create_events_df(rows):
    """Helper to construct canonical events DataFrame fixture."""
    records = []
    for r in rows:
        records.append({
            "timestamp_ms": int(r["timestamp_ms"]),
            "sequence_id": int(r.get("seq", 1)),
            "market_id": str(r.get("market_id", "mkt1")),
            "asset_id": str(r.get("asset_id", "ast1")),
            "bid": float(r.get("bid", 0.40)),
            "ask": float(r.get("ask", 0.50)),
            "bid_size": float(r.get("bid_size", 100.0)),
            "ask_size": float(r.get("ask_size", 200.0)),
            "session_id": str(r.get("session_id", "sess1")),
        })
    df = pd.DataFrame(records)
    return df.astype({
        "timestamp_ms": "int64",
        "sequence_id": "int64",
        "market_id": "string",
        "asset_id": "string",
        "bid": "float64",
        "ask": "float64",
        "bid_size": "float64",
        "ask_size": "float64",
        "session_id": "string",
    })


# 1. Irregular timestamps: verify target is based on physical milliseconds, not row position
def test_irregular_timestamps_physical_not_row_position(labeler):
    """
    Physical millisecond selection vs row position:
    Current observation is at T = 10000 ms.
    If rows were irregularly spaced:
      row 0: 10000ms (current)
      row 1: 10500ms
      row 2: 11000ms
      row 3: 12000ms
      row 4: 15200ms (physical delta = 5200ms, row index = 4)
      row 5: 18000ms (physical delta = 8000ms, row index = 5)
    A 5-row shift would pick row 5 (18000ms, delta=8000ms, INVALID!).
    Physical matching must pick row 4 (15200ms, delta=5200ms, VALID!).
    """
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([
        {"timestamp_ms": 10500, "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 11000, "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 12000, "bid": 0.40, "ask": 0.50},
        {"timestamp_ms": 15200, "bid": 0.45, "ask": 0.55},
        {"timestamp_ms": 18000, "bid": 0.60, "ask": 0.70},
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)

    assert len(clean_df) == 1
    assert len(audit_df) == 1
    assert audit_df.iloc[0]["target_timestamp"] == 15200
    assert audit_df.iloc[0]["physical_horizon_ms"] == 5200


# 2. Exact 5-second target: 5000ms => valid
def test_exact_5000ms_target(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([{"timestamp_ms": 15000}])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert audit_df.iloc[0]["physical_horizon_ms"] == 5000
    assert audit_df.iloc[0]["target_timestamp"] == 15000


# 3. 5001ms target: valid
def test_5001ms_target(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([{"timestamp_ms": 15001}])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert audit_df.iloc[0]["physical_horizon_ms"] == 5001


# 4. 6999ms target: valid
def test_6999ms_target(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([{"timestamp_ms": 16999}])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert audit_df.iloc[0]["physical_horizon_ms"] == 6999


# 5. 7000ms target: valid
def test_7000ms_target(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([{"timestamp_ms": 17000}])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert audit_df.iloc[0]["physical_horizon_ms"] == 7000


# 6. 7001ms target: invalid / dropped
def test_7001ms_target_invalid_dropped(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([{"timestamp_ms": 17001}])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 0
    assert len(audit_df) == 0
    assert report.dropped_rows == 1


# 7. 4999ms target: must NOT be selected
def test_4999ms_target_must_not_be_selected(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([
        {"timestamp_ms": 14999},  # 4999ms -> too early
        {"timestamp_ms": 15500},  # 5500ms -> valid!
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert audit_df.iloc[0]["target_timestamp"] == 15500
    assert audit_df.iloc[0]["target_timestamp"] != 14999
    assert audit_df.iloc[0]["physical_horizon_ms"] == 5500


# 8. Multiple future events: select FIRST event >= T+5000ms
def test_multiple_future_events_select_first(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([
        {"timestamp_ms": 15200, "bid": 0.42, "ask": 0.52},
        {"timestamp_ms": 15400, "bid": 0.43, "ask": 0.53},
        {"timestamp_ms": 16000, "bid": 0.44, "ask": 0.54},
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert audit_df.iloc[0]["target_timestamp"] == 15200
    assert audit_df.iloc[0]["physical_horizon_ms"] == 5200


# 9. Stale current observation: no label
def test_stale_current_observation_no_label(labeler):
    resampled = create_resampled_df([
        {"timestamp_ms": 10000, "is_stale": True},
        {"timestamp_ms": 11000, "is_stale": False},
    ])
    events = create_events_df([
        {"timestamp_ms": 15000},
        {"timestamp_ms": 16000},
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert report.stale_rows_excluded == 1
    # Only 11000ms was labeled
    assert clean_df.iloc[0]["grid_timestamp_ms"] == 11000


# 10. Session boundary: target must never come from another session
def test_session_boundary_isolation(labeler):
    resampled = create_resampled_df([
        {"timestamp_ms": 10000, "session_id": "sess_A"},
    ])
    events = create_events_df([
        {"timestamp_ms": 15000, "session_id": "sess_B"},  # Target in different session!
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    # Target from session B must NOT be matched to session A
    assert len(clean_df) == 0
    assert report.session_boundary_violations == 0


# 11. Constant price: should correctly produce FLAT
def test_constant_price_produces_flat(labeler):
    resampled = create_resampled_df([
        {"timestamp_ms": 10000, "bid": 0.40, "ask": 0.50},  # mid = 0.45, spread = 0.10, thresh = 0.05
    ])
    events = create_events_df([
        {"timestamp_ms": 15000, "bid": 0.40, "ask": 0.50},  # target mid = 0.45, delta = 0.0
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["label"] == "FLAT"
    assert audit_df.iloc[0]["delta"] == 0.0
    assert report.flat_count == 1


# 12. UP classification
def test_up_classification(labeler):
    resampled = create_resampled_df([
        {"timestamp_ms": 10000, "bid": 0.49, "ask": 0.51},  # mid = 0.50, spread = 0.02, thresh = 0.01
    ])
    events = create_events_df([
        {"timestamp_ms": 15000, "bid": 0.52, "ask": 0.54},  # target mid = 0.53, delta = +0.03 > 0.01
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["label"] == "UP"
    assert audit_df.iloc[0]["label"] == "UP"
    assert report.up_count == 1


# 13. DOWN classification
def test_down_classification(labeler):
    resampled = create_resampled_df([
        {"timestamp_ms": 10000, "bid": 0.49, "ask": 0.51},  # mid = 0.50, spread = 0.02, thresh = 0.01
    ])
    events = create_events_df([
        {"timestamp_ms": 15000, "bid": 0.45, "ask": 0.47},  # target mid = 0.46, delta = -0.04 < -0.01
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["label"] == "DOWN"
    assert audit_df.iloc[0]["label"] == "DOWN"
    assert report.down_count == 1


# 14. Future columns absent from exported labeled dataset
def test_future_columns_absent_from_training_dataset(labeler):
    resampled = create_resampled_df([{"timestamp_ms": 10000}])
    events = create_events_df([{"timestamp_ms": 15000}])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)

    forbidden_cols = [
        "target_timestamp", "target_timestamp_ms",
        "future_timestamp", "future_timestamp_ms",
        "target_mid", "future_mid",
        "target_bid", "future_bid",
        "target_ask", "future_ask",
        "delta", "future_delta",
        "threshold", "label_threshold",
        "physical_horizon_ms", "horizon_ms",
    ]
    for col in forbidden_cols:
        assert col not in clean_df.columns

    for col in clean_df.columns:
        assert not col.startswith("future_")
        assert not col.startswith("target_")

    assert report.future_column_violations == 0


# 15. No shift(-5) implementation
def test_no_shift_minus_5_in_source_code():
    source = inspect.getsource(PhysicalHorizonLabeler)
    assert "shift(-5)" not in source
    assert ".shift(" not in source


# 16. Physical horizon assertion: 5000 <= horizon_ms <= 7000
def test_physical_horizon_assertion_on_all_rows(labeler):
    np.random.seed(123)
    curr_times = [10000, 11000, 12000, 13000, 14000]
    resampled = create_resampled_df([{"timestamp_ms": t} for t in curr_times])

    # Future events with varied valid horizons between 5000 and 7000
    events = create_events_df([
        {"timestamp_ms": 15123},
        {"timestamp_ms": 16500},
        {"timestamp_ms": 17200},
        {"timestamp_ms": 18900},
        {"timestamp_ms": 20999},
    ])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)
    assert len(clean_df) > 0
    assert report.physical_horizon_violations == 0

    horizons = audit_df["physical_horizon_ms"].to_numpy()
    assert (horizons >= 5000).all()
    assert (horizons <= 7000).all()


# 17. Provenance: current and target source timestamps remain auditable
def test_provenance_auditability(labeler):
    resampled = create_resampled_df([{
        "timestamp_ms": 10000,
        "source_ts": 9850,
        "seq": 101,
        "market_id": "mkt_poly",
        "asset_id": "token_yes",
    }])
    events = create_events_df([{
        "timestamp_ms": 15432,
        "seq": 555,
        "market_id": "mkt_poly",
        "asset_id": "token_yes",
    }])

    clean_df, audit_df, report = labeler.label_dataset(resampled, events_df=events)

    assert len(audit_df) == 1
    aud = audit_df.iloc[0]
    assert aud["current_timestamp"] == 10000
    assert aud["target_timestamp"] == 15432
    assert aud["physical_horizon_ms"] == 5432
    assert aud["market_id"] == "mkt_poly"
    assert aud["asset_id"] == "token_yes"
    assert aud["target_sequence_id"] == 555
