import pandas as pd
import pytest

from market_data.track_a import CONFIG, validate_chronological_splits, validate_feature_frame
from market_data.track_b_audit import MARKET_ID, audit_raw_sessions, build_verified_market_events


def frame(times):
    return pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True), "mid_price_up": [0.5] * len(times), "spread_up": [0.01] * len(times), "spread_down": [0.01] * len(times), "mid_price_change": [0.0] * len(times), "label_5m": ["UP"] * len(times)})


def test_feature_validation_rejects_future_column():
    unsafe = frame(["2026-01-01T00:00:00Z"]).rename(columns={"mid_price_up": "target_close"})
    with pytest.raises(ValueError, match="Missing configured features"):
        validate_feature_frame(unsafe)


def test_chronological_split_validation_rejects_overlap():
    splits = {"train": frame(["2026-01-01T00:00:00Z"]), "validation": frame(["2026-01-01T00:00:00Z"]), "test": frame(["2026-01-01T00:02:00Z"])}
    with pytest.raises(ValueError, match="overlaps"):
        validate_chronological_splits(splits)


def test_raw_market_audit_preserves_observed_market_and_events(tmp_path):
    from market_data.track_b_audit import SESSION_FILES
    event = {"received_at": "2026-01-01T00:00:00Z", "asset_id": "token", "event": {"market": MARKET_ID, "asset_id": "token", "bids": [], "asks": []}}
    for name in SESSION_FILES:
        (tmp_path / name).write_text(__import__("json").dumps(event) + "\n", encoding="utf-8")
    audit = audit_raw_sessions(tmp_path)
    assert audit["observed_market_ids"] == [MARKET_ID]
    assert audit["files"][0]["event_count"] == 1


def test_verified_event_builder_attaches_only_explicit_mapping(tmp_path):
    from market_data.track_b_audit import SESSION_FILES
    event = {"received_at": "2026-01-01T00:00:00Z", "asset_id": "token", "event": {"market": MARKET_ID, "asset_id": "token", "bids": [{"price": "0.4"}], "asks": [{"price": "0.6"}]}}
    for name in SESSION_FILES:
        (tmp_path / name).write_text(__import__("json").dumps(event) + "\n", encoding="utf-8")
    result = build_verified_market_events(tmp_path, {"token": {"outcome": "Down"}}, "Down")
    assert len(result) == 3
    assert result.is_winning_token.all()
