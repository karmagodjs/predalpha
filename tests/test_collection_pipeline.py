import json

from market_data.track_a_collector import TrackACollectionConfig, TrackACollector, process_track_a_raw
from market_data.track_b_collector import TrackBCollectionConfig, TrackBCollector, verified_mapping


class Response:
    def __init__(self, payload, status_code=200): self.payload, self.status_code = payload, status_code
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError(f"http_{self.status_code}")
    def json(self): return self.payload


class Session:
    def __init__(self, responses): self.responses = iter(responses)
    def get(self, *args, **kwargs): return next(self.responses)


def test_track_a_append_resume_and_offline_processing(tmp_path):
    candle = [[1000, "1", "2", "0.5", "1.5", "7", 59999]]
    config = TrackACollectionConfig(output_dir=tmp_path / "raw", processed_dir=tmp_path / "processed", max_retries=1)
    assert TrackACollector(config, Session([Response(candle)])).collect_once()
    assert not TrackACollector(config, Session([Response(candle)])).collect_once()  # same source candle on resume
    quality = process_track_a_raw(config.output_dir, config.processed_dir, 60)
    assert quality["processed_rows"] == 1 and quality["labels_created"] is False


def test_track_a_missing_payload_is_logged_not_fabricated(tmp_path):
    config = TrackACollectionConfig(output_dir=tmp_path, max_retries=1)
    collector = TrackACollector(config, Session([Response([])]))
    assert not collector.collect_once()
    assert (tmp_path / "collection_errors.jsonl").exists()
    assert not (tmp_path / "btc_observations.jsonl").exists()


def metadata(condition="c1", winner=True):
    return {"condition_id": condition, "question": "Bitcoin Up or Down", "closed": True, "tokens": [{"token_id": "up", "outcome": "Up", "winner": False}, {"token_id": "down", "outcome": "Down", "winner": winner}]}


def test_track_b_authoritative_mapping_event_and_settlement(tmp_path):
    collector = TrackBCollector(TrackBCollectionConfig(output_dir=tmp_path), Session([Response(metadata()), Response(metadata())]))
    official = collector.initialize_market("c1")
    mapping = verified_mapping(official)
    assert collector.record_event("c1", {"market": "c1", "asset_id": "down", "timestamp": "12", "event_type": "book"}, mapping)
    result = collector.capture_settlement("c1")
    assert result["status"] == "VERIFIED" and result["winner_outcome"] == "Down"


def test_track_b_rejects_unknown_token_and_marks_unresolved(tmp_path):
    collector = TrackBCollector(TrackBCollectionConfig(output_dir=tmp_path), Session([Response(metadata(winner=False))]))
    assert not collector.record_event("c1", {"market": "c1", "asset_id": "unknown", "timestamp": "12"}, {"up": "Up", "down": "Down"})
    result = collector.capture_settlement("c1")
    assert result["status"] == "UNRESOLVED"


def test_track_b_reconnects_stream_factory(tmp_path):
    collector = TrackBCollector(TrackBCollectionConfig(output_dir=tmp_path, max_reconnects=1))
    calls = {"count": 0}
    def stream():
        calls["count"] += 1
        if calls["count"] == 1: raise ConnectionError("dropped")
        return iter([{"market": "c1", "asset_id": "up", "timestamp": "12"}])
    assert collector.consume_streams("c1", {"up": "Up"}, stream) == 1
    assert calls["count"] == 2


def test_mapping_requires_explicit_outcomes():
    try:
        verified_mapping({"tokens": [{"token_id": "x"}, {"token_id": "y", "outcome": "Down"}]})
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("missing outcome accepted")
