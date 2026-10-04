"""Metadata-first multi-market Polymarket collector with auditable settlement evidence."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from market_data.collection_common import AppendOnlyJsonl, append_error, jsonl_records, timestamp_quality, utc_now

CLOB_MARKET_URL = "https://clob.polymarket.com/markets/{condition_id}"
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(frozen=True)
class TrackBCollectionConfig:
    output_dir: Path = Path("data/raw/track_b")
    gamma_url: str = GAMMA_MARKETS_URL
    clob_url_template: str = CLOB_MARKET_URL
    timeout_seconds: float = 10.0
    max_reconnects: int = 3


def verified_mapping(metadata: dict[str, Any]) -> dict[str, str]:
    tokens = metadata.get("tokens")
    if not isinstance(tokens, list) or len(tokens) < 2:
        raise ValueError("metadata_has_no_complete_token_list")
    mapping = {str(token.get("token_id")): token.get("outcome") for token in tokens}
    if None in mapping.values() or any(not token_id or token_id == "None" for token_id in mapping):
        raise ValueError("metadata_token_mapping_incomplete")
    return mapping


class TrackBCollector:
    def __init__(self, config: TrackBCollectionConfig, session: requests.Session | Any | None = None):
        self.config, self.session = config, session or requests.Session()

    def fetch_metadata(self, condition_id: str) -> dict[str, Any]:
        response = self.session.get(self.config.clob_url_template.format(condition_id=condition_id), timeout=self.config.timeout_seconds)
        response.raise_for_status()
        metadata = response.json()
        if str(metadata.get("condition_id")) != condition_id:
            raise ValueError("official_metadata_condition_id_mismatch")
        verified_mapping(metadata)
        return metadata

    def discover(self, limit: int = 100) -> list[dict[str, Any]]:
        response = self.session.get(self.config.gamma_url, params={"active": "true", "closed": "false", "limit": limit}, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        markets = response.json()
        return [market for market in markets if "bitcoin" in str(market.get("question", "")).lower() and "up or down" in str(market.get("question", "")).lower()]

    def _writers(self, condition_id: str) -> tuple[AppendOnlyJsonl, AppendOnlyJsonl, AppendOnlyJsonl, AppendOnlyJsonl]:
        root = self.config.output_dir / condition_id
        return (AppendOnlyJsonl(root / "official_metadata.jsonl"), AppendOnlyJsonl(root / "events.jsonl"), AppendOnlyJsonl(root / "settlement_evidence.jsonl"), AppendOnlyJsonl(root / "collection_errors.jsonl"))

    def initialize_market(self, condition_id: str) -> dict[str, Any]:
        metadata_writer, _, _, errors = self._writers(condition_id)
        try:
            metadata = self.fetch_metadata(condition_id)
            metadata_writer.append({"source": "official_clob", "endpoint": self.config.clob_url_template.format(condition_id=condition_id), "condition_id": condition_id, "metadata": metadata})
            return metadata
        except (requests.RequestException, ValueError) as error:
            append_error(errors, "fetch_metadata", error, condition_id=condition_id)
            raise

    def record_event(self, condition_id: str, event: dict[str, Any], mapping: dict[str, str]) -> bool:
        _, event_writer, _, errors = self._writers(condition_id)
        try:
            token_id = str(event["asset_id"])
            if str(event.get("market")) != condition_id or token_id not in mapping:
                raise ValueError("event_market_or_token_not_in_verified_mapping")
            source_timestamp = event.get("timestamp")
            if source_timestamp is None:
                raise ValueError("event_missing_source_timestamp")
            return event_writer.append({"source": "polymarket_clob_websocket", "condition_id": condition_id, "token_id": token_id, "outcome": mapping[token_id], "source_timestamp": str(source_timestamp), "event_type": event.get("event_type"), "event": event})
        except (KeyError, ValueError) as error:
            append_error(errors, "record_event", error, condition_id=condition_id)
            return False

    def capture_settlement(self, condition_id: str) -> dict[str, Any]:
        _, _, settlement_writer, errors = self._writers(condition_id)
        try:
            metadata = self.fetch_metadata(condition_id)
            mapping = verified_mapping(metadata)
            winners = [outcome for token_id, outcome in mapping.items() if next(token for token in metadata["tokens"] if str(token["token_id"]) == token_id).get("winner") is True]
            result = {"source": "official_clob", "endpoint": self.config.clob_url_template.format(condition_id=condition_id), "condition_id": condition_id, "retrieved_at": utc_now(), "market_closed": metadata.get("closed"), "winner_outcome": winners[0] if len(winners) == 1 else None, "status": "VERIFIED" if len(winners) == 1 else "UNRESOLVED", "tokens": metadata.get("tokens"), "question": metadata.get("question"), "end_date_iso": metadata.get("end_date_iso")}
            settlement_writer.append(result)
            return result
        except (requests.RequestException, ValueError) as error:
            append_error(errors, "capture_settlement", error, condition_id=condition_id)
            return {"condition_id": condition_id, "status": "UNRESOLVED", "winner_outcome": None, "error": str(error)}

    def consume_streams(self, condition_id: str, mapping: dict[str, str], stream_factory: Callable[[], Iterable[dict[str, Any]]]) -> int:
        """Testable reconnect loop; a live websocket adapter can supply each stream."""
        accepted = 0
        for attempt in range(self.config.max_reconnects + 1):
            try:
                for event in stream_factory():
                    accepted += int(self.record_event(condition_id, event, mapping))
                return accepted
            except (ConnectionError, OSError) as error:
                append_error(self._writers(condition_id)[3], "websocket_reconnect", error, attempt=attempt)
        return accepted


def track_b_status(raw_dir: Path) -> dict[str, Any]:
    markets = [path for path in raw_dir.iterdir() if path.is_dir()] if raw_dir.exists() else []
    status = {"markets": len(markets), "resolved": 0, "unresolved": 0, "events": 0, "collection_errors": 0, "date_ranges": {}}
    for market in markets:
        events = jsonl_records(market / "events.jsonl")
        settlements = jsonl_records(market / "settlement_evidence.jsonl")
        errors = jsonl_records(market / "collection_errors.jsonl")
        status["events"] += len(events); status["collection_errors"] += len(errors)
        latest = settlements[-1] if settlements else {"status": "UNRESOLVED"}
        status["resolved" if latest.get("status") == "VERIFIED" else "unresolved"] += 1
        status["date_ranges"][market.name] = timestamp_quality(events, "collected_at")
    return status


async def websocket_events(asset_ids: list[str], max_messages: int):
    """Small live adapter; callers retain reconnect ownership in the collector."""
    import websockets
    async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as websocket:
        await websocket.send(json.dumps({"assets_ids": asset_ids, "type": "market", "custom_feature_enabled": True}))
        for _ in range(max_messages):
            payload = json.loads(await websocket.recv())
            for event in payload if isinstance(payload, list) else [payload]:
                yield event


async def capture_live_websocket(collector: TrackBCollector, condition_id: str, mapping: dict[str, str], max_messages: int) -> int:
    """Record a bounded live session, reconnecting after a dropped connection."""
    accepted = 0
    for attempt in range(collector.config.max_reconnects + 1):
        try:
            async for event in websocket_events(list(mapping), max_messages - accepted):
                accepted += int(collector.record_event(condition_id, event, mapping))
                if accepted >= max_messages:
                    return accepted
            return accepted
        except (ConnectionError, OSError, asyncio.TimeoutError) as error:
            append_error(collector._writers(condition_id)[3], "websocket_reconnect", error, attempt=attempt)
    return accepted
