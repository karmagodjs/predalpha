"""Auditable, non-labeling inspection for the specified Polymarket market."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


MARKET_ID = "0x0b1e1b2f032a7668cc96455e75939666e25eceab070d4b9162da5a9da7784b68"
REQUESTED_TOKEN_IDS = [
    "32664731353322490016663940157951165038326534411865610104571929420891932752044",
    "83368144530639073518031297526937851295808514385655078316894643930648829402",
]
SESSION_FILES = [
    "btc_oct3_up_down_session_20261002_01.jsonl",
    "btc_oct3_up_down_session_20261002_03.jsonl",
    "btc_oct3_up_down_session_20261002_04.jsonl",
]


def audit_raw_sessions(raw_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    all_assets: set[str] = set()
    all_markets: set[str] = set()
    for filename in SESSION_FILES:
        events = [json.loads(line) for line in (raw_dir / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        event_types = Counter(event.get("event", {}).get("event_type", event.get("event", {}).get("type", "book")) for event in events)
        assets = sorted({str(event.get("asset_id", event.get("event", {}).get("asset_id"))) for event in events})
        markets = sorted({str(event.get("event", {}).get("market")) for event in events})
        timestamps = [event.get("received_at") for event in events if event.get("received_at")]
        fields = sorted({key for event in events for key in event})
        nested_fields = sorted({key for event in events for key in event.get("event", {})})
        resolution_keys = sorted({key for event in events for key in event.get("event", {}) if any(term in key.lower() for term in ("outcome", "resolution", "winner", "settle"))})
        files.append({"file": filename, "event_count": len(events), "event_types": dict(event_types), "asset_ids": assets,
                      "market_ids": markets, "first_received_at": min(timestamps), "last_received_at": max(timestamps),
                      "top_level_fields": fields, "event_fields": nested_fields, "resolution_related_fields": resolution_keys})
        all_assets.update(assets)
        all_markets.update(markets)
    return {"market_id_expected": MARKET_ID, "files": files, "observed_asset_ids": sorted(all_assets), "observed_market_ids": sorted(all_markets)}


def build_verified_market_events(raw_dir: Path, token_mapping: dict[str, dict[str, Any]], settlement_outcome: str) -> pd.DataFrame:
    """Flatten raw book events only after official mapping and settlement are explicit."""
    rows: list[dict[str, Any]] = []
    for filename in SESSION_FILES:
        for line in (raw_dir / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            event = record["event"]
            token_id = str(event["asset_id"])
            if token_id not in token_mapping:
                raise ValueError(f"No verified mapping for raw token {token_id}")
            bids = [float(level["price"]) for level in event.get("bids", [])]
            asks = [float(level["price"]) for level in event.get("asks", [])]
            rows.append({
                "timestamp": pd.to_datetime(record["received_at"], utc=True), "source_file": filename,
                "market_id": event["market"], "token_id": token_id, "token_outcome": token_mapping[token_id]["outcome"],
                "best_bid": max(bids) if bids else None, "best_ask": min(asks) if asks else None,
                "market_settlement_outcome": settlement_outcome,
                "is_winning_token": token_mapping[token_id]["outcome"] == settlement_outcome,
            })
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "token_id", "token_outcome", "market_settlement_outcome", "is_winning_token"}
    if frame.empty or frame[list(required)].isna().any().any() or set(frame["market_id"]) != {MARKET_ID}:
        raise ValueError("Verified Track B event validation failed")
    return frame
