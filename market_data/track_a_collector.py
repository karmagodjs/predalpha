"""Restartable BTC spot/candle collector. It intentionally creates no labels."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from market_data.collection_common import AppendOnlyJsonl, append_error, jsonl_records, timestamp_quality, utc_now


@dataclass(frozen=True)
class TrackACollectionConfig:
    symbol: str = "BTCUSDT"
    interval_seconds: float = 60.0
    output_dir: Path = Path("data/raw/track_a")
    processed_dir: Path = Path("data/processed/track_a")
    endpoint: str = "https://api.binance.com/api/v3/klines"
    timeout_seconds: float = 10.0
    max_retries: int = 3


class TrackACollector:
    def __init__(self, config: TrackACollectionConfig, session: requests.Session | Any | None = None):
        self.config = config
        self.session = session or requests.Session()
        self.raw_writer = AppendOnlyJsonl(config.output_dir / "btc_observations.jsonl")
        self.error_writer = AppendOnlyJsonl(config.output_dir / "collection_errors.jsonl")

    def fetch_once(self) -> dict[str, Any] | None:
        """Fetch one completed 1m candle; retry transient HTTP/network errors."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.get(self.config.endpoint, params={"symbol": self.config.symbol, "interval": "1m", "limit": 1}, timeout=self.config.timeout_seconds)
                if getattr(response, "status_code", 200) == 429:
                    raise RuntimeError("rate_limited_http_429")
                response.raise_for_status()
                rows = response.json()
                if not isinstance(rows, list) or not rows or len(rows[0]) < 7:
                    raise ValueError("missing_or_malformed_kline")
                candle = rows[0]
                return {"source": "binance", "symbol": self.config.symbol, "source_timestamp": pd.Timestamp(int(candle[6]), unit="ms", tz="UTC").isoformat(), "open_time": pd.Timestamp(int(candle[0]), unit="ms", tz="UTC").isoformat(), "open": candle[1], "high": candle[2], "low": candle[3], "close": candle[4], "volume": candle[5], "source_payload": candle}
            except (requests.RequestException, ValueError, RuntimeError) as error:
                append_error(self.error_writer, "fetch_btc", error, attempt=attempt, symbol=self.config.symbol)
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
        return None

    def collect_once(self, dry_run: bool = False) -> bool:
        if dry_run:
            return self.raw_writer.append({"source": "dry_run", "symbol": self.config.symbol, "source_timestamp": utc_now(), "open": "0", "high": "0", "low": "0", "close": "0", "volume": "0", "dry_run": True})
        record = self.fetch_once()
        return record is not None and self.raw_writer.append(record)

    def run(self, start_time: float | None = None, end_time: float | None = None, dry_run: bool = False) -> int:
        start_time = time.time() if start_time is None else start_time
        if start_time > time.time() and not dry_run:
            time.sleep(start_time - time.time())
        collected = 0
        while end_time is None or time.time() < end_time:
            if self.collect_once(dry_run=dry_run):
                collected += 1
            if dry_run:
                break
            next_run = start_time + (collected * self.config.interval_seconds)
            time.sleep(max(0.0, next_run - time.time()))
        return collected


def process_track_a_raw(raw_dir: Path, processed_dir: Path, expected_gap_seconds: float) -> dict[str, Any]:
    """Write a separate cleaned parquet; raw JSONL remains untouched."""
    records = [record for record in jsonl_records(raw_dir / "btc_observations.jsonl") if "_parse_error" not in record and not record.get("dry_run")]
    required = ["source", "symbol", "source_timestamp", "collected_at", "open", "high", "low", "close", "volume"]
    valid = [record for record in records if all(record.get(field) is not None for field in required)]
    frame = pd.DataFrame(valid)
    if not frame.empty:
        frame["source_timestamp"] = pd.to_datetime(frame["source_timestamp"], utc=True)
        # Preserve the raw nested payload as immutable JSON while keeping parquet schema stable.
        if "source_payload" in frame:
            frame["source_payload"] = frame["source_payload"].map(lambda value: __import__("json").dumps(value, separators=(",", ":")))
        frame = frame.sort_values("source_timestamp").drop_duplicates(["source", "symbol", "source_timestamp"], keep="first")
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_dir / "btc_observations.parquet", index=False)
    quality = timestamp_quality(valid, "source_timestamp", expected_gap_seconds)
    quality.update({"raw_records": len(records), "valid_records": len(valid), "missing_required_fields": len(records) - len(valid), "processed_rows": len(frame), "labels_created": False})
    (processed_dir / "status.json").write_text(__import__("json").dumps(quality, indent=2), encoding="utf-8")
    return quality
