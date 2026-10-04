"""Level 2 Orderbook Collector, Manager, and Recorder for PredAlpha-HFT.

Supports:
- Genuine Level 2 snapshot capture with exchange timestamps and sequence numbers (lastUpdateId).
- Incremental depth updates with strict sequence gap detection and resynchronization.
- Bid/Ask depth storage with explicit price and quantity levels.
- Non-negative, positive-price, and uncrossed-book validation.
- Derived microstructure features: order flow imbalance, spread, volume-weighted microprice.
- Polymarket L2 interface adapter with explicit documentation of exchange sequence-tracking limitations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from market_data.collection_common import (
    AppendOnlyJsonl,
    append_error,
    jsonl_records,
    utc_now,
)

logger = logging.getLogger(__name__)


class OrderBookSequenceGapError(Exception):
    """Raised when an incremental orderbook update sequence is broken (packet drop or out-of-order)."""

    def __init__(self, expected_start: int, actual_start: int, symbol: str):
        super().__init__(
            f"Orderbook sequence gap detected for {symbol}: "
            f"expected first update ID <= {expected_start}, got {actual_start}."
        )
        self.expected_start = expected_start
        self.actual_start = actual_start
        self.symbol = symbol


class InvalidOrderBookError(ValueError):
    """Raised when an orderbook has crossed bids/asks or negative prices/quantities."""
    pass


@dataclass(frozen=True)
class L2Snapshot:
    """Immutable Level 2 orderbook depth snapshot."""

    symbol: str
    last_update_id: int
    bids: tuple[tuple[float, float], ...]  # ((price, size), ...) sorted desc by price
    asks: tuple[tuple[float, float], ...]  # ((price, size), ...) sorted asc by price
    exchange_timestamp: str | None = None   # UTC ISO string if provided by exchange
    received_at: str = field(default_factory=utc_now)
    source: str = "binance_l2_rest"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "exchange_timestamp": self.exchange_timestamp,
            "received_at": self.received_at,
            "bids": [[p, s] for p, s in self.bids],
            "asks": [[p, s] for p, s in self.asks],
        }


@dataclass(frozen=True)
class L2DeltaUpdate:
    """Immutable Level 2 incremental orderbook update event."""

    symbol: str
    first_update_id: int
    final_update_id: int
    bids: tuple[tuple[float, float], ...]  # ((price, size), ...) size 0 means remove level
    asks: tuple[tuple[float, float], ...]  # ((price, size), ...) size 0 means remove level
    prev_final_update_id: int | None = None
    exchange_timestamp: str | None = None
    received_at: str = field(default_factory=utc_now)
    source: str = "binance_l2_ws"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "first_update_id": self.first_update_id,
            "final_update_id": self.final_update_id,
            "prev_final_update_id": self.prev_final_update_id,
            "exchange_timestamp": self.exchange_timestamp,
            "received_at": self.received_at,
            "bids": [[p, s] for p, s in self.bids],
            "asks": [[p, s] for p, s in self.asks],
        }


class L2OrderBook:
    """High-performance orderbook depth manager with sequence and crossed-book verification.

    Bids are sorted descending by price; Asks are sorted ascending by price.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self.bids: dict[float, float] = {}  # price -> quantity
        self.asks: dict[float, float] = {}  # price -> quantity
        self.last_update_id: int = 0
        self.last_exchange_timestamp: str | None = None
        self.last_received_at: str | None = None

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_update_id = 0
        self.last_exchange_timestamp = None
        self.last_received_at = None

    def apply_snapshot(self, snapshot: L2Snapshot) -> None:
        """Initialize or resynchronize the orderbook from an authoritative snapshot."""
        if snapshot.symbol != self.symbol:
            raise ValueError(f"Snapshot symbol '{snapshot.symbol}' does not match book symbol '{self.symbol}'.")
        if snapshot.last_update_id <= 0:
            raise ValueError(f"Invalid snapshot last_update_id: {snapshot.last_update_id}")

        new_bids: dict[float, float] = {}
        for p, s in snapshot.bids:
            price = float(p)
            size = float(s)
            if price <= 0:
                raise InvalidOrderBookError(f"Bid price must be strictly positive: {price}")
            if size < 0:
                raise InvalidOrderBookError(f"Bid quantity cannot be negative: {size}")
            if size > 0:
                new_bids[price] = size

        new_asks: dict[float, float] = {}
        for p, s in snapshot.asks:
            price = float(p)
            size = float(s)
            if price <= 0:
                raise InvalidOrderBookError(f"Ask price must be strictly positive: {price}")
            if size < 0:
                raise InvalidOrderBookError(f"Ask quantity cannot be negative: {size}")
            if size > 0:
                new_asks[price] = size

        # Verify crossed book
        if new_bids and new_asks:
            best_bid = max(new_bids.keys())
            best_ask = min(new_asks.keys())
            if best_bid > best_ask:
                raise InvalidOrderBookError(f"Crossed orderbook in snapshot: best_bid={best_bid} > best_ask={best_ask}")

        self.bids = new_bids
        self.asks = new_asks
        self.last_update_id = snapshot.last_update_id
        self.last_exchange_timestamp = snapshot.exchange_timestamp
        self.last_received_at = snapshot.received_at

    def apply_delta(self, delta: L2DeltaUpdate) -> bool:
        """Apply an incremental depth update event with strict sequence gap checking.

        Returns True if the update was applied, False if it was older than the snapshot.
        Raises OrderBookSequenceGapError if a gap is detected.
        """
        if delta.symbol != self.symbol:
            raise ValueError(f"Delta symbol '{delta.symbol}' does not match book symbol '{self.symbol}'.")

        # 1. Check if update is completely in the past
        if delta.final_update_id <= self.last_update_id:
            return False  # Already covered by current state

        # 2. Sequence gap verification:
        # The first update ID in this event must be <= self.last_update_id + 1
        if self.last_update_id > 0 and delta.first_update_id > self.last_update_id + 1:
            raise OrderBookSequenceGapError(
                expected_start=self.last_update_id + 1,
                actual_start=delta.first_update_id,
                symbol=self.symbol,
            )

        # 3. Apply bids updates
        for p, s in delta.bids:
            price = float(p)
            size = float(s)
            if price <= 0:
                raise InvalidOrderBookError(f"Bid price must be strictly positive: {price}")
            if size < 0:
                raise InvalidOrderBookError(f"Bid quantity cannot be negative: {size}")
            if size == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size

        # 4. Apply asks updates
        for p, s in delta.asks:
            price = float(p)
            size = float(s)
            if price <= 0:
                raise InvalidOrderBookError(f"Ask price must be strictly positive: {price}")
            if size < 0:
                raise InvalidOrderBookError(f"Ask quantity cannot be negative: {size}")
            if size == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size

        # 5. Check crossed book
        if self.bids and self.asks:
            best_bid = max(self.bids.keys())
            best_ask = min(self.asks.keys())
            if best_bid > best_ask:
                raise InvalidOrderBookError(
                    f"Crossed orderbook after delta update: best_bid={best_bid} > best_ask={best_ask}"
                )

        self.last_update_id = delta.final_update_id
        self.last_exchange_timestamp = delta.exchange_timestamp
        self.last_received_at = delta.received_at
        return True

    @property
    def best_bid(self) -> float | None:
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks.keys()) if self.asks else None

    @property
    def mid_price(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        return ba - bb if bb is not None and ba is not None else None

    @property
    def spread_bps(self) -> float | None:
        s, m = self.spread, self.mid_price
        return (s / m) * 10_000.0 if s is not None and m is not None and m > 0 else None

    def top_bids(self, levels: int = 10) -> list[tuple[float, float]]:
        """Return the top N bid levels sorted descending by price [(price, size), ...]."""
        sorted_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        return [(p, self.bids[p]) for p in sorted_prices]

    def top_asks(self, levels: int = 10) -> list[tuple[float, float]]:
        """Return the top N ask levels sorted ascending by price [(price, size), ...]."""
        sorted_prices = sorted(self.asks.keys())[:levels]
        return [(p, self.asks[p]) for p in sorted_prices]

    def order_book_imbalance(self, levels: int = 5) -> float:
        """Compute Order Flow / Depth Imbalance across top N levels: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
        bid_vol = sum(size for _, size in self.top_bids(levels))
        ask_vol = sum(size for _, size in self.top_asks(levels))
        denom = bid_vol + ask_vol
        return (bid_vol - ask_vol) / denom if denom > 0 else 0.0

    def volume_weighted_microprice(self, levels: int = 5) -> float | None:
        """Compute volume-weighted microprice across top N levels."""
        top_b = self.top_bids(levels)
        top_a = self.top_asks(levels)
        if not top_b or not top_a:
            return self.mid_price

        # Microprice weighted by opposing depth: (bid_vol * best_ask + ask_vol * best_bid) / (bid_vol + ask_vol)
        best_b = top_b[0][0]
        best_a = top_a[0][0]
        b_vol = sum(s for _, s in top_b)
        a_vol = sum(s for _, s in top_a)
        total_vol = b_vol + a_vol
        if total_vol <= 0:
            return (best_b + best_a) / 2.0
        return (b_vol * best_a + a_vol * best_b) / total_vol

    def snapshot_state(self, levels: int = 10) -> dict[str, Any]:
        """Generate a complete dictionary snapshot of current orderbook metrics."""
        return {
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "order_book_imbalance_5": self.order_book_imbalance(5),
            "microprice_5": self.volume_weighted_microprice(5),
            "bid_depth_total": sum(self.bids.values()),
            "ask_depth_total": sum(self.asks.values()),
            "top_bids": self.top_bids(levels),
            "top_asks": self.top_asks(levels),
            "exchange_timestamp": self.last_exchange_timestamp,
            "received_at": self.last_received_at,
        }


class BinanceL2OrderBookRecorder:
    """Manages raw immutable persistence and REST polling / WS ingestion for Level 2 orderbook data."""

    def __init__(
        self,
        output_dir: Path,
        symbol: str = "BTCUSDT",
        endpoint: str = "https://api.binance.com/api/v3/depth",
        session: requests.Session | Any | None = None,
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self.snapshot_writer = AppendOnlyJsonl(self.output_dir / "orderbook_snapshots.jsonl")
        self.update_writer = AppendOnlyJsonl(self.output_dir / "orderbook_updates.jsonl")
        self.error_writer = AppendOnlyJsonl(self.output_dir / "collection_errors.jsonl")

        self.book = L2OrderBook(symbol=symbol)

    def fetch_snapshot_rest(self, limit: int = 100) -> L2Snapshot | None:
        """Fetch a genuine Level 2 orderbook depth snapshot from Binance REST API."""
        params = {"symbol": self.symbol, "limit": limit}
        for attempt in range(1, self.max_retries + 1):
            try:
                recv_time = utc_now()
                response = self.session.get(self.endpoint, params=params, timeout=self.timeout_seconds)
                if getattr(response, "status_code", 200) == 429:
                    raise RuntimeError("rate_limited_http_429")
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict from depth endpoint, got {type(data)}")

                last_update_id = int(data["lastUpdateId"])
                bids = tuple((float(p), float(s)) for p, s in data.get("bids", []))
                asks = tuple((float(p), float(s)) for p, s in data.get("asks", []))

                # Optional exchange timestamp from response headers if present
                date_header = getattr(response, "headers", {}).get("Date")
                exchange_ts = None
                if date_header:
                    try:
                        exchange_ts = pd.to_datetime(date_header, utc=True).isoformat()
                    except Exception:
                        exchange_ts = None

                snapshot = L2Snapshot(
                    symbol=self.symbol,
                    last_update_id=last_update_id,
                    bids=bids,
                    asks=asks,
                    exchange_timestamp=exchange_ts,
                    received_at=recv_time,
                    source="binance_l2_rest",
                )
                return snapshot
            except Exception as error:
                append_error(self.error_writer, "fetch_l2_snapshot", error, attempt=attempt, symbol=self.symbol)
        return None

    def record_snapshot(self, snapshot: L2Snapshot) -> bool:
        """Record an L2 snapshot to immutable JSONL and apply it to the internal book."""
        try:
            self.book.apply_snapshot(snapshot)
            record = snapshot.to_dict()
            return self.snapshot_writer.append(record)
        except Exception as error:
            append_error(self.error_writer, "record_snapshot_failed", error, symbol=self.symbol)
            return False

    def record_delta(self, delta: L2DeltaUpdate) -> bool:
        """Record an incremental L2 update to immutable JSONL and update the internal book."""
        try:
            applied = self.book.apply_delta(delta)
            if applied:
                record = delta.to_dict()
                return self.update_writer.append(record)
            return False
        except OrderBookSequenceGapError as gap_err:
            append_error(self.error_writer, "orderbook_sequence_gap", gap_err, symbol=self.symbol)
            logger.warning("Orderbook sequence break detected: %s. Resynchronization required.", gap_err)
            # Re-fetch snapshot to restore causal state
            fresh_snapshot = self.fetch_snapshot_rest()
            if fresh_snapshot:
                self.record_snapshot(fresh_snapshot)
                logger.info("Resynchronized orderbook via fresh snapshot (lastUpdateId=%d).", fresh_snapshot.last_update_id)
            return False
        except Exception as error:
            append_error(self.error_writer, "record_delta_failed", error, symbol=self.symbol)
            return False


class PolymarketL2Adapter:
    """Interface adapter for Polymarket CLOB WebSocket book events.

    ARCHITECTURAL LIMITATION & BLOCKER DOCUMENTATION:
    -------------------------------------------------
    Polymarket CLOB WebSocket emits:
    1. 'book' event: Initial top-of-book and level snapshot.
    2. 'price_change' event: Level updates with side ('BUY'/'SELL'), price, and size.

    CRITICAL LIMITATION:
    Unlike Binance and regulated derivatives exchanges, Polymarket CLOB WebSocket does NOT
    provide monotonic sequence numbers (e.g., lastUpdateId, U, u, pu). If a WebSocket message
    is dropped, reordered by network buffering, or delayed during reconnection, the client
    has NO mathematical mechanism to detect missing deltas without polling a full snapshot.
    Furthermore, historical Polymarket datasets in Track B contain only official settlement
    records and completely lack continuous orderbook depth streams.

    Genuine sequence-audited Level 2 capture is therefore supported and implemented on
    Binance BTCUSDT (Track A). This adapter provides the interface for Polymarket should
    authoritative sequencing become available.
    """

    def __init__(self, asset_id: str, symbol: str = "BTC-BINARY"):
        self.asset_id = asset_id
        self.symbol = symbol
        self.book = L2OrderBook(symbol=symbol)

    def process_book_event(self, data: dict[str, Any]) -> L2Snapshot | None:
        """Process an initial 'book' snapshot event from Polymarket CLOB."""
        if not isinstance(data, dict):
            return None
        event_type = data.get("event_type") or data.get("type")
        if event_type != "book":
            return None

        raw_bids = data.get("bids", [])
        raw_asks = data.get("asks", [])
        bids = []
        for level in raw_bids:
            p = float(level.get("price", 0))
            s = float(level.get("size", 0))
            if p > 0 and s > 0:
                bids.append((p, s))

        asks = []
        for level in raw_asks:
            p = float(level.get("price", 0))
            s = float(level.get("size", 0))
            if p > 0 and s > 0:
                asks.append((p, s))

        # Synthetic sequence number because Polymarket does not provide one
        snapshot = L2Snapshot(
            symbol=self.symbol,
            last_update_id=int(datetime.now(timezone.utc).timestamp() * 1000),
            bids=tuple(sorted(bids, key=lambda x: x[0], reverse=True)),
            asks=tuple(sorted(asks, key=lambda x: x[0])),
            exchange_timestamp=data.get("timestamp"),
            received_at=utc_now(),
            source="polymarket_clob_ws",
        )
        self.book.apply_snapshot(snapshot)
        return snapshot

    def process_price_change(self, data: dict[str, Any]) -> bool:
        """Process a 'price_change' event from Polymarket CLOB."""
        if not isinstance(data, dict):
            return False
        event_type = data.get("event_type") or data.get("type")
        if event_type != "price_change":
            return False

        changes = data.get("price_changes", [])
        for ch in changes:
            if ch.get("asset_id") != self.asset_id:
                continue
            price = float(ch.get("price", 0))
            size = float(ch.get("size", 0))
            side = str(ch.get("side", "")).upper()
            if price <= 0:
                continue
            if side == "BUY":
                if size == 0:
                    self.book.bids.pop(price, None)
                else:
                    self.book.bids[price] = size
            elif side == "SELL":
                if size == 0:
                    self.book.asks.pop(price, None)
                else:
                    self.book.asks[price] = size
        return True
