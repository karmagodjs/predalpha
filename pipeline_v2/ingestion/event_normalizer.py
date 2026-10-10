"""
Canonical Event Normalizer for Pipeline V2.

This module implements Phase 11 of the PredAlpha-HFT clean rebuild:
Reads raw market event JSONL feeds, strictly validates quote events,
preserves raw provenance, maintains L2 top-of-book state, enforces
deterministic ordering (timestamp_ms ASC, sequence_id ASC), deduplicates
events with payload conflict tracking, and outputs canonical event records.

Zero dependency on legacy market_data/ or deprecated schemas.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger("pipeline_v2.event_normalizer")


class RejectReason(str, Enum):
    """Enumeration of strict record rejection reasons."""
    MALFORMED_JSON = "MALFORMED_JSON"
    EMPTY_RECORD = "EMPTY_RECORD"
    NON_QUOTE_EVENT = "NON_QUOTE_EVENT"
    MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    OUT_OF_BOUNDS_TIMESTAMP = "OUT_OF_BOUNDS_TIMESTAMP"
    MISSING_MARKET_ID = "MISSING_MARKET_ID"
    MISSING_ASSET_ID = "MISSING_ASSET_ID"
    MISSING_PRICE = "MISSING_PRICE"
    INVALID_PRICE = "INVALID_PRICE"
    IMPOSSIBLE_PRICE = "IMPOSSIBLE_PRICE"
    CROSSED_BOOK = "CROSSED_BOOK"
    MISSING_SIZE = "MISSING_SIZE"
    INVALID_SIZE = "INVALID_SIZE"
    NON_POSITIVE_DEPTH = "NON_POSITIVE_DEPTH"
    ORDERBOOK_NOT_INITIALIZED = "ORDERBOOK_NOT_INITIALIZED"


@dataclass(frozen=True)
class CanonicalEvent:
    """
    Standardized, validated canonical market quote event.

    Attributes:
        timestamp_ms: Exact physical millisecond timestamp (int64).
        sequence_id: Deterministic ordering identifier (int64).
                     Note: The raw feed does not provide an exchange sequence number;
                     this is a deterministic local arrival/line identifier.
        market_id: Market condition/contract identifier.
        asset_id: Specific outcome token identifier.
        bid: Top-of-book best bid price (float64).
        ask: Top-of-book best ask price (float64).
        bid_size: Available volume/size at the best bid (float64).
        ask_size: Available volume/size at the best ask (float64).
        source_file: Provenance: origin file path or identifier.
        source_line: Provenance: 1-indexed line number in source file.
        source_record_idx: Provenance: 0-indexed sub-event within source line.
    """
    timestamp_ms: int
    sequence_id: int
    market_id: str
    asset_id: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    source_file: str
    source_line: int
    source_record_idx: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert canonical event to dictionary representation."""
        return asdict(self)

    def dedup_key(self) -> Tuple[str, str, int, int]:
        """
        Deduplication key per requirement:
        (market_id, asset_id, timestamp_ms, sequence_id)
        """
        return (self.market_id, self.asset_id, self.timestamp_ms, self.sequence_id)

    def payload_tuple(self) -> Tuple[float, float, float, float]:
        """Comparable payload tuple for detecting conflicting duplicates."""
        return (
            round(self.bid, 8),
            round(self.ask, 8),
            round(self.bid_size, 8),
            round(self.ask_size, 8),
        )


@dataclass
class ConflictingDuplicateAudit:
    """Audit entry for duplicate records arriving with conflicting payloads."""
    key: Tuple[str, str, int, int]
    first_payload: Tuple[float, float, float, float]
    conflicting_payload: Tuple[float, float, float, float]
    first_provenance: Tuple[str, int, int]
    conflicting_provenance: Tuple[str, int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": {
                "market_id": self.key[0],
                "asset_id": self.key[1],
                "timestamp_ms": self.key[2],
                "sequence_id": self.key[3],
            },
            "first_payload": {
                "bid": self.first_payload[0],
                "ask": self.first_payload[1],
                "bid_size": self.first_payload[2],
                "ask_size": self.first_payload[3],
            },
            "conflicting_payload": {
                "bid": self.conflicting_payload[0],
                "ask": self.conflicting_payload[1],
                "bid_size": self.conflicting_payload[2],
                "ask_size": self.conflicting_payload[3],
            },
            "first_provenance": {
                "source_file": self.first_provenance[0],
                "source_line": self.first_provenance[1],
                "source_record_idx": self.first_provenance[2],
            },
            "conflicting_provenance": {
                "source_file": self.conflicting_provenance[0],
                "source_line": self.conflicting_provenance[1],
                "source_record_idx": self.conflicting_provenance[2],
            },
        }


@dataclass
class LocalOrderBook:
    """
    Self-contained L2 order book tracking top-of-book for an asset.
    Preserves exact price and size levels from book snapshots and delta updates.
    """
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    initialized: bool = False

    def reset(self, bids_data: List[Dict[str, Any]], asks_data: List[Dict[str, Any]]) -> None:
        """Reset book with full snapshot levels."""
        self.bids.clear()
        self.asks.clear()

        for b in bids_data:
            try:
                p = float(b["price"])
                s = float(b["size"])
                if p > 0.0 and s > 0.0:
                    self.bids[p] = s
            except (ValueError, TypeError, KeyError):
                continue

        for a in asks_data:
            try:
                p = float(a["price"])
                s = float(a["size"])
                if p > 0.0 and s > 0.0:
                    self.asks[p] = s
            except (ValueError, TypeError, KeyError):
                continue

        self.initialized = True

    def update_bid(self, price: float, size: float) -> None:
        """Update or remove a bid level."""
        if size <= 0.0:
            self.bids.pop(price, None)
        else:
            self.bids[price] = size

    def update_ask(self, price: float, size: float) -> None:
        """Update or remove an ask level."""
        if size <= 0.0:
            self.asks.pop(price, None)
        else:
            self.asks[price] = size

    def get_bbo(self) -> Tuple[Optional[float], Optional[float], float, float]:
        """
        Extract top-of-book best bid, best ask, and their respective sizes.
        Returns:
            (best_bid, best_ask, best_bid_size, best_ask_size)
        """
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None

        best_bid_size = self.bids[best_bid] if best_bid is not None else 0.0
        best_ask_size = self.asks[best_ask] if best_ask is not None else 0.0

        return best_bid, best_ask, best_bid_size, best_ask_size


@dataclass
class NormalizationReport:
    """Summary and audit verification of event normalization."""
    total_input_rows: int = 0
    total_records_parsed: int = 0
    valid_canonical_events: int = 0
    rejected_by_reason: Dict[str, int] = field(default_factory=dict)
    duplicates_dropped_identical: int = 0
    duplicates_dropped_conflicting: int = 0
    min_timestamp_ms: Optional[int] = None
    max_timestamp_ms: Optional[int] = None
    is_monotonic_increasing: bool = True
    conflicting_duplicates: List[ConflictingDuplicateAudit] = field(default_factory=list)

    @property
    def total_duplicates_dropped(self) -> int:
        return self.duplicates_dropped_identical + self.duplicates_dropped_conflicting

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input_rows": self.total_input_rows,
            "total_records_parsed": self.total_records_parsed,
            "valid_canonical_events": self.valid_canonical_events,
            "rejected_by_reason": dict(self.rejected_by_reason),
            "duplicates_dropped_identical": self.duplicates_dropped_identical,
            "duplicates_dropped_conflicting": self.duplicates_dropped_conflicting,
            "total_duplicates_dropped": self.total_duplicates_dropped,
            "min_timestamp_ms": self.min_timestamp_ms,
            "max_timestamp_ms": self.max_timestamp_ms,
            "is_monotonic_increasing": self.is_monotonic_increasing,
            "conflicting_duplicates_count": len(self.conflicting_duplicates),
            "conflicting_duplicates": [c.to_dict() for c in self.conflicting_duplicates],
        }

    def to_markdown(self) -> str:
        """Format report into GitHub-flavored markdown."""
        rejection_rows = ""
        for reason, count in sorted(self.rejected_by_reason.items(), key=lambda x: -x[1]):
            rejection_rows += f"| `{reason}` | {count} |\n"
        if not rejection_rows:
            rejection_rows = "| None | 0 |\n"

        conflict_note = f"⚠️ {self.duplicates_dropped_conflicting} conflicting duplicate(s) detected!" if self.duplicates_dropped_conflicting > 0 else "None (clean)"

        return (
            f"# Canonical Event Normalization Report\n\n"
            f"- **Total Input Raw Rows / Lines**: {self.total_input_rows:,}\n"
            f"- **Total Candidate Records Parsed**: {self.total_records_parsed:,}\n"
            f"- **Valid Canonical Events**: {self.valid_canonical_events:,}\n"
            f"- **Identical Duplicates Dropped**: {self.duplicates_dropped_identical:,}\n"
            f"- **Conflicting Duplicates Dropped**: {self.duplicates_dropped_conflicting:,} ({conflict_note})\n"
            f"- **Earliest Event Timestamp (`min_timestamp_ms`)**: {self.min_timestamp_ms}\n"
            f"- **Latest Event Timestamp (`max_timestamp_ms`)**: {self.max_timestamp_ms}\n"
            f"- **Monotonic Non-Decreasing Verification**: `{'PASS' if self.is_monotonic_increasing else 'FAIL'}`\n\n"
            f"### Rejected Records by Reason\n\n"
            f"| Rejection Reason | Count |\n"
            f"| :--- | :--- |\n"
            f"{rejection_rows}"
        )


class EventNormalizer:
    """
    Streaming canonical event parser, validator, deduplicator, and normalizer.

    Features:
    - Parses all Polymarket envelope formats (Envelope A, Envelope B, direct records).
    - Maintains local L2 order books for delta updates to compute exact BBO and depth.
    - Strictly validates timestamps, prices, sizes, and crossed books.
    - Preserves provenance (source_file, source_line, source_record_idx).
    - Sorts deterministically by (timestamp_ms ASC, sequence_id ASC).
    - Enforces deduplication key (market_id, asset_id, timestamp_ms, sequence_id).
    - Emits normalization audit report.
    """

    def __init__(
        self,
        target_asset_id: Optional[str] = None,
        min_price: float = 0.0,
        max_price: Optional[float] = 1.0,
        min_valid_timestamp_ms: int = 1_577_836_800_000,  # 2020-01-01
        max_valid_timestamp_ms: int = 2_051_222_400_000,  # 2035-01-01
    ):
        """
        Initialize normalizer.

        Args:
            target_asset_id: If specified, filters events exclusively to this asset.
            min_price: Minimum allowed price boundary (exclusive). Default 0.0.
            max_price: Maximum allowed price boundary (exclusive) for binary outcome tokens. Default 1.0.
            min_valid_timestamp_ms: Earliest valid unix ms.
            max_valid_timestamp_ms: Latest valid unix ms.
        """
        self.target_asset_id = str(target_asset_id) if target_asset_id is not None else None
        self.min_price = min_price
        self.max_price = max_price
        self.min_valid_timestamp_ms = min_valid_timestamp_ms
        self.max_valid_timestamp_ms = max_valid_timestamp_ms

        self.books: Dict[str, LocalOrderBook] = {}
        self.seen_keys: Dict[Tuple[str, str, int, int], CanonicalEvent] = {}
        self.report = NormalizationReport()

    def reset_state(self) -> None:
        """Reset internal books, seen keys, and report counters."""
        self.books.clear()
        self.seen_keys.clear()
        self.report = NormalizationReport()

    def _record_rejection(self, reason: Union[RejectReason, str]) -> None:
        """Increment count for a rejection reason."""
        reason_str = reason.value if isinstance(reason, RejectReason) else str(reason)
        self.report.rejected_by_reason[reason_str] = (
            self.report.rejected_by_reason.get(reason_str, 0) + 1
        )

    def validate_and_create(
        self,
        timestamp_ms_raw: Any,
        sequence_id_raw: Any,
        market_id_raw: Any,
        asset_id_raw: Any,
        bid_raw: Any,
        ask_raw: Any,
        bid_size_raw: Any,
        ask_size_raw: Any,
        source_file: str,
        source_line: int,
        source_record_idx: int,
    ) -> Tuple[Optional[CanonicalEvent], Optional[str]]:
        """
        Execute strict validation checks on candidate fields.
        Returns (CanonicalEvent, None) if valid, or (None, rejection_reason) if rejected.
        """
        # 1. Timestamp validation
        if timestamp_ms_raw is None or timestamp_ms_raw == "":
            self._record_rejection(RejectReason.MISSING_TIMESTAMP)
            return None, RejectReason.MISSING_TIMESTAMP.value

        try:
            timestamp_ms = int(timestamp_ms_raw)
        except (ValueError, TypeError):
            self._record_rejection(RejectReason.INVALID_TIMESTAMP)
            return None, RejectReason.INVALID_TIMESTAMP.value

        if not (self.min_valid_timestamp_ms <= timestamp_ms <= self.max_valid_timestamp_ms):
            self._record_rejection(RejectReason.OUT_OF_BOUNDS_TIMESTAMP)
            return None, RejectReason.OUT_OF_BOUNDS_TIMESTAMP.value

        # 2. Sequence ID validation (deterministic arrival/line index)
        try:
            sequence_id = int(sequence_id_raw)
        except (ValueError, TypeError):
            sequence_id = (source_line * 1000) + source_record_idx

        # 3. Market and Asset identifiers
        if not market_id_raw or str(market_id_raw).strip() == "":
            self._record_rejection(RejectReason.MISSING_MARKET_ID)
            return None, RejectReason.MISSING_MARKET_ID.value
        market_id = str(market_id_raw).strip()

        if not asset_id_raw or str(asset_id_raw).strip() == "":
            self._record_rejection(RejectReason.MISSING_ASSET_ID)
            return None, RejectReason.MISSING_ASSET_ID.value
        asset_id = str(asset_id_raw).strip()

        # Target asset filter
        if self.target_asset_id is not None and asset_id != self.target_asset_id:
            # Asset filtered out intentionally; not counted as invalid record
            return None, None

        # 4. Price validation
        if bid_raw is None or ask_raw is None:
            self._record_rejection(RejectReason.MISSING_PRICE)
            return None, RejectReason.MISSING_PRICE.value

        try:
            bid = float(bid_raw)
            ask = float(ask_raw)
        except (ValueError, TypeError):
            self._record_rejection(RejectReason.INVALID_PRICE)
            return None, RejectReason.INVALID_PRICE.value

        if math.isnan(bid) or math.isnan(ask) or math.isinf(bid) or math.isinf(ask):
            self._record_rejection(RejectReason.INVALID_PRICE)
            return None, RejectReason.INVALID_PRICE.value

        # Price range bounds
        if bid <= self.min_price or ask <= self.min_price:
            self._record_rejection(RejectReason.IMPOSSIBLE_PRICE)
            return None, RejectReason.IMPOSSIBLE_PRICE.value

        if self.max_price is not None:
            if bid >= self.max_price or ask >= self.max_price:
                self._record_rejection(RejectReason.IMPOSSIBLE_PRICE)
                return None, RejectReason.IMPOSSIBLE_PRICE.value

        # Crossed or locked book: bid >= ask is strictly invalid in an orderly book
        if bid >= ask:
            self._record_rejection(RejectReason.CROSSED_BOOK)
            return None, RejectReason.CROSSED_BOOK.value

        # 5. Size and Depth validation
        if bid_size_raw is None or ask_size_raw is None:
            self._record_rejection(RejectReason.MISSING_SIZE)
            return None, RejectReason.MISSING_SIZE.value

        try:
            bid_size = float(bid_size_raw)
            ask_size = float(ask_size_raw)
        except (ValueError, TypeError):
            self._record_rejection(RejectReason.INVALID_SIZE)
            return None, RejectReason.INVALID_SIZE.value

        if math.isnan(bid_size) or math.isnan(ask_size) or math.isinf(bid_size) or math.isinf(ask_size):
            self._record_rejection(RejectReason.INVALID_SIZE)
            return None, RejectReason.INVALID_SIZE.value

        if bid_size <= 0.0 or ask_size <= 0.0:
            self._record_rejection(RejectReason.NON_POSITIVE_DEPTH)
            return None, RejectReason.NON_POSITIVE_DEPTH.value

        event = CanonicalEvent(
            timestamp_ms=timestamp_ms,
            sequence_id=sequence_id,
            market_id=market_id,
            asset_id=asset_id,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            source_file=source_file,
            source_line=source_line,
            source_record_idx=source_record_idx,
        )
        return event, None

    def _get_or_create_book(self, asset_id: str) -> LocalOrderBook:
        """Retrieve or initialize the order book for an asset."""
        if asset_id not in self.books:
            self.books[asset_id] = LocalOrderBook()
        return self.books[asset_id]

    def parse_event_dict(
        self,
        event_dict: Dict[str, Any],
        source_line: int,
        source_record_idx: int,
        source_file: str,
        outer_asset_id: Optional[str] = None,
    ) -> List[Tuple[Optional[CanonicalEvent], Optional[str]]]:
        """
        Parse an individual event dictionary from Polymarket or synthetic test fixture.
        Returns a list of candidate results (event, rejection_reason).
        """
        self.report.total_records_parsed += 1

        if not isinstance(event_dict, dict):
            self._record_rejection(RejectReason.EMPTY_RECORD)
            return [(None, RejectReason.EMPTY_RECORD.value)]

        # Process standard Polymarket protocol event types first
        event_type = event_dict.get("event_type")
        market_id = event_dict.get("market")
        ts_raw = event_dict.get("timestamp")
        seq_raw = event_dict.get("sequence_id", (source_line * 1000) + source_record_idx)

        if event_type == "book":
            asset_id = event_dict.get("asset_id", outer_asset_id)
            if not asset_id:
                self._record_rejection(RejectReason.MISSING_ASSET_ID)
                return [(None, RejectReason.MISSING_ASSET_ID.value)]

            book = self._get_or_create_book(asset_id)
            book.reset(event_dict.get("bids", []), event_dict.get("asks", []))
            best_bid, best_ask, bid_size, ask_size = book.get_bbo()

            ev, rej = self.validate_and_create(
                timestamp_ms_raw=ts_raw,
                sequence_id_raw=seq_raw,
                market_id_raw=market_id,
                asset_id_raw=asset_id,
                bid_raw=best_bid,
                ask_raw=best_ask,
                bid_size_raw=bid_size,
                ask_size_raw=ask_size,
                source_file=source_file,
                source_line=source_line,
                source_record_idx=source_record_idx,
            )
            return [(ev, rej)]

        elif event_type == "price_change":
            results = []
            price_changes = event_dict.get("price_changes", [])
            if not price_changes:
                self._record_rejection(RejectReason.NON_QUOTE_EVENT)
                return [(None, RejectReason.NON_QUOTE_EVENT.value)]

            for pc_idx, change in enumerate(price_changes):
                asset_id = change.get("asset_id")
                if not asset_id:
                    self._record_rejection(RejectReason.MISSING_ASSET_ID)
                    results.append((None, RejectReason.MISSING_ASSET_ID.value))
                    continue

                if self.target_asset_id is not None and asset_id != self.target_asset_id:
                    continue

                book = self._get_or_create_book(asset_id)
                side = change.get("side", "").upper()
                try:
                    price = float(change["price"])
                    size = float(change["size"])
                except (ValueError, TypeError, KeyError):
                    self._record_rejection(RejectReason.INVALID_PRICE)
                    results.append((None, RejectReason.INVALID_PRICE.value))
                    continue

                if side == "BUY":
                    book.update_bid(price, size)
                elif side == "SELL":
                    book.update_ask(price, size)

                best_bid, best_ask, bid_size, ask_size = book.get_bbo()
                effective_seq = (source_line * 1000) + source_record_idx + pc_idx

                ev, rej = self.validate_and_create(
                    timestamp_ms_raw=ts_raw,
                    sequence_id_raw=change.get("sequence_id", effective_seq),
                    market_id_raw=market_id,
                    asset_id_raw=asset_id,
                    bid_raw=best_bid,
                    ask_raw=best_ask,
                    bid_size_raw=bid_size,
                    ask_size_raw=ask_size,
                    source_file=source_file,
                    source_line=source_line,
                    source_record_idx=source_record_idx + pc_idx,
                )
                results.append((ev, rej))

            return results

        elif event_type in ("last_trade_price", "trade"):
            # Execution event; does not contain quote book updates
            self._record_rejection(RejectReason.NON_QUOTE_EVENT)
            return [(None, RejectReason.NON_QUOTE_EVENT.value)]

        elif event_type is not None:
            self._record_rejection(RejectReason.NON_QUOTE_EVENT)
            return [(None, RejectReason.NON_QUOTE_EVENT.value)]

        # Direct flat quote record fallback (synthetic test fixtures or flat quote feeds)
        is_direct_quote = (
            ("bid" in event_dict or "best_bid" in event_dict)
            and ("ask" in event_dict or "best_ask" in event_dict)
        )
        if is_direct_quote:
            ts_raw = event_dict.get("timestamp_ms", event_dict.get("timestamp"))
            seq_raw = event_dict.get("sequence_id", (source_line * 1000) + source_record_idx)
            mkt_raw = event_dict.get("market_id", event_dict.get("market"))
            ast_raw = event_dict.get("asset_id", outer_asset_id)
            bid_raw = event_dict.get("bid", event_dict.get("best_bid"))
            ask_raw = event_dict.get("ask", event_dict.get("best_ask"))
            bs_raw = event_dict.get("bid_size", event_dict.get("best_bid_size"))
            as_raw = event_dict.get("ask_size", event_dict.get("best_ask_size"))

            ev, rej = self.validate_and_create(
                timestamp_ms_raw=ts_raw,
                sequence_id_raw=seq_raw,
                market_id_raw=mkt_raw,
                asset_id_raw=ast_raw,
                bid_raw=bid_raw,
                ask_raw=ask_raw,
                bid_size_raw=bs_raw,
                ask_size_raw=as_raw,
                source_file=source_file,
                source_line=source_line,
                source_record_idx=source_record_idx,
            )
            return [(ev, rej)]

        self._record_rejection(RejectReason.NON_QUOTE_EVENT)
        return [(None, RejectReason.NON_QUOTE_EVENT.value)]

    def parse_line(
        self,
        line: str,
        source_line: int,
        source_file: str,
    ) -> List[Tuple[Optional[CanonicalEvent], Optional[str]]]:
        """
        Parse a single JSONL line into candidate canonical events.
        """
        line_clean = line.strip()
        if not line_clean:
            self._record_rejection(RejectReason.EMPTY_RECORD)
            return [(None, RejectReason.EMPTY_RECORD.value)]

        try:
            payload = json.loads(line_clean)
        except json.JSONDecodeError:
            self._record_rejection(RejectReason.MALFORMED_JSON)
            return [(None, RejectReason.MALFORMED_JSON.value)]

        if not isinstance(payload, dict):
            self._record_rejection(RejectReason.EMPTY_RECORD)
            return [(None, RejectReason.EMPTY_RECORD.value)]

        # Envelope A: {"received_at": ..., "asset_id": ..., "event": dict | list}
        if "event" in payload:
            outer_asset_id = payload.get("asset_id")
            ev_data = payload["event"]
            items = ev_data if isinstance(ev_data, list) else [ev_data]
            results = []
            for sub_idx, item in enumerate(items):
                results.extend(
                    self.parse_event_dict(
                        event_dict=item,
                        source_line=source_line,
                        source_record_idx=sub_idx,
                        source_file=source_file,
                        outer_asset_id=outer_asset_id,
                    )
                )
            return results

        # Envelope B: {"timestamp": ..., "data": dict | list}
        if "data" in payload:
            d_data = payload["data"]
            if not d_data:
                self._record_rejection(RejectReason.EMPTY_RECORD)
                return [(None, RejectReason.EMPTY_RECORD.value)]
            items = d_data if isinstance(d_data, list) else [d_data]
            results = []
            for sub_idx, item in enumerate(items):
                results.extend(
                    self.parse_event_dict(
                        event_dict=item,
                        source_line=source_line,
                        source_record_idx=sub_idx,
                        source_file=source_file,
                    )
                )
            return results

        # Direct event dictionary
        return self.parse_event_dict(
            event_dict=payload,
            source_line=source_line,
            source_record_idx=0,
            source_file=source_file,
        )

    def process_candidate(self, candidate: CanonicalEvent) -> Optional[CanonicalEvent]:
        """
        Deduplication and conflict audit:
        Key: (market_id, asset_id, timestamp_ms, sequence_id)
        Keep first received event.
        Record audit entry if duplicate payload differs.
        """
        key = candidate.dedup_key()
        if key in self.seen_keys:
            existing = self.seen_keys[key]
            if candidate.payload_tuple() == existing.payload_tuple():
                self.report.duplicates_dropped_identical += 1
            else:
                self.report.duplicates_dropped_conflicting += 1
                conflict_audit = ConflictingDuplicateAudit(
                    key=key,
                    first_payload=existing.payload_tuple(),
                    conflicting_payload=candidate.payload_tuple(),
                    first_provenance=(existing.source_file, existing.source_line, existing.source_record_idx),
                    conflicting_provenance=(candidate.source_file, candidate.source_line, candidate.source_record_idx),
                )
                self.report.conflicting_duplicates.append(conflict_audit)
                logger.warning(
                    f"Conflicting duplicate detected for key {key}: "
                    f"First payload {existing.payload_tuple()} vs New {candidate.payload_tuple()}"
                )
            return None

        self.seen_keys[key] = candidate
        return candidate

    def normalize_stream(
        self,
        lines_iterable: Iterable[str],
        source_file: str = "stream",
    ) -> List[CanonicalEvent]:
        """
        Stream, parse, validate, deduplicate, and deterministically sort events.
        """
        collected_events: List[CanonicalEvent] = []

        for line_no, line in enumerate(lines_iterable, start=1):
            self.report.total_input_rows += 1
            candidates = self.parse_line(line=line, source_line=line_no, source_file=source_file)
            for ev, _ in candidates:
                if ev is not None:
                    accepted = self.process_candidate(ev)
                    if accepted is not None:
                        collected_events.append(accepted)

        # Deterministic ordering: timestamp_ms ASC, sequence_id ASC
        collected_events.sort(key=lambda e: (e.timestamp_ms, e.sequence_id))

        self.report.valid_canonical_events = len(collected_events)

        if collected_events:
            self.report.min_timestamp_ms = collected_events[0].timestamp_ms
            self.report.max_timestamp_ms = max(e.timestamp_ms for e in collected_events)
            self.report.is_monotonic_increasing = all(
                collected_events[i].timestamp_ms <= collected_events[i + 1].timestamp_ms
                for i in range(len(collected_events) - 1)
            )
        else:
            self.report.min_timestamp_ms = None
            self.report.max_timestamp_ms = None
            self.report.is_monotonic_increasing = True

        return collected_events

    def normalize_file(
        self,
        input_file: Union[Path, str],
        output_parquet: Optional[Union[Path, str]] = None,
    ) -> Tuple[pd.DataFrame, NormalizationReport]:
        """
        Normalize a raw JSONL file into canonical events and optionally persist to Parquet.
        """
        in_path = Path(input_file)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        source_name = in_path.name
        self.reset_state()

        with open(in_path, "r", encoding="utf-8") as f:
            events = self.normalize_stream(f, source_file=source_name)

        if events:
            df = pd.DataFrame([e.to_dict() for e in events])
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
        else:
            df = pd.DataFrame(columns=[
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
            ]).astype({
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

        if output_parquet is not None:
            out_path = Path(output_parquet)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False, engine="pyarrow")
            logger.info(f"Saved {len(df)} canonical events to {out_path}")

        return df, self.report


def main() -> None:
    """Command-line interface for canonical event normalization."""
    import argparse

    parser = argparse.ArgumentParser(description="Normalize raw Polymarket JSONL into canonical events.")
    parser.add_argument("--input", "-i", required=True, help="Path to input raw JSONL file")
    parser.add_argument("--output", "-o", default=None, help="Path to output canonical Parquet file")
    parser.add_argument("--target-asset", "-a", default=None, help="Target asset ID to filter (optional)")
    parser.add_argument("--report", "-r", default=None, help="Path to save markdown report (optional)")

    args = parser.parse_args()

    normalizer = EventNormalizer(target_asset_id=args.target_asset)
    df, report = normalizer.normalize_file(args.input, output_parquet=args.output)

    print("\n" + report.to_markdown())

    if args.report:
        rep_path = Path(args.report)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        rep_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\nReport written to {rep_path}")


if __name__ == "__main__":
    main()

