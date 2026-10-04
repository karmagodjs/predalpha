"""Shared immutable-storage and validation helpers for collection jobs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppendOnlyJsonl:
    """Append records safely and skip only exact, previously written record ids."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids = self._load_ids()

    def _load_ids(self) -> set[str]:
        if not self.path.exists():
            return set()
        ids = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if "record_id" in record:
                    ids.add(record["record_id"])
            except json.JSONDecodeError:
                # Existing corrupt lines remain immutable; callers record the issue separately.
                continue
        return ids

    @staticmethod
    def record_id(record: dict[str, Any]) -> str:
        basis = {key: value for key, value in record.items() if key not in {"collected_at", "record_id"}}
        encoded = json.dumps(basis, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def append(self, record: dict[str, Any]) -> bool:
        record = dict(record)
        record.setdefault("collected_at", utc_now())
        record.setdefault("record_id", self.record_id(record))
        if record["record_id"] in self._ids:
            return False
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self._ids.add(record["record_id"])
        return True


def append_error(error_writer: AppendOnlyJsonl, stage: str, error: Exception | str, **context: Any) -> None:
    error_writer.append({"stage": stage, "error": str(error), "context": context})


def jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            records.append({"_parse_error": str(error), "_line_number": line_number})
    return records


def timestamp_quality(records: Iterable[dict[str, Any]], timestamp_field: str, expected_gap_seconds: float | None = None) -> dict[str, Any]:
    records = list(records)
    values = [record.get(timestamp_field) for record in records]
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    valid = sorted(value for value in parsed if not pd.isna(value))
    gaps = [(right - left).total_seconds() for left, right in zip(valid, valid[1:])]
    return {
        "rows": len(records), "missing_or_invalid_timestamps": int(pd.isna(parsed).sum()),
        "duplicate_timestamps": int(pd.Series(parsed).duplicated().sum()),
        "first_timestamp": str(valid[0]) if valid else None, "last_timestamp": str(valid[-1]) if valid else None,
        "out_of_order_records": sum(left > right for left, right in zip(parsed, parsed[1:]) if not pd.isna(left) and not pd.isna(right)),
        "gaps_over_expected": 0 if expected_gap_seconds is None else sum(gap > expected_gap_seconds * 1.5 for gap in gaps),
        "max_gap_seconds": max(gaps) if gaps else 0.0,
    }
