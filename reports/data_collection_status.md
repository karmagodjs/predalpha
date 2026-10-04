# Data collection status

## Track A

{
  "rows": 0,
  "missing_or_invalid_timestamps": 0,
  "duplicate_timestamps": 0,
  "first_timestamp": null,
  "last_timestamp": null,
  "out_of_order_records": 0,
  "gaps_over_expected": 0,
  "max_gap_seconds": 0.0,
  "collection_errors": 0
}

## Track B

{
  "markets": 0,
  "resolved": 0,
  "unresolved": 0,
  "events": 0,
  "collection_errors": 0,
  "date_ranges": {}
}

Repeated snapshots are events, not independent market samples. A market counts as resolved only when official CLOB settlement evidence has exactly one token marked `winner: true`.
