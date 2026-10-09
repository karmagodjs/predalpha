# PredAlpha Phase 3 — Session Validation Report: `session_live_002`

- **Validation Time**: 2026-10-04T14:56:12.675387+00:00
- **Overall Status**: ✅ PASSED
- **Raw Observations**: 60
- **Valid Observations**: 60
- **Depth Snapshots**: 61
- **Time Range**: 2026-10-04 14:55:14.094252+00:00 to 2026-10-04 14:56:12.158505+00:00

## Integrity Checks Breakdown

| Check Name | Status | Details |
|:-----------|:-------|:--------|
| `malformed_json` | ✅ PASS | malformed_count=0 |
| `required_fields` | ✅ PASS | missing_count=0 |
| `positive_prices` | ✅ PASS | invalid_count=0 |
| `high_low_validity` | ✅ PASS | invalid_count=0 |
| `non_negative_volume` | ✅ PASS | invalid_count=0 |
| `duplicate_records` | ✅ PASS | duplicate_count=0 |
| `timestamp_ordering` | ✅ PASS | out_of_order_count=0 |
| `duplicate_timestamps` | ✅ PASS | duplicate_count=0 |
| `missing_intervals` | ✅ PASS | gaps_over_expected=0, max_gap_seconds=1.63832 |
| `candle_source_timestamps` | ✅ PASS | out_of_order_count=0, intra_candle_updates=58, unique_candle_count=2 |
| `l2_uncrossed_books` | ✅ PASS | crossed_count=0 |
| `l2_monotonic_sequence` | ✅ PASS | non_monotonic_count=0 |
| `l2_valid_levels` | ✅ PASS | invalid_count=0 |
