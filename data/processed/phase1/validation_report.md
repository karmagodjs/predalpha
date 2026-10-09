# PredAlpha Phase 1 — Dataset Validation Report

- **Dataset**: Canonical PredAlpha Dataset (Phase 1)
- **Validation Time**: 2026-10-04T13:54:22.554498
- **Overall Status**: ✅ PASSED
- **Checks Passed**: 8/8
- **Errors**: 0 | **Warnings**: 0

## Summary of Validation Checks

| Check Name | Status | Details |
|:-----------|:-------|:--------|
| `missing_values` | ✅ PASS | No missing values found across 11 columns. |
| `duplicate_records` | ✅ PASS | No duplicate records found. |
| `timestamp_integrity` | ✅ PASS | Timestamps are ordered, unique, and continuous. |
| `market_data_integrity` | ✅ PASS | Market prices and orderbook integrity valid. |
| `finite_values` | ✅ PASS | All 9 numeric columns contain only finite values. |
| `constant_features` | ✅ PASS | All 9 features exhibit sufficient variability. |
| `label_distribution` | ✅ PASS | Label distribution valid with 3 classes: {'FLAT': 244, 'UP': 33, 'DOWN': 12}. |
| `target_leakage` | ✅ PASS | No target leakage or future information detected in features. |

## Split Boundary Verification

- **Train Bound**: 2026-10-02 12:44:12+00:00 to 2026-10-02 12:47:33+00:00 (202 rows)
- **Train -> Val Gap**: 6.0s (minimum purge required: 5s)
- **Val Bound**: 2026-10-02 12:47:39+00:00 to 2026-10-02 12:48:16+00:00 (38 rows)
- **Val -> Test Gap**: 6.0s (minimum purge required: 5s)
- **Test Bound**: 2026-10-02 12:48:22+00:00 to 2026-10-02 12:49:00+00:00 (39 rows)
