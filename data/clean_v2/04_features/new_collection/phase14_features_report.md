# Phase 14: Causal Feature Engineering Audit Report

**Generated UTC**: `2026-10-06T03:37:38.425052+00:00`  
**Pipeline**: `pipeline_v2/features`  
**Execution Runtime**: 0.73 seconds  
**Dataset SHA-256**: `b070849e02607778a5b0622658ba6b1d00cea962ae2691b43dcef86008d70392`  
**Maximum Feature Lookback**: 5000 ms (5 seconds)

---

## 1. Executive Summary

Phase 14 causal feature engineering was executed across all **19 retained production markets** from Phase 13.
- **Input Dataset**: `data/clean_v2/03_labeled_5s/new_collection/` (4,970 rows)
- **Output Dataset**: `data/clean_v2/04_features/new_collection/` (4,970 rows, 21 columns)
- **Whitelisted Causal Features**: 11 features generated strictly using backward-looking operations.
- **Warm-up Rows Tracked**: 190 rows (5 warm-up rows per asset stream, exactly 38 asset streams = 190 warm-up rows). Zero backward filling or data fabrication.

---

## 2. Invariant & Causality Verification

| Invariant | Requirement | Actual Status |
| :--- | :--- | :--- |
| **Strict Causality** | Feature at $T$ uses ONLY data $\le T$ | `PASS (strictly causal, zero lookahead)` |
| **Chronological Ordering** | Monotonically ascending timestamps | `PASS (strictly chronological)` |
| **Label Separation** | Target label quarantined from feature inputs | `PASS (0 target/future columns ingested)` |
| **Label Preservation** | Phase 13 physical 5-second labels untouched | `PASS (100% bitwise label match)` |
| **Microstructure Bounds** | $\text{bid} \le \text{mid} \le \text{ask}$, $\text{spread} > 0$, $\text{imb} \in [-1, 1]$ | `PASS (all bounds verified)` |
| **Constant Feature Check** | Zero constant / zero-variance features | `PASS (0 constant features)` |
| **Duplicate Timestamps** | 0 duplicate timestamps per stream | `PASS (0 duplicates)` |
| **Infinite Values** | 0 Inf / -Inf values | `PASS (0 Infs)` |

---

## 3. Whitelisted Causal Features & Lookback Specification

| Feature | Formula | Lookback | Unique | Mean | Std | Range | NaNs | Infs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ask_change_1s` | `ask_t - ask_t-1` | 1s (1000 ms) | 228 | -0.0000 | 0.0399 | [-0.4800, 0.5000] | 38 | 0 |
| `bid_change_1s` | `bid_t - bid_t-1` | 1s (1000 ms) | 233 | 0.0000 | 0.0399 | [-0.5000, 0.4800] | 38 | 0 |
| `depth_imbalance` | `(bid_size - ask_size) / (bid_size + ask_size)` | 0s (0 ms) | 4,925 | -0.0000 | 0.7080 | [-0.9994, 0.9994] | 0 | 0 |
| `microprice` | `(bid * ask_size + ask * bid_size) / (bid_size + ask_size)` | 0s (0 ms) | 4,942 | 0.5000 | 0.3059 | [0.0020, 0.9980] | 0 | 0 |
| `mid_price` | `(bid + ask) / 2` | 0s (0 ms) | 261 | 0.5000 | 0.3056 | [0.0020, 0.9980] | 0 | 0 |
| `mid_return_1s` | `(mid_t - mid_t-1) / mid_t-1` | 1s (1000 ms) | 1,325 | 0.0034 | 0.1662 | [-0.9630, 4.2609] | 38 | 0 |
| `mid_return_3s` | `(mid_t - mid_t-3) / mid_t-3` | 3s (3000 ms) | 1,776 | 0.0098 | 0.2933 | [-0.9639, 5.1538] | 114 | 0 |
| `mid_return_5s` | `(mid_t - mid_t-5) / mid_t-5` | 5s (5000 ms) | 1,981 | 0.0159 | 0.3783 | [-0.9639, 5.0909] | 190 | 0 |
| `mid_volatility_5s` | `rolling_std(mid_return_1s, window=5, center=False)` | 5s (5000 ms) | 3,511 | 0.0892 | 0.1345 | [0.0000, 1.9150] | 190 | 0 |
| `spread` | `ask - bid` | 0s (0 ms) | 50 | 0.0103 | 0.0039 | [0.0010, 0.0800] | 0 | 0 |
| `spread_bps` | `(spread / mid_price) * 10000` | 0s (0 ms) | 257 | 627.1327 | 1222.9664 | [10.0251, 15000.0000] | 0 | 0 |


---

## 4. Target Label Integrity

Phase 13 physical 5-second labels are preserved exactly as target variables:
- **`UP`**: **2,109** (42.43%)
- **`DOWN`**: **2,108** (42.41%)
- **`FLAT`**: **753** (15.15%)
- **Exact Preservation Status**: `PASS (100% bitwise label match)`

---

## 5. Per-Market Breakdown (19 Retained Markets)

| Market Stem | Input Rows | Output Rows | Warm-up Rows | Total NaNs | Total Infs | Sanity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `btc-updown-5m-1791204600` | 524 | 524 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791204900` | 586 | 586 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791205200` | 274 | 274 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791205500` | 560 | 560 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791205800` | 454 | 454 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791206100` | 426 | 426 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791206400` | 330 | 330 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791206700` | 238 | 238 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791207000` | 254 | 254 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791207300` | 84 | 84 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791207600` | 246 | 246 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791207900` | 112 | 112 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791208200` | 116 | 116 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791208500` | 220 | 220 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791209400` | 84 | 84 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791209700` | 144 | 144 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791210000` | 60 | 60 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791210300` | 144 | 144 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791210600` | 114 | 114 | 10 | 32 | 0 | `PASS` |


---

## 6. Deliverables & Artifact Verification

- **Per-Market Feature Datasets**: `data/clean_v2/04_features/new_collection/*_features.parquet` (19 files)
- **Combined Feature Dataset**: `data/clean_v2/04_features/new_collection/features_production.parquet` (4,970 rows)
- **Metadata Summary**: `data/clean_v2/04_features/new_collection/phase14_features_metadata.json`
- **Validation Report**: `data/clean_v2/04_features/new_collection/phase14_features_report.md`
