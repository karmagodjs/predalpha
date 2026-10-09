# Phase 14: Causal Feature Engineering Audit Report

**Generated UTC**: `2026-10-07T02:59:46.096988+00:00`  
**Pipeline**: `pipeline_v2/features`  
**Execution Runtime**: 3.88 seconds  
**Dataset SHA-256**: `057dd9b526161e3cf57d47dddd201cd0d8b8e2071ea366f9b877ce96efe728c2`  
**Maximum Feature Lookback**: 5000 ms (5 seconds)

---

## 1. Executive Summary

Phase 14 causal feature engineering was executed across all **46 retained production markets** from Phase 13.
- **Input Dataset**: `data\clean_v2\03_labeled_5s\expanded_collection` (11,352 rows)
- **Output Dataset**: `data\clean_v2\04_features\expanded_collection` (11,352 rows, 21 columns)
- **Whitelisted Causal Features**: 11 features generated strictly using backward-looking operations.
- **Warm-up Rows Tracked**: 460 rows. Zero backward filling or data fabrication.

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
| `ask_change_1s` | `ask_t - ask_t-1` | 1s (1000 ms) | 269 | 0.0000 | 0.0382 | [-0.7000, 0.7000] | 92 | 0 |
| `bid_change_1s` | `bid_t - bid_t-1` | 1s (1000 ms) | 270 | -0.0000 | 0.0382 | [-0.7000, 0.7000] | 92 | 0 |
| `depth_imbalance` | `(bid_size - ask_size) / (bid_size + ask_size)` | 0s (0 ms) | 11,203 | 0.0000 | 0.6981 | [-0.9994, 0.9994] | 0 | 0 |
| `microprice` | `(bid * ask_size + ask * bid_size) / (bid_size + ask_size)` | 0s (0 ms) | 11,248 | 0.5000 | 0.2959 | [0.0020, 0.9980] | 0 | 0 |
| `mid_price` | `(bid + ask) / 2` | 0s (0 ms) | 282 | 0.5000 | 0.2957 | [0.0020, 0.9980] | 0 | 0 |
| `mid_return_1s` | `(mid_t - mid_t-1) / mid_t-1` | 1s (1000 ms) | 1,835 | 0.0019 | 0.1421 | [-0.9630, 4.2609] | 92 | 0 |
| `mid_return_3s` | `(mid_t - mid_t-3) / mid_t-3` | 3s (3000 ms) | 2,507 | 0.0055 | 0.2516 | [-0.9639, 5.1538] | 276 | 0 |
| `mid_return_5s` | `(mid_t - mid_t-5) / mid_t-5` | 5s (5000 ms) | 2,883 | 0.0091 | 0.3334 | [-0.9693, 8.1905] | 460 | 0 |
| `mid_volatility_5s` | `rolling_std(mid_return_1s, window=5, center=False)` | 5s (5000 ms) | 7,649 | 0.0750 | 0.1160 | [0.0000, 1.9150] | 460 | 0 |
| `spread` | `ask - bid` | 0s (0 ms) | 57 | 0.0104 | 0.0039 | [0.0010, 0.1000] | 0 | 0 |
| `spread_bps` | `(spread / mid_price) * 10000` | 0s (0 ms) | 305 | 580.2401 | 1130.9509 | [10.0251, 15000.0000] | 0 | 0 |


---

## 4. Target Label Integrity

Phase 13 physical 5-second labels are preserved exactly as target variables:
- **`UP`**: **4,716** (41.54%)
- **`DOWN`**: **4,711** (41.50%)
- **`FLAT`**: **1,925** (16.96%)
- **Exact Preservation Status**: `PASS (100% bitwise label match)`

---

## 5. Per-Market Breakdown (46 Retained Markets)

| Market Stem | Input Rows | Output Rows | Warm-up Rows | Total NaNs | Total Infs | Sanity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `btc-updown-5m-1791062400` | 306 | 306 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791062700` | 176 | 176 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791063900` | 222 | 222 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791064200` | 120 | 120 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791064500` | 302 | 302 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791065400` | 238 | 238 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791065700` | 100 | 100 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791066600` | 154 | 154 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791066900` | 482 | 482 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791067200` | 444 | 444 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791067500` | 208 | 208 | 10 | 32 | 0 | `PASS` |
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
| `btc-updown-5m-1791291600` | 350 | 350 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791291900` | 270 | 270 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791292200` | 62 | 62 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791292500` | 196 | 196 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791292800` | 376 | 376 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791293100` | 262 | 262 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791293400` | 122 | 122 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791293700` | 104 | 104 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791294000` | 180 | 180 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791294300` | 174 | 174 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791294600` | 206 | 206 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791294900` | 198 | 198 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791295200` | 184 | 184 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791295500` | 156 | 156 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791296400` | 438 | 438 | 10 | 32 | 0 | `PASS` |
| `btc-updown-5m-1791296700` | 352 | 352 | 10 | 32 | 0 | `PASS` |


---

## 6. Deliverables & Artifact Verification

- **Per-Market Feature Datasets**: `data\clean_v2\04_features\expanded_collection/*_features.parquet` (46 files)
- **Combined Feature Dataset**: `data\clean_v2\04_features\expanded_collection/features_production.parquet` (11,352 rows)
- **Metadata Summary**: `data\clean_v2\04_features\expanded_collection/phase14_features_metadata.json`
- **Validation Report**: `data\clean_v2\04_features\expanded_collection/phase14_features_report.md`
