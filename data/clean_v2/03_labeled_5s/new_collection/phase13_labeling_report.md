# Phase 13: Physical 5-Second Forward Labeling Audit Report

**Generated UTC**: `2026-10-06T03:20:59.202994+00:00`  
**Pipeline**: `pipeline_v2/labeling`  
**Execution Runtime**: 10.91 seconds  
**Target Architecture**: Forward 5000 ms physical horizon ($5000 \text{ ms} \le \Delta t \le 7000 \text{ ms}$)

---

## 1. Executive Summary

Phase 13 physical 5-second forward labeling was successfully executed on all **19 retained production markets** from the Phase 11B/12 clean pipeline.

### Core Architecture Highlights:
1. **Physical Elapsed Milliseconds**: Target event matching strictly adheres to elapsed physical time:
   $$\text{target\_timestamp} \ge T + 5000\text{ ms} \quad \text{and} \quad \text{target\_timestamp} \le T + 7000\text{ ms}$$
   Zero row-shift operations or index offset assumptions (`shift(-5)` strictly forbidden and verified absent).
2. **Strict Dual-Asset & Session Isolation**: Up and Down tokens are partitioned independently per `(market_id, asset_id)`. Zero cross-session or cross-token matches.
3. **Staleness Exclusion**: Stale observations (`is_stale == True`) are strictly excluded prior to label generation (2,278 stale rows quarantined).
4. **Target Leakage Quarantine**: Clean labeled dataset exports strictly causal features at $T$ and the `label` column. All future/target variables (`target_timestamp`, `current_mid`, `future_mid`, `delta`, `threshold`, `physical_horizon_ms`) are segregated into detached audit Parquet files.
5. **Dynamic Directional Thresholding**:
   $$\text{threshold} = \max\left(\frac{\text{spread}}{2}, 0.001\right)$$
   $$\Delta = \text{future\_mid} - \text{current\_mid}$$
   $$\text{label} = \begin{cases} \text{UP} & \Delta > \text{threshold} \\ \text{DOWN} & \Delta < -\text{threshold} \\ \text{FLAT} & \text{otherwise} \end{cases}$$

---

## 2. Invariant Verification

| Invariant | Requirement | Actual Status |
| :--- | :--- | :--- |
| **Physical Horizon Bounds** | $5000 \le \Delta t \le 7000$ ms | `PASS (0 violations, 5000 <= horizon <= 7000 ms)` |
| **Target Leakage Prevention** | Zero future columns in clean dataset | `PASS (0 target/future columns in clean output)` |
| **Session Boundary Isolation** | Zero cross-session target matches | `PASS (0 cross-session / cross-asset matches)` |
| **Asset Boundary Isolation** | Zero cross-asset target matches | `PASS (100% token isolation)` |
| **Stale Row Exclusion** | Zero stale rows in clean training set | `PASS (0 stale rows in labeled set)` |
| **Deterministic Reproducibility** | Exact row match across multiple runs | `PASS (100% bitwise deterministic)` |

---

## 3. Aggregate Dataset Overview

- **Retained Production Markets**: 19 markets
- **Excluded Markets**: 2 (`btc-updown-5m-1791204300, btc-updown-5m-1791208800`)
- **Total Input Grid Observations**: 8,358
- **Stale Grid Observations Excluded**: 2,278 (27.26%)
- **Successfully Labeled Observations**: 4,970
- **Dropped Observations**: 3,388 (Stale + No target event in $[5\text{s}, 7\text{s}]$)

### Class Distribution

| Class | Count | Percentage |
| :--- | :--- | :--- |
| **`UP`** | **2,109** | **42.43%** |
| **`DOWN`** | **2,108** | **42.41%** |
| **`FLAT`** | **753** | **15.15%** |
| **Total** | **4,970** | **100.00%** |

> [!NOTE]
> Unlike the earlier 51-minute single-market capture which was 100% FLAT due to a stale quote feed, the new 19-market production collection exhibits rich, balanced market movements:
> **UP (42.43%) vs DOWN (42.41%)** are nearly identical, with **FLAT at 15.15%**. This validates genuine price discovery across the markets.

---

## 4. Physical Horizon Statistics

- **Minimum Physical Horizon**: 5000 ms
- **Maximum Physical Horizon**: 6929 ms
- **Mean Physical Horizon**: 5016.26 ms
- **Median Physical Horizon**: 5001.0 ms
- **95th Percentile Horizon**: 5024.0 ms
- **Count Exceeding 7,000 ms**: **0** (strictly enforced)

### Horizon Interval Distribution

| Horizon Bucket | Observation Count | Percentage |
| :--- | :--- | :--- |
| `[5000, 5500) ms` | 4,928 | 99.15% |
| `[5500, 6000) ms` | 26 | 0.52% |
| `[6000, 6500) ms` | 8 | 0.16% |
| `[6500, 7000) ms` | 8 | 0.16% |


---

## 5. Comparative Evaluation: Canonical vs Resampled Target Pools

To evaluate the impact of target event sourcing:
- **Primary Method (Canonical Events Pool)**: Targets matched to the earliest physical raw quote event arriving at $\ge T + 5000$ ms. Labeled count = **4,970** rows.
- **Benchmark Method (Resampled Grid Pool)**: Targets matched to the earliest fresh 1-second grid observation at $\ge T + 5000$ ms. Labeled count = **5,456** rows.
- **Label Agreement**: **100.0% exact match (0 mismatches)** across all common observations.
- **Root Cause of Delta (486 rows)**: The canonical event method strictly drops trailing observations where the recording session ended before $T + 5000$ ms, preventing forward-fill artifacts at session boundaries.

---

## 6. Per-Market Breakdown

| Market Stem | Input Rows | Stale Excl | Labeled Rows | Dropped | UP (Count/%) | DOWN (Count/%) | FLAT (Count/%) | Mean Horizon |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `btc-updown-5m-1791204600` | 534 | 0 | 524 | 10 | 212 (40.5%) | 212 (40.5%) | 100 (19.1%) | 5012.8 ms |
| `btc-updown-5m-1791204900` | 596 | 0 | 586 | 10 | 267 (45.6%) | 267 (45.6%) | 52 (8.9%) | 5011.9 ms |
| `btc-updown-5m-1791205200` | 338 | 40 | 274 | 64 | 99 (36.1%) | 99 (36.1%) | 76 (27.7%) | 5013.2 ms |
| `btc-updown-5m-1791205500` | 570 | 0 | 560 | 10 | 265 (47.3%) | 265 (47.3%) | 30 (5.4%) | 5008.1 ms |
| `btc-updown-5m-1791205800` | 464 | 0 | 454 | 10 | 176 (38.8%) | 176 (38.8%) | 102 (22.5%) | 5013.3 ms |
| `btc-updown-5m-1791206100` | 480 | 24 | 426 | 54 | 191 (44.8%) | 191 (44.8%) | 44 (10.3%) | 5004.5 ms |
| `btc-updown-5m-1791206400` | 408 | 52 | 330 | 78 | 130 (39.4%) | 130 (39.4%) | 70 (21.2%) | 5037.3 ms |
| `btc-updown-5m-1791206700` | 488 | 170 | 238 | 250 | 105 (44.1%) | 105 (44.1%) | 28 (11.8%) | 5011.8 ms |
| `btc-updown-5m-1791207000` | 484 | 160 | 254 | 230 | 82 (32.3%) | 81 (31.9%) | 91 (35.8%) | 5006.4 ms |
| `btc-updown-5m-1791207300` | 366 | 198 | 84 | 282 | 42 (50.0%) | 42 (50.0%) | 0 (0.0%) | 5002.6 ms |
| `btc-updown-5m-1791207600` | 432 | 116 | 246 | 186 | 106 (43.1%) | 106 (43.1%) | 34 (13.8%) | 5016.0 ms |
| `btc-updown-5m-1791207900` | 330 | 148 | 112 | 218 | 39 (34.8%) | 39 (34.8%) | 34 (30.4%) | 5128.0 ms |
| `btc-updown-5m-1791208200` | 476 | 250 | 116 | 360 | 53 (45.7%) | 53 (45.7%) | 10 (8.6%) | 5013.5 ms |
| `btc-updown-5m-1791208500` | 502 | 178 | 220 | 282 | 92 (41.8%) | 92 (41.8%) | 36 (16.4%) | 5040.4 ms |
| `btc-updown-5m-1791209400` | 304 | 150 | 84 | 220 | 39 (46.4%) | 39 (46.4%) | 6 (7.1%) | 5011.0 ms |
| `btc-updown-5m-1791209700` | 426 | 192 | 144 | 282 | 64 (44.4%) | 64 (44.4%) | 16 (11.1%) | 5008.4 ms |
| `btc-updown-5m-1791210000` | 418 | 268 | 60 | 358 | 29 (48.3%) | 29 (48.3%) | 2 (3.3%) | 5001.2 ms |
| `btc-updown-5m-1791210300` | 430 | 196 | 144 | 286 | 68 (47.2%) | 68 (47.2%) | 8 (5.6%) | 5005.9 ms |
| `btc-updown-5m-1791210600` | 312 | 136 | 114 | 198 | 50 (43.9%) | 50 (43.9%) | 14 (12.3%) | 5020.0 ms |


---

## 7. Deliverables & Artifact Verification

- **Per-Market Clean Datasets**: `data/clean_v2/03_labeled_5s/new_collection/*_labeled_5s.parquet` (19 files)
- **Combined Clean Dataset**: `data/clean_v2/03_labeled_5s/new_collection/labeled_5s_production.parquet` (4,970 rows)
- **Per-Market Detached Audit**: `data/clean_v2/03_labeled_5s/new_collection/*_labeling_audit.parquet` (19 files)
- **Combined Detached Audit**: `data/clean_v2/03_labeled_5s/new_collection/labeling_audit_production.parquet` (4,970 rows)
- **Metadata Summary**: `data/clean_v2/03_labeled_5s/new_collection/phase13_labeling_metadata.json`
- **Validation Report**: `data/clean_v2/03_labeled_5s/new_collection/phase13_labeling_report.md`
