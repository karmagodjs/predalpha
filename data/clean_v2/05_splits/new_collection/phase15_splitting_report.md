# Phase 15: Purged Temporal Train/Validation/Test Split Audit Report

**Generated UTC**: `2026-10-06T03:51:09.873569+00:00`  
**Pipeline**: `pipeline_v2/splitting`  
**Execution Runtime**: 0.4 seconds  
**Input Dataset**: `C:\Users\karma\OneDrive\Desktop\predalpha\data\clean_v2\04_features\new_collection\features_production.parquet`  
**Output Directory**: `C:\Users\karma\OneDrive\Desktop\predalpha\data\clean_v2\05_splits\new_collection`  

---

## 1. Executive Summary

Phase 15 strictly chronological purged temporal splitting was executed across the **19 retained production markets**.
- **Input Rows**: 4,970
- **Train Partition**: **3,392 rows** (68.25%) across 8 markets
- **Validation Partition**: **812 rows** (16.34%) across 5 markets
- **Test Partition**: **766 rows** (15.41%) across 6 markets
- **Intra-Market Purged Rows**: **0 rows** (0.00% sample loss due to exact market-boundary snapping)

---

## 2. Dependency Calculation & Required Purge Gap

The required minimum purge gap is mathematically derived from upstream feature lookback, label horizon, and downstream sequence lookback:

$$\begin{aligned}
\text{Required Purge} &= \text{max\_label\_horizon} + \text{feature\_lookback} + \text{future\_sequence\_lookback} \\
&= 7000\text{ ms} + 5000\text{ ms} + 10000\text{ ms} \\
&= 22000\text{ ms} \quad (22.0\text{ seconds})
\end{aligned}$$

### Purge Intervals Actually Applied:
- **Purge Gap 1 (Train $\to$ Validation)**: **88,000 ms (88.0 seconds)** $\ge 22,000$ ms `PASS (88,000 ms >= 22,000 ms)`
- **Purge Gap 2 (Validation $\to$ Test)**: **69,000 ms (69.0 seconds)** $\ge 22,000$ ms `PASS (69,000 ms >= 22,000 ms)`

> [!NOTE]
> By aligning temporal split cuts to market session boundaries where recording naturally paused between 5-minute contracts, the actual purge intervals (88s and 69s) exceed the 22-second mathematical minimum by **4.0x** and **3.1x** respectively, while eliminating artificial sample loss.

---

## 3. Boundary Invariant Checks

| Invariant | Requirement | Actual Status |
| :--- | :--- | :--- |
| **Chronological Ordering** | $\text{max(Train)} < \text{min(Val)} < \text{max(Val)} < \text{min(Test)}$ | `PASS (strictly chronological)` |
| **Purge Gap 1 (Train $\to$ Val)** | $\Delta t \ge 22,000$ ms | `PASS (88,000 ms >= 22,000 ms)` |
| **Purge Gap 2 (Val $\to$ Test)** | $\Delta t \ge 22,000$ ms | `PASS (69,000 ms >= 22,000 ms)` |
| **Timestamp Overlap** | $\text{Train} \cap \text{Val} = \emptyset, \dots$ | `PASS (0 timestamp overlaps)` |
| **Duplicate Rows Across Splits** | 0 duplicated row hashes | `PASS (0 duplicate rows across splits)` |
| **Label Horizon Leakage** | $\text{Train\_ts} + 7\text{s} \le \text{Val\_min}$ | `PASS (0 label boundary violations)` |
| **Feature Lookback Leakage** | $\text{Val\_min} - 5\text{s} \ge \text{Train\_max}$ | `PASS (0 lookback boundary violations)` |
| **Market Session Isolation** | 0 markets shared across splits | `PASS (100% market isolation, 0 shared markets)` |
| **Deterministic Reproducibility** | Bitwise reproducible partition outputs | `PASS (SHA-256 verified)` |

---

## 4. Chronological Boundaries & Partitions

| Partition | Row Count | Ratio (%) | Start Timestamp | End Timestamp | Market Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3,392 | 68.25% | `1791204601000` | `1791206913000` | 8 markets |
| *Purge Gap 1* | *0* | *0.00%* | *1791206913000* | *1791207001000* | *88,000 ms gap* |
| **Validation** | 812 | 16.34% | `1791207001000` | `1791208433000` | 5 markets |
| *Purge Gap 2* | *0* | *0.00%* | *1791208433000* | *1791208502000* | *69,000 ms gap* |
| **Test** | 766 | 15.41% | `1791208502000` | `1791210723000` | 6 markets |

---

## 5. Directional Class Preservation per Split

| Split Partition | `UP` Count (%) | `DOWN` Count (%) | `FLAT` Count (%) | Total Rows |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | **1,445 (42.6%)** | **1,445 (42.6%)** | **502 (14.8%)** | **3,392** |
| **Validation** | **322 (39.66%)** | **321 (39.53%)** | **169 (20.81%)** | **812** |
| **Test** | **342 (44.65%)** | **342 (44.65%)** | **82 (10.7%)** | **766** |

> [!NOTE]
> In all three partitions, the directional balance between **UP and DOWN** is virtually identical (Train: 42.60% vs 42.60%; Val: 39.66% vs 39.53%; Test: 44.65% vs 44.65%), with healthy FLAT transition frequencies (10.7% to 20.8%).

---

## 6. Market Allocation Summary

- **Train Partition (8 Markets)**:
  `0x1513370f3c00f9..., 0x2e222470f8d209..., 0x67d7111399311b..., 0x7944f37d88ba36..., 0x9523a610624fc2..., 0xae59df8807f60d..., 0xde773924630e62..., 0xece8b19e50895f...`
- **Validation Partition (5 Markets)**:
  `0x06d0b3cfe98dd6..., 0x178ac9dd3e21a3..., 0xa9a650633de03b..., 0xc00c0afd4a0daa..., 0xe1948ec2ba75cc...`
- **Test Partition (6 Markets)**:
  `0x1d0fb9a895dc5d..., 0x4283a2a38aa82a..., 0x7c470e1d274ed6..., 0xba005b7e518925..., 0xd18098910284bd..., 0xe8d6899ca1b3d8...`

---

## 7. Deliverables & Checksums

- **Train Partition**: `data/clean_v2/05_splits/new_collection/train.parquet` (3,392 rows)  
  `SHA-256: 6234e04192ab68c57e987b93a0770a8e9c6111895e31169d88607a47f274b164`
- **Validation Partition**: `data/clean_v2/05_splits/new_collection/validation.parquet` (812 rows)  
  `SHA-256: 8826c1c7d555e01db01eff71f5901f6e68c78b7392c6ba25d539d10aad0ccdbb`
- **Test Partition**: `data/clean_v2/05_splits/new_collection/test.parquet` (766 rows)  
  `SHA-256: e8122c8cbd04c4b42788cd421b79f4d528233a74dc9d872d183a4f28dda6844b`
- **Metadata**: `data/clean_v2/05_splits/new_collection/phase15_splitting_metadata.json`
- **Validation Report**: `data/clean_v2/05_splits/new_collection/phase15_splitting_report.md`
