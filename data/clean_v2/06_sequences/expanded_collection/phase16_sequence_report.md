# Phase 16 — Causal Sequence Construction Audit Report

- **Execution Timestamp (UTC)**: `2026-10-07T03:15:20.678844+00:00`
- **Execution Duration**: `3.92s`
- **Input Directory**: `data\clean_v2\05_splits\expanded_collection`
- **Output Directory**: `data\clean_v2\06_sequences\expanded_collection`
- **Sequence Length ($L$)**: `10` steps (10-second causal lookback on 1s grid)
- **Feature Dimension ($D$)**: `11` causal features
- **Final Verdict**: `**PASS**`

---

## 1. Sequence Construction & Warm-Up Summary

| Partition | Input Rows | Active Streams | Warm-Up Dropped (<10s) | Valid Sequences Generated | Data Retention Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 8,072 | 62 | 558 | **7,514** | 93.09% |
| **Validation** | 1,572 | 16 | 144 | **1,428** | 90.84% |
| **Test** | 1,708 | 14 | 126 | **1,582** | 92.62% |
| **Total** | 11,352 | 92 | 828 | **10,524** | 92.71% |

---

## 2. 3D Tensor Specifications

All sequence tensors adhere strictly to `(N_samples, Sequence_Length, Feature_Dim)` format:

- **Train $X$ Shape**: `(7514, 10, 11)` | $y$ Shape: `(7514,)`
- **Validation $X$ Shape**: `(1428, 10, 11)` | $y$ Shape: `(1428,)`
- **Test $X$ Shape**: `(1582, 10, 11)` | $y$ Shape: `(1582,)`

### Feature Columns (D=11 Safe Causal Features)

1. `mid_price`
2. `spread`
3. `spread_bps`
4. `mid_return_1s`
5. `mid_return_3s`
6. `mid_return_5s`
7. `mid_volatility_5s`
8. `bid_change_1s`
9. `ask_change_1s`
10. `microprice`
11. `depth_imbalance`

---

## 3. Label Alignment & Class Distribution

Labels are strictly aligned with sequence endpoints $T$ using the Phase 13 physical 5-second forward label.

| Partition | Total Sequences | UP Count (Pct) | DOWN Count (Pct) | FLAT Count (Pct) | UP/DOWN Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 7,514 | 3,050 (40.59%) | 3,050 (40.59%) | 1,414 (18.82%) | 1.000 |
| **Validation** | 1,428 | 617 (43.21%) | 616 (43.14%) | 195 (13.66%) | 1.002 |
| **Test** | 1,582 | 709 (44.82%) | 705 (44.56%) | 168 (10.62%) | 1.006 |
| **Combined** | 10,524 | 4,376 (41.58%) | 4,371 (41.53%) | 1,777 (16.89%) | 1.001 |

---

## 4. Causal Warm-Up NaN Accounting

Warm-up NaNs from Phase 14 causal lookback features are preserved causally without row deletion, interpolation, or backward filling. All NaNs are strictly confined to initial sequence steps ($T-9\text{s}$ through $T-5\text{s}$). The endpoint step $T$ contains 0 NaNs.

### NaN Count by Split
- **Train**: 2,418 NaNs
- **Validation**: 624 NaNs
- **Test**: 546 NaNs
- **Total**: 3,588 NaNs

### NaN Count by Sequence Step
| Step Index | Time Relative to Endpoint | Total NaN Count |
| :--- | :--- | :--- |
| Step 0 | T - 9s | 1,472 |
| Step 1 | T - 8s | 920 |
| Step 2 | T - 7s | 644 |
| Step 3 | T - 6s | 368 |
| Step 4 | T - 5s | 184 |
| Step 5 | T - 4s | 0 |
| Step 6 | T - 3s | 0 |
| Step 7 | T - 2s | 0 |
| Step 8 | T - 1s | 0 |
| Step 9 | T (Endpoint) | 0 |

### NaN Count by Feature
| Feature Name | Total NaN Count |
| :--- | :--- |
| `mid_price` | 0 |
| `spread` | 0 |
| `spread_bps` | 0 |
| `mid_return_1s` | 92 |
| `mid_return_3s` | 552 |
| `mid_return_5s` | 1,380 |
| `mid_volatility_5s` | 1,380 |
| `bid_change_1s` | 92 |
| `ask_change_1s` | 92 |
| `microprice` | 0 |
| `depth_imbalance` | 0 |

---

## 5. Invariant Verification & Boundary Audits

| Invariant | Requirement | Observed Status | Verdict |
| :--- | :--- | :--- | :--- |
| **Sequence Length** | Exactly L=10 steps for all sequences | min=10, max=10 | `PASS` |
| **Feature Dimensionality** | Exactly D=11 safe causal features | D=11 | `PASS` |
| **No Forbidden Columns** | Zero targets or future audit columns in X | forbidden_found=0 | `PASS` |
| **Strict Causality** | All observation timestamps in sequence <= endpoint T | violations=0 | `PASS` |
| **Chronological Monotonicity** | Strict step-by-step time monotonicity within sequences | violations=0 | `PASS` |
| **Cross-Split Isolation** | 0 sequences crossing Train / Val / Test partitions | violations=0 | `PASS` |
| **Cross-Market Isolation** | 0 sequences crossing market_id boundaries | violations=0 | `PASS` |
| **Cross-Asset Isolation** | 0 sequences crossing asset_id (UP vs DOWN) boundaries | violations=0 | `PASS` |
| **Endpoint Deduplication** | 0 duplicate sequences for any (market, asset, T) | duplicates=0 | `PASS` |
| **Target Endpoint Alignment** | Target label matches endpoint row physical 5s label | mismatches=0 | `PASS` |
| **Deterministic Verification** | Bitwise identical regeneration | status=PASS | `PASS` |

---

## 6. Artifact Hashes & Storage

### Input Partitions (Phase 15)
- `train.parquet` (SHA-256): `bce005c8f67f809a386b0c12f9201fc45e6eab338a6d1bd4d9d04b49761e0af1`
- `validation.parquet` (SHA-256): `a38bff9fbbd31564900e9a94aa5dca397569d0824ef44d5c774557a562d49390`
- `test.parquet` (SHA-256): `faa2952096545a8edf0fcf22ca655493e9ab27f80539b358da2ce112e2c05bfa`

### Generated Sequence Artifacts (Phase 16)
- `train_sequences.parquet` (SHA-256): `20d73ec929fdeb0843d08ffe197fa15da7c535eadba84eb92b119cac8910a46e`
- `validation_sequences.parquet` (SHA-256): `51eac3800dec70024b747558854694b1f98e75b14ce80215ccd657396809a787`
- `test_sequences.parquet` (SHA-256): `2c2abe39c3eb278d8a98918ae50e4611e807ae4d194850e9ce4749313f01b2e6`
- `train_sequences.npz` (SHA-256): `f49c97683c1844847770bcf7c1fe81b0e67cc35a82e86b11156b60573558569b`
- `validation_sequences.npz` (SHA-256): `caabc3d101b2307932e08de8334528615159b5b8db98ed875352b7fff5dd9cf4`
- `test_sequences.npz` (SHA-256): `cbfb24c462de2fe868f2016b9e9f00165032f793137a7b7b71fc8f0c5236ffab`
- `sequences_production.npz` (SHA-256): `ceb1737020de56a353f9c2f0d0e80c339dbfc37bae9e7d295c30662ba7f86b52`
