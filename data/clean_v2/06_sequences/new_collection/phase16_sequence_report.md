# Phase 16 — Causal Sequence Construction Audit Report

- **Execution Timestamp (UTC)**: `2026-10-06T04:10:44.235856+00:00`
- **Execution Duration**: `0.94s`
- **Input Directory**: `data\clean_v2\05_splits\new_collection`
- **Output Directory**: `data\clean_v2\06_sequences\new_collection`
- **Sequence Length ($L$)**: `10` steps (10-second causal lookback on 1s grid)
- **Feature Dimension ($D$)**: `11` causal features
- **Final Verdict**: `**PASS**`

---

## 1. Sequence Construction & Warm-Up Summary

| Partition | Input Rows | Active Streams | Warm-Up Dropped (<10s) | Valid Sequences Generated | Data Retention Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 3,392 | 16 | 144 | **3,248** | 95.75% |
| **Validation** | 812 | 10 | 90 | **722** | 88.92% |
| **Test** | 766 | 12 | 108 | **658** | 85.90% |
| **Total** | 4,970 | 38 | 342 | **4,628** | 93.12% |

---

## 2. 3D Tensor Specifications

All sequence tensors adhere strictly to `(N_samples, Sequence_Length, Feature_Dim)` format:

- **Train $X$ Shape**: `(3248, 10, 11)` | $y$ Shape: `(3248,)`
- **Validation $X$ Shape**: `(722, 10, 11)` | $y$ Shape: `(722,)`
- **Test $X$ Shape**: `(658, 10, 11)` | $y$ Shape: `(658,)`

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
| **Train** | 3,248 | 1,386 (42.67%) | 1,386 (42.67%) | 476 (14.66%) | 1.000 |
| **Validation** | 722 | 282 (39.06%) | 281 (38.92%) | 159 (22.02%) | 1.004 |
| **Test** | 658 | 291 (44.22%) | 291 (44.22%) | 76 (11.55%) | 1.000 |
| **Combined** | 4,628 | 1,959 (42.33%) | 1,958 (42.31%) | 711 (15.36%) | 1.000 |

---

## 4. Invariant Verification & Boundary Audits

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

## 5. Artifact Hashes & Storage

### Input Partitions (Phase 15)
- `train.parquet` (SHA-256): `51ad84ec1e8ffa819fb5fc3c3531e3292ecab1def07ae73596093ef366a969b8`
- `validation.parquet` (SHA-256): `d19e16d3bf83abfbde12cc1a8b6fbbb40335725a6e61034247b627b14b1ca6e2`
- `test.parquet` (SHA-256): `ea03ff1a6f8907b8796a3d7bcc2a103275d347ae8169a66b6c4c610df531b9b0`

### Generated Sequence Artifacts (Phase 16)
- `train_sequences.parquet` (SHA-256): `c93ae311c1f6b6a12e09eb4d6375a0954ca14718c43013926e53f9be395fb86d`
- `validation_sequences.parquet` (SHA-256): `03040fc8218c088c99a96362b305d7fb69b8d11def09066f43819cf6b8ec8acd`
- `test_sequences.parquet` (SHA-256): `bc7e709468b41fb309344cafe9b2148fd021367e542f86e64fc931a03fe828c3`
- `train_sequences.npz` (SHA-256): `eb8f9823fae6e1eaa81f737143745b5643085daeeaff68b41e5523d8d142348b`
- `validation_sequences.npz` (SHA-256): `ef383211a6d7dc26e5848fd534dfd57e9f64085bc7b06cf6de9af8936401ff7b`
- `test_sequences.npz` (SHA-256): `36e2d8e2e7d5393b3e4e47449f3e189e22e22f3fe6f25f42629b5ffc0a3f9b85`
- `sequences_production.npz` (SHA-256): `12fa2a1c0372b191accac1c72bb73a44d2505364bab881b6e880ab815787385b`
