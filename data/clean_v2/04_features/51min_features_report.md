# Causal Feature Engineering Validation Report

- **Input Observations**: 375
- **Output Feature Rows**: 375
- **Warm-up Rows (Incomplete Rolling History)**: 5
- **Feature Columns Generated**: 11 features

### Invariant Validations

- **Causality & Backward-Only Alignment**: `PASS (strictly causal, zero lookahead)`
- **Chronological Ordering**: `PASS (strictly chronological)`
- **Duplicate Timestamp Rejection**: `PASS (0 duplicates)`
- **Forbidden Target Column Ingestion**: `PASS (0 forbidden columns ingested)`
- **Microstructure Bounds & Sanity Checks**: `PASS (all bounds verified)`

### Feature Health & Missing Values

| Feature Name | NaN Count | Inf Count |
| :--- | :--- | :--- |
| `mid_price` | 0 | 0 |
| `spread` | 0 | 0 |
| `spread_bps` | 0 | 0 |
| `mid_return_1s` | 1 | 0 |
| `mid_return_3s` | 3 | 0 |
| `mid_return_5s` | 5 | 0 |
| `mid_volatility_5s` | 5 | 0 |
| `bid_change_1s` | 1 | 0 |
| `ask_change_1s` | 1 | 0 |
| `microprice` | 0 | 0 |
| `depth_imbalance` | 0 | 0 |

- **Output File**: `data\clean_v2\04_features\features_51min.parquet`
