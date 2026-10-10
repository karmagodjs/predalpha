# Physical 5-Second Labeling Validation Report

- **Input Observations**: 3,074
- **Stale Observations Excluded**: 1,664
- **Successfully Labeled Rows**: 375
- **Dropped Rows (Stale + No Target in [5s, 7s])**: 2,699

### Horizon Statistics

- **Min Physical Horizon**: 5000 ms
- **Max Physical Horizon**: 7000 ms
- **Mean Physical Horizon**: 5906.14 ms
- **Physical Horizon Invariant [5000, 7000] ms**: `PASS (0 violations, all in [5000, 7000] ms)`
- **Session Boundary Isolation**: `PASS (0 cross-session target matches)`
- **Target Leakage / Future Column Invariant**: `PASS (0 future columns in training dataset)`

### Class Distribution

| Class | Count | Percentage |
| :--- | :--- | :--- |
| `UP` | 0 | 0.00% |
| `DOWN` | 0 | 0.00% |
| `FLAT` | 375 | 100.00% |

### Horizon Interval Distribution

| Horizon Bucket (ms) | Count |
| :--- | :--- |
| `[5000, 5500) ms` | 121 |
| `[5500, 6000) ms` | 69 |
| `[6000, 6500) ms` | 101 |
| `[6500, 7000) ms` | 61 |
| `[7000, 7500) ms` | 23 |

- **Training Dataset Output**: `data\clean_v2\03_labeled_5s\labeled_dataset_51min.parquet`
- **Detached Audit Output**: `data\clean_v2\03_labeled_5s\labeling_audit_51min.parquet`
