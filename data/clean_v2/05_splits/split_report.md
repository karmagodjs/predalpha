# Purged Temporal Splitting Validation Report

- **Input Total Rows**: 375
- **Train Partition Rows**: 258 (68.80%)
- **Validation Partition Rows**: 56 (14.93%)
- **Test Partition Rows**: 55 (14.67%)
- **Purged Boundary Rows**: 6 (1.60%)

### Temporal Partitions & Purge Gaps

- **Train Range**: `1790916233000 -> 1790918535000`
- **Purge Gap 1 (Train -> Val)**: `PASS (29,000 ms >= 20,000 ms)`
- **Validation Range**: `1790918564000 -> 1790918937000`
- **Purge Gap 2 (Val -> Test)**: `PASS (29,000 ms >= 20,000 ms)`
- **Test Range**: `1790918966000 -> 1790919281000`

### Boundary Invariant Checks

- **Timestamp Overlap**: `PASS (0 overlaps)`
- **Duplicate Samples Across Splits**: `PASS (0 duplicate rows)`
- **Label Horizon Cross-Boundary Leakage**: `PASS (0 label boundary violations)`
- **Feature Lookback Cross-Boundary Leakage**: `PASS (0 lookback boundary violations)`
- **Session Boundary Isolation**: `PASS (0 session violations)`
- **Deterministic Execution**: `PASS`
- **Final Split Verdict**: `**PASS**`
