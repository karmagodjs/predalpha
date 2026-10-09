# Causal 1-Second Resampling Validation Report

- **Input Canonical Events**: 498
- **Output 1-Second Grid Rows**: 3,074
- **Grid Timestamp Range**: 1790916233000 -> 1790919306000
- **Matched Source Timestamp Range**: 1790916232243 -> 1790919304160
- **Mean Event Age**: 11349.28 ms
- **Maximum Event Age**: 150,533 ms
- **Stale Rows (`age > 5000 ms`)**: 1,664 (54.13%)
- **Future-Event Violations (`age < 0`)**: `PASS (0 violations)`
- **Duplicate Grid Timestamps**: `PASS (0 duplicates)`
- **Timestamp Monotonicity**: `PASS (strictly monotonic)`
- **Session Boundary Isolation**: `PASS (0 cross-session matches)`
