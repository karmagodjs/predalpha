# Causal Sequence Construction Report

- **Sequence Length ($L$)**: 20 steps (10-second lookback on causal grid)
- **Feature Dimension ($D$)**: 11 features
- **Feature Columns**: `mid_price, spread, spread_bps, mid_return_1s, mid_return_3s, mid_return_5s, mid_volatility_5s, bid_change_1s, ask_change_1s, microprice, depth_imbalance`
- **Total Sequences Constructed**: 9,604

### Partition Counts & Class Distribution

| Split | Input Rows | Sequences Generated | Class Distribution |
| :--- | :--- | :--- | :--- |
| **Train** | 8,072 | 6,894 | {'DOWN': 2787, 'UP': 2787, 'FLAT': 1320} |
| **Validation** | 1,572 | 1,268 | {'DOWN': 545, 'UP': 545, 'FLAT': 178} |
| **Test** | 1,708 | 1,442 | {'UP': 640, 'DOWN': 636, 'FLAT': 166} |

### Sequence Invariant Audits

- **Duplicate Sequence Endpoints**: `PASS (0 duplicates)`
- **Cross-Split Boundary Violations**: `PASS (0 cross-split)`
- **Cross-Session Boundary Violations**: `PASS (0 cross-session)`
- **Causality Violations**: `PASS (0 future features)`
- **Deterministic Repeatability**: `PASS`
- **Final Sequence Construction Verdict**: `**PASS**`
