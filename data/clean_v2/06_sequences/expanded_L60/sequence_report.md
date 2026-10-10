# Causal Sequence Construction Report

- **Sequence Length ($L$)**: 60 steps (10-second lookback on causal grid)
- **Feature Dimension ($D$)**: 11 features
- **Feature Columns**: `mid_price, spread, spread_bps, mid_return_1s, mid_return_3s, mid_return_5s, mid_volatility_5s, bid_change_1s, ask_change_1s, microprice, depth_imbalance`
- **Total Sequences Constructed**: 6,150

### Partition Counts & Class Distribution

| Split | Input Rows | Sequences Generated | Class Distribution |
| :--- | :--- | :--- | :--- |
| **Train** | 7,722 | 4,338 | {'UP': 1717, 'DOWN': 1717, 'FLAT': 904} |
| **Validation** | 1,922 | 930 | {'DOWN': 398, 'UP': 397, 'FLAT': 135} |
| **Test** | 1,708 | 882 | {'UP': 391, 'DOWN': 387, 'FLAT': 104} |

### Sequence Invariant Audits

- **Duplicate Sequence Endpoints**: `PASS (0 duplicates)`
- **Cross-Split Boundary Violations**: `PASS (0 cross-split)`
- **Cross-Session Boundary Violations**: `PASS (0 cross-session)`
- **Causality Violations**: `PASS (0 future features)`
- **Deterministic Repeatability**: `PASS`
- **Final Sequence Construction Verdict**: `**PASS**`
