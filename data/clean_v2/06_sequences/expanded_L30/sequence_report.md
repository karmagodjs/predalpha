# Causal Sequence Construction Report

- **Sequence Length ($L$)**: 30 steps (10-second lookback on causal grid)
- **Feature Dimension ($D$)**: 11 features
- **Feature Columns**: `mid_price, spread, spread_bps, mid_return_1s, mid_return_3s, mid_return_5s, mid_volatility_5s, bid_change_1s, ask_change_1s, microprice, depth_imbalance`
- **Total Sequences Constructed**: 8,684

### Partition Counts & Class Distribution

| Split | Input Rows | Sequences Generated | Class Distribution |
| :--- | :--- | :--- | :--- |
| **Train** | 8,072 | 6,274 | {'DOWN': 2507, 'UP': 2507, 'FLAT': 1260} |
| **Validation** | 1,572 | 1,108 | {'DOWN': 472, 'UP': 472, 'FLAT': 164} |
| **Test** | 1,708 | 1,302 | {'UP': 580, 'DOWN': 576, 'FLAT': 146} |

### Sequence Invariant Audits

- **Duplicate Sequence Endpoints**: `PASS (0 duplicates)`
- **Cross-Split Boundary Violations**: `PASS (0 cross-split)`
- **Cross-Session Boundary Violations**: `PASS (0 cross-session)`
- **Causality Violations**: `PASS (0 future features)`
- **Deterministic Repeatability**: `PASS`
- **Final Sequence Construction Verdict**: `**PASS**`
