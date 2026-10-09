# Causal Sequence Construction Report

- **Sequence Length ($L$)**: 10 steps (10-second lookback on causal grid)
- **Feature Dimension ($D$)**: 11 features
- **Feature Columns**: `mid_price, spread, spread_bps, mid_return_1s, mid_return_3s, mid_return_5s, mid_volatility_5s, bid_change_1s, ask_change_1s, microprice, depth_imbalance`
- **Total Sequences Constructed**: 342

### Partition Counts & Class Distribution

| Split | Input Rows | Sequences Generated | Class Distribution |
| :--- | :--- | :--- | :--- |
| **Train** | 258 | 249 | {'FLAT': 249} |
| **Validation** | 56 | 47 | {'FLAT': 47} |
| **Test** | 55 | 46 | {'FLAT': 46} |

### Sequence Invariant Audits

- **Duplicate Sequence Endpoints**: `PASS (0 duplicates)`
- **Cross-Split Boundary Violations**: `PASS (0 cross-split)`
- **Cross-Session Boundary Violations**: `PASS (0 cross-session)`
- **Causality Violations**: `PASS (0 future features)`
- **Deterministic Repeatability**: `PASS`
- **Final Sequence Construction Verdict**: `**PASS**`
