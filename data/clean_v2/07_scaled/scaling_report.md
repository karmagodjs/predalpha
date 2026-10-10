# Train-Only Feature Scaling Report

- **Scaler Type**: `standard` (z-score standardization)
- **Fit Source**: `train` (strictly TRAIN only)
- **Feature Dimension ($D$)**: 11 features
- **Sequence Length ($L$)**: 10 steps
- **Zero-Variance Features Handled**: `mid_price, spread, spread_bps, mid_return_1s, mid_return_3s, mid_return_5s, mid_volatility_5s, bid_change_1s, ask_change_1s`

### Scaled Dataset Partitions

| Split | Input Sequences | Scaled Shape | Target Untouched |
| :--- | :--- | :--- | :--- |
| **Train** | 249 | `(249, 10, 11)` | `PASS` |
| **Validation** | 47 | `(47, 10, 11)` | `PASS` |
| **Test** | 46 | `(46, 10, 11)` | `PASS` |

### Invariant & Leakage Audits

- **Fit Source Verification**: `PASS (fitted strictly on train)`
- **Validation Leakage Check**: `PASS (validation data cannot affect scaler)`
- **Test Leakage Check**: `PASS (test data cannot affect scaler)`
- **Sequence Shape Preservation**: `PASS`
- **Target Isolation**: `PASS (labels untouched)`
- **Deterministic Output**: `PASS`
- **Final Scaling Verdict**: `**PASS**`
