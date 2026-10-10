# Train-Only Feature Scaling Report

- **Scaler Type**: `standard` (z-score standardization)
- **Fit Source**: `train` (strictly TRAIN only)
- **Feature Dimension ($D$)**: 11 features
- **Sequence Length ($L$)**: 30 steps
- **Zero-Variance Features Handled**: `None`

### Scaled Dataset Partitions

| Split | Input Sequences | Scaled Shape | Target Untouched |
| :--- | :--- | :--- | :--- |
| **Train** | 6,274 | `(6274, 30, 11)` | `PASS` |
| **Validation** | 1,108 | `(1108, 30, 11)` | `PASS` |
| **Test** | 1,302 | `(1302, 30, 11)` | `PASS` |

### Invariant & Leakage Audits

- **Fit Source Verification**: `PASS (fitted strictly on train)`
- **Validation Leakage Check**: `PASS (validation data cannot affect scaler)`
- **Test Leakage Check**: `PASS (test data cannot affect scaler)`
- **Sequence Shape Preservation**: `PASS`
- **Target Isolation**: `PASS (labels untouched)`
- **Deterministic Output**: `PASS`
- **Final Scaling Verdict**: `**PASS**`
