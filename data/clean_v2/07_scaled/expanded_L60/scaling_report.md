# Train-Only Feature Scaling Report

- **Scaler Type**: `standard` (z-score standardization)
- **Fit Source**: `train` (strictly TRAIN only)
- **Feature Dimension ($D$)**: 11 features
- **Sequence Length ($L$)**: 60 steps
- **Zero-Variance Features Handled**: `None`

### Scaled Dataset Partitions

| Split | Input Sequences | Scaled Shape | Target Untouched |
| :--- | :--- | :--- | :--- |
| **Train** | 4,338 | `(4338, 60, 11)` | `PASS` |
| **Validation** | 930 | `(930, 60, 11)` | `PASS` |
| **Test** | 882 | `(882, 60, 11)` | `PASS` |

### Invariant & Leakage Audits

- **Fit Source Verification**: `PASS (fitted strictly on train)`
- **Validation Leakage Check**: `PASS (validation data cannot affect scaler)`
- **Test Leakage Check**: `PASS (test data cannot affect scaler)`
- **Sequence Shape Preservation**: `PASS`
- **Target Isolation**: `PASS (labels untouched)`
- **Deterministic Output**: `PASS`
- **Final Scaling Verdict**: `**PASS**`
