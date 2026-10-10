# PredAlpha-HFT — Training Pipeline Smoke Test Report

> [!IMPORTANT]
> **DISCLAIMER**: PIPELINE SMOKE TEST ONLY — NOT EVIDENCE OF PREDICTIVE EDGE OR PROFITABILITY. This run confirms end-to-end model training, scaling isolation, and evaluation pipeline integrity on existing canonical data splits. No statistical significance or trading viability is claimed.

- **Execution Timestamp**: `2026-10-04T14:37:50.711931+00:00`
- **Random Seed**: `42`
- **Model**: `LogisticRegression` (`class_weight=balanced`, `solver=lbfgs`)
- **Preprocessing**: `StandardScaler (fitted strictly on train)`
- **Status**: `COMPLETED_SUCCESSFULLY`

## 1. Dataset & Label Confirmation

- **Dataset Source**: Phase 1 Canonical Splits (`data/processed/phase1/`)
- **Total Rows Across Splits**: `279`
- **Target Column**: `label` (5-second forward horizon return direction)
- **Target Classes**: `['DOWN', 'FLAT', 'UP']`
- **Approved Features (9 cols)**: `['mid_price', 'spread', 'spread_bps', 'mid_return_1s', 'mid_return_3s', 'mid_return_5s', 'mid_volatility_5s', 'bid_change_1s', 'ask_change_1s']`
- **Forbidden Leakage Columns Checked (7 cols)**: `['future_delta', 'future_mid', 'future_return', 'label_threshold', 'price_change_5m', 'target_close', 'target_time']`

### Split Boundaries & Class Distributions

| Split | Rows | Time Range (UTC) | Class Distribution |
|:------|:-----|:-----------------|:-------------------|
| **Train** | 202 | 2026-10-02 12:44:12 to 2026-10-02 12:47:33 | `{'FLAT': 164, 'UP': 28, 'DOWN': 10}` |
| **Validation** | 38 | 2026-10-02 12:47:39 to 2026-10-02 12:48:16 | `{'FLAT': 36, 'UP': 2}` |
| **Test** | 39 | 2026-10-02 12:48:22 to 2026-10-02 12:49:00 | `{'FLAT': 37, 'DOWN': 2}` |

> [!NOTE]
> Validation data contains **0 DOWN** samples. Test data contains **0 UP** samples.
> This truncation is an inherent artifact of the short continuous Phase 1 session (~4.8 minutes).

## 2. Evaluation Summary

| Split | Model | Accuracy | Balanced Accuracy | Macro F1 |
|:------|:------|:---------|:------------------|:---------|
| **Train** | Smoke Test (LogisticRegression) | `0.5396` | `0.6316` | `0.4312` |
| **Train** | Baseline (DummyClassifier) | `0.8119` | `0.3333` | `0.2987` |
| **Validation** | Smoke Test (LogisticRegression) | `0.2895` | `0.1528` | `0.1560` |
| **Validation** | Baseline (DummyClassifier) | `0.9474` | `0.5000` | `0.3243` |
| **Test** | Smoke Test (LogisticRegression) | `0.4359` | `0.7027` | `0.2436` |
| **Test** | Baseline (DummyClassifier) | `0.9487` | `0.5000` | `0.3246` |

## 3. Confusion Matrices (Smoke Test Model)

### Train Split
```text
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               9         0         1        10
FLAT              53        87        24       164
UP                13         2        13        28
---------------------------------------------------
Total             75        89        38       202
```

### Validation Split
```text
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               0         0         0         0
FLAT              22        11         3        36
UP                 2         0         0         2
---------------------------------------------------
Total             24        11         3        38
```

### Test Split
```text
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               2         0         0         2
FLAT              22        15         0        37
UP                 0         0         0         0
---------------------------------------------------
Total             24        15         0        39
```

## 4. Pipeline Integrity Verification

- **Train-Only Scaling**: Verified. `StandardScaler` was fit strictly on the 202 training samples.
- **Split Boundaries**: Verified. Exact temporal purge gap (5s) and boundaries preserved.
- **Data Leakage Exclusion**: Verified. No quote columns, target-derived variables, or future timestamps were passed to the model.
- **Artifact Isolation**: Verified. Output saved exclusively to `data/processed/phase3_smoke_test/`. Phase 1 and Phase 2 artifacts were unmodified.

## 5. Statistical Caveats & Limitations

- This run is strictly a pipeline smoke test to verify training and evaluation execution.
- Sample size of 279 total rows is insufficient for statistical generalization.
- Validation ground truth contains 0 DOWN instances; test ground truth contains 0 UP instances.
- Balanced class weights result in high false-positive predictions on rare classes.
- No edge, predictive power, or commercial profitability is claimed.
