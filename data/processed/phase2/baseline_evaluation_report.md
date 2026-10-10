# PredAlpha-HFT — Phase 2: Baseline Models & Evaluation Report

**Date**: 2026-10-04T11:24:43.663488+00:00
**Random Seed**: `42`
**Candidate Model Selection**: `dummy_classifier` (FAILED_BASELINE)

## 1. Executive Summary & Core Finding

Phase 2 evaluated whether 9 causal orderbook and return features computed on 1-second BBO market data 
contain predictive signal for 5-second future directional returns beyond a majority-class baseline. 

**Key Conclusion**: **No genuine predictive signal is present.** Logistic Regression with balanced class weighting 
fails to beat the `DummyClassifier(strategy='most_frequent')` baseline on the validation split. 
- **Validation Accuracy**: Dummy = `0.9474` vs LogisticRegression = `0.2895`
- **Validation Balanced Accuracy**: Dummy = `0.5000` vs LogisticRegression = `0.1528`
- **Validation Macro F1**: Dummy = `0.3243` vs LogisticRegression = `0.1560`

Class weighting heavily incentivizes the model to predict rare classes (`DOWN` and `UP`), leading to severe 
over-prediction of `DOWN` (24 out of 38 predictions in validation, where 0 `DOWN` instances actually occur). 
On the test split, Logistic Regression achieves only 43.59% accuracy with an 8.3% precision on the `DOWN` class. 

**Decision**: **Do NOT proceed to deep learning or neural architectures.** Further modeling on this small dataset 
is statistically ungrounded. The project must first collect substantially more continuous market sessions.

---

## 2. Dataset & Split Specifications

- **Total Rows Across Splits**: `279`
- **Train Split**: `202` rows (`{'FLAT': 164, 'UP': 28, 'DOWN': 10}`)
- **Validation Split**: `38` rows (`{'FLAT': 36, 'UP': 2}`)
- **Test Split**: `39` rows (`{'FLAT': 37, 'DOWN': 2}`)

### Feature Columns (9)

```
mid_price, spread, spread_bps, mid_return_1s, mid_return_3s, mid_return_5s, mid_volatility_5s, bid_change_1s, ask_change_1s
```

### Excluded Non-Feature Columns

- `timestamp`: Temporal index; excluded to prevent spurious temporal memorization.
- `asset_id`: Contract identifier (`btc_88000`); constant across dataset.
- `bid`, `ask`: Absolute price level quotes; excluded in favor of stationary spread and return features.
- `label`: Target directional classification (`DOWN`, `FLAT`, `UP`).
- Forbidden leakage columns (`future_mid`, `future_delta`, etc.): strictly purged in Phase 1.

---

## 3. Baseline Model Architectures & Preprocessing

### Model A: DummyClassifier (Reference Baseline)

- **Strategy**: `most_frequent`
- Always predicts `FLAT` (the dominant majority class representing 81.2% of train, 94.7% of val, 94.9% of test).

### Model B: LogisticRegression (Candidate Model)

- **Pipeline**: `StandardScaler -> LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`
- **Preprocessing Isolation**: `StandardScaler` is fitted **strictly on the Training split**. Validation and test data are strictly transformed without leaking mean or variance statistics.
- **Missing / Non-Finite Handling**: All inputs are checked prior to training; 0 missing values and 0 non-finite values permitted.

---

## 4. Validation Results & Model Selection Protocol

Model selection is conducted strictly on the **Validation Split** (38 rows). The Test Split is completely untouched during this phase.

### Validation Metric Comparison

| Metric | DummyClassifier (Baseline) | LogisticRegression (Candidate) | Difference (LR - Dummy) |
|:-------|:---------------------------|:-------------------------------|:------------------------|
| **Accuracy** | `0.9474` | `0.2895` | `-0.6579` |
| **Balanced Accuracy** | `0.5000` | `0.1528` | `-0.3472` |
| **Macro F1** | `0.3243` | `0.1560` | `-0.1683` |
| **Log Loss** | `None` | `None` | N/A (Ground truth lacks DOWN) |

*Note on Log Loss*: Log loss is omitted because ground truth in the validation split completely lacks the `DOWN` class (absent: `['DOWN']`). Computing standard cross-entropy across the 3-class simplex on an incomplete split produces misleading metrics.

### Validation Per-Class Breakdown

#### Model A: DummyClassifier
| Class | Precision | Recall | F1-Score | Support | Ground Truth Present |
|:------|:----------|:-------|:---------|:--------|:---------------------|
| `DOWN` | `0.0000` | `0.0000` | `0.0000` | `0` | `False` |
| `FLAT` | `0.9474` | `1.0000` | `0.9730` | `36` | `True` |
| `UP` | `0.0000` | `0.0000` | `0.0000` | `2` | `True` |

#### Model B: LogisticRegression
| Class | Precision | Recall | F1-Score | Support | Ground Truth Present |
|:------|:----------|:-------|:---------|:--------|:---------------------|
| `DOWN` | `0.0000` | `0.0000` | `0.0000` | `0` | `False` |
| `FLAT` | `1.0000` | `0.3056` | `0.4681` | `36` | `True` |
| `UP` | `0.0000` | `0.0000` | `0.0000` | `2` | `True` |

### Validation Confusion Matrices

```
--- DummyClassifier (Validation) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               0         0         0         0
FLAT               0        36         0        36
UP                 0         2         0         2
---------------------------------------------------
Total              0        38         0        38

--- LogisticRegression (Validation) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               0         0         0         0
FLAT              22        11         3        36
UP                 2         0         0         2
---------------------------------------------------
Total             24        11         3        38
```

### Selection Verdict

- **Verdict**: `FAILED_BASELINE`
- **Selected Candidate**: `dummy_classifier`
- **Reasoning**: LogisticRegression failed to demonstrate credible improvement over DummyClassifier on validation. Balanced Accuracy: LR=0.1528 vs Dummy=0.5000; Macro F1: LR=0.1560 vs Dummy=0.3243; Accuracy: LR=0.2895 vs Dummy=0.9474. Class weighting caused excessive false-positive predictions of rare classes on validation.

---

## 5. Test Split Evaluation (Evaluated Exactly Once)

Out-of-sample evaluation on the test split (39 rows) was executed exactly once following model selection.

### Test Metric Comparison

| Metric | DummyClassifier (Baseline) | LogisticRegression | Difference (LR - Dummy) |
|:-------|:---------------------------|:-------------------|:------------------------|
| **Accuracy** | `0.9487` | `0.4359` | `-0.5128` |
| **Balanced Accuracy** | `0.5000` | `0.7027` | `+0.2027` |
| **Macro F1** | `0.3246` | `0.2436` | `-0.0810` |
| **Log Loss** | `None` | `None` | N/A (Ground truth lacks UP) |

### Test Confusion Matrices

```
--- DummyClassifier (Test) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               0         2         0         2
FLAT               0        37         0        37
UP                 0         0         0         0
---------------------------------------------------
Total              0        39         0        39

--- LogisticRegression (Test) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN               2         0         0         2
FLAT              22        15         0        37
UP                 0         0         0         0
---------------------------------------------------
Total             24        15         0        39
```

### Test Per-Class Breakdown (LogisticRegression)

| Class | Precision | Recall | F1-Score | Support | Ground Truth Present |
|:------|:----------|:-------|:---------|:--------|:---------------------|
| `DOWN` | `0.0833` | `1.0000` | `0.1538` | `2` | `True` |
| `FLAT` | `1.0000` | `0.4054` | `0.5769` | `37` | `True` |
| `UP` | `0.0000` | `0.0000` | `0.0000` | `0` | `False` |

---

## 6. Critical Statistical Limitations & Discussion

1. **Severe Split Truncation & Absent Classes**:
   - The validation split contains **0 DOWN samples**.
   - The test split contains **0 UP samples**.
   - Evaluating 3-class precision, recall, and Macro F1 is structurally compromised by missing support.
2. **Pathology of Class-Weighted Training on Small Datasets**:
   - In training data, `DOWN` represents only 4.95% (10/202 rows). Class balancing applies an inverse weight of ~6.7x to DOWN and ~2.4x to UP.
   - Because the underlying 9 features possess almost no linear correlation with future 5-second returns, the model lowers its decision threshold drastically to avoid the heavy penalty on DOWN.
   - Consequently, in validation, it predicts `DOWN` 24 times (all false positives). In test, it predicts `DOWN` 24 times (2 true positives, 22 false positives; precision = 8.33%).
3. **Zero Statistical Significance**:
   - The total sample represents less than 5 minutes of trading. P-values and standard errors cannot establish significance.
4. **No Commercial or Profitability Claims**:
   - The model is not commercially viable and would lose significant capital through excessive trading fees and false-positive turnover.

---

## 7. Decision & Next Steps

- **Immediate Decision**: **HALT MODELING**. Do not implement LSTM, Transformer, or neural architectures on this dataset.
- **Recommended Next Step**: Prioritize data collection to capture at least 50-100 full trading sessions with synchronized depth orderbook events before resuming machine learning experimentation.
