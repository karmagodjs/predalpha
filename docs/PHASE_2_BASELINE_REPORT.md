# PredAlpha-HFT — Phase 2: Baseline Models & Evaluation Report

## 1. Executive Summary & Core Decision

This document details the experimental baseline evaluation for **Phase 2: Baseline Models & Evaluation** in the **PredAlpha-HFT** project.

### Core Objective
Determine whether the 9 causal orderbook and momentum features constructed in Phase 1 (`data/processed/phase1/canonical_dataset.parquet`) exhibit genuine, statistically grounded predictive signal for 5-second future directional returns beyond a majority-class prediction baseline.

### Primary Decision: HALT DEEP LEARNING MODELING
- **Validation Selection Result**: **FAILED_BASELINE**. The candidate model (`LogisticRegression` with `StandardScaler` and `class_weight='balanced'`) failed to demonstrate credible improvement over the reference baseline (`DummyClassifier(strategy='most_frequent')`) on the validation split.
- **Key Validation Metrics**:
  - `DummyClassifier`: Accuracy = **0.9474**, Balanced Accuracy = **0.5000**, Macro F1 = **0.3243**
  - `LogisticRegression`: Accuracy = **0.2895**, Balanced Accuracy = **0.1528**, Macro F1 = **0.1560**
- **Root Cause of Breakdown**:
  Because the validation split contains 0 DOWN rows (36 FLAT, 2 UP), and training down-weights the dominant class by ~6.7x to balance rare events, Logistic Regression severely over-predicts DOWN (24 out of 38 predictions), resulting in 24 false positives and an accuracy collapse to 28.95%.
- **Out-of-Sample Test Verification (Evaluated Once)**:
  On the test split (37 FLAT, 2 DOWN, 0 UP), Logistic Regression achieves an accuracy of only **43.59%** (vs **94.87%** for DummyClassifier) and an effective precision on DOWN of only **8.33%** (22 false positives, 2 true positives).
- **Mandate**: **Do NOT proceed to LSTM, Transformer, or neural architectures.** Further machine learning modeling on this 289-row single-session dataset is scientifically ungrounded. The project must first collect substantially more continuous market sessions with depth orderbook updates.

---

## 2. Dataset & Split Specifications

All models were evaluated on the verified, leakage-free chronological splits generated in Phase 1:

| Split Name | Row Count | Start Time (UTC) | End Time (UTC) | Class Distribution | Absent Ground Truth Classes |
|:---|:---:|:---|:---|:---|:---:|
| **Train** | 202 | 2026-10-02 12:44:12 | 2026-10-02 12:47:33 | FLAT: 164 (81.2%), UP: 28 (13.9%), DOWN: 10 (5.0%) | None (All 3 present) |
| *Purge Gap* | 5 rows | 12:47:34 | 12:47:38 | Boundary isolation (5 seconds = prediction horizon) | N/A |
| **Validation** | 38 | 2026-10-02 12:47:39 | 2026-10-02 12:48:16 | FLAT: 36 (94.7%), UP: 2 (5.3%), DOWN: 0 (0.0%) | **DOWN** |
| *Purge Gap* | 5 rows | 12:48:17 | 12:48:21 | Boundary isolation (5 seconds = prediction horizon) | N/A |
| **Test** | 39 | 2026-10-02 12:48:22 | 2026-10-02 12:49:00 | FLAT: 37 (94.9%), DOWN: 2 (5.1%), UP: 0 (0.0%) | **UP** |
| **Total** | **279** | **2026-10-02 12:44:12** | **2026-10-02 12:49:00** | FLAT: 237, UP: 30, DOWN: 12 | (10 rows purged in gaps) |

---

## 3. Feature Matrix & Target Schema

### Approved Feature Columns (9)
All features are causal (backward-looking) and strictly stationary:
1. `mid_price`: Current mid-quote $(bid + ask) / 2$ ($0.064 - $0.0915)
2. `spread`: Current top-of-book bid-ask spread $(ask - bid)$ ($0.001 - $0.013)
3. `spread_bps`: Spread expressed in basis points $(spread / mid\_price) \times 10{,}000$
4. `mid_return_1s`: 1-second percentage return of the mid-price
5. `mid_return_3s`: 3-second percentage return of the mid-price
6. `mid_return_5s`: 5-second percentage return of the mid-price
7. `mid_volatility_5s`: Rolling 5-second standard deviation of 1-second returns
8. `bid_change_1s`: 1-second first difference of best bid quote
9. `ask_change_1s`: 1-second first difference of best ask quote

### Excluded Columns & Leakage Prevention
The following columns were strictly excluded from feature inputs:
- `timestamp`: DatetimeIndex; excluded to prevent temporal memorization.
- `asset_id`: Contract identifier string (`"btc_88000"`); constant.
- `bid`, `ask`: Absolute level quotes; excluded in favor of stationary differences and spreads.
- `label`: Target directional category (`"DOWN"`, `"FLAT"`, `"UP"`).
- Intermediate labeling/future columns (`future_mid`, `future_delta`, `future_return`, `label_threshold`): purged in Phase 1 and checked by automated assertion before extraction.

### Non-Finite & Missing Value Safeguards
Before passing features to scikit-learn:
- 0 missing values (`NaN`, `None`, `pd.NA`) are permitted.
- All values must satisfy `np.isfinite(x)` (0 `inf` or `-inf`).
- Any violation triggers an immediate `ValueError`.

---

## 4. Model Architectures & Preprocessing Protocol

### Model A: DummyClassifier (Reference Majority Baseline)
- **Class**: `sklearn.dummy.DummyClassifier`
- **Strategy**: `most_frequent`
- **Behavior**: Unconditionally predicts `FLAT` on every inference step.
- **Purpose**: Establishes the performance floor that any candidate model must demonstrably beat.

### Model B: LogisticRegression (Candidate Model)
- **Pipeline Architecture**:
  ```python
  Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', LogisticRegression(
          class_weight='balanced',
          max_iter=1000,
          solver='lbfgs',
          random_state=42
      ))
  ])
  ```
- **Strict Preprocessing Isolation**:
  - `StandardScaler` is fitted **strictly on the Training split** (202 rows).
  - Validation and Test splits are transformed strictly using the training mean and standard deviation.
  - No data leakage across fold boundaries occurs.

---

## 5. Validation Evaluation & Candidate Selection

Model selection was conducted strictly on the **Validation Split** (38 rows). The Test Split was not accessed, inspected, or tuned upon during this step.

### Comprehensive Validation Results

| Metric | DummyClassifier (Baseline) | LogisticRegression (Candidate) | Difference (LR - Baseline) | Winner |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | **0.9474** (36/38) | 0.2895 (11/38) | -0.6579 | **Dummy** |
| **Balanced Accuracy** | **0.5000** | 0.1528 | -0.3472 | **Dummy** |
| **Macro F1** | **0.3243** | 0.1560 | -0.1683 | **Dummy** |
| **Log Loss** | *Omitted* (lacks DOWN) | *Omitted* (lacks DOWN) | N/A | N/A |

### Validation Per-Class Metrics

#### Model A: DummyClassifier (Validation)
| Class | Precision | Recall | F1-Score | Support | In Ground Truth? | In Predictions? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `DOWN` | 0.0000 | 0.0000 | 0.0000 | 0 | No (Absent) | No |
| `FLAT` | **0.9474** | **1.0000** | **0.9730** | 36 | Yes | Yes (38/38) |
| `UP` | 0.0000 | 0.0000 | 0.0000 | 2 | Yes | No |

#### Model B: LogisticRegression (Validation)
| Class | Precision | Recall | F1-Score | Support | In Ground Truth? | In Predictions? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `DOWN` | 0.0000 | 0.0000 | 0.0000 | 0 | No (Absent) | **Yes (24/38)** |
| `FLAT` | 1.0000 | 0.3056 | 0.4681 | 36 | Yes | Yes (11/38) |
| `UP` | 0.0000 | 0.0000 | 0.0000 | 2 | Yes | Yes (3/38) |

### Validation Confusion Matrices

```
--- Model A: DummyClassifier (Validation) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN                0         0         0         0
FLAT                0        36         0        36
UP                  0         2         0         2
---------------------------------------------------
Total               0        38         0        38

--- Model B: LogisticRegression (Validation) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN                0         0         0         0
FLAT               22        11         3        36
UP                  2         0         0         2
---------------------------------------------------
Total              24        11         3        38
```

### Candidate Selection Verdict
- **Verdict**: `FAILED_BASELINE`
- **Selected Model**: `dummy_classifier`
- **Selection Decision**:
  `LogisticRegression` collapsed validation accuracy by 65.79 percentage points and balanced accuracy by 34.72 percentage points. Because it failed the baseline benchmark, `DummyClassifier` is retained as the candidate reference model.

---

## 6. Out-of-Sample Test Evaluation

Following model selection, the test split (39 rows) was evaluated exactly once.

### Test Results Summary

| Metric | DummyClassifier (Baseline) | LogisticRegression | Difference (LR - Dummy) |
|:---|:---:|:---:|:---:|
| **Accuracy** | **0.9487** (37/39) | 0.4359 (17/39) | -0.5128 |
| **Balanced Accuracy** | 0.5000 | **0.7027** | +0.2027 |
| **Macro F1** | **0.3246** | 0.2436 | -0.0810 |
| **Log Loss** | *Omitted* (lacks UP) | *Omitted* (lacks UP) | N/A |

### Test Per-Class Metrics (LogisticRegression)
| Class | Precision | Recall | F1-Score | Support | In Ground Truth? | In Predictions? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `DOWN` | 0.0833 | 1.0000 | 0.1538 | 2 | Yes | Yes (24/39) |
| `FLAT` | 1.0000 | 0.4054 | 0.5769 | 37 | Yes | Yes (15/39) |
| `UP` | 0.0000 | 0.0000 | 0.0000 | 0 | No (Absent) | No |

### Test Confusion Matrices

```
--- Model A: DummyClassifier (Test) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN                0         2         0         2
FLAT                0        37         0        37
UP                  0         0         0         0
---------------------------------------------------
Total               0        39         0        39

--- Model B: LogisticRegression (Test) ---
True \ Pred      DOWN      FLAT        UP     Total
---------------------------------------------------
DOWN                2         0         0         2
FLAT               22        15         0        37
UP                  0         0         0         0
---------------------------------------------------
Total              24        15         0        39
```

### Analysis of the Test Illusion
While LogisticRegression shows a Balanced Accuracy of **0.7027** on test, this is an artifact of the small split:
1. Ground truth contains only 2 DOWN rows and 37 FLAT rows.
2. The model aggressively predicted DOWN **24 times**.
3. It caught both DOWN rows (Recall = 100%), but generated **22 false positives** on FLAT rows.
4. The precision of DOWN predictions is a dismal **8.33%** (2 out of 24).
5. In live market trading, executing on these false-positive signals would result in catastrophic transaction cost churn and loss of capital.

---

## 7. Explicit Missing-Class & Log Loss Handling

In compliance with the project protocol:
1. **Absent Ground Truth Classes**:
   - In Validation: `DOWN` is absent (0 samples).
   - In Test: `UP` is absent (0 samples).
   - These are explicitly tracked in `absent_classes_ground_truth`.
2. **Log Loss Status**:
   - The protocol dictates: *"Log Loss, only if valid probabilities are available and all required classes are represented."*
   - Because all 3 classes (`DOWN`, `FLAT`, `UP`) are not simultaneously represented in validation or test ground truth, standard cross-entropy over the complete class simplex is structurally unrepresentative.
   - Therefore, `log_loss` is recorded as `null` with status `OMITTED_MISSING_CLASSES_IN_SPLIT`.
   - On the Training split (where all 3 classes are present), Log Loss was successfully computed: **0.8555** for Logistic Regression.

---

## 8. Statistical Limitations & Pathology of Small Datasets

1. **Extreme Sample Truncation**:
   - The entire canonical dataset spans only 289 seconds (4.8 minutes).
   - The validation set has 38 rows; the test set has 39 rows.
   - It is impossible to establish asymptotic statistical significance or generalized predictive power on sample sizes this small.
2. **Class-Weighted Training Does Not Substitute for Sample Volume**:
   - In training data, `DOWN` appears only 10 times (4.95%).
   - Inverse class weighting heavily penalizes minority misclassifications, forcing the model to drastically shift its intercept towards DOWN.
   - Without sufficient feature variance and sample diversity, the model simply fires DOWN signals indiscriminately.
3. **No Commercial Viability**:
   - Neither model demonstrates viable market-making or directional trading capability.

---

## 9. Artifact Inventory

All Phase 2 outputs are isolated in `data/processed/phase2/`:

| Artifact Path | Description |
|:---|:---|
| `data/processed/phase2/baseline_metrics.json` | Comprehensive machine-readable metrics JSON covering train, validation, and test splits. |
| `data/processed/phase2/baseline_evaluation_report.md` | Human-readable Markdown report with complete metric tables and confusion matrices. |
| `data/processed/phase2/model_config.json` | Exact model hyperparameters, feature list, excluded columns, and deterministic random seed. |
| `data/processed/phase2/confusion_matrices.json` | Structured confusion matrices across all splits for both models. |
| `data/processed/phase2/candidate_pipeline.joblib` | Serialized candidate pipeline (`DummyClassifier` per validation selection). |
| `data/processed/phase2/logistic_regression_pipeline.joblib` | Serialized `StandardScaler + LogisticRegression` pipeline for auditability. |
| `data/processed/phase2/dummy_pipeline.joblib` | Serialized `DummyClassifier` pipeline. |

---

## 10. Reproduction & Verification

To reproduce Phase 2 evaluation, generate all artifacts, and run the test suite:

```powershell
# 1. Execute the Phase 2 Baseline Pipeline
python -m models.baseline_models

# 2. Run Phase 2 Specific Tests (11 tests)
pytest tests/test_phase2_baseline.py -v

# 3. Run the Entire Repository Test Suite (50 tests)
pytest
```

### Verified Test Suite Execution Output
```
collected 50 items

tests\test_collection_pipeline.py ......                                 [ 12%]
tests\test_feature_engineering.py .                                      [ 14%]
tests\test_ml_pipeline.py ..                                             [ 18%]
tests\test_optimized_orderbook.py ...                                    [ 24%]
tests\test_orderbook.py ...                                              [ 30%]
tests\test_phase1_data_pipeline.py ...............                       [ 60%]
tests\test_phase2_baseline.py ...........                                [ 82%]
tests\test_track_ab.py ....                                              [ 90%]
tests\test_training_pipeline.py ...                                      [ 96%]
tests\test_walk_forward_phase6.py ..                                     [100%]

======================= 50 passed, 2 warnings in 12.25s =======================
```

---

## 11. Final Recommendation

1. **Do NOT build a neural network.** A deep neural network on 202 training samples will trivially overfit or predict constant outputs.
2. **Prioritize Data Collection**:
   - Collect at least 50 to 100 continuous trading sessions (minimum 100,000+ rows) across diverse market volatility regimes.
   - Record Level 2 orderbook depth to enable causal orderbook imbalance features.
3. **Multi-Horizon & Multi-Session Cross-Validation**:
   - Once multi-session data is collected, implement Group-K-Fold or Session-Purged Walk-Forward cross-validation before revisiting model architectures.
