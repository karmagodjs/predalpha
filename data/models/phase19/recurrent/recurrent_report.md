# Phase 19B — Small Recurrent Sequence Modeling Report

> [!IMPORTANT]
> **TEMPORAL VALUE TEST VERDICT**:  
> **DECISION CATEGORY: `B` — `LSTM clearly improves over MLP`**  
> - **Small GRU (3-Seed Mean Macro F1)**: **0.5342 ± 0.0049** (Best: **0.5411**, Δ vs MLP: **+0.0253**)
> - **Small LSTM (3-Seed Mean Macro F1)**: **0.5543 ± 0.0043** (Best: **0.5580**, Δ vs MLP: **+0.0454**)
> - **Phase 19A MLP Reference**: **0.5089 ± 0.0028** (Best: **0.5128**)
> - **Recommended Architecture**: **`Small LSTM`**
> - **Test Partition Integrity**: **100% UNTOUCHED & LOCKED** (zero evaluations, zero tuning on test data)

- **Execution Timestamp (UTC)**: `2026-10-07T12:52:17.985956+00:00`
- **Training Samples ($N_{train}$)**: `7,514`
- **Validation Samples ($N_{val}$)**: `1,428`
- **Sequence Dimension**: `L = 10 timesteps x D = 11 causal features`
- **Masking Mechanism**: Fixed 10-step lookback window; warm-up missing values at $t < 5$ were zero-filled by Phase 17 standard scaling (representing mean/neutral prior) with zero NaN/Inf entering the recurrent network. Endpoint $t=9$ contains 100% complete causal data.

---

## 1. Multi-Model Benchmark Comparison Table

| Model Architecture | Input Dim | Params | Train Acc | Val Acc | Val Bal Acc | Val Macro F1 | F1 (DOWN) | F1 (FLAT) | F1 (UP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline (Phase 19A)** | 0 | 0 | 40.59% | **43.21%** | 33.33% | **0.2011** | 0.0000 | 0.0000 | 0.6034 |
| **Logistic Regression (Phase 19A)** | 110 | 333 | 51.13% | **56.09%** | 48.45% | **0.4995** | 0.5891 | 0.3284 | 0.5811 |
| **Small MLP (Best: Seed 999)** | 110 | 9,283 | 55.32% | **55.95%** | 49.98% | **0.5128** | 0.5891 | 0.3742 | 0.5750 |
| **Small MLP (3-Seed Mean ± Std)** | 110 | 9,283 | 54.56% | **56.35%** | 49.55% | **0.5089** | 0.5955 | 0.3512 | 0.5798 |
| **Small GRU (Best: Seed 999)** | 10x11 | 4,419 | 55.06% | **57.91%** | 52.55% | **0.5411** | 0.6082 | 0.4281 | 0.5869 |
| **Small GRU (3-Seed Mean ± Std)** | 10x11 | 4,419 | 56.61% | **57.05%** | 52.04% | **0.5342** | 0.5868 | 0.4253 | 0.5905 |
| **Small LSTM (Best: Seed 999)** | 10x11 | 5,859 | 56.20% | **59.10%** | 54.05% | **0.5580** | 0.5998 | 0.4601 | 0.6142 |
| **Small LSTM (3-Seed Mean ± Std)** | 10x11 | 5,859 | 56.67% | **59.10%** | 53.85% | **0.5543** | 0.5991 | 0.4452 | 0.6185 |

---

## 2. Seed-by-Seed Breakdown (Seeds: 42, 123, 999)

### Model A: Small GRU (Hidden Size = 32, 1 Layer, 4,419 Parameters)
| Seed | Best Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val Bal Acc | Val Macro F1 | F1 (DOWN) | F1 (FLAT) | F1 (UP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `42` | 45 | 0.9055 | 0.9012 | 57.27% | **56.44%** | 51.88% | **0.5319** | 0.5754 | 0.4329 | 0.5873 |
| `123` | 52 | 0.8990 | 0.9037 | 57.49% | **56.79%** | 51.68% | **0.5297** | 0.5768 | 0.4149 | 0.5974 |
| `999` | 30 | 0.9268 | 0.9044 | 55.06% | **57.91%** | 52.55% | **0.5411** | 0.6082 | 0.4281 | 0.5869 |

- **GRU 3-Seed Aggregate**:
  - Validation Accuracy: `57.05% ± 0.63%`
  - Validation Balanced Accuracy: `52.04% ± 0.37%`
  - Validation Macro F1: `0.5342 ± 0.0049`
  - Best Seed: `Seed 999` (Val Macro F1 = `0.5411`)

---

### Model B: Small LSTM (Hidden Size = 32, 1 Layer, 5,859 Parameters)
| Seed | Best Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Val Bal Acc | Val Macro F1 | F1 (DOWN) | F1 (FLAT) | F1 (UP) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `42` | 26 | 0.9184 | 0.8991 | 55.75% | **59.17%** | 54.10% | **0.5566** | 0.6032 | 0.4514 | 0.6153 |
| `123` | 47 | 0.8787 | 0.8953 | 58.05% | **59.03%** | 53.41% | **0.5482** | 0.5944 | 0.4241 | 0.6261 |
| `999` | 31 | 0.9104 | 0.8967 | 56.20% | **59.10%** | 54.05% | **0.5580** | 0.5998 | 0.4601 | 0.6142 |

- **LSTM 3-Seed Aggregate**:
  - Validation Accuracy: `59.10% ± 0.06%`
  - Validation Balanced Accuracy: `53.85% ± 0.31%`
  - Validation Macro F1: `0.5543 ± 0.0043`
  - Best Seed: `Seed 999` (Val Macro F1 = `0.5580`)

---

## 3. Temporal Value Test Analysis

### Primary Research Question:
*Does explicit recurrent sequential modeling over 10 causal steps outperform the flattened Small MLP?*

1. **Performance Comparison**:
   - Small MLP 3-Seed Mean: Macro F1 = `0.5089`, Accuracy = `56.35%`
   - Small GRU 3-Seed Mean: Macro F1 = `0.5342`, Accuracy = `57.05%` (Δ F1: `+0.0253`)
   - Small LSTM 3-Seed Mean: Macro F1 = `0.5543`, Accuracy = `59.10%` (Δ F1: `+0.0454`)

2. **Categorical Assessment**:
   - **Verdict**: **Category `B`** (`LSTM clearly improves over MLP`).
   - **Parameter Efficiency**: Small GRU achieves its performance using only **4,419 parameters** (52.4% fewer parameters than Small MLP's 9,283 parameters). Small LSTM uses **5,859 parameters** (36.9% fewer parameters).
   - **Directional Discrimination**: Both GRU and LSTM maintain balanced discrimination on UP and DOWN directional movements with no collapse onto majority class.

---

## 4. Overfitting and Stability Audit

| Diagnostic Check | Small GRU | Small LSTM | Assessment |
| :--- | :---: | :---: | :--- |
| **Train / Val Accuracy Gap** | `-0.44%` | `-2.43%` | `PASS (Narrow gap, no severe train overfit)` |
| **Val / Train Loss Gap** | `-0.0073` | `-0.0055` | `PASS (Well-regularized)` |
| **Seed Stability (Macro F1 Std)** | `0.0049` | `0.0043` | `PASS (Std <= 0.015 across seeds)` |
| **Seed Stability (Accuracy Std)** | `0.0063` | `0.0006` | `PASS (Consistent convergence)` |
| **Class Collapse Detected** | `No` | `No` | `PASS (All 3 classes predicted with positive recall)` |
| **Overfitting Risk Level** | **LOW** | **LOW** | `PASS` |

---

## 5. Confusion Matrices (Validation Partition, N = 1,428)

### Best Small GRU (Seed 999)
| True \ Pred | **Pred DOWN** | **Pred FLAT** | **Pred UP** |
| :--- | :---: | :---: | :---: |
| **True DOWN** | 392 | 25 | 199 |
| **True FLAT** | 58 | 67 | 70 |
| **True UP** | 223 | 26 | 368 |

### Best Small LSTM (Seed 999)
| True \ Pred | **Pred DOWN** | **Pred FLAT** | **Pred UP** |
| :--- | :---: | :---: | :---: |
| **True DOWN** | 374 | 22 | 220 |
| **True FLAT** | 62 | 72 | 61 |
| **True UP** | 195 | 24 | 398 |

---

## 6. Official Recommendation

- **Temporal Modeling Value**: Evaluated under rigorous 3-seed protocol.
- **Recommended Model**: **`Small LSTM`**.
- **Test Set Status**: The test partition (`test_scaled.npz`, $N=1,582$) remains **100% LOCKED AND UNTOUCHED**.
