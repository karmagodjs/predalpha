# Phase 23C — Temperature Scaling & Probability Calibration Report

> [!IMPORTANT]
> **VALIDATION-ONLY EVALUATION**: The locked test partition was **NEVER** accessed, read, evaluated, or tuned against.
> **EXPERIMENT SCOPE**: Evaluates probability calibration for the frozen Phase 19B SmallLSTM checkpoint (`best_lstm.pt`) using multiclass temperature scaling fitted strictly on an earlier chronological calibration subset and evaluated out-of-sample on a later disjoint evaluation subset.
> **RESEARCH-GRADE DISCLAIMER**: A lower validation loss or calibration error is strictly evidence about the evaluated partition. It does NOT constitute evidence of trading alpha, execution viability, or live market profitability.

---

## 1. Executive Summary & Recommendation

- **Implementation Status**: **PASS / COMPLETE**
- **Decision Verdict**: **NO-GO FOR PRODUCTION CONVERSION** (Calibration did not improve probability quality on the out-of-sample evaluation partition; see Section 4).
- **Fitted Temperature ($T$)**: `0.7951` (bounds: `[0.01, 10.0]`, converged: `True`)
- **Evaluation Out-of-Sample Results**:
  - **Negative Log-Likelihood (NLL)**: Baseline `0.9006` $\\rightarrow$ Calibrated `0.9047` ($\Delta = +0.0041$)
  - **Brier Score**: Baseline `0.5472` $\\rightarrow$ Calibrated `0.5496` ($\Delta = +0.0025$)
  - **Expected Calibration Error (ECE)**: Baseline `0.0494` $\\rightarrow$ Calibrated `0.0567` ($\Delta = +0.0073$)
  - **Accuracy / Macro F1**: Exactly identical (`0.5889` / `0.5832`) due to the mathematical order-preserving property of temperature scaling.

---

## 2. Temporal Partitioning & Leakage Elimination

To avoid temporal lookahead and data leakage, sequences are partitioned strictly chronologically based on sequence endpoint timestamps. Between the calibration set and evaluation set, an architectural purge gap is enforced to guarantee zero overlapping lookbacks or label horizons.

| Parameter | Calibration Subset | Purge Boundary | Out-of-Sample Evaluation Subset |
| :--- | :---: | :---: | :---: |
| **Sample Count** | `832` (58.3%) | `0` samples | `596` (41.7%) |
| **Start Endpoint (ms)** | `1791291910000` | — | `1791293110000` |
| **End Endpoint (ms)** | `1791293029000` | — | `1791294206000` |
| **Observed Gap** | — | **81000 ms** (81.0 s) | — |
| **Required Purge Gap** | — | **22000 ms** (22.0 s) | — |
| **Purge Satisfied** | — | **YES (81000 ms $\ge$ 22000 ms)** | — |

### Class Distributions Across Splits
- **Calibration Subset**: DOWN = `378`, FLAT = `75`, UP = `379`
- **Evaluation Subset**: DOWN = `238`, FLAT = `120`, UP = `238`

---

## 3. Optimization & Calibration Fitting

Multiclass temperature scaling scales logits $z \\in \\mathbb{R}^3$ by a scalar $T > 0$:
$$P_k = \\frac{\\exp(z_k / T)}{\\sum_{j=1}^3 \\exp(z_j / T)}$$

The scalar $T$ was optimized by minimizing multiclass negative log-likelihood strictly over the **Calibration Subset** via bounded scalar minimization:

- **Optimizer**: `scipy.optimize.minimize_scalar(bounded)`
- **Search Range**: `[0.01, 10.0]`
- **Convergence Status**: `True` (Solution found.)
- **Evaluations**: `17` iterations
- **Calibration Subset In-Sample NLL**: `0.8939` $\\rightarrow$ `0.8871` ($\Delta = -0.0067$)
- **In-Sample Effect**: On the calibration set, $T = 0.7951 < 1.0$ indicates that the raw model outputs were slightly underconfident relative to calibration empirical label frequencies, sharpening predictions and lowering NLL.

---

## 4. Evaluation Performance Comparison (Out-of-Sample)

The fitted temperature $T = 0.7951$ was then frozen and applied to the **Evaluation Subset**. All metrics use identical evaluation indices ($N = 596$).

| Metric | Baseline (Uncalibrated, $T=1.0$) | Calibrated ($T=0.7951$) | Difference (Cal - Base) | Direction / Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Multiclass NLL** | `0.900622` | `0.904737` | `+0.004115` | Lower is better (Degraded out-of-sample) |
| **Brier Score** | `0.547153` | `0.549632` | `+0.002479` | Lower is better (Degraded out-of-sample) |
| **ECE (10 Bins)** | `0.049436` | `0.056700` | `+0.007264` | Lower is better (Degraded out-of-sample) |
| **Mean Confidence** | `0.5548` | `0.5981` | `+0.0433` | Mean maximum predicted probability |
| **Median Confidence** | `0.5343` | `0.5787` | `+0.0443` | Median maximum predicted probability |
| **Accuracy** | `0.5889` | `0.5889` | `+0.0000` | **Invariant** (Order-preserving) |
| **Balanced Accuracy** | `0.5729` | `0.5729` | `+0.0000` | **Invariant** (Order-preserving) |
| **Macro F1** | `0.5832` | `0.5832` | `+0.0000` | **Invariant** (Order-preserving) |

### Per-Class Performance Breakdown (Invariant)
| Class | Support | Precision | Recall | F1 Score |
| :---: | :---: | :---: | :---: | :---: |
| **DOWN** | `238` | `0.5875` | `0.5924` | `0.5900` |
| **FLAT** | `120` | `0.6484` | `0.4917` | `0.5592` |
| **UP** | `238` | `0.5698` | `0.6345` | `0.6004` |

### Confusion Matrix
```
[[141, 15, 82], [29, 59, 32], [70, 17, 151]]
```
*(Rows = Actual [DOWN, FLAT, UP], Columns = Predicted [DOWN, FLAT, UP])*

---

## 5. Reliability Diagram Binning Analysis

Binning convention: 10 uniform intervals across $[0.0, 1.0]$.

| Bin Range | Base Count (Frac) | Base Conf | Base Acc | Base Abs Err | Calib Count (Frac) | Calib Conf | Calib Acc | Calib Abs Err |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `[0.0, 0.1)` | 0 (0.0%) | 0.000 | 0.000 | 0.000 | 0 (0.0%) | 0.000 | 0.000 | 0.000 |
| `[0.1, 0.2)` | 0 (0.0%) | 0.000 | 0.000 | 0.000 | 0 (0.0%) | 0.000 | 0.000 | 0.000 |
| `[0.2, 0.3)` | 0 (0.0%) | 0.000 | 0.000 | 0.000 | 0 (0.0%) | 0.000 | 0.000 | 0.000 |
| `[0.3, 0.4)` | 8 (1.3%) | 0.377 | 0.750 | 0.373 | 6 (1.0%) | 0.380 | 0.667 | 0.286 |
| `[0.4, 0.5)` | 177 (29.7%) | 0.466 | 0.554 | 0.088 | 105 (17.6%) | 0.470 | 0.552 | 0.082 |
| `[0.5, 0.6)` | 252 (42.3%) | 0.545 | 0.563 | 0.019 | 219 (36.7%) | 0.546 | 0.562 | 0.015 |
| `[0.6, 0.7)` | 112 (18.8%) | 0.639 | 0.598 | 0.041 | 163 (27.3%) | 0.644 | 0.552 | 0.092 |
| `[0.7, 0.8)` | 35 (5.9%) | 0.755 | 0.800 | 0.045 | 64 (10.7%) | 0.734 | 0.719 | 0.015 |
| `[0.8, 0.9)` | 12 (2.0%) | 0.828 | 0.833 | 0.005 | 34 (5.7%) | 0.842 | 0.735 | 0.107 |
| `[0.9, 1.0)` | 0 (0.0%) | 0.000 | 0.000 | 0.000 | 5 (0.8%) | 0.906 | 1.000 | 0.094 |

---

## 6. Analytical Findings & Statistical Uncertainty

1. **In-Sample vs Out-of-Sample Calibration Divergence**:
   - On the chronological calibration partition, temperature scaling found $T = 0.7951$, lowering NLL from `0.8939` to `0.8871`.
   - However, when applied out-of-sample to the subsequent evaluation partition, NLL degraded from `0.9006` to `0.9047` ($\Delta = +0.0041$), ECE increased from `0.0494` to `0.0567`, and Brier score worsened from `0.5472` to `0.5496`.
   - This occurs because non-stationary market regimes caused the evaluation partition to have different class-conditional logit scales than the calibration partition. Calibrating on earlier time-series points overconfidently sharpened logits on later points.

2. **Argmax Invariance**:
   - Because scalar temperature scaling is a strictly monotonic transformation for any positive scalar $T > 0$:
     $$\\arg\\max_k (z_k / T) = \\arg\\max_k z_k$$
   - Consequently, classification decisions, accuracy, balanced accuracy, and F1 scores remain mathematically identical before and after calibration. Calibration adjusts probabilistic confidence, not classification boundaries.

3. **Statistical Uncertainty & Risk**:
   - Any observed shift in calibration quality on this validation partition is within ordinary market drift noise.
   - Because $T < 1.0$ increases model confidence (from `0.5548` to `0.5981`), adopting this temperature scaling parameter would expose downstream sizing modules to overconfidence risk during regime shifts.

---

## 7. Verification Invariants & Provenance Manifest

- **Model Weights Mutation Check**: `PASSED` (SHA-256 fingerprint verified identical before and after inference).
- **Locked Test Set Access**: `VERIFIED UNTOUCHED` (`test_scaled.npz` was never loaded, evaluated, or tuned against).
- **Baseline Alignment**: `PASSED` (Uncalibrated baseline probabilities match Phase 20 `validation_predictions.npz` to within float32 tolerance `1.192e-7`).
- **Zero Leakage**: `PASSED` (Evaluation data was strictly withheld from optimization).
- **Chronological Purge**: `PASSED` (Disjoint interval with verified 81000 ms gap $\ge$ 22000 ms).

---
*Report generated automatically by `pipeline_v2/models/phase23c_temperature_scaling.py`.*
