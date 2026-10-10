# Phase 22 — Temporal Context Validation Report

> [!IMPORTANT]
> **VALIDATION-ONLY EVALUATION**: Test partition was NEVER loaded or evaluated.
> **STATUS**: Phase 22 Validation Experiments Completed.

## 1. Summary Comparison: L10 Baseline vs. Longer Lookback Contexts

| Lookback Window | Model | Val Accuracy (Mean ± Std) | Val Balanced Accuracy (Mean ± Std) | Val Macro F1 (Mean ± Std) | Δ Macro F1 vs L10 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **L10 Baseline** | SmallLSTM | 59.10% ± 0.06% | 53.85% ± 0.31% | 0.5543 ± 0.0043 | baseline |
| **L20** | SmallLSTM | 57.91% ± 0.13% | 52.77% ± 0.43% | 0.5411 ± 0.0025 | -0.0132 |
| **L30** | SmallLSTM | 58.12% ± 0.83% | 53.88% ± 0.90% | 0.5495 ± 0.0098 | -0.0048 |
| **L60** | SmallLSTM | 55.41% ± 1.27% | 48.11% ± 2.15% | 0.4865 ± 0.0230 | -0.0678 |

---

## 2. Seed-by-Seed Performance Breakdown

| Context | Seed | Val Accuracy | Val Balanced Accuracy | Val Macro F1 |
| :---: | :---: | :---: | :---: | :---: |
| L20 | `42` | 57.97% | 52.90% | 0.5416 |
| L20 | `123` | 58.04% | 52.20% | 0.5378 |
| L20 | `999` | 57.73% | 53.22% | 0.5439 |
| L30 | `42` | 56.95% | 52.65% | 0.5358 |
| L30 | `123` | 58.57% | 54.19% | 0.5546 |
| L30 | `999` | 58.84% | 54.80% | 0.5581 |
| L60 | `42` | 55.38% | 48.57% | 0.4923 |
| L60 | `123` | 53.87% | 45.27% | 0.4558 |
| L60 | `999` | 56.99% | 50.48% | 0.5113 |

---

## 3. Analysis & Key Findings

- **Best Mean Validation Accuracy**: **L30** (58.12%)
- **Best Mean Balanced Accuracy**: **L30** (53.88%)
- **Best Mean Macro F1**: **L30** (0.5495)

### Temporal Context Verdict:
- Longer temporal context does NOT improve over the L10 baseline (0.5495 vs 0.5543). L10 remains the superior or more parsimonious lookback window.

---

## 4. Verification Check
- **Test partition touched**: **NO** (test_scaled.npz was completely locked and unread)
- **Phase 22 Status**: **PASS**