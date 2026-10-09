# PredAlpha-HFT

PredAlpha-HFT is a research-grade pipeline for studying short-horizon directional prediction from high-frequency prediction-market order-book data.

The project emphasizes causal data construction, physical-time labeling, strict temporal isolation, leakage prevention, train-only scaling, and locked out-of-sample evaluation.

> **Status: Final model frozen and locked test evaluation complete.**
>
> The project demonstrates directional predictive signal, but the final model has **not** been demonstrated to be a profitable or production-ready trading strategy.

---

## Final Result

| Metric | Result |
|---|---:|
| Model | SmallLSTM |
| Sequence length | 10 seconds |
| Features | 11 |
| Hidden size | 32 |
| Layers | 1 |
| Dropout | 0.10 |
| Parameters | 5,859 |
| Seed | 999 |
| Test samples | 1,582 |
| Accuracy | **52.72%** |
| Balanced Accuracy | **42.49%** |
| Macro F1 | **0.4315** |
| Weighted F1 | **0.5140** |

### Per-class test performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| DOWN | 53.96% | 56.03% | 0.5498 | 705 |
| FLAT | 38.89% | 12.50% | 0.1892 | 168 |
| UP | 52.51% | 58.96% | 0.5555 | 709 |

Confusion matrix:

```text
              Predicted
             DOWN FLAT UP
Actual DOWN    395  14 296
       FLAT     65  21  82
       UP      272  19 418