# Track A baseline results

Features: `mid_price_up`, `spread_up`, `spread_down`, `mid_price_change`. Future-derived columns are prohibited by `market_data.track_a.CONFIG` and validated before fitting.

Models were fitted once on chronological train only; validation and test were scored without tuning. Metrics are macro precision/recall/F1.

## majority_class

- validation: accuracy=0.071 | balanced_accuracy=0.500 | precision_macro=0.036 | recall_macro=0.500 | f1_macro=0.067; class distribution={'DOWN': 1, 'UP': 13}; labels=['DOWN', 'UP']; confusion matrix=[[1, 0], [13, 0]]
- test: accuracy=0.200 | balanced_accuracy=0.500 | precision_macro=0.100 | recall_macro=0.500 | f1_macro=0.167; class distribution={'DOWN': 3, 'UP': 12}; labels=['DOWN', 'UP']; confusion matrix=[[3, 0], [12, 0]]
## logistic_regression

- validation: accuracy=0.857 | balanced_accuracy=0.462 | precision_macro=0.462 | recall_macro=0.462 | f1_macro=0.462; class distribution={'DOWN': 1, 'UP': 13}; labels=['DOWN', 'UP']; confusion matrix=[[0, 1], [1, 12]]
- test: accuracy=0.533 | balanced_accuracy=0.708 | precision_macro=0.650 | recall_macro=0.708 | f1_macro=0.525; class distribution={'DOWN': 3, 'UP': 12}; labels=['DOWN', 'UP']; confusion matrix=[[3, 0], [7, 5]]

## Limitations

Only 94 observations and seven label transitions exist. Train has only 9 UP labels while validation/test are UP-heavy; test has 15 rows. Accuracy can be misleading, and neither result supports a predictive-performance claim. More independently resolved markets and a purged/walk-forward design are required before meaningful modeling.
