# PredAlpha-HFT Pipeline V2 Data Quality Gate Report

- **Pipeline Correctness (`pipeline_valid`)**: `**PASS (Valid)**`
- **Data Quality Standards (`data_quality_pass`)**: `**FAIL**`
- **Downstream Training Readiness (`training_ready`)**: `**NO (NOT Ready)**`

### Executive Summary

- **Total Sequences**: 342 (Train: 249, Val: 47, Test: 46)
- **Class Distribution**: `{'FLAT': np.int64(342)}`
- **Unique Mid Prices**: 1
- **Directional Transitions**: 0
- **Stale Observations Percentage**: 54.13%
- **Total Infs / NaNs**: 39
- **Constant Features Count**: 9 / 11

### Failed Quality & Integrity Criteria (9)

| Check Name | Category | Required | Observed | Diagnostic Details |
| :--- | :--- | :--- | :--- | :--- |
| `canonical_observations_count` | `data_quality` | `>= 50,000` | `498` | Minimum number of canonical market events required for institutional training. |
| `labeled_observations_count` | `data_quality` | `>= 10,000` | `375` | Minimum physical labeled observations required before sequence construction. |
| `sequence_counts_sufficiency` | `data_quality` | `total >= 10,000, train >= 7,000, val >= 1,000, test >= 1,000` | `total=342 (train=249, val=47, test=46)` | Partition sample counts must meet minimum thresholds for neural convergence. |
| `zero_nan_values` | `data_quality` | `0` | `39` | Scaled feature matrices must contain zero NaN values. |
| `required_classes_present` | `data_quality` | `['UP', 'DOWN', 'FLAT']` | `['FLAT']` | All required outcome classes (UP, DOWN, FLAT) must be represented in target labels. |
| `class_balance_minimum` | `data_quality` | `Each class >= 5.0%` | `{'UP': 0.0, 'DOWN': 0.0, 'FLAT': 1.0} (Violations: {'UP': '0.00% < 5.00%', 'DOWN': '0.00% < 5.00%'})` | Each target label class must comprise at least 5% of all samples. |
| `price_diversity` | `data_quality` | `>= 25 unique prices` | `1` | Underlying market observations must exhibit distinct price levels to avoid degenerate models. |
| `directional_transitions` | `data_quality` | `>= 100 transitions` | `0` | Dataset must exhibit frequent directional state switches to train non-trivial dynamics. |
| `staleness_threshold` | `data_quality` | `<= 20.0%` | `54.13%` | Proportion of stale resampled grid observations must remain within threshold. |

### Advisory Warnings (1)

| Warning Item | Observed Value | Rationale |
| :--- | :--- | :--- |
| `feature_variance_audit` | `9 constant features: [np.str_('mid_price'), np.str_('spread'), np.str_('spread_bps'), np.str_('mid_return_1s'), np.str_('mid_return_3s'), np.str_('mid_return_5s'), np.str_('mid_volatility_5s'), np.str_('bid_change_1s'), np.str_('ask_change_1s')]` | Features with zero variance reflect static market conditions and provide zero signal. |

