# PredAlpha-HFT — Phase 1: Data Pipeline Foundation

## 1. Executive Summary & Repository Inspection

This document records the audit, analysis, canonical dataset decision, data quality validation, and reproducible preparation pipeline completed for **Phase 1: Data Pipeline Foundation** inside the **PredAlpha-HFT** repository.

### Critical Constraints Maintained
- Existing project architecture and directory conventions preserved.
- No external APIs invoked; no live collection started.
- Zero modifications, renames, or deletions of existing raw or processed datasets.
- All baseline raw and processed datasets verified bit-for-bit unchanged via SHA-256 hashes.
- All newly generated canonical and split artifacts isolated in `data/processed/phase1/`.
- No model training or validation claimed in this phase.

---

## 2. Comprehensive Dataset Inventory

An automated audit scanned all 114 data files (Parquet, CSV, JSONL) across the repository. The data landscape splits into seven distinct historical and functional families:

### Family Overview

| Dataset Category | Key File Paths | Rows | Price Behavior | Labels / Target | Assessment |
|:---|:---|:---:|:---:|:---:|:---|
| **BTC 88000 Session 00 (Reference BBO)** | `data/processed/btc_88000_bbo_1s.parquet`<br>`data/processed/btc_88000_labeled_5s.parquet`<br>`data/processed/btc_88000_features_5s.parquet` | 299<br>294<br>289 | Dynamic ($0.064 - $0.0915)<br>29 unique levels | 5s horizon; FLAT (84.4%), UP (11.4%), DOWN (4.2%) | **Selected for Phase 1 Canonical Prep** (after purging target leakage). Valid for experimentation only. |
| **BTC 88000 Session 01** | `data/processed/btc_88000_session_20261002_01_bbo_1s.parquet`<br>`..._features_5s.parquet` | 492<br>482 | Dynamic ($0.069 - $0.0865)<br>39 unique levels | FLAT (96.3%), DOWN (2.1%), UP (1.7%) | **Rejected**: Severe class imbalance; validation split contains 100% FLAT (single class); target leakage present. |
| **Older Experiment (Sept 27, 2026)** | `data/processed/market_data.parquet`<br>`data/processed/labeled_market_data.parquet`<br>`data/processed/train.parquet`<br>`data/processed/val.parquet`<br>`data/processed/test.parquet` | 273<br>223<br>156<br>33<br>34 | **100% CONSTANT**<br>mid_price = $0.03650<br>spread = $0.00100 | Microprice return (3e-6 threshold); labeled set has only class 1 | **Rejected**: Fatal defects (zero price movement, single class, target leakage in `train/val/test`, token ID float precision corruption). |
| **Extended Replay & Test Snaps** | `data/processed/market_data_extended.parquet`<br>`data/processed/market_data_test.parquet`<br>`data/processed/market_data_multi_10m.parquet` | 498<br>1834<br>540 | **100% CONSTANT** ($0.02550 & $0.03650) | Unlabeled snapshots | **Rejected**: Unlabeled, frozen quotes, 0 price variance. |
| **Polymarket Oct 3 Up/Down** | `data/processed/btc_oct3_up_down_session_20261002_04_bbo.parquet`<br>`..._combined.parquet`<br>`..._final_features.parquet` | 190<br>190<br>95 | Dynamic ($0.085 - $0.915)<br>38 unique levels | Unlabeled on disk; 94-row external candle research set has 7 transitions | **Rejected**: Paired complementary token book events without on-disk resolution labels; irregular horizon when aligned to candles. |
| **Track A Raw Pilot** | `data/raw/btc_observations.jsonl`<br>`data/raw/track_a/btc_observations.jsonl` | 12 | Binance 1m candles ($85,022.00 - $85,022.01) | Unlabeled | **Rejected**: Only 12 rows (1 hour) of sparse pilot data. |
| **Track B Raw Settlement** | `data/raw/track_b/` (16 condition ID subdirectories) | 16 mkts | N/A (Settlement metadata) | Settlement outcome (Up: 5, Down: 11) | **Rejected**: Official settlement records only; no order-book or tick event streams available. |
| **Reference Candles** | `data/reference/btcusdt_1m_oct2_oct3.csv`<br>`data/reference/btcusdt_1m_oct2_oct3_resolution.csv` | 453<br>1441 | External Binance 1m candles ($83,000 - $89,000) | Oct 3 single market resolution (DOWN) | Reference benchmark only. |

---

## 3. Data Source & Schema Analysis

### Cross-Dataset Compatibility Analysis
1. **Timestamp Incompatibility**:
   - `btc_88000` datasets use regular 1-second UTC intervals with explicit DatetimeIndexes.
   - Older `market_data.parquet` and `market_data_extended.parquet` use irregular microsecond event timestamps.
   - Binance reference candles use 1-minute open/close timestamps.
   - Combining datasets across these intervals introduces severe temporal misalignment and artificial autocorrelation.
2. **Asset ID Discrepancies**:
   - `btc_88000` represents a single Binary Prediction contract ("Bitcoin to hit $88,000").
   - `btc_oct3_up_down` contains complementary UP/DOWN token pairs (`326647...` and `833681...`).
   - `market_data.parquet` stored large integer asset IDs coerced into float scientific notation (`3.233822e+76`), permanently corrupting token identification.
3. **Session Independence**:
   - `btc_88000_bbo_1s.parquet` (Session 00: 12:44:07 - 12:49:05 UTC) and `btc_88000_session_20261002_01_bbo_1s.parquet` (Session 01: 13:18:21 - 13:26:32 UTC) are separated by a 29-minute gap. Merging them into a single continuous time series creates an artificial 1,760-second jump that severely distorts rolling returns and volatility indicators.
4. **Resolution Limitations**:
   - Track A contains only 12 pilot rows.
   - Track B contains 16 verified settlement outcomes, but completely lacks order-book updates, depth snapshots, or trade execution streams.

---

## 4. Canonical Dataset Decision

### Selection
- **Selected Dataset Source**: `data/processed/btc_88000_bbo_1s.parquet`
- **Output Canonical Dataset**: `data/processed/phase1/canonical_dataset.parquet`
- **Suitability Classification**: **SUITABLE FOR EXPERIMENTATION & TESTING ONLY**
- **Production Status**: **NOT SUITABLE FOR PRODUCTION MODEL TRAINING**

### Rationale
1. **Empirical Price Variability**: Displays 29 unique mid-price levels spanning $0.06400 to $0.09150 over 299 seconds (unlike older datasets which had 0 price changes).
2. **Valid Orderbook Integrity**: 100% of rows have strictly positive bids and asks ($0.059 - $0.096), strictly positive spreads ($0.001 - $0.013), and 0 crossed books (`bid > ask` = 0).
3. **Multi-Class Directional Balance**: Under a 5-second prediction horizon with a dynamic half-spread threshold clipped at 0.001, the 289 valid feature rows contain all three classes:
   - `FLAT`: 244 (84.4%)
   - `UP`: 33 (11.4%)
   - `DOWN`: 12 (4.2%)
4. **Target Leakage Elimination**: In previous implementations (`market_data/build_features.py`), intermediate labeling columns (`future_mid`, `future_delta`, `label_threshold`) were inadvertently preserved in the feature file. The Phase 1 pipeline completely purges all future columns.

### Detailed Rejection of Alternatives
- **Older Experiment (`market_data.parquet`, root `train/val/test`)**: Permanently rejected. Mid-price is completely frozen at $0.03650. Labels were manufactured from minute microprice fluctuations (±0.00001), directly violating `training/data_quality.py`. In addition, future return columns were saved in feature frames.
- **Extended Snapshots (`market_data_extended.parquet`, `market_data_test.parquet`)**: Permanently rejected. Zero price movement (frozen at $0.02550 and $0.03650); no target labels.
- **BTC 88000 Session 01 (`btc_88000_session_20261002_01_features_5s.parquet`)**: Rejected. Severe class imbalance (96.3% FLAT). When split chronologically, validation contains 100% FLAT (67 rows, 0 UP, 0 DOWN), rendering validation completely uninformative.
- **Oct 3 Up/Down Sessions**: Rejected. Book events are split across token pairs without resolution labels. External candle alignment yielded only 94 rows and 7 label transitions.
- **Track A & Track B Raw**: Rejected. Insufficient observation volume (Track A: 12 rows; Track B: 0 event files).

---

## 5. Canonical Schema Definition

The canonical dataset is saved to `data/processed/phase1/canonical_dataset.parquet` conforming to the following explicit schema:

```
Index: DatetimeIndex (freq=1s, tz=UTC, name='timestamp')
Columns (14):
  1. timestamp          : datetime64[ns, UTC] — Explicit UTC timestamp
  2. asset_id           : string               — Unique asset identifier ("btc_88000")
  3. bid                : float64              — Best bid quote
  4. ask                : float64              — Best ask quote
  5. mid_price          : float64              — (bid + ask) / 2
  6. spread             : float64              — ask - bid
  7. spread_bps         : float64              — (spread / mid_price) * 10,000
  8. mid_return_1s      : float64              — 1-second causal mid-price return
  9. mid_return_3s      : float64              — 3-second causal mid-price return
 10. mid_return_5s      : float64              — 5-second causal mid-price return
 11. mid_volatility_5s  : float64              — 5-second rolling standard deviation of returns
 12. bid_change_1s      : float64              — 1-second first difference of best bid
 13. ask_change_1s      : float64              — 1-second first difference of best ask
 14. label              : string (Categorical) — Directional target ('FLAT', 'UP', 'DOWN')
```

### Mathematical Definitions
- **Spread in Basis Points**: $\text{spread\_bps} = \frac{\text{ask} - \text{bid}}{\text{mid\_price}} \times 10{,}000$
- **Causal Rolling Volatility**: $\text{mid\_volatility\_5s}_t = \text{std}(\text{mid\_return\_1s}_{t-4 \dots t})$
- **Directional Target**:
  $$\text{threshold}_t = \max\left(\frac{\text{spread}_t}{2}, 0.001\right)$$
  $$\Delta \text{mid}_t = \text{mid}_{t+5} - \text{mid}_t$$
  $$\text{label}_t = \begin{cases} \text{UP} & \text{if } \Delta \text{mid}_t > \text{threshold}_t \\ \text{DOWN} & \text{if } \Delta \text{mid}_t < -\text{threshold}_t \\ \text{FLAT} & \text{otherwise} \end{cases}$$

---

## 6. Data Quality & Integrity Findings

The automated validation suite (`market_data/validation.py`) ran 8 test suites against the canonical dataset:

| Check Name | Status | Observed Findings | Severity |
|:---|:---:|:---|:---:|
| `missing_values` | ✅ PASS | 0 missing values across all 14 canonical columns | ERROR |
| `duplicate_records` | ✅ PASS | 0 duplicate records across timestamps and feature rows | ERROR |
| `timestamp_integrity` | ✅ PASS | Strictly ascending, continuous 1-second UTC timestamps (12:44:12 to 12:49:00 UTC) | ERROR |
| `market_data_integrity` | ✅ PASS | All quotes positive ($0.062 - $0.096); all spreads positive ($0.001 - $0.013); 0 crossed books | ERROR |
| `finite_values` | ✅ PASS | 0 infinite (`inf` / `-inf`) values in numeric features | ERROR |
| `constant_features` | ✅ PASS | All 9 features exhibit non-zero variance; no constant columns | ERROR |
| `label_distribution` | ✅ PASS | 3 active classes (FLAT: 244, UP: 33, DOWN: 12) | INFO |
| `target_leakage` | ✅ PASS | 0 forbidden columns or future return patterns present in feature matrix | ERROR |

---

## 7. Target Leakage & Causal Integrity Audit

### Historical Leakage Identified
During inspection of previous artifacts (`btc_88000_features_5s.parquet` and `splits/train.parquet`), the following future-derived columns were found directly in the saved feature files:
- `future_mid`: The exact mid-price 5 seconds into the future.
- `future_delta`: The exact future price change.
- `label_threshold`: The target boundary calculation.
- `future_return`, `future_return_50`: Present in older `market_data` splits.

### Phase 1 Remediation
1. `CanonicalSchemaConfig` defines `forbidden_cols = frozenset({'future_mid', 'future_delta', 'future_return', 'label_threshold', 'target_close', ...})`.
2. Intermediate future columns used during label calculation are strictly purged prior to dataset serialization.
3. Automated unit tests (`test_leakage_prevention_catches_future_columns`) enforce rejection if any future-looking column enters the feature pipeline.

---

## 8. Chronological Split Methodology

To prevent temporal leakage in time-series prediction, deterministic chronological splitting is enforced with explicit boundary purge gaps:

- **Split Ratio**: 70% Train, 15% Validation, 15% Test
- **Purge Gap**: 5 seconds (5 rows at 1 Hz), exactly equal to the 5-second prediction horizon. This ensures that the future mid-price evaluated at the end of the training set ($\text{mid}_{t+5}$) does not overlap with the earliest prediction timestamp in the validation set.

### Verified Split Statistics

```
========================================================================================
Split        Rows   Start Time (UTC)           End Time (UTC)             Class Distribution
========================================================================================
Train         202   2026-10-02 12:44:12+00:00  2026-10-02 12:47:33+00:00  FLAT: 164, UP: 28, DOWN: 10
[Purge Gap]     5   (5.0 seconds between Train end and Validation start)
Validation     38   2026-10-02 12:47:39+00:00  2026-10-02 12:48:16+00:00  FLAT: 36, UP: 2, DOWN: 0
[Purge Gap]     5   (5.0 seconds between Validation end and Test start)
Test           39   2026-10-02 12:48:22+00:00  2026-10-02 12:49:00+00:00  FLAT: 37, UP: 0, DOWN: 2
========================================================================================
```

**Non-Overlap Proof**:
- $\max(\text{Train}) < \min(\text{Val})$: `12:47:33 < 12:47:39` (Gap: 6.0s $\ge$ 5.0s purge)
- $\max(\text{Val}) < \min(\text{Test})$: `12:48:16 < 12:48:22` (Gap: 6.0s $\ge$ 5.0s purge)

---

## 9. Reproduction Commands

To reproduce the complete Phase 1 audit, preparation pipeline, and tests:

```powershell
# 1. Execute the Phase 1 Canonical Dataset Pipeline
python -m market_data.canonical_pipeline

# 2. Run the Phase 1 Test Suite
pytest tests/test_phase1_data_pipeline.py -v

# 3. Run the Entire Repository Test Suite (39 tests)
pytest
```

All generated artifacts are written to `data/processed/phase1/`:
- `data/processed/phase1/canonical_dataset.parquet`
- `data/processed/phase1/train.parquet`
- `data/processed/phase1/validation.parquet`
- `data/processed/phase1/test.parquet`
- `data/processed/phase1/dataset_metadata.json`
- `data/processed/phase1/validation_report.json`
- `data/processed/phase1/validation_report.md`
- `data/processed/phase1/dataset_selection_decision.md`

---

## 10. Test Execution Results

All 39 repository tests passed without regression:

```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\karma\OneDrive\Desktop\predalpha-clean
configfile: pytest.ini
testpaths: tests
collected 39 items

tests\test_collection_pipeline.py ......                                 [ 15%]
tests\test_feature_engineering.py .                                      [ 17%]
tests\test_ml_pipeline.py ..                                             [ 23%]
tests\test_optimized_orderbook.py ...                                    [ 30%]
tests\test_orderbook.py ...                                              [ 38%]
tests\test_phase1_data_pipeline.py ...............                       [ 76%]
tests\test_track_ab.py ....                                              [ 87%]
tests\test_training_pipeline.py ...                                      [ 94%]
tests\test_walk_forward_phase6.py ..                                     [100%]

============================= 39 passed in 11.54s =============================
```

### Coverage of Phase 1 Tests (`tests/test_phase1_data_pipeline.py`)
1. `test_schema_validation_passes_valid_canonical_data`
2. `test_schema_validation_rejects_missing_required_columns`
3. `test_timestamp_parsing_and_ordering`
4. `test_timestamp_continuity_detects_unexpected_gaps`
5. `test_duplicate_detection`
6. `test_missing_values_detection`
7. `test_invalid_market_data_detection` (crossed book, non-positive price, negative spread)
8. `test_finite_values_validation` (`inf`, `-inf`)
9. `test_constant_features_detection`
10. `test_single_class_label_detection`
11. `test_chronological_splitting_and_purge_gap`
12. `test_chronological_splits_rejects_overlap`
13. `test_leakage_prevention_catches_future_columns`
14. `test_reproducibility`
15. `test_preservation_of_existing_files` (SHA-256 hash preservation)

---

## 11. Known Limitations & Blockers

1. **Ultra-Short Session Sample (289 Rows)**:
   - The selected canonical dataset spans only 4 minutes and 48 seconds of a single trading session.
   - It is sufficient to validate pipeline mechanics, feature code, and baseline architectures, but **it cannot produce statistically significant predictive models**.
2. **Split Class Imbalance**:
   - Because the session is short, directional movements cluster in time: the validation split has 0 DOWN labels (36 FLAT, 2 UP), and the test split has 0 UP labels (37 FLAT, 2 DOWN).
   - Evaluating multi-class classification metrics (Macro F1, Precision, Recall) on these test splits is heavily biased.
3. **High Serial Autocorrelation**:
   - High-frequency 1-second BBO quotes exhibit strong autocorrelation. Without hundreds of independent sessions, standard error estimates are understated.
4. **Track A & Track B Blockers**:
   - Track A currently has only 12 pilot rows.
   - Track B has settlement records for 16 markets, but zero streaming order-book or trade event captures. Production multi-market modeling remains blocked until full event logs are recorded.

---

## 12. Exact Next Steps for Phase 2

1. **Feature Expansion & Scaling**:
   - Build causal order-book depth imbalance features if depth is available.
   - Implement fold-safe feature scalers fitted strictly on training splits without test contamination.
2. **Multi-Horizon Labeling**:
   - Extend the labeling framework to support multiple prediction horizons ($H \in \{3\text{s}, 5\text{s}, 10\text{s}\}$).
3. **Walk-Forward Validation**:
   - Implement expanding-window walk-forward splits once longer recording sessions are collected.
4. **Baseline Benchmark Model**:
   - Fit majority class and regularized logistic regression baselines on `data/processed/phase1/train.parquet` to establish a performance floor before introducing deep learning models.
