# PredAlpha Phase 1 — Canonical Dataset Selection Decision

## Decision Summary

- **Selected Canonical Dataset**: `data/processed/phase1/canonical_dataset.parquet` (prepared from `data/processed/btc_88000_bbo_1s.parquet`)
- **Suitability Classification**: **SUITABLE FOR EXPERIMENTATION & TESTING ONLY**
- **Production Status**: **NOT SUITABLE FOR PRODUCTION MODEL TRAINING**
- **Reason for Selection**: It is the only continuous, high-frequency (1s) market dataset in the repository that exhibits genuine mid-price variability (29 distinct price levels), valid positive spreads, no crossed books, all three directional target classes (FLAT: 84.4%, UP: 11.4%, DOWN: 4.2%), and clean causal feature generation once leakage columns are purged.

## Alternatives Evaluated & Reasons for Rejection

1. **Older Experiment (`market_data.parquet`, `labeled_market_data.parquet`, root `train.parquet`, `val.parquet`, `test.parquet`)**:
   - **Fatal Defect**: 100% CONSTANT mid-price ($0.03650), best bid ($0.036), best ask ($0.037), and spread ($0.001) across all rows.
   - **Invalid Synthetic Labels**: Labels were generated from tiny orderbook depth fluctuations moving microprice by $0.00001, violating the project's explicit rule in `training/data_quality.py` ("Mid-price is constant. Do not generate directional labels.").
   - **Target Leakage**: `future_return`, `future_return_50`, and `future_return_100` were saved directly alongside feature matrices.
   - **Type Corruption**: `asset_id` was cast to float64, losing integer token precision.
   - **Rejection**: **Fatal defects; rejected permanently.**

2. **`market_data_extended.parquet` & `market_data_test.parquet`**:
   - **Fatal Defect**: `market_data_extended.parquet` has 100% constant mid-price ($0.02550); `market_data_test.parquet` has 100% constant mid-price ($0.03650). Both lack target labels.
   - **Rejection**: **Unlabeled and zero price movement; rejected.**

3. **`btc_oct3_up_down_session_*` (Sessions 01-04)**:
   - **Defect**: Interleaved UP/DOWN token book events without on-disk settlement/outcome labels.
   - **Alignment Artifacts**: Aligning with Binance 1m candles produced only 94 valid rows with variable event-to-target horizons (1 to 5 minutes) and only 7 label transitions.
   - **Rejection**: **Unsuitable for order-book HFT training; rejected.**

4. **`btc_88000_session_20261002_01_features_5s.parquet` (Session 1)**:
   - **Defect**: Extreme class imbalance (96.3% FLAT, 2.1% DOWN, 1.7% UP).
   - **Single-Class Validation Set**: Chronological splitting produces a validation set with 100% FLAT labels (0 UP, 0 DOWN), making validation uninformative.
   - **Target Leakage**: Future columns (`future_mid`, `future_delta`, `label_threshold`) were saved into feature files.
   - **Rejection**: **Single-class validation set and extreme imbalance; rejected.**

5. **Track A Pilot (`data/raw/btc_observations.jsonl`)**:
   - **Defect**: Contains only 12 rows of 1-minute BTC candles (1 hour of sparse data).
   - **Rejection**: **Insufficient sample size for any statistical or ML pipeline.**

6. **Track B Settlement Evidence (`data/raw/track_b/`)**:
   - **Defect**: Contains 16 market settlement outcome records but zero event or order-book streaming files.
   - **Rejection**: **Settlement metadata only; no tick/BBO data available for training.**

## Known Limitations of Selected Canonical Dataset

1. **Ultra-Short Duration**: 289 rows of 1-second BBO observations spans only 4 minutes and 48 seconds of continuous trading.
2. **Split Class Coverage**: Chronological splitting with a 5-second purge gap creates a validation split lacking DOWN samples and a test split lacking UP samples due to regime clustering in this short session.
3. **Autocorrelation**: Consecutive 1-second snapshots exhibit high serial dependency.

## Canonical Schema Definition

- `timestamp`: UTC Datetime (`datetime64[ns, UTC]`)
- `asset_id`: String (`btc_88000`)
- `bid`: Float64
- `ask`: Float64
- `mid_price`: Float64
- `spread`: Float64
- `spread_bps`: Float64
- `mid_return_1s`: Float64
- `mid_return_3s`: Float64
- `mid_return_5s`: Float64
- `mid_volatility_5s`: Float64
- `bid_change_1s`: Float64
- `ask_change_1s`: Float64
- `label`: Categorical string (`FLAT`, `UP`, `DOWN`)
