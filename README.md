# PredAlpha-HFT: High-Frequency Directional Forecasting for Prediction Markets

PredAlpha-HFT is a quantitative research codebase engineered for high-frequency directional prediction on decentralized binary prediction markets (Polymarket order books). The framework enforces strict point-in-time mathematical causality, physical-time forward labeling, train-only feature normalization, and an isolated locked out-of-sample evaluation protocol.

> **Important Quantitative Disclaimer:**
> Validation metrics demonstrate statistical directional signal above random guessing, but **they do not constitute proof of trading profitability**. In backtested market-making simulations (Phase 27 calibration), maker adverse selection resulted in negative net PnL across a 5,000 ms holding horizon. Live order routing, WebSockets, capital deployment, and prospective Session 001 execution remain **strictly unauthorized**.

---

## 1. System Architecture & Methodology

```
Raw L1 Events (Order Book Snapshots)
        │
        ▼
Point-in-Time 1-Second Grid Resampling (Zero Lookahead)
        │
        ▼
Causal Feature Construction (11 Microstructure Features)
        │
        ▼
Physical-Time 5-Second Labeling (Δt = 5,000 ms, Stale-Tick Guard)
        │
        ▼
Chronological Purged Train / Validation Splitting
        │
        ▼
Train-Only Standardization (CausalSequenceScaler, Zero Validation Leakage)
        │
        ▼
Recurrent Deep Learning (SmallLSTM, L=10 Steps, 32 Hidden Units)
        │
        ▼
Hypothetical Passive Maker Execution Simulation (Phase 27 Protocol)
```

### Key Engineering Invariants
- **Zero Forward Leakage:** Mathematical causality is strictly enforced; all features at grid timestamp $T$ depend exclusively on historical data $\le T$. Negative shifts (`shift(-k)`), backward filling (`bfill()`), and centered rolling operators are barred.
- **Physical-Time Target Definition:** Targets are evaluated across an exact physical 5,000 ms forward window (not tick count), classifying market displacement into three mutually exclusive regimes: `DOWN` (0), `FLAT` (1), and `UP` (2).
- **Leakage-Free Feature Scaling:** `CausalSequenceScaler` computes mean and standard deviation strictly from training sequence partitions. Validation and test partitions are transformed strictly using learned training parameters.
- **Locked Test Isolation:** The final test set (`test_scaled.npz`, 182,401 bytes) remains locked and isolated to prevent test-set tuning and evaluation leakage.

---

## 2. Core Model Architecture & Baseline Specification

The primary modeling engine is a compact, regularization-focused Recurrent Neural Network:

- **Model Class:** `SmallLSTM`
- **Sequence Length ($L$):** 10 steps (10-second causal rolling history on a 1 Hz grid)
- **Feature Dimension ($D$):** 11 microstructure features
- **Hidden Units:** 32 (1 recurrent layer)
- **Output Classes:** 3 (`DOWN`, `FLAT`, `UP`)
- **Total Parameters:** 5,859
- **Regularization & Optimization:** Dropout $p = 0.10$, AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$), batch size 256, validation early stopping (patience = 25).

### Baseline 11 Feature Whitelist
1. `mid_price`: Top-of-book midpoint $(b + a) / 2$
2. `spread`: Absolute bid-ask spread $(a - b)$
3. `spread_bps`: Relative spread in basis points $(spread / mid) \times 10,000$
4. `mid_return_1s`: 1-second midpoint price return
5. `mid_return_3s`: 3-second midpoint price return
6. `mid_return_5s`: 5-second midpoint price return
7. `mid_volatility_5s`: 5-second rolling standard deviation of 1-second returns
8. `bid_change_1s`: 1-second change in best bid price
9. `ask_change_1s`: 1-second change in best ask price
10. `microprice`: Queue-weighted volume price $(b \cdot q_a + a \cdot q_b) / (q_b + q_a)$
11. `depth_imbalance`: Relative top-of-book depth imbalance $(q_b - q_a) / (q_b + q_a)$

---

## 3. Controlled Empirical Experiments

Three controlled, multi-seed experiments were conducted on the chronological validation split ($N = 1,428$ sequences, Train $N = 7,514$ sequences) across identical random seeds (42, 123, 999) and identical training budgets.

### Multi-Seed Experimental Results Summary

| Experiment / Configuration | Features ($D$) | Accuracy (Mean) | Macro-F1 (Mean) | Directional Inversions ($0 \leftrightarrow 2$) | Primary Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Reference SmallLSTM)** | **11** | **59.10%** | **0.5543** | **408.7** | Stable, calibrated baseline representation |
| **Experiment 1: Class Weighting** | 11 | 58.40% | 0.5512 | 409.0 | Minority class weighting degraded overall accuracy |
| **Experiment 1: Cost-Sensitive Penalty** | 11 | **59.22%** | **0.5533** | **399.0** | **Highest accuracy & cuts severe inversions by -9.7** |
| **Experiment 2: Microprice Collinearity Fix** | 11 | 58.52% | 0.5485 | 412.7 | Decoupled collinearity (575×), but increased seed variance |
| **Experiment 3: Baseline + Quote OFI** | 12 | 58.24% | 0.5478 | 422.3 | High-frequency quote noise degraded decision boundaries |
| **Experiment 3: OFI (No Depth Imbalance)** | 11 | 56.32% | 0.5387 | 452.0 | Static depth imbalance accounts for >97% of queue signal |

### Analysis of Experimental Findings

1. **Cost-Sensitive Loss Dynamics (Experiment 1):**
   - The cost-sensitive formulation introduces an explicit penalty ($\lambda = 0.50$) for severe directional inversions (predicting `UP` when the outcome is `DOWN`, or vice versa).
   - While this arm achieved the **highest mean validation accuracy (59.22%)** and successfully reduced severe directional inversions from $408.7$ down to $399.0$, it produced a **slightly lower mean Macro-F1 (0.5533 vs. 0.5543)** due to trade-offs in the unweighted `FLAT` minority cohort.
   - **Conclusion:** It represents an objective-specific trade-off (prioritizing tail-risk mitigation) rather than an unambiguous overall model improvement.

2. **Microprice Collinearity (Experiment 2):**
   - In level space, `microprice` and `mid_price` exhibit extreme empirical collinearity ($r = 0.999918$, covariance condition number $= 361,009$).
   - Replacing `microprice` with $\Delta_{\text{micro}} = \text{microprice} - \text{mid\_price} = \frac{1}{2} \cdot \text{spread} \cdot \text{depth\_imbalance}$ dropped the condition number to $627.44$ (575× reduction).
   - However, removing level redundancy increased model sensitivity to random weight initialization, reducing aggregate 3-seed accuracy to $58.52\%$.

3. **Quote-Based Order Flow Imbalance (Experiment 3):**
   - Level 1 quote-based OFI (Cont, Kukanov & Stoikov 2014) was computed causally and depth-normalized.
   - Ablation demonstrated that adding OFI degraded accuracy ($58.24\%$) and increased directional inversions ($422.3$). Substituting OFI for static depth imbalance caused accuracy to drop precipitously to $56.32\%$.
   - **Conclusion:** In 1 Hz resampled prediction-market books, static queue depth imbalance provides a significantly cleaner predictive signal than discrete quote-delta flow.

---

## 4. Execution Modeling & Trading Profitability Limitations

Phase 27 defines the protocol for simulated maker execution (`2.0_PHASE27_PREDECLARED_FROZEN`):
- **Entry:** Passive maker limit order placed at touch.
- **Exit:** Aggressive market taker exit at an exact 5,000 ms horizon.
- **Fill Certainty Accounting:** Conservative (`CONFIRMED_PRICE_THROUGH`) and optimistic (`AMBIGUOUS_QUOTE_RETREAT`) cohorts are tracked separately.

### Historical Calibration Evidence
In historical backtest calibration (`data/models/phase27/phase27_calibration_report.json`), the strategy **failed declared profitability hurdles**:
- **ITT Net PnL:** $-0.00085$ units per signal (vs. declared acceptance hurdle of $\ge +0.0025$).
- **Filled Trade Net PnL:** $-0.00102$ units per trade (vs. declared acceptance hurdle of $\ge +0.0030$).
- **Mechanism:** Fills were disproportionately adverse-selected (orders filled when informed traders crossed the spread against the resting limit).

**Status:** Software readiness and protocol verification are complete; however, trading profitability is unproven. Live trading remains unauthorized.

---

## 5. Repository Structure & Artifacts

```
predalpha/
├── pipeline_v2/              # Modular causal architecture
│   ├── ingestion/            # Raw JSON event normalizer
│   ├── resampling/           # Point-in-time 1s grid resampler
│   ├── labeling/             # Physical 5s forward labeling
│   ├── features/             # Causal microstructure feature builders
│   ├── splitting/            # Temporal purged train/val/test splitter
│   ├── sequences/            # 3D (N, 10, D) tensor window builder
│   ├── scaling/              # Train-only CausalSequenceScaler
│   └── models/               # PyTorch models, metrics, and execution simulator
├── experiments/              # Controlled experimental runners
│   ├── controlled_experiment_1.py  # Loss weighting & cost-sensitive loss
│   ├── controlled_experiment_2.py  # Microprice collinearity analysis
│   └── controlled_experiment_3.py  # Quote-based OFI evaluation
├── data/models/              # Checkpoints and persisted experiment results
│   ├── phase19/recurrent/    # Frozen model checkpoint (best_lstm.pt)
│   ├── experiment1_accuracy/ # Exp 1 JSON results and logs
│   ├── experiment2_features/ # Exp 2 JSON results and logs
│   └── experiment3_ofi/      # Exp 3 JSON results and logs
└── RELEASE_MANIFEST.md       # Predeclared protocol freeze manifest
```

---

## 6. Offline Reproduction

All experimental workflows are designed for offline reproduction using fixed random seeds (42, 123, 999) and train-only preprocessing without network access or live market feeds:

```bash
# Verify frozen checkpoint integrity
python -c "import hashlib; assert hashlib.sha256(open('data/models/phase19/recurrent/best_lstm.pt','rb').read()).hexdigest() == 'd8b75bdbaffbb7f2c4b398beea484f83d851b37380bccd93ca07199a3d9c025d'; print('Checkpoint verified.')"

# Reproduce Controlled Experiment 1 (Loss Weighting & Cost-Sensitive Loss)
python experiments/controlled_experiment_1.py

# Reproduce Controlled Experiment 2 (Feature Engineering & Collinearity)
python experiments/controlled_experiment_2.py

# Reproduce Controlled Experiment 3 (Quote-Based Order Flow Imbalance)
python experiments/controlled_experiment_3.py
```
