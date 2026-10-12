# PredAlpha — Release Manifest

**Protocol Specification:** `2.0_PHASE27_PREDECLARED_FROZEN`  
**Milestone:** Phase 27 Final Protocol Reconciliation and Repository Freeze  
**Execution Architecture:** Passive Limit Maker Entry at Touch / Market Taker Exit at 5,000 ms Horizon  
**Primary Evaluation Cohort:** Intention-to-Treat (ITT) Master Ledger  
**Release Gate Verdict:** **SOFTWARE-RELEASE-READY**  
**Prospective Execution Status:** **Session 001 UNAUTHORIZED (Offline)**

---

## 1. Frozen Checkpoint & Protected Data Integrity

- **Frozen Model Architecture:** `SmallLSTM(input_dim=11, hidden_dim=32, num_layers=1, num_classes=3)`
- **Primary Checkpoint Path:** `data/models/phase19/recurrent/best_lstm.pt`
- **Backup Checkpoint Path:** `data/models/phase19/recurrent/checkpoints/best_lstm.pt`
- **Immutable SHA-256 Checksum:** `d8b75bdbaffbb7f2c4b398beea484f83d851b37380bccd93ca07199a3d9c025d` *(Verified Invariant)*
- **Locked Test Dataset:** `data/clean_v2/07_scaled/expanded_collection/test_scaled.npz` (182,401 bytes)
- **Data Isolation Guarantee:** Strict zero-access enforced. The locked test set has never been loaded, opened, or evaluated across any development or audit phase.

---

## 2. Predeclared Protocol Hurdles & Completion Criteria

### Acceptance Hurdles (Non-Fungible Conjunction)
- **Hurdle 1 (ITT Mean Net PnL):** $\ge +0.0025$ units per eligible signal (unfilled = $0.0$)
- **Hurdle 2 (Filled Mean Net PnL):** $\ge +0.0030$ units per executed trade
- **Hurdle 3 (Profit Factor Floor):** $\ge 1.25$ (nominal target $\ge 1.30$)
- **Hurdle 4 (Decisive Win Rate):** $\ge 50.0\%$ (Wins / [Wins + Losses])
- **Hurdle 5 (Overall Win Rate):** $\ge 48.0\%$ (accounting for scratches)
- **Hurdle 6 (Directional Asymmetry):** $\le 0.35$ index across aggregate cohort
- **Hurdle 7 (Cluster Significance):** Cluster-robust $t$-statistic $t_{\text{CR}} \ge 2.0$

### Mandatory Completion Conditions
- **C1 (Executed Trades Floor):** $N_{\text{filled}} \ge 400$ completed maker trades
- **C2 (ITT Signals Floor):** $N_{\text{ITT}} \ge 600$ eligible decisions
- **C3 (Burst Sampling Floor):** $C \ge 50$ session-scoped recording bursts
- **C4 (Fixed Evaluation Horizon):** Completed prospective sessions $\ge 5$
- **C5 (Zero Active Circuit Breakers):** `has_active_hard_stop == False`
- **Target (Cluster Degrees of Freedom):** Effective cluster count $N_{\text{eff}} \ge 40.0$ ($G^* / C \ge 0.80$)

### Hard-Stopping Circuit Breakers
- **Consecutive Losses Limit:** $\ge 8$ filled losses $\rightarrow$ `ABORTED_HARD_STOP`, fails C5.
- **Session Max Drawdown Limit:** $\ge 2.50$ units $\rightarrow$ `ABORTED_HARD_STOP`, fails C5.
- **Missing / Unquotable Rate:** $> 25.0\%$ of ticks $\rightarrow$ `ABORTED_HARD_STOP`, fails C5.
- **Average Staleness:** $> 2,000\,\text{ms}$ $\rightarrow$ `ABORTED_HARD_STOP`, fails C5.
- **Rolling Directional Asymmetry:** $> 0.40$ (over trailing 50 filled trades) $\rightarrow$ `ABORTED_HARD_STOP`, fails C5.
- **Global Strategy Aborts:** $\ge 3$ cumulative aborted sessions $\rightarrow$ Permanent strategy rejection.

---

## 3. Fill Certainty Scenarios

Execution simulations report both conservative and optimistic scenarios independently:
- **Conservative Cohort:** Restricts filled trades strictly to `CONFIRMED_PRICE_THROUGH` (Ask $\le$ limit price for BUY; Bid $\ge$ limit price for SELL). Ambiguous quote retreats are evaluated as unfilled ($0.0$ PnL).
- **Optimistic Cohort:** Admits `AMBIGUOUS_QUOTE_RETREAT` (quote retreated without confirmed trade-through).
- Both cohorts are segregated in `fill_certainty_scenarios` to prevent fill-rate fabrication or double-counting.

---

## 4. Test Suite Verification Totals

Executed completely offline via pytest without network or WebSocket dependencies:
```bash
pytest tests/pipeline_v2/test_phase27_maker_execution.py \
       tests/pipeline_v2/test_phase27_prospective_collector.py \
       tests/pipeline_v2/test_phase26_forward_shadow.py \
       tests/pipeline_v2/test_phase26_prospective_collector.py -v
```

- `tests/pipeline_v2/test_phase27_maker_execution.py`: **27 passed**
- `tests/pipeline_v2/test_phase27_prospective_collector.py`: **27 passed**
- `tests/pipeline_v2/test_phase26_forward_shadow.py`: **12 passed**
- `tests/pipeline_v2/test_phase26_prospective_collector.py`: **55 passed**
- **Total Test Count:** **121 passed, 0 failed, 0 skipped (100% pass rate)**

---

## 5. Phase 27.10 Audit Findings Reconciliation

All findings from the Phase 27.10 audit have been remediated or reconciled:

- **F-01 (Prospective Burst Clustering Definition):** Fixed. Replaced tick-based burst progression with deterministic quiet-interval clustering (`inter-signal gap > 5,000 ms` or 5-minute contract rollover). Successive signals within 5 seconds share a cluster; clusters are never artificially manufactured. Verified by `test_deterministic_prospective_burst_clustering_on_signal_gaps`.
- **F-02 (Fill Certainty Propagation & Cohort Accounting):** Fixed. Ensured `fill_certainty` (`CONFIRMED_PRICE_THROUGH` vs `AMBIGUOUS_QUOTE_RETREAT`) is fully propagated across all terminal order states (fill, cancellation, settlement) and populated in offline evaluation cohorts. Verified by `test_offline_fill_certainty_propagation_and_conservative_cohort`.
- **F-03 (DOWN-Token Execution Alignment):** Fixed & Reconciled. `MakerExecutionSimulator` now supports native BUY limit execution at `down_bid` when separate DOWN quotes exist. For historical data where only UP-token order books are recorded, the inverted UP proxy is explicitly retained, and dual-book backtest comparisons are formally documented as `UNSUPPORTED_HISTORICAL_DOWN_BOOK_DATA_UNAVAILABLE` rather than fabricating synthetic order books. Verified by `test_down_token_execution_alignment_in_maker_simulator`.
- **F-04 (Five-Session Fixed Horizon Enforcement):** Verified. `aggregate_prospective_sessions` strictly enforces `completed_sessions >= 5` before protocol acceptance. Favorable intermediate sessions cannot trigger early acceptance. Circuit breakers permit early termination only to abort and fail the strategy.
- **F-05 (Pre-Rollover Entry Blackout Gate):** Fixed. Added 5-second entry blackout window (`time_to_rollover_ms <= 5,000 ms`) in `process_buffered_grid_step` suppressing boundary orders under `ROLLOVER_BLACKOUT_WINDOW` to eliminate artificial forced liquidation penalties. Verified by `test_pre_rollover_entry_blackout_window_rejects_boundary_orders`.
- **F-06 (Rolling Directional Asymmetry Hard Stop Inactivity):** Verified. Hard stop triggers strictly when $|N_{\text{up}} - N_{\text{down}}| / 50 > 0.40$ over a completed 50-trade rolling window and remains strictly inactive when fewer than 50 filled trades have accumulated. Verified by `test_incomplete_window_asymmetry_inactive`.

---

## 6. Known Limitations & Profitability Disclaimers

1. **Profitability Is Unproven:**
   - **Passing deterministic software tests does NOT demonstrate trading alpha.**
   - Historical Period 2 calibration (`data/models/phase27/phase27_calibration_report.json`) failed declared profitability hurdles across all three evaluated approaches (Raw Baseline: $-0.00085$ ITT PnL; Temperature Scaling: $+0.000255$ ITT PnL; Directional Gating: $+0.000409$ ITT PnL).
   - Predeclared hurdles were never relaxed. Prospective collection serves strictly as an empirical hypothesis test.
2. **Inter-Burst Serial Independence Assumption:**
   - Cluster-robust standard errors group trades by recording burst ($> 5,000\,\text{ms}$ quiet intervals) and scope cluster IDs by session.
   - The inference model assumes residual correlation decays across $\ge 5,000\,\text{ms}$ gaps. Multi-hour macro market trends may introduce residual autocorrelation. Session-level clustering is precluded because $C_{\text{sessions}} = 5$ lacks asymptotic degrees of freedom.
3. **Effective Degrees of Freedom Bound ($N_{\text{eff}} \le C$):**
   - Per Cameron-Miller / Carter-Schnepel-Steigerwald formulation, $N_{\text{eff}}$ quantifies balanced cluster degrees of freedom ($G^*$), not individual trades. By Jensen's Inequality, $N_{\text{eff}} \le C$ strictly holds.
4. **Session 001 Authorization:**
   - Prospective collection remains **strictly UNAUTHORIZED**. Any execution requires separate explicit operator authorization.
