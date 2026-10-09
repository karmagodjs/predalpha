# Phase 12 — Causal 1-Second Grid Resampling Audit Report

**Execution Timestamp**: `2026-10-06T03:07:46.078134+00:00`
**Pipeline Component**: `pipeline_v2/resampling/point_in_time_grid.py`
**Input Dataset**: `data/clean_v2/01_canonical_events/new_collection/`
**Output Dataset**: `data/clean_v2/02_resampled_1s/new_collection/`
**Overall Status**: `ALL CHECKS PASSED`

## 1. Executive Summary & Verification Matrix

| Audit Dimension | Value / Metric | Requirement | Evaluation |
| :--- | :--- | :--- | :--- |
| **Retained Production Markets** | `19` markets | Exactly 19 audited markets | PASS |
| **Excluded Non-Production Markets** | `2` markets | Exactly 2 markets (`1791204300`, `1791208800`) | PASS |
| **Total Input Canonical Events** | `7,812,923` quotes | 100% accounted from Phase 11B | PASS |
| **Total 1-Second Grid Rows** | `8,358` rows | 1 row per second per asset | PASS |
| **Fresh Grid Observations** | `6,080` (72.74%) | Age $\le$ 5,000 ms | PASS |
| **Stale Grid Observations** | `2,278` (27.26%) | Age $>$ 5,000 ms (flagged `is_stale=True`) | PASS |
| **Future-Event Access Violations** | `0` | Strictly `event_age_ms >= 0` | PASS |
| **Cross-Session Violations** | `0` | Zero state carry-over | PASS |
| **Cross-Asset Violations** | `0` | Independent resampling per asset | PASS |
| **Monotonicity (grid_timestamp_ms)** | `100.0%` | Strictly increasing by 1,000 ms | PASS |
| **Duplicate Grid Timestamps** | `0` | 0 duplicates per (market_id, asset_id) | PASS |
| **Deterministic Reproducibility** | `VERIFIED (100.0% identical)` | Bit-for-bit duplicate pass match | PASS |

## 2. Market Scope: Inclusions and Exclusions

### Excluded Markets
- **`btc-updown-5m-1791204300`**: Zero valid canonical events (resolved/expired contract, 100% one-sided book).
- **`btc-updown-5m-1791208800`**: Truncated recording duration (54.2s < 60s); insufficient length for sequence model.

### Retained Markets (19 Sessions)
The 19 retained markets represent `7,812,923` valid canonical quote events, spanning `4160.0` seconds (~69.3 minutes) of active two-sided orderbook dynamics.

## 3. Data-Size & Expansion Verification

### Why did 4.2M raw records expand into 7.91M canonical events in Phase 11B?
- In Polymarket's CLOB websocket, a single JSONL event packet (`event_type == 'price_change'`) contains price level updates for multiple assets simultaneously (both the UP token and the DOWN token).
- `EventNormalizer` updates the local L2 orderbook and produces top-of-book canonical quotes for each affected outcome asset independently, with complete raw provenance (`source_file`, `source_line`, `source_record_idx`).
- Thus, ~4.2M raw messages containing multi-asset updates naturally expand into ~7.91M valid canonical quote events (~1.9 quotes per raw line).
### Why does Phase 12 output ~10,000 grid rows?
- Phase 12 resamples the continuous canonical quote stream onto an exact integer 1-second physical grid ($T \in \{1000, 2000, \dots\}$). At each integer second $T$, **backward as-of semantics** select strictly the latest quote where $\tau \le T$.
- For each 5-minute (~200–300 second) session, this produces ~200–300 grid observations per asset, totaling ~500–600 rows per market across both assets.
- Across the 19 retained markets, this yields exactly `8,358` synchronized point-in-time observations. There is **zero double-counting** and **zero synthetic fabrication**.

## 4. Per-Market, Per-Asset Resampling Ledger

| # | Market Stem | Asset / Token | Input Events | Grid Rows | Fresh | Stale | Stale % | Mean Age (ms) | Max Age (ms) | Mean Ev/Sec | Parquet Size |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `btc-updown-5m-1791204600` | `103004105738...` | 336,883 | 267 | 267 | 0 | 0.0% | 16.48 | 1011 | 1261.4 | 29.5 KB |
| 1 | `btc-updown-5m-1791204600` | `464089472127...` | 336,883 | 267 | 267 | 0 | 0.0% | 16.48 | 1011 | 1261.4 | 29.5 KB |
| 2 | `btc-updown-5m-1791204900` | `351825476268...` | 462,684 | 298 | 298 | 0 | 0.0% | 11.47 | 1065 | 1552.1 | 30.9 KB |
| 2 | `btc-updown-5m-1791204900` | `473660730856...` | 462,684 | 298 | 298 | 0 | 0.0% | 11.47 | 1065 | 1552.1 | 30.9 KB |
| 3 | `btc-updown-5m-1791205200` | `157381305981...` | 104,956 | 169 | 149 | 20 | 11.83% | 1913.0 | 24865 | 620.8 | 20.9 KB |
| 3 | `btc-updown-5m-1791205200` | `613083876120...` | 104,956 | 169 | 149 | 20 | 11.83% | 1913.0 | 24865 | 620.8 | 20.9 KB |
| 4 | `btc-updown-5m-1791205500` | `802876477434...` | 478,365 | 285 | 285 | 0 | 0.0% | 11.03 | 503 | 1677.9 | 29.9 KB |
| 4 | `btc-updown-5m-1791205500` | `925896058669...` | 478,365 | 285 | 285 | 0 | 0.0% | 11.03 | 503 | 1677.9 | 29.9 KB |
| 5 | `btc-updown-5m-1791205800` | `100773438282...` | 227,345 | 232 | 232 | 0 | 0.0% | 7.13 | 143 | 979.5 | 26.6 KB |
| 5 | `btc-updown-5m-1791205800` | `109229399631...` | 227,345 | 232 | 232 | 0 | 0.0% | 7.13 | 143 | 979.5 | 26.6 KB |
| 6 | `btc-updown-5m-1791206100` | `509947144324...` | 271,284 | 240 | 228 | 12 | 5.0% | 634.22 | 16895 | 1129.5 | 26.2 KB |
| 6 | `btc-updown-5m-1791206100` | `600287239860...` | 271,284 | 240 | 228 | 12 | 5.0% | 634.22 | 16895 | 1129.5 | 26.2 KB |
| 7 | `btc-updown-5m-1791206400` | `481608493785...` | 168,057 | 204 | 178 | 26 | 12.75% | 2391.59 | 30291 | 822.2 | 22.9 KB |
| 7 | `btc-updown-5m-1791206400` | `948152936681...` | 168,057 | 204 | 178 | 26 | 12.75% | 2391.59 | 30291 | 822.2 | 22.9 KB |
| 8 | `btc-updown-5m-1791206700` | `932847378880...` | 189,064 | 244 | 159 | 85 | 34.84% | 5663.55 | 27958 | 774.0 | 22.0 KB |
| 8 | `btc-updown-5m-1791206700` | `998334914417...` | 189,064 | 244 | 159 | 85 | 34.84% | 5663.55 | 27958 | 774.0 | 22.0 KB |
| 9 | `btc-updown-5m-1791207000` | `637866452947...` | 160,095 | 242 | 162 | 80 | 33.06% | 6383.59 | 35997 | 661.0 | 22.4 KB |
| 9 | `btc-updown-5m-1791207000` | `706119029889...` | 160,095 | 242 | 162 | 80 | 33.06% | 6383.59 | 35997 | 661.0 | 22.4 KB |
| 10 | `btc-updown-5m-1791207300` | `648387816960...` | 132,409 | 183 | 84 | 99 | 54.1% | 9719.13 | 32422 | 718.7 | 17.1 KB |
| 10 | `btc-updown-5m-1791207300` | `674677772807...` | 132,409 | 183 | 84 | 99 | 54.1% | 9719.13 | 32422 | 718.7 | 17.1 KB |
| 11 | `btc-updown-5m-1791207600` | `290867883413...` | 175,513 | 216 | 158 | 58 | 26.85% | 4066.41 | 25211 | 812.2 | 21.9 KB |
| 11 | `btc-updown-5m-1791207600` | `872034412152...` | 175,513 | 216 | 158 | 58 | 26.85% | 4066.41 | 25211 | 812.2 | 21.9 KB |
| 12 | `btc-updown-5m-1791207900` | `491291375149...` | 120,267 | 165 | 91 | 74 | 44.85% | 8106.74 | 29609 | 725.7 | 17.1 KB |
| 12 | `btc-updown-5m-1791207900` | `605296424063...` | 120,267 | 165 | 91 | 74 | 44.85% | 8106.74 | 29609 | 725.7 | 17.1 KB |
| 13 | `btc-updown-5m-1791208200` | `509908347043...` | 193,165 | 238 | 113 | 125 | 52.52% | 9494.47 | 32573 | 801.2 | 19.3 KB |
| 13 | `btc-updown-5m-1791208200` | `902597675900...` | 193,166 | 238 | 113 | 125 | 52.52% | 9494.47 | 32573 | 801.2 | 19.3 KB |
| 14 | `btc-updown-5m-1791208500` | `106423082807...` | 192,905 | 251 | 162 | 89 | 35.46% | 5600.67 | 26276 | 768.2 | 22.4 KB |
| 14 | `btc-updown-5m-1791208500` | `756705874599...` | 192,906 | 251 | 162 | 89 | 35.46% | 5600.67 | 26276 | 768.2 | 22.4 KB |
| 15 | `btc-updown-5m-1791209400` | `710166823939...` | 115,107 | 152 | 77 | 75 | 49.34% | 8975.54 | 32775 | 748.1 | 16.3 KB |
| 15 | `btc-updown-5m-1791209400` | `736296543675...` | 115,108 | 152 | 77 | 75 | 49.34% | 8975.54 | 32775 | 748.1 | 16.3 KB |
| 16 | `btc-updown-5m-1791209700` | `336877258341...` | 165,622 | 213 | 117 | 96 | 45.07% | 7929.58 | 31707 | 776.7 | 19.5 KB |
| 16 | `btc-updown-5m-1791209700` | `374863261333...` | 165,622 | 213 | 117 | 96 | 45.07% | 7929.58 | 31707 | 776.7 | 19.5 KB |
| 17 | `btc-updown-5m-1791210000` | `100772786011...` | 121,435 | 209 | 75 | 134 | 64.11% | 14504.01 | 43436 | 579.1 | 17.0 KB |
| 17 | `btc-updown-5m-1791210000` | `661787603649...` | 121,435 | 209 | 75 | 134 | 64.11% | 14504.01 | 43436 | 579.1 | 17.0 KB |
| 18 | `btc-updown-5m-1791210300` | `411271742330...` | 178,343 | 215 | 117 | 98 | 45.58% | 8122.98 | 30988 | 824.8 | 19.4 KB |
| 18 | `btc-updown-5m-1791210300` | `818503514135...` | 178,344 | 215 | 117 | 98 | 45.58% | 8122.98 | 30988 | 824.8 | 19.4 KB |
| 19 | `btc-updown-5m-1791210600` | `513347628137...` | 112,961 | 156 | 88 | 68 | 43.59% | 7424.65 | 28660 | 719.8 | 17.0 KB |
| 19 | `btc-updown-5m-1791210600` | `913814958601...` | 112,960 | 156 | 88 | 68 | 43.59% | 7424.65 | 28660 | 719.8 | 17.0 KB |

## 5. Causal Timestamp & Duplicate Handling Statistics

- **Backward Point-In-Time Semantics**: Every grid snapshot at integer second $T$ selects the latest event with $\tau \le T$. The event age is computed as $\Delta t = T - \tau \ge 0$.
- **Future Access Violations**: `0` (asserted for 100% of rows across all 19 markets).
- **Duplicate Source Timestamps**: In raw high-frequency feeds, multiple order book updates frequently arrive within the same physical millisecond. In Phase 11B, canonical events retain deterministic ordering via `sequence_id ASC`. At grid time $T$, `PointInTimeResampler` uses `pd.merge_asof(direction='backward')`, which deterministically selects the **latest sequence update** at or before $T$, completely eliminating ambiguity without discarding intermediate book states.
- **Event Density within 1-Second Windows**: An average of `892.3` source canonical quotes arrive during each 1-second grid window, confirming high liquidity and active book depth updating across the retained sessions.
