# Phase 11B — Expanded Polymarket Collection Ingestion & Audit Report

**Execution Timestamp**: `2026-10-06T16:54:43.206325+00:00`  
**Total Raw Markets Processed**: `52` (19 original retained + 2 original excluded + 31 new expansion)  
**Total Markets Retained**: `46` (19 original baseline + 27 new expansion)  
**Total Markets Excluded**: `6` (2 original excluded + 4 new excluded)  
**Total Raw Input Rows**: `8,921,784`  
**Total Valid Canonical Events**: `16,718,713`  
**Total Rejected Records**: `1,000,312`  
**Total Identical Duplicates Dropped**: `0`  
**Total Conflicting Duplicates Dropped**: `0`  
**Raw Data Immutability**: `VERIFIED (0 mutations)`  
**Session Isolation**: `VERIFIED (52 isolated files, orderbook state cleanly reset per market)`  

---

## 1. Aggregate Quality Verification

| Metric | Aggregate Result | Evaluation |
| :--- | :--- | :--- |
| All Raw Files Immutable (SHA-256) | `True` | PASS |
| Total Canonical Parquet Files Generated | `52` | PASS |
| Monotonic Non-Decreasing Timestamps | `True` | PASS |
| Bid < Ask Enforcement (Accepted Events) | `100.0%` (0 crossed / 0 non-positive spreads) | PASS |
| Non-Zero Depth Enforcement (Accepted Events) | `100.0%` (bid_size > 0, ask_size > 0) | PASS |
| Total Conflicting Duplicates Detected | `0` | PASS |
| Normalization Processing Runtime | `146.67s` | PASS |

---

## 2. Rejection Reasons Breakdown (Aggregate)

| Rejection Reason | Total Count | % of Parsed Records | Description |
| :--- | :--- | :--- | :--- |
| `MISSING_PRICE` | 948,860 | 10.635% | Top of book lacked valid two-sided bid/ask quote |
| `NON_QUOTE_EVENT` | 41,432 | 0.464% | Non-quote trade or empty book update payload |
| `CROSSED_BOOK` | 10,020 | 0.112% | Crossed market condition (bid >= ask) strictly rejected |

---

## 3. Cohort Retention & Exclusion Summary

| Cohort | Total Files | Retained Markets | Excluded Markets | Total Raw Rows | Valid Canonical Events |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original Baseline (Session 1)** | 21 | 19 | 2 | 4,205,376 | 7,910,699 |
| **New Expansion (Historical)** | 12 | 11 | 1 | 1,168,880 | 2,130,593 |
| **New Expansion (Recent)** | 19 | 17 | 2 | 3,547,528 | 6,677,421 |
| **All Combined** | **52** | **47** | **5** | **8,921,784** | **16,718,713** |

---

## 4. Per-Market Ingestion & Audit Ledger (All 52 Markets)

| # | Market Slug | Cohort | Status | Raw Rows | Canonical Events | Rejected | Span (s) | Density (ev/s) | Parquet Size | Reason / Note |
| :- | :--- | :--- | :---: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| 1 | `btc-updown-5m-1791062400` | NEW_HISTORICAL | `RETAINED` | 125,583 | 240,788 | 8,185 | 157.8 | 1526.3 | 3.13 MB | Dense truncated session (157.8s span, 240,788 canonical events, both assets active). |
| 2 | `btc-updown-5m-1791062700` | NEW_HISTORICAL | `RETAINED` | 78,623 | 155,820 | 383 | 93.2 | 1672.0 | 2.26 MB | Dense truncated session (93.2s span, 155,820 canonical events, both assets active). |
| 3 | `btc-updown-5m-1791063900` | NEW_HISTORICAL | `RETAINED` | 43,414 | 67,138 | 18,796 | 135.5 | 495.4 | 0.94 MB | Dense truncated session (135.5s span, 67,138 canonical events, both assets active). |
| 4 | `btc-updown-5m-1791064200` | NEW_HISTORICAL | `RETAINED` | 52,080 | 103,330 | 228 | 65.0 | 1588.7 | 1.43 MB | Dense truncated session (65.0s span, 103,330 canonical events, both assets active). |
| 5 | `btc-updown-5m-1791064500` | NEW_HISTORICAL | `RETAINED` | 164,705 | 316,394 | 11,277 | 170.8 | 1852.7 | 3.96 MB | Dense truncated session (170.8s span, 316,394 canonical events, both assets active). |
| 6 | `btc-updown-5m-1791064800` | NEW_HISTORICAL | `EXCLUDED` | 11,931 | 23,614 | 61 | 19.1 | 1236.1 | 0.32 MB | Minimal duration (19.1s span < 50s threshold). |
| 7 | `btc-updown-5m-1791065400` | NEW_HISTORICAL | `RETAINED` | 122,340 | 230,288 | 12,832 | 160.4 | 1435.3 | 3.03 MB | Dense truncated session (160.4s span, 230,288 canonical events, both assets active). |
| 8 | `btc-updown-5m-1791065700` | NEW_HISTORICAL | `RETAINED` | 40,358 | 79,264 | 380 | 55.1 | 1439.5 | 1.10 MB | Dense truncated session (55.1s span, 79,264 canonical events, both assets active). |
| 9 | `btc-updown-5m-1791066600` | NEW_HISTORICAL | `RETAINED` | 102,558 | 200,912 | 1,120 | 112.8 | 1781.8 | 2.82 MB | Dense truncated session (112.8s span, 200,912 canonical events, both assets active). |
| 10 | `btc-updown-5m-1791066900` | NEW_HISTORICAL | `RETAINED` | 133,360 | 254,634 | 10,108 | 246.6 | 1032.4 | 3.28 MB | Complete active trading session (246.6s span, 254,634 canonical events). |
| 11 | `btc-updown-5m-1791067200` | NEW_HISTORICAL | `RETAINED` | 199,990 | 388,811 | 6,659 | 277.8 | 1399.5 | 5.11 MB | Complete active trading session (277.8s span, 388,811 canonical events). |
| 12 | `btc-updown-5m-1791067500` | NEW_HISTORICAL | `RETAINED` | 93,938 | 186,682 | 316 | 109.6 | 1702.6 | 2.67 MB | Dense truncated session (109.6s span, 186,682 canonical events, both assets active). |
| 13 | `btc-updown-5m-1791204300` | ORIGINAL_2_EXCLUDED | `EXCLUDED` | 8,557 | 0 | 17,091 | 0.0 | 0.0 | 0.01 MB | 0 valid canonical events (contract post-resolution). |
| 14 | `btc-updown-5m-1791204600` | ORIGINAL_19_RETAINED | `RETAINED` | 343,369 | 673,766 | 8,859 | 266.5 | 2528.3 | 8.12 MB | Original Phase 11B production market (immutable baseline). |
| 15 | `btc-updown-5m-1791204900` | ORIGINAL_19_RETAINED | `RETAINED` | 467,917 | 925,368 | 3,463 | 298.1 | 3103.8 | 11.02 MB | Original Phase 11B production market (immutable baseline). |
| 16 | `btc-updown-5m-1791205200` | ORIGINAL_19_RETAINED | `RETAINED` | 122,173 | 209,912 | 32,443 | 168.8 | 1243.5 | 2.96 MB | Original Phase 11B production market (immutable baseline). |
| 17 | `btc-updown-5m-1791205500` | ORIGINAL_19_RETAINED | `RETAINED` | 485,682 | 956,730 | 7,266 | 284.8 | 3359.4 | 11.58 MB | Original Phase 11B production market (immutable baseline). |
| 18 | `btc-updown-5m-1791205800` | ORIGINAL_19_RETAINED | `RETAINED` | 237,034 | 454,690 | 16,128 | 231.9 | 1960.7 | 5.83 MB | Original Phase 11B production market (immutable baseline). |
| 19 | `btc-updown-5m-1791206100` | ORIGINAL_19_RETAINED | `RETAINED` | 287,637 | 542,568 | 28,751 | 239.7 | 2263.9 | 6.70 MB | Original Phase 11B production market (immutable baseline). |
| 20 | `btc-updown-5m-1791206400` | ORIGINAL_19_RETAINED | `RETAINED` | 186,873 | 336,114 | 34,973 | 204.0 | 1647.9 | 4.39 MB | Original Phase 11B production market (immutable baseline). |
| 21 | `btc-updown-5m-1791206700` | ORIGINAL_19_RETAINED | `RETAINED` | 198,037 | 378,128 | 15,261 | 243.7 | 1551.5 | 4.78 MB | Original Phase 11B production market (immutable baseline). |
| 22 | `btc-updown-5m-1791207000` | ORIGINAL_19_RETAINED | `RETAINED` | 177,441 | 320,190 | 31,763 | 241.9 | 1323.6 | 4.37 MB | Original Phase 11B production market (immutable baseline). |
| 23 | `btc-updown-5m-1791207300` | ORIGINAL_19_RETAINED | `RETAINED` | 144,861 | 264,818 | 22,997 | 182.6 | 1450.2 | 3.55 MB | Original Phase 11B production market (immutable baseline). |
| 24 | `btc-updown-5m-1791207600` | ORIGINAL_19_RETAINED | `RETAINED` | 188,949 | 351,026 | 24,273 | 215.7 | 1627.0 | 4.57 MB | Original Phase 11B production market (immutable baseline). |
| 25 | `btc-updown-5m-1791207900` | ORIGINAL_19_RETAINED | `RETAINED` | 147,065 | 240,534 | 52,041 | 165.2 | 1456.0 | 3.23 MB | Original Phase 11B production market (immutable baseline). |
| 26 | `btc-updown-5m-1791208200` | ORIGINAL_19_RETAINED | `RETAINED` | 198,717 | 386,331 | 8,752 | 237.8 | 1624.8 | 4.86 MB | Original Phase 11B production market (immutable baseline). |
| 27 | `btc-updown-5m-1791208500` | ORIGINAL_19_RETAINED | `RETAINED` | 205,888 | 385,811 | 22,815 | 251.5 | 1534.1 | 5.00 MB | Original Phase 11B production market (immutable baseline). |
| 28 | `btc-updown-5m-1791208800` | ORIGINAL_2_EXCLUDED | `EXCLUDED` | 49,527 | 97,776 | 399 | 54.2 | 1802.7 | 1.36 MB | Original Phase 11B excluded market (truncated 54.2s recording). |
| 29 | `btc-updown-5m-1791209400` | ORIGINAL_19_RETAINED | `RETAINED` | 120,261 | 230,215 | 8,608 | 152.4 | 1510.1 | 3.17 MB | Original Phase 11B production market (immutable baseline). |
| 30 | `btc-updown-5m-1791209700` | ORIGINAL_19_RETAINED | `RETAINED` | 182,624 | 331,244 | 31,380 | 212.8 | 1556.4 | 4.31 MB | Original Phase 11B production market (immutable baseline). |
| 31 | `btc-updown-5m-1791210000` | ORIGINAL_19_RETAINED | `RETAINED` | 129,945 | 242,870 | 15,707 | 208.5 | 1165.0 | 3.27 MB | Original Phase 11B production market (immutable baseline). |
| 32 | `btc-updown-5m-1791210300` | ORIGINAL_19_RETAINED | `RETAINED` | 188,828 | 356,687 | 18,781 | 215.0 | 1659.3 | 4.56 MB | Original Phase 11B production market (immutable baseline). |
| 33 | `btc-updown-5m-1791210600` | ORIGINAL_19_RETAINED | `RETAINED` | 133,991 | 225,921 | 40,558 | 156.3 | 1445.7 | 3.14 MB | Original Phase 11B production market (immutable baseline). |
| 34 | `btc-updown-5m-1791291300` | NEW_RECENT | `EXCLUDED` | 15,609 | 5,036 | 25,951 | 11.8 | 425.8 | 0.07 MB | Insufficient span (11.8s) or zero valid events. |
| 35 | `btc-updown-5m-1791291600` | NEW_RECENT | `RETAINED` | 220,060 | 421,144 | 15,622 | 244.7 | 1721.1 | 5.47 MB | Complete active trading session (244.7s span, 421,144 canonical events). |
| 36 | `btc-updown-5m-1791291900` | NEW_RECENT | `RETAINED` | 227,188 | 431,136 | 20,248 | 238.7 | 1806.0 | 5.46 MB | Complete active trading session (238.7s span, 431,136 canonical events). |
| 37 | `btc-updown-5m-1791292200` | NEW_RECENT | `RETAINED` | 67,797 | 117,482 | 16,895 | 73.1 | 1606.7 | 1.74 MB | Dense truncated session (73.1s span, 117,482 canonical events, both assets active). |
| 38 | `btc-updown-5m-1791292500` | NEW_RECENT | `RETAINED` | 191,575 | 340,422 | 40,147 | 195.2 | 1743.5 | 4.21 MB | Dense truncated session (195.2s span, 340,422 canonical events, both assets active). |
| 39 | `btc-updown-5m-1791292800` | NEW_RECENT | `RETAINED` | 204,480 | 381,268 | 24,722 | 233.7 | 1631.1 | 4.79 MB | Complete active trading session (233.7s span, 381,268 canonical events). |
| 40 | `btc-updown-5m-1791293100` | NEW_RECENT | `RETAINED` | 188,115 | 349,076 | 24,529 | 234.1 | 1491.1 | 4.32 MB | Complete active trading session (234.1s span, 349,076 canonical events). |
| 41 | `btc-updown-5m-1791293400` | NEW_RECENT | `RETAINED` | 163,543 | 247,270 | 77,247 | 165.1 | 1497.8 | 3.18 MB | Dense truncated session (165.1s span, 247,270 canonical events, both assets active). |
| 42 | `btc-updown-5m-1791293700` | NEW_RECENT | `RETAINED` | 204,407 | 364,740 | 41,025 | 220.8 | 1652.2 | 4.46 MB | Complete active trading session (220.8s span, 364,740 canonical events). |
| 43 | `btc-updown-5m-1791294000` | NEW_RECENT | `RETAINED` | 188,525 | 342,667 | 31,894 | 210.8 | 1625.3 | 4.33 MB | Complete active trading session (210.8s span, 342,667 canonical events). |
| 44 | `btc-updown-5m-1791294300` | NEW_RECENT | `RETAINED` | 237,723 | 468,576 | 3,501 | 262.3 | 1786.1 | 5.88 MB | Complete active trading session (262.3s span, 468,576 canonical events). |
| 45 | `btc-updown-5m-1791294600` | NEW_RECENT | `RETAINED` | 223,141 | 433,732 | 10,137 | 230.9 | 1878.7 | 5.27 MB | Complete active trading session (230.9s span, 433,732 canonical events). |
| 46 | `btc-updown-5m-1791294900` | NEW_RECENT | `RETAINED` | 164,207 | 288,478 | 37,759 | 214.6 | 1344.4 | 3.77 MB | Complete active trading session (214.6s span, 288,478 canonical events). |
| 47 | `btc-updown-5m-1791295200` | NEW_RECENT | `RETAINED` | 234,287 | 447,510 | 18,253 | 268.7 | 1665.3 | 5.33 MB | Complete active trading session (268.7s span, 447,510 canonical events). |
| 48 | `btc-updown-5m-1791295500` | NEW_RECENT | `RETAINED` | 141,395 | 280,374 | 747 | 83.8 | 3345.6 | 3.58 MB | Dense truncated session (83.8s span, 280,374 canonical events, both assets active). |
| 49 | `btc-updown-5m-1791296100` | NEW_RECENT | `EXCLUDED` | 14,177 | 0 | 28,313 | 0.0 | 0.0 | 0.01 MB | 0 valid canonical events (empty order books, 100% missing prices). |
| 50 | `btc-updown-5m-1791296400` | NEW_RECENT | `RETAINED` | 456,582 | 877,252 | 29,984 | 241.5 | 3632.6 | 11.16 MB | Complete active trading session (241.5s span, 877,252 canonical events). |
| 51 | `btc-updown-5m-1791296700` | NEW_RECENT | `RETAINED` | 323,624 | 603,364 | 40,254 | 219.0 | 2754.5 | 8.01 MB | Complete active trading session (219.0s span, 603,364 canonical events). |
| 52 | `btc-updown-5m-1791297000` | NEW_RECENT | `EXCLUDED` | 81,093 | 160,812 | 430 | 41.1 | 3912.3 | 2.37 MB | Truncated recording (<50s threshold: 41.1s span) to guarantee sequence length. |

---

## 5. Architectural Guarantees & Verification

1. **Zero Raw Mutation**: All 52 raw JSONL files were SHA-256 hashed before and after ingestion. Zero raw file modifications detected.
2. **Session Isolation**: Each market was normalized independently in an isolated process with L2 order book state reset. Zero cross-market or cross-session state contamination.
3. **Token Representation**: Both UP and DOWN tokens are explicitly verified and mapped for every retained market.
4. **Strict Monotonicity & Causal Ordering**: 100% of generated canonical parquet files exhibit non-decreasing timestamp series.
5. **Zero Crossed Quotes**: 100% of canonical records strictly enforce bid < ask with positive depth on both sides.
6. **Interrupted Collector Run (1791295500)**: Fully normalized into 280,374 canonical events across 83.8 seconds with active two-sided quotes, confirming full data persistence.
