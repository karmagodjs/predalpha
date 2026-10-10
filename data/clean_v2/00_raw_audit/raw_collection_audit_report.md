# Raw Polymarket Collection Integrity Audit Report

**Audit Execution Timestamp**: `2026-10-06T17:04:29.895337+00:00`  
**Execution Runtime**: `118.08s`  
**Total Raw Files Scanned**: `52`  
**Total Raw Event Lines**: `8,921,784`  
**Total Raw File Size**: `8.71 GB` (`9,352,191,998` bytes)  
**Malformed JSON Lines**: `0`  
**Original Production Files Immutability**: `VERIFIED (0 mutations)`  

---

## 1. Executive Summary & Core Audit Answers

| Audit Question | Exact Finding | Status |
| :--- | :--- | :--- |
| **Number of new markets** | **31 new markets** (12 historical `179106xxxx` + 19 recent `179129xxxx`) [or **33 non-retained** if including the 2 original excluded] | VERIFIED |
| **Number of complete new markets** | **18 complete markets** (observed span $\ge 200$s, both tokens active) | VERIFIED |
| **Number of truncated/incomplete new markets** | **13 markets** (11 truncated with $\ge 50$s; 2 minimal/degenerate) | VERIFIED |
| **Total raw events across collection** | **8,921,784 raw events** (Original 19: 4,147,292; New 31: 4,716,408; Excluded 2: 58,084) | VERIFIED |
| **Estimated usable new markets** | **29 usable new markets** (18 complete + 11 truncated with dense quotes) | VERIFIED |
| **Interrupted Run (1791295500) Persistence** | **141,395 raw events** persisted (141.19 MB, 83.81s span) | VERIFIED |
| **Sufficient to Proceed to Phase 11B** | **YES — Collection is highly sufficient** (Combines to 48 usable markets, >8.5M events, satisfying the 50k observation threshold) | **READY** |

---

## 2. Cohort Breakdown & Inventory

| Cohort | Count | Raw Size (MB) | Raw Events | Complete | Truncated | Minimal/Degenerate | Usable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original 19 Retained** | 19 | 4,147.7 MB | 4,147,292 | 19 | 0 | 0 | **19** |
| **Original 2 Excluded** | 2 | 58.2 MB | 58,084 | 0 | 2 | 0 | **2** |
| **New Historical (179106xxxx)** | 12 | 1,170.1 MB | 1,168,880 | 4 | 7 | 1 | **11** |
| **New Recent (179129xxxx)** | 19 | 3,543.0 MB | 3,547,528 | 14 | 4 | 1 | **18** |
| **Total New Expansion (31)** | 31 | 4,713.1 MB | 4,716,408 | 18 | 11 | 2 | **29** |
| **All 52 Markets Combined** | 52 | 8,919.0 MB | 8,921,784 | 37 | 13 | 2 | **50** |

---

## 3. Newly Collected Markets Detailed Ledger (31 Markets)

| # | Market Slug | Cohort | Size (MB) | Events | Duration (s) | Both Tokens? | Classification | Usable? | Note |
| :- | :--- | :--- | :-: | :-: | :-: | :-: | :--- | :-: | :--- |
| 1 | `btc-updown-5m-1791062400` | NEW_HISTORICAL | 125.62 | 125,583 | 187.6 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 187.6s duration, both assets active, 125583 events. |
| 2 | `btc-updown-5m-1791062700` | NEW_HISTORICAL | 78.29 | 78,623 | 93.19 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 93.19s duration, both assets active, 78623 events. |
| 3 | `btc-updown-5m-1791063900` | NEW_HISTORICAL | 43.9 | 43,414 | 194.51 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 194.51s duration, both assets active, 43414 events. |
| 4 | `btc-updown-5m-1791064200` | NEW_HISTORICAL | 51.79 | 52,080 | 65.04 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 65.04s duration, both assets active, 52080 events. |
| 5 | `btc-updown-5m-1791064500` | NEW_HISTORICAL | 163.53 | 164,705 | 239.74 | YES | `COMPLETE` | **YES** | Full active session: 239.74s duration, both assets active, 164128 quote events. |
| 6 | `btc-updown-5m-1791064800` | NEW_HISTORICAL | 11.92 | 11,931 | 19.1 | YES | `MINIMAL_TRUNCATED` | NO | Very short duration (<50s): 19.1s, insufficient for sustained multi-step sequence modeling. |
| 7 | `btc-updown-5m-1791065400` | NEW_HISTORICAL | 121.75 | 122,340 | 217.46 | YES | `COMPLETE` | **YES** | Full active session: 217.46s duration, both assets active, 121822 quote events. |
| 8 | `btc-updown-5m-1791065700` | NEW_HISTORICAL | 41.07 | 40,358 | 55.06 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 55.06s duration, both assets active, 40358 events. |
| 9 | `btc-updown-5m-1791066600` | NEW_HISTORICAL | 103.97 | 102,558 | 112.76 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 112.76s duration, both assets active, 102558 events. |
| 10 | `btc-updown-5m-1791066900` | NEW_HISTORICAL | 133.51 | 133,360 | 293.84 | YES | `COMPLETE` | **YES** | Full active session: 293.84s duration, both assets active, 132698 quote events. |
| 11 | `btc-updown-5m-1791067200` | NEW_HISTORICAL | 201.71 | 199,990 | 298.55 | YES | `COMPLETE` | **YES** | Full active session: 298.55s duration, both assets active, 198491 quote events. |
| 12 | `btc-updown-5m-1791067500` | NEW_HISTORICAL | 93.07 | 93,938 | 109.64 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 109.64s duration, both assets active, 93938 events. |
| 13 | `btc-updown-5m-1791291300` | NEW_RECENT | 15.61 | 15,609 | 104.41 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 104.41s duration, both assets active, 15609 events. |
| 14 | `btc-updown-5m-1791291600` | NEW_RECENT | 220.67 | 220,060 | 299.15 | YES | `COMPLETE` | **YES** | Full active session: 299.15s duration, both assets active, 218944 quote events. |
| 15 | `btc-updown-5m-1791291900` | NEW_RECENT | 226.83 | 227,188 | 298.83 | YES | `COMPLETE` | **YES** | Full active session: 298.83s duration, both assets active, 226192 quote events. |
| 16 | `btc-updown-5m-1791292200` | NEW_RECENT | 68.19 | 67,797 | 123.26 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 123.26s duration, both assets active, 67797 events. |
| 17 | `btc-updown-5m-1791292500` | NEW_RECENT | 190.59 | 191,575 | 298.58 | YES | `COMPLETE` | **YES** | Full active session: 298.58s duration, both assets active, 190714 quote events. |
| 18 | `btc-updown-5m-1791292800` | NEW_RECENT | 204.84 | 204,480 | 298.6 | YES | `COMPLETE` | **YES** | Full active session: 298.6s duration, both assets active, 203492 quote events. |
| 19 | `btc-updown-5m-1791293100` | NEW_RECENT | 188.17 | 188,115 | 298.75 | YES | `COMPLETE` | **YES** | Full active session: 298.75s duration, both assets active, 187242 quote events. |
| 20 | `btc-updown-5m-1791293400` | NEW_RECENT | 163.66 | 163,543 | 298.61 | YES | `COMPLETE` | **YES** | Full active session: 298.61s duration, both assets active, 162688 quote events. |
| 21 | `btc-updown-5m-1791293700` | NEW_RECENT | 205.08 | 204,407 | 298.82 | YES | `COMPLETE` | **YES** | Full active session: 298.82s duration, both assets active, 203394 quote events. |
| 22 | `btc-updown-5m-1791294000` | NEW_RECENT | 188.32 | 188,525 | 298.73 | YES | `COMPLETE` | **YES** | Full active session: 298.73s duration, both assets active, 187697 quote events. |
| 23 | `btc-updown-5m-1791294300` | NEW_RECENT | 237.66 | 237,723 | 298.62 | YES | `COMPLETE` | **YES** | Full active session: 298.62s duration, both assets active, 236606 quote events. |
| 24 | `btc-updown-5m-1791294600` | NEW_RECENT | 221.27 | 223,141 | 298.7 | YES | `COMPLETE` | **YES** | Full active session: 298.7s duration, both assets active, 222340 quote events. |
| 25 | `btc-updown-5m-1791294900` | NEW_RECENT | 164.02 | 164,207 | 298.53 | YES | `COMPLETE` | **YES** | Full active session: 298.53s duration, both assets active, 163482 quote events. |
| 26 | `btc-updown-5m-1791295200` | NEW_RECENT | 233.23 | 234,287 | 298.81 | YES | `COMPLETE` | **YES** | Full active session: 298.81s duration, both assets active, 233356 quote events. |
| 27 | `btc-updown-5m-1791295500` | NEW_RECENT | 141.19 | 141,395 | 83.81 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 83.81s duration, both assets active, 141395 events. |
| 28 | `btc-updown-5m-1791296100` | NEW_RECENT | 13.82 | 14,177 | 89.79 | YES | `TRUNCATED` | **YES** | Partial/interrupted session: 89.79s duration, both assets active, 14177 events. |
| 29 | `btc-updown-5m-1791296400` | NEW_RECENT | 456.84 | 456,582 | 298.76 | YES | `COMPLETE` | **YES** | Full active session: 298.76s duration, both assets active, 454608 quote events. |
| 30 | `btc-updown-5m-1791296700` | NEW_RECENT | 322.24 | 323,624 | 298.85 | YES | `COMPLETE` | **YES** | Full active session: 298.85s duration, both assets active, 322410 quote events. |
| 31 | `btc-updown-5m-1791297000` | NEW_RECENT | 80.75 | 81,093 | 41.1 | YES | `MINIMAL_TRUNCATED` | NO | Very short duration (<50s): 41.1s, insufficient for sustained multi-step sequence modeling. |

---

## 4. Original 21 Markets Immutability & Status Ledger

| # | Market Slug | Status in Phase 11B | Size (MB) | Raw Events | SHA-256 Match | Immutability |
| :- | :--- | :--- | :-: | :-: | :-: | :--- |
| 1 | `btc-updown-5m-1791204300` | `EXCLUDED` | 8.33 | 8,557 | `MATCH` | **IMMUTABLE** |
| 2 | `btc-updown-5m-1791204600` | `RETAINED` | 343.82 | 343,369 | `MATCH` | **IMMUTABLE** |
| 3 | `btc-updown-5m-1791204900` | `RETAINED` | 467.46 | 467,917 | `MATCH` | **IMMUTABLE** |
| 4 | `btc-updown-5m-1791205200` | `RETAINED` | 122.86 | 122,173 | `MATCH` | **IMMUTABLE** |
| 5 | `btc-updown-5m-1791205500` | `RETAINED` | 486.07 | 485,682 | `MATCH` | **IMMUTABLE** |
| 6 | `btc-updown-5m-1791205800` | `RETAINED` | 238.2 | 237,034 | `MATCH` | **IMMUTABLE** |
| 7 | `btc-updown-5m-1791206100` | `RETAINED` | 287.64 | 287,637 | `MATCH` | **IMMUTABLE** |
| 8 | `btc-updown-5m-1791206400` | `RETAINED` | 187.08 | 186,873 | `MATCH` | **IMMUTABLE** |
| 9 | `btc-updown-5m-1791206700` | `RETAINED` | 197.9 | 198,037 | `MATCH` | **IMMUTABLE** |
| 10 | `btc-updown-5m-1791207000` | `RETAINED` | 178.55 | 177,441 | `MATCH` | **IMMUTABLE** |
| 11 | `btc-updown-5m-1791207300` | `RETAINED` | 144.59 | 144,861 | `MATCH` | **IMMUTABLE** |
| 12 | `btc-updown-5m-1791207600` | `RETAINED` | 188.68 | 188,949 | `MATCH` | **IMMUTABLE** |
| 13 | `btc-updown-5m-1791207900` | `RETAINED` | 146.14 | 147,065 | `MATCH` | **IMMUTABLE** |
| 14 | `btc-updown-5m-1791208200` | `RETAINED` | 198.09 | 198,717 | `MATCH` | **IMMUTABLE** |
| 15 | `btc-updown-5m-1791208500` | `RETAINED` | 206.8 | 205,888 | `MATCH` | **IMMUTABLE** |
| 16 | `btc-updown-5m-1791208800` | `EXCLUDED` | 49.83 | 49,527 | `MATCH` | **IMMUTABLE** |
| 17 | `btc-updown-5m-1791209400` | `RETAINED` | 120.2 | 120,261 | `MATCH` | **IMMUTABLE** |
| 18 | `btc-updown-5m-1791209700` | `RETAINED` | 182.85 | 182,624 | `MATCH` | **IMMUTABLE** |
| 19 | `btc-updown-5m-1791210000` | `RETAINED` | 129.33 | 129,945 | `MATCH` | **IMMUTABLE** |
| 20 | `btc-updown-5m-1791210300` | `RETAINED` | 188.05 | 188,828 | `MATCH` | **IMMUTABLE** |
| 21 | `btc-updown-5m-1791210600` | `RETAINED` | 133.39 | 133,991 | `MATCH` | **IMMUTABLE** |

---

## 5. Collection Integrity Findings

1. **Zero Raw Mutation**: All 21 original files match their Phase 11B SHA-256 hashes bit-for-bit. Raw files have remained completely untouched.
2. **Zero JSON Syntax Corruption**: 0 malformed lines out of all raw event records scanned across 52 files.
3. **Companion Metadata Integrity**: All 52 market files have companion `.meta.json` files specifying `market`, `tokens.UP`, and `tokens.DOWN`.
4. **Duplicate Slugs / IDs**: 0 duplicate slugs detected. 0 market ID collisions.
5. **Interrupted Run Verification**: `btc-updown-5m-1791295500.jsonl` contains exactly `141,395` lines (141.19 MB, span = 83.81s), verifying that the ~141k events are fully persisted.
6. **Usable New Data**: Of the 31 new markets, **29 markets** have valid two-sided quote series and represent active trading sessions.
