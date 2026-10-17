# Prospective Shadow Collector Health Report — Session `session_live_smoke_001`

> [!IMPORTANT]
> **RESEARCH-ONLY PROSPECTIVE TELEMETRY**: Zero real capital deployed; zero live orders sent to exchange.
> All hypothetical PnL figures are simulated paper executions under conservative bid/ask crossing and 5 bps taker fees.

---

## 1. Session Health & Telemetry

| Telemetry Metric | Session Value | Quality Benchmark |
| :--- | :---: | :--- |
| **Start Timestamp (UTC)** | `2026-10-09T12:05:47.548721+00:00` | Recorded |
| **End Timestamp (UTC)** | `2026-10-09T12:06:22.612573+00:00` | Recorded |
| **WebSocket Raw Messages** | `24,137` | Active feed connectivity |
| **1-Second Decisions Logged** | `15` | Grid synchronization |
| **Trades Triggered (EXECUTE)** | `1` | Evaluated under frozen regime gates |
| **Trades Suppressed by Gates** | `14` | Strict regime gating |
| **Predictions Skipped (FLAT)** | `0` | Non-directional model output |
| **Stale Quote Events** | `0` | Quote age > 2000ms threshold |

---

## 2. Invariants & Safety Verification

- [x] **Zero Live Orders**: Process operates strictly as a read-only WebSocket subscriber.
- [x] **Model Weights Frozen**: SmallLSTM weights remained bitwise identical (SHA-256 verified).
- [x] **Locked Test Isolation**: `test_scaled.npz` was never accessed or opened.
- [x] **Append-Only Logging**: Every observation hashed with SHA-256 rolling digest.
- [x] **Observation Log File**: `session_live_smoke_001_observations.jsonl` (SHA-256: `835f68834c59588b...`).

---

## 3. How to Verify Freshness & Completeness

1. Inspect the append-only JSONL log: each record contains `decision_timestamp_utc` and matching `quote_timestamp_ms`.
2. Confirm that `quote_age_ms` is small (< 1000ms under normal network conditions).
3. Verify that `integrity_hash` matches `sha256(json_representation)` for every line.

